# train_dice_with_logging_and_resume.py

import os
import argparse
import itertools
import torch
import re
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DDPMScheduler, AutoencoderKL
from diffusers.optimization import get_cosine_schedule_with_warmup
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
from tqdm.auto import tqdm
import segmentation_models_pytorch as smp
import bitsandbytes as bnb
from monai.losses import DiceLoss
from torch.utils.tensorboard import SummaryWriter
import json # <--- ADD THIS LINE for saving metadata

def parse_args():
    parser = argparse.ArgumentParser(description="DreamBooth fine-tuning with Dice loss.")
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-2-base", help="Pretrained model ID.")
    parser.add_argument("--instance_data_dir", type=str, required=True, help="Path to your training images.")
    parser.add_argument("--instance_mask_dir", type=str, required=True, help="Path to your training segmentation masks.")
    parser.add_argument("--output_dir", type=str, default="./results_dreambooth_dice", help="Directory to save the model and logs.")
    parser.add_argument("--seg_model_weights", type=str, required=True, help="Path to pre-trained segmentation model weights.")
    parser.add_argument("--unique_token", type=str, default="<nih-xray>", help="Unique token for your concept.")
    parser.add_argument("--class_token", type=str, default="x-ray", help="General class for prior preservation.")
    parser.add_argument("--class_data_dir", type=str, default="./class_images_xray", help="Directory to cache generated class images.")
    parser.add_argument("--num_class_images", type=int, default=200, help="Number of class images for prior preservation.")
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="Weight for prior preservation loss.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=2e-6, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--dice_loss_weight", type=float, default=0.1, help="Weight for the Dice loss.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of steps to accumulate gradients before updating.")
    return parser.parse_args()

class DreamBoothDataset(Dataset):
    def __init__(self, instance_data_root, instance_mask_root, class_data_root, tokenizer, instance_prompt, class_prompt, size=512):
        self.tokenizer = tokenizer
        self.instance_prompt = instance_prompt
        self.class_prompt = class_prompt
        self.size = size
        self.instance_images_path = [os.path.join(instance_data_root, f) for f in os.listdir(instance_data_root) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.instance_masks_path = [os.path.join(instance_mask_root, f) for f in os.listdir(instance_mask_root) if f.endswith(('.png', '.jpg', '.jpeg'))]
        # Filter mask paths to only include masks for existing images (assuming file names match)
        instance_filenames = set(os.path.basename(p) for p in self.instance_images_path)
        self.instance_masks_path = [p for p in self.instance_masks_path if os.path.basename(p) in instance_filenames]
        self.num_instance_images = len(self.instance_images_path)
        self.class_images_path = [os.path.join(class_data_root, f) for f in os.listdir(class_data_root) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.num_class_images = len(self.class_images_path)
        self._length = max(self.num_instance_images, self.num_class_images)
        self.image_transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5]),
        ])
        self.mask_transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop(size), transforms.ToTensor(),
        ])

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        # Ensure image and mask path indexing wraps correctly and are aligned
        instance_image_path = self.instance_images_path[index % self.num_instance_images]
        
        # Simple mask path derivation (assumes mask file name matches image file name)
        # This is a potential point of failure if file names are not strictly aligned.
        # A safer approach would be to store (image_path, mask_path) tuples.
        instance_mask_path = os.path.join(os.path.dirname(self.instance_masks_path[0]), os.path.basename(instance_image_path))


        instance_image = Image.open(instance_image_path).convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_prompt_ids"] = self.tokenizer(self.instance_prompt, padding="do_not_pad", truncation=True, max_length=self.tokenizer.model_max_length).input_ids
        
        try:
            instance_mask = Image.open(instance_mask_path).convert("L")
        except FileNotFoundError:
            # Handle case where mask might not exist for a given image, though required by the setup.
            # In a real scenario, this should raise an error or be handled during data prep.
            # Here, we create a black mask to avoid crashing (suboptimal).
            print(f"Warning: Mask not found for {os.path.basename(instance_image_path)}. Using black mask.")
            instance_mask = Image.new('L', (self.size, self.size), 0)
            
        example["instance_masks"] = self.mask_transforms(instance_mask)

        if self.num_class_images > 0:
            class_image = Image.open(self.class_images_path[index % self.num_class_images]).convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenizer(self.class_prompt, padding="do_not_pad", truncation=True, max_length=self.tokenizer.model_max_length).input_ids
        return example

def collate_fn(examples, with_prior_preservation=True):
    input_ids = [e["instance_prompt_ids"] for e in examples]
    pixel_values = [e["instance_images"] for e in examples]
    masks = [e["instance_masks"] for e in examples]
    if with_prior_preservation:
        input_ids += [e["class_prompt_ids"] for e in examples]
        pixel_values += [e["class_images"] for e in examples]
    pixel_values = torch.stack(pixel_values).to(memory_format=torch.contiguous_format).float()
    
    # Only instance masks are stacked. Class images will have no corresponding mask in this batch
    # which is handled later by only using the first half (instance data) for the Dice Loss.
    masks = torch.stack(masks)
    
    padded_input_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(ids, dtype=torch.long) for ids in input_ids], batch_first=True, padding_value=49407)
    return {"pixel_values": pixel_values, "input_ids": padded_input_ids, "masks": masks}


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.class_data_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))

    # --- MODEL LOADING AND SETUP ---
    # NOTE: The segmentation model's normalization is assumed to be handled by the fix below.
    seg_model = smp.Unet(encoder_name="resnet34", in_channels=1, classes=1).to(device)
    seg_model.load_state_dict(torch.load(args.seg_model_weights, map_location=device))
    seg_model.eval()
    for param in seg_model.parameters():
        param.requires_grad = False
    print("Loaded and froze pre-trained segmentation model.")
    dice_loss_fn = DiceLoss(sigmoid=True)
    instance_prompt = f"a photo of {args.unique_token}"
    class_prompt = f"a photo of a {args.class_token}"
    tokenizer = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.model_id, subfolder="text_encoder").to(device)
    num_added_tokens = tokenizer.add_tokens(args.unique_token)
    if num_added_tokens == 0:
        # Re-initialize tokenizer and text_encoder to handle the case where the token was already added 
        # but the token embedding was not correctly initialized in a previous run.
        # We proceed assuming the token is correctly handled by the checkpoint logic later.
        pass 
    else:
        text_encoder.resize_token_embeddings(len(tokenizer))
        # Initializing the new token embedding to the class token embedding
        token_embeds = text_encoder.get_input_embeddings().weight.data
        class_token_id = tokenizer.encode(args.class_token, add_special_tokens=False)[0]
        new_token_id = len(tokenizer) - 1
        token_embeds[new_token_id] = token_embeds[class_token_id]
        print(f"Added new token '{args.unique_token}' and initialized its embedding.")

    unet = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet").to(device)
    unet.enable_gradient_checkpointing()
    try:
        unet.enable_xformers_memory_efficient_attention()
        print("xFormers enabled for memory efficiency.")
    except ImportError:
        print("xFormers is not installed. For better memory efficiency, consider installing it.")
    vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae").to(device)
    vae.requires_grad_(False)
    noise_scheduler = DDPMScheduler.from_pretrained(args.model_id, subfolder="scheduler")

    # --- CLASS IMAGE GENERATION ---
    if args.num_class_images > 0:
        class_images_dir = args.class_data_dir
        if not os.path.exists(class_images_dir):
            os.makedirs(class_images_dir)
        cur_class_images = len(os.listdir(class_images_dir))
        if cur_class_images < args.num_class_images:
            pipeline = StableDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch.float16).to(device)
            pipeline.set_progress_bar_config(disable=True)
            num_new_images = args.num_class_images - cur_class_images
            print(f"Generating {num_new_images} class images...")
            for i in tqdm(range(num_new_images)):
                image = pipeline(class_prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
                image.save(os.path.join(class_images_dir, f"{i + cur_class_images}.jpg"))
            del pipeline
            torch.cuda.empty_cache()
    
    train_dataset = DreamBoothDataset(args.instance_data_dir, args.instance_mask_dir, args.class_data_dir, tokenizer, instance_prompt, class_prompt)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda examples: collate_fn(examples))
    
    # --- OPTIMIZER AND SCALER INITIALIZATION (BEFORE RESUME) ---
    optimizer = bnb.optim.AdamW8bit(
        itertools.chain(unet.parameters(), text_encoder.parameters()),
        lr=args.learning_rate,
    )
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_dataloader) * args.num_epochs,
    )
    scaler = torch.amp.GradScaler('cuda')

    # --- CHECK FOR CHECKPOINTS AND RESUME ---
    initial_epoch = 0
    global_step_resume = 0
    if os.path.isdir(args.output_dir):
        checkpoint_dirs = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
        if checkpoint_dirs:
            latest_epoch = -1
            latest_checkpoint_dir = None
            for d in checkpoint_dirs:
                try:
                    match = re.search(r'checkpoint-(\d+)', d)
                    if match:
                        epoch_num = int(match.group(1))
                        if epoch_num > latest_epoch:
                            latest_epoch = epoch_num
                            latest_checkpoint_dir = d
                except (AttributeError, ValueError):
                    continue
            
            if latest_checkpoint_dir:
                checkpoint_path = os.path.join(args.output_dir, latest_checkpoint_dir)
                print(f"Resuming training from checkpoint: {checkpoint_path}")
                
                # 1. Load Models
                unet = UNet2DConditionModel.from_pretrained(checkpoint_path, subfolder="unet").to(device)
                text_encoder = CLIPTextModel.from_pretrained(checkpoint_path, subfolder="text_encoder").to(device)

                # 2. Load Optimizer, Scheduler, and Scaler State
                optim_state_path = os.path.join(checkpoint_path, "optimizer.pt")
                scaler_state_path = os.path.join(checkpoint_path, "scaler.pt")
                metadata_path = os.path.join(checkpoint_path, "metadata.json")

                if os.path.exists(optim_state_path) and os.path.exists(scaler_state_path) and os.path.exists(metadata_path):
                    optimizer.load_state_dict(torch.load(optim_state_path, map_location=device))
                    scaler.load_state_dict(torch.load(scaler_state_path, map_location=device))
                    
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    initial_epoch = metadata.get("epoch", latest_epoch)
                    global_step_resume = metadata.get("global_step", initial_epoch * len(train_dataloader))
                    
                    # Update LR scheduler's state
                    lr_scheduler = get_cosine_schedule_with_warmup(
                        optimizer=optimizer,
                        num_warmup_steps=0,
                        num_training_steps=len(train_dataloader) * args.num_epochs,
                    )
                    # Advance the scheduler to the loaded step
                    for _ in range(global_step_resume):
                        lr_scheduler.step()
                    
                    print(f"Loaded optimizer, scaler, and scheduler state. Starting from epoch {initial_epoch}, global step {global_step_resume}")
                else:
                    initial_epoch = latest_epoch
                    global_step_resume = initial_epoch * len(train_dataloader)
                    print("Could not find optimizer/scaler state files, starting from the beginning of the epoch with a fresh optimizer/scaler.")
    # --- END OF RESUME LOGIC ---

    # --- TRAINING LOOP ---
    global_step = global_step_resume
    for epoch in range(initial_epoch, args.num_epochs):
        unet.train()
        text_encoder.train()
        progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}")
        for step, batch in enumerate(train_dataloader):
            
            # Skip steps already processed if resuming mid-epoch
            if step < (global_step_resume % len(train_dataloader)) and epoch == initial_epoch:
                progress_bar.update(1)
                continue

            with torch.no_grad():
                latents = vae.encode(batch["pixel_values"].to(device, dtype=torch.float32)).latent_dist.sample() * vae.config.scaling_factor
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            with torch.amp.autocast('cuda'):
                # 1. Calculate DreamBooth Loss
                encoder_hidden_states = text_encoder(batch["input_ids"].to(device))[0]
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                
                noise_pred_instance, noise_pred_class = noise_pred.chunk(2, dim=0)
                noise_instance, noise_class = noise.chunk(2, dim=0)
                
                instance_loss = torch.nn.functional.mse_loss(noise_pred_instance.float(), noise_instance.float())
                class_loss = torch.nn.functional.mse_loss(noise_pred_class.float(), noise_class.float())
                dreambooth_loss = instance_loss + args.prior_loss_weight * class_loss

                # 2. Calculate Dice Loss
                # Estimate the original image (x_0) from the predicted noise
                alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)
                sqrt_alpha_prod = alphas_cumprod[timesteps].sqrt()
                sqrt_alpha_prod = sqrt_alpha_prod.flatten()
                while len(sqrt_alpha_prod.shape) < len(noisy_latents.shape):
                    sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
                
                sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]).sqrt()
                sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
                while len(sqrt_one_minus_alpha_prod.shape) < len(noisy_latents.shape):
                    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
                
                pred_original_sample = (noisy_latents - sqrt_one_minus_alpha_prod * noise_pred) / sqrt_alpha_prod
                
                # Only use instance data for the segmentation mask comparison (first half of batch)
                pred_instance_latents, _ = pred_original_sample.chunk(2, dim=0) 
                
                # Decode predicted instance latents to pixel space (output is in [-1, 1])
                with torch.no_grad():
                    pred_images = vae.decode(pred_instance_latents / vae.config.scaling_factor, return_dict=False)[0]
                
                # *** THE CRITICAL FIX IS HERE ***
                # Rescale VAE output from [-1, 1] to [0, 1] for the segmentation model
                pred_images_unnormalized = (pred_images / 2 + 0.5).clamp(0, 1) 
                
                gt_masks = batch["masks"].to(device)
                
                # Convert to Grayscale (1 channel)
                pred_images_gray = transforms.functional.rgb_to_grayscale(pred_images_unnormalized)
                
                # Get mask prediction logits from frozen segmentation model
                pred_mask_logits = seg_model(pred_images_gray)
                
                # Calculate Dice Loss
                dice_loss = dice_loss_fn(pred_mask_logits, gt_masks)
                
                # Total Loss
                total_loss = dreambooth_loss + args.dice_loss_weight * dice_loss
                
                # --- LOGGING ---
                writer.add_scalar("Loss/total", total_loss.detach().item(), global_step)
                writer.add_scalar("Loss/dreambooth", dreambooth_loss.detach().item(), global_step)
                writer.add_scalar("Loss/instance", instance_loss.detach().item(), global_step)
                writer.add_scalar("Loss/class", class_loss.detach().item(), global_step)
                writer.add_scalar("Loss/dice", dice_loss.detach().item(), global_step)
                writer.add_scalar("LearningRate", lr_scheduler.get_last_lr()[0], global_step)
                
                with torch.no_grad():
                    # Calculate Dice Score metric (not part of loss)
                    pred_masks_prob = torch.sigmoid(pred_mask_logits)
                    intersection = torch.sum(pred_masks_prob * gt_masks)
                    union = torch.sum(pred_masks_prob) + torch.sum(gt_masks)
                    dice_score = (2. * intersection) / (union + 1e-6)
                    writer.add_scalar("Metric/DiceScore", dice_score.item(), global_step)
                
                loss = total_loss / args.gradient_accumulation_steps
            
            # --- BACKWARD PASS ---
            scaler.scale(loss).backward()
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            
            progress_bar.set_postfix(loss=total_loss.detach().item())
            progress_bar.update(1)
            global_step += 1
        
        progress_bar.close()
        
        # --- CHECKPOINT SAVING LOGIC (SAVE OPTIMIZER AND SCALER) ---
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{epoch + 1}")
        
        # 1. Save Models
        # Need to ensure the tokenizer from the checkpoint loading is passed if it was updated
        pipeline = StableDiffusionPipeline.from_pretrained(args.model_id, unet=unet, text_encoder=text_encoder, tokenizer=tokenizer)
        pipeline.save_pretrained(checkpoint_dir)
        
        # 2. Save Optimizer, Scaler, and Metadata State
        torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
        torch.save(scaler.state_dict(), os.path.join(checkpoint_dir, "scaler.pt"))
        
        metadata = {"epoch": epoch + 1, "global_step": global_step}
        with open(os.path.join(checkpoint_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f)
            
        print(f"Saved checkpoint for epoch {epoch + 1} at {checkpoint_dir}")
        
    print("Saving final model...")
    pipeline = StableDiffusionPipeline.from_pretrained(args.model_id, unet=unet, text_encoder=text_encoder, tokenizer=tokenizer)
    pipeline.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")
    writer.close()

if __name__ == "__main__":
    main()
