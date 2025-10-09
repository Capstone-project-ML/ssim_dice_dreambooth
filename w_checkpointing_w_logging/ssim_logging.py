# train_ssim_with_logging_and_resume.py

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
import bitsandbytes as bnb
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torch.utils.tensorboard import SummaryWriter

def parse_args():
    parser = argparse.ArgumentParser(description="DreamBooth fine-tuning with SSIM loss.")
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-2-base", help="Pretrained model ID.")
    parser.add_argument("--instance_data_dir", type=str, required=True, help="Path to your training images.")
    parser.add_argument("--output_dir", type=str, default="./results_ssim", help="Directory to save the model and logs.")
    parser.add_argument("--unique_token", type=str, default="<nih-xray>", help="Unique token for your concept.")
    parser.add_argument("--class_token", type=str, default="x-ray", help="General class for prior preservation.")
    parser.add_argument("--class_data_dir", type=str, default="./class_images_xray", help="Directory to cache generated class images.")
    parser.add_argument("--num_class_images", type=int, default=200, help="Number of class images for prior preservation.")
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="Weight for prior preservation loss.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=2e-6, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--ssim_loss_weight", type=float, default=0.1, help="Weight for the SSIM loss.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of steps to accumulate gradients before updating.")
    return parser.parse_args()

class DreamBoothDataset(Dataset):
    def __init__(self, instance_data_root, class_data_root, tokenizer, instance_prompt, class_prompt, size=512):
        self.tokenizer = tokenizer
        self.instance_prompt = instance_prompt
        self.class_prompt = class_prompt
        self.size = size
        self.instance_images_path = [os.path.join(instance_data_root, f) for f in os.listdir(instance_data_root) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.num_instance_images = len(self.instance_images_path)

        self.class_images_path = [os.path.join(class_data_root, f) for f in os.listdir(class_data_root) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.num_class_images = len(self.class_images_path)
        self._length = max(self.num_instance_images, self.num_class_images)
        self.image_transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image_path = self.instance_images_path[index % self.num_instance_images]
        instance_image = Image.open(instance_image_path).convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_prompt_ids"] = self.tokenizer(self.instance_prompt, padding="do_not_pad", truncation=True, max_length=self.tokenizer.model_max_length).input_ids

        if self.num_class_images > 0:
            class_image = Image.open(self.class_images_path[index % self.num_class_images]).convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenizer(self.class_prompt, padding="do_not_pad", truncation=True, max_length=self.tokenizer.model_max_length).input_ids

        return example

def collate_fn(examples, with_prior_preservation=True):
    input_ids = [e["instance_prompt_ids"] for e in examples]
    pixel_values = [e["instance_images"] for e in examples]

    if with_prior_preservation:
        input_ids += [e["class_prompt_ids"] for e in examples]
        pixel_values += [e["class_images"] for e in examples]

    pixel_values = torch.stack(pixel_values).to(memory_format=torch.contiguous_format).float()
    padded_input_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(ids, dtype=torch.long) for ids in input_ids], batch_first=True, padding_value=49407)

    return {"pixel_values": pixel_values, "input_ids": padded_input_ids}


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.class_data_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))

    instance_prompt = f"a photo of {args.unique_token}"
    class_prompt = f"a photo of a {args.class_token}"

    tokenizer = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.model_id, subfolder="text_encoder").to(device)
    
    num_added_tokens = tokenizer.add_tokens(args.unique_token)
    if num_added_tokens == 0:
        raise ValueError(f"The tokenizer already contains the token {args.unique_token}. Please pass a different `unique_token`.")
    text_encoder.resize_token_embeddings(len(tokenizer))
    token_embeds = text_encoder.get_input_embeddings().weight.data
    token_embeds[token_embeds.shape[0] - 1] = token_embeds[tokenizer.encode(args.class_token, add_special_tokens=False)[0]]

    unet = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet").to(device)
    unet.enable_gradient_checkpointing() 
    
    try:
        unet.enable_xformers_memory_efficient_attention()
        print("xFormers enabled for memory efficiency.")
    except ImportError:
        print("xFormers is not installed. For better memory efficiency, consider installing it.")

    vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae").to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(args.model_id, subfolder="scheduler")
    
    vae.requires_grad_(False)
    
    if args.num_class_images > 0:
        class_images_dir = args.class_data_dir
        if not os.path.exists(class_images_dir): os.makedirs(class_images_dir)
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

    train_dataset = DreamBoothDataset(args.instance_data_dir, args.class_data_dir, tokenizer, instance_prompt, class_prompt)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda examples: collate_fn(examples))
    
    optimizer = bnb.optim.AdamW8bit(
        itertools.chain(unet.parameters(), text_encoder.get_input_embeddings().parameters()),
        lr=args.learning_rate,
    )
    
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_dataloader) * args.num_epochs,
    )

    scaler = torch.amp.GradScaler('cuda')
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    # --- [NEW] CHECK FOR CHECKPOINTS AND RESUME ---
    initial_epoch = 0
    global_step_resume = 0
    if os.path.isdir(args.output_dir):
        checkpoint_dirs = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
        if checkpoint_dirs:
            latest_epoch = -1
            latest_checkpoint_dir = None
            for d in checkpoint_dirs:
                try:
                    epoch_num = int(re.search(r'checkpoint-(\d+)', d).group(1))
                    if epoch_num > latest_epoch:
                        latest_epoch = epoch_num
                        latest_checkpoint_dir = d
                except (AttributeError, ValueError):
                    continue
            
            if latest_checkpoint_dir:
                checkpoint_path = os.path.join(args.output_dir, latest_checkpoint_dir)
                print(f"Resuming training from checkpoint: {checkpoint_path}")
                # Load the models from the checkpoint
                unet = UNet2DConditionModel.from_pretrained(checkpoint_path, subfolder="unet").to(device)
                text_encoder = CLIPTextModel.from_pretrained(checkpoint_path, subfolder="text_encoder").to(device)
                # Set the starting epoch for the training loop
                initial_epoch = latest_epoch
                # Set the global step to continue logging correctly
                global_step_resume = initial_epoch * len(train_dataloader)
    # --- END OF RESUME LOGIC ---

    # --- TRAINING LOOP ---
    global_step = global_step_resume
    for epoch in range(initial_epoch, args.num_epochs):
        unet.train()
        text_encoder.get_input_embeddings().train()

        progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}")

        for step, batch in enumerate(train_dataloader):
            with torch.no_grad():
                latents = vae.encode(batch["pixel_values"].to(device, dtype=torch.float32)).latent_dist.sample() * vae.config.scaling_factor

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            with torch.amp.autocast('cuda'):
                encoder_hidden_states = text_encoder(batch["input_ids"].to(device))[0]
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                noise_pred_instance, noise_pred_class = noise_pred.chunk(2, dim=0)
                noise_instance, noise_class = noise.chunk(2, dim=0)

                instance_loss = torch.nn.functional.mse_loss(noise_pred_instance.float(), noise_instance.float())
                class_loss = torch.nn.functional.mse_loss(noise_pred_class.float(), noise_class.float())
                dreambooth_loss = instance_loss + args.prior_loss_weight * class_loss

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
                pred_instance_latents, _ = pred_original_sample.chunk(2, dim=0)

                with torch.no_grad():
                    pred_images = vae.decode(pred_instance_latents / vae.config.scaling_factor, return_dict=False)[0]
                    pred_images_norm = (pred_images + 1.0) / 2.0 
                
                gt_images, _ = batch["pixel_values"].chunk(2, dim=0)
                gt_images_norm = (gt_images.to(device) + 1.0) / 2.0

                ssim_score = ssim_metric(pred_images_norm, gt_images_norm)
                ssim_loss = 1.0 - ssim_score 

                total_loss = dreambooth_loss + args.ssim_loss_weight * ssim_loss
                
                writer.add_scalar("Loss/total", total_loss.detach().item(), global_step)
                writer.add_scalar("Loss/dreambooth", dreambooth_loss.detach().item(), global_step)
                writer.add_scalar("Loss/ssim", ssim_loss.detach().item(), global_step)
                writer.add_scalar("Metric/SSIM_Score", ssim_score.detach().item(), global_step)
                writer.add_scalar("LearningRate", lr_scheduler.get_last_lr()[0], global_step)

                loss = total_loss / args.gradient_accumulation_steps

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

        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{epoch + 1}")
        pipeline = StableDiffusionPipeline.from_pretrained(args.model_id, unet=unet, text_encoder=text_encoder, tokenizer=tokenizer)
        pipeline.save_pretrained(checkpoint_dir)
        print(f"Saved checkpoint for epoch {epoch + 1} at {checkpoint_dir}")

    print("Saving final model...")
    pipeline = StableDiffusionPipeline.from_pretrained(args.model_id, unet=unet, text_encoder=text_encoder, tokenizer=tokenizer)
    pipeline.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")

    writer.close()

if __name__ == "__main__":
    main()