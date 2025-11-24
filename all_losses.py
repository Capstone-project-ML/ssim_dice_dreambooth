# train_dreambooth_holistic_loss.py

import os
import argparse
import itertools
import torch
import re
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import vgg16, VGG16_Weights
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DDPMScheduler, AutoencoderKL
from diffusers.optimization import get_cosine_schedule_with_warmup
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
from tqdm.auto import tqdm
from torchmetrics.functional import structural_similarity_index_measure as ssim
import bitsandbytes as bnb
from monai.losses import DiceLoss
import segmentation_models_pytorch as smp
from torch.utils.tensorboard import SummaryWriter

# --- NEW LOSS CLASSES ---

class VGGPerceptualLoss(torch.nn.Module):
    """Computes Perceptual Loss using VGG16 features."""
    def __init__(self, device):
        super().__init__()
        # Load VGG16, drop the classifier, use only features
        self.vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16].to(device).eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        # ImageNet normalization statistics
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))

    def forward(self, input, target):
        # Input is [-1, 1], convert to [0, 1] for VGG
        input = (input + 1) / 2
        target = (target + 1) / 2
        
        # Normalize for VGG
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        
        input_features = self.vgg(input)
        target_features = self.vgg(target)
        return F.mse_loss(input_features, target_features)

def compute_edge_loss(pred, target):
    """Computes loss based on Laplacian Edge Detection."""
    # Convert to grayscale for edge detection
    if pred.shape[1] == 3:
        weights = torch.tensor([0.299, 0.587, 0.114], device=pred.device).view(1, 3, 1, 1)
        pred_gray = (pred * weights).sum(dim=1, keepdim=True)
        target_gray = (target * weights).sum(dim=1, keepdim=True)
    else:
        pred_gray = pred
        target_gray = target

    # Laplacian Kernel
    kernel = torch.tensor([[[[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]]], device=pred.device)
    
    pred_edges = F.conv2d(pred_gray, kernel, padding=1)
    target_edges = F.conv2d(target_gray, kernel, padding=1)
    
    return F.mse_loss(pred_edges, target_edges)

def compute_freq_loss(pred, target):
    """Computes Frequency Loss using FFT."""
    # FFT requires real inputs
    pred_fft = torch.fft.rfft2(pred)
    target_fft = torch.fft.rfft2(target)
    
    # Compare magnitude spectra
    loss = F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))
    return loss

# ------------------------

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="DreamBooth fine-tuning with Holistic Loss Function.")
    
    # Core DreamBooth arguments
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-2-base", help="Pretrained model ID.")
    parser.add_argument("--instance_data_dir", type=str, required=True, help="Path to your training images.")
    parser.add_argument("--output_dir", type=str, default="./results_holistic", help="Directory to save the model and logs.")
    parser.add_argument("--unique_token", type=str, default="<nih-xray>", help="Unique token for your concept.")
    
    # Prior Preservation arguments (The 'Cls' Loss)
    parser.add_argument("--class_token", type=str, default="x-ray", help="General class for prior preservation.")
    parser.add_argument("--class_data_dir", type=str, default="./class_images_xray", help="Directory to cache generated class images.")
    parser.add_argument("--num_class_images", type=int, default=200, help="Number of class images for prior preservation.")
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="Weight for prior preservation loss (Cls).")
    
    # Training hyperparameters
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=2e-6, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of steps to accumulate gradients.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")

    # Segmentation (Dice) Args
    parser.add_argument("--instance_mask_dir", type=str, required=True, help="Path to training segmentation masks.")
    parser.add_argument("--seg_model_weights", type=str, required=True, help="Path to pre-trained segmentation model weights.")
    
    # --- WEIGHTS FOR NEW LOSSES ---
    parser.add_argument("--weight_seg", type=float, default=0.1, help="Weight for Segmentation (Dice) loss.")
    parser.add_argument("--weight_ssim", type=float, default=0.1, help="Weight for SSIM loss.")
    parser.add_argument("--weight_pix", type=float, default=0.2, help="Weight for Pixel Reconstruction (L1/MSE) loss.")
    parser.add_argument("--weight_perc", type=float, default=0.1, help="Weight for Perceptual (VGG) loss.")
    parser.add_argument("--weight_edge", type=float, default=0.1, help="Weight for Edge/Gradient loss.")
    parser.add_argument("--weight_freq", type=float, default=0.1, help="Weight for Frequency/Spectrum loss.")
    
    return parser.parse_args()

class DreamBoothDataset(Dataset):
    """Dataset for DreamBooth with instance images, class images, and instance masks."""
    def __init__(self, instance_data_root, instance_mask_root, class_data_root, tokenizer, instance_prompt, class_prompt, size=512):
        self.tokenizer = tokenizer
        self.instance_prompt = instance_prompt
        self.class_prompt = class_prompt
        self.size = size
        
        self.instance_images_path = [os.path.join(instance_data_root, f) for f in os.listdir(instance_data_root) if f.endswith(('.png', '.jpg', '.jpeg'))]
        # Match masks to images by filename
        self.instance_masks_path = [os.path.join(instance_mask_root, os.path.basename(f)) for f in self.instance_images_path]
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
        
        instance_image_path = self.instance_images_path[index % self.num_instance_images]
        instance_image = Image.open(instance_image_path).convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_prompt_ids"] = self.tokenizer(self.instance_prompt, padding="do_not_pad", truncation=True, max_length=self.tokenizer.model_max_length).input_ids

        instance_mask_path = self.instance_masks_path[index % self.num_instance_images]
        instance_mask = Image.open(instance_mask_path).convert("L")
        example["instance_masks"] = self.mask_transforms(instance_mask)

        if self.num_class_images > 0:
            class_image = Image.open(self.class_images_path[index % self.num_class_images]).convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenizer(self.class_prompt, padding="do_not_pad", truncation=True, max_length=self.tokenizer.model_max_length).input_ids
        
        return example

def collate_fn(examples, pad_token_id, with_prior_preservation=True):
    """Collates examples into a batch, including masks."""
    input_ids = [e["instance_prompt_ids"] for e in examples]
    pixel_values = [e["instance_images"] for e in examples]
    masks = [e["instance_masks"] for e in examples]

    if with_prior_preservation:
        input_ids += [e["class_prompt_ids"] for e in examples]
        pixel_values += [e["class_images"] for e in examples]
    
    pixel_values = torch.stack(pixel_values).to(memory_format=torch.contiguous_format).float()
    masks = torch.stack(masks)
    
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(ids, dtype=torch.long) for ids in input_ids],
        batch_first=True,
        padding_value=pad_token_id
    )
    
    return {"pixel_values": pixel_values, "input_ids": input_ids, "masks": masks}


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.class_data_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))

    # --- 1. SETUP AUXILIARY MODELS ---
    
    # Segmentation (Seg)
    print("Loading segmentation model...")
    seg_model = smp.Unet(encoder_name="resnet34", in_channels=1, classes=1).to(device)
    seg_model.load_state_dict(torch.load(args.seg_model_weights, map_location=device))
    seg_model.eval()
    for param in seg_model.parameters(): param.requires_grad = False
    dice_loss_fn = DiceLoss(sigmoid=True)
    
    # Perceptual (Perc)
    print("Loading VGG for perceptual loss...")
    perceptual_loss_fn = VGGPerceptualLoss(device)

    # --- 2. SETUP DIFFUSION MODEL COMPONENTS ---
    instance_prompt = f"a photo of {args.unique_token}"
    class_prompt = f"a photo of a {args.class_token}"

    tokenizer = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    text_encoder = CLIPTextModel.from_pretrained(args.model_id, subfolder="text_encoder").to(device)

    num_added_tokens = tokenizer.add_tokens(args.unique_token)
    if num_added_tokens > 0:
        text_encoder.resize_token_embeddings(len(tokenizer))
        token_embeds = text_encoder.get_input_embeddings().weight.data
        class_token_id = tokenizer.convert_tokens_to_ids(args.class_token)
        new_token_embed = token_embeds[class_token_id]
        new_token_id = tokenizer.convert_tokens_to_ids(args.unique_token)
        token_embeds[new_token_id] = new_token_embed.clone()
    
    unet = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet").to(device)
    vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae").to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(args.model_id, subfolder="scheduler")
    
    vae.requires_grad_(False)
    
    unet.enable_gradient_checkpointing()
    try:
        unet.enable_xformers_memory_efficient_attention()
        print("xFormers enabled.")
    except ImportError:
        pass

    # --- 3. GENERATE CLASS IMAGES ---
    num_current_class_images = len(os.listdir(args.class_data_dir))
    if num_current_class_images < args.num_class_images:
        print(f"Generating {args.num_class_images - num_current_class_images} class images...")
        class_pipeline = StableDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch.float16, safety_checker=None).to(device)
        class_pipeline.set_progress_bar_config(disable=True)
        for i in tqdm(range(args.num_class_images - num_current_class_images)):
            image = class_pipeline(class_prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
            image.save(os.path.join(args.class_data_dir, f"{i + num_current_class_images}.png"))
        del class_pipeline
        torch.cuda.empty_cache()

    # --- 4. SETUP DATALOADER AND OPTIMIZER ---
    train_dataset = DreamBoothDataset(args.instance_data_dir, args.instance_mask_dir, args.class_data_dir, tokenizer, instance_prompt, class_prompt)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, tokenizer.pad_token_id)
    )

    optimizer = bnb.optim.AdamW8bit(
        itertools.chain(unet.parameters(), text_encoder.parameters()),
        lr=args.learning_rate,
    )

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_dataloader) * args.num_epochs,
    )
    
    scaler = torch.cuda.amp.GradScaler()

    # Resume Logic (Condensed)
    initial_epoch = 0
    global_step = 0
    if os.path.isdir(args.output_dir):
        checkpoint_dirs = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
        if checkpoint_dirs:
            # Sort loosely by number
            checkpoint_dirs.sort(key=lambda x: int(x.split('-')[1]))
            latest_dir = checkpoint_dirs[-1]
            state_file = os.path.join(args.output_dir, latest_dir, "training_state.pth")
            if os.path.isfile(state_file):
                print(f"Resuming from {state_file}")
                cp = torch.load(state_file, map_location=device)
                unet.load_state_dict(cp['unet_state_dict'])
                text_encoder.load_state_dict(cp['text_encoder_state_dict'])
                optimizer.load_state_dict(cp['optimizer_state_dict'])
                initial_epoch = cp['epoch']
                global_step = cp['global_step']

    # --- 5. TRAINING LOOP ---
    for epoch in range(initial_epoch, args.num_epochs):
        unet.train()
        text_encoder.train()
        progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}")

        for step, batch in enumerate(train_dataloader):
            # 1. Base Diffusion Process (DDPM)
            with torch.no_grad():
                latents = vae.encode(batch["pixel_values"].to(device, dtype=torch.float32)).latent_dist.sample() * vae.config.scaling_factor

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            with torch.cuda.amp.autocast():
                encoder_hidden_states = text_encoder(batch["input_ids"].to(device))[0]
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                
                noise_pred_instance, noise_pred_class = noise_pred.chunk(2, dim=0)
                noise_instance, noise_class = noise.chunk(2, dim=0)
                
                # Loss 1: DDPM Instance Loss
                loss_ddpm_inst = torch.nn.functional.mse_loss(noise_pred_instance.float(), noise_instance.float())
                
                # Loss 2: Cls (Prior Preservation) Loss
                loss_cls = torch.nn.functional.mse_loss(noise_pred_class.float(), noise_class.float())
                
                loss_dreambooth = loss_ddpm_inst + args.prior_loss_weight * loss_cls

                # --- RECONSTRUCTION FOR AUXILIARY LOSSES ---
                # To compute Pix, Perc, Edge, Freq, SSIM, Seg, we must decode the predicted latents.
                # Use the analytic scheduler approximation (x_0 prediction)
                alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)
                sqrt_alpha_prod = alphas_cumprod[timesteps]**0.5
                sqrt_alpha_prod = sqrt_alpha_prod.flatten()
                while len(sqrt_alpha_prod.shape) < len(noisy_latents.shape):
                    sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

                sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps])**0.5
                sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
                while len(sqrt_one_minus_alpha_prod.shape) < len(noisy_latents.shape):
                    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
                
                # Predicted Original Latent (x_0)
                pred_original_sample = (noisy_latents - sqrt_one_minus_alpha_prod * noise_pred) / sqrt_alpha_prod
                pred_instance_latents, _ = pred_original_sample.chunk(2, dim=0)

                # Decode to Image Space
                pred_images = vae.decode(pred_instance_latents / vae.config.scaling_factor, return_dict=False)[0]
                
                # Get Ground Truth Instance Images
                pixel_values_instance, _ = batch["pixel_values"].to(device).chunk(2, dim=0)
                
                # Normalize to [0, 1] for metric calculations where needed
                pred_images_norm = torch.clamp((pred_images + 1) / 2, 0, 1)
                gt_images_norm = torch.clamp((pixel_values_instance + 1) / 2, 0, 1)

                # --- AUXILIARY LOSSES ---
                
                # Loss 3: SSIM
                val_ssim = ssim(pred_images_norm, gt_images_norm, data_range=1.0)
                loss_ssim = 1.0 - val_ssim

                # Loss 4: Seg (Dice Loss)
                gt_masks = batch["masks"].to(device)
                pred_images_gray = transforms.functional.rgb_to_grayscale(pred_images) # for UNet
                pred_mask_logits = seg_model(pred_images_gray)
                loss_seg = dice_loss_fn(pred_mask_logits, gt_masks)
                
                # Loss 5: Pix (Pixel Reconstruction L1)
                loss_pix = torch.nn.functional.l1_loss(pred_images, pixel_values_instance)
                
                # Loss 6: Perc (Perceptual VGG)
                loss_perc = perceptual_loss_fn(pred_images, pixel_values_instance)

                # Loss 7: Edge (Gradient Loss)
                loss_edge = compute_edge_loss(pred_images_norm, gt_images_norm)

                # Loss 8: Freq (Frequency Loss)
                loss_freq = compute_freq_loss(pred_images, pixel_values_instance)

                # --- TOTAL LOSS AGGREGATION ---
                total_loss = (
                    loss_dreambooth + 
                    (args.weight_ssim * loss_ssim) +
                    (args.weight_seg * loss_seg) +
                    (args.weight_pix * loss_pix) +
                    (args.weight_perc * loss_perc) +
                    (args.weight_edge * loss_edge) +
                    (args.weight_freq * loss_freq)
                )

                # --- LOGGING ---
                writer.add_scalar("Loss/Total", total_loss.item(), global_step)
                writer.add_scalar("Loss/DDPM_Inst", loss_ddpm_inst.item(), global_step)
                writer.add_scalar("Loss/Cls_Prior", loss_cls.item(), global_step)
                writer.add_scalar("Loss/SSIM", loss_ssim.item(), global_step)
                writer.add_scalar("Loss/Seg_Dice", loss_seg.item(), global_step)
                writer.add_scalar("Loss/Pix_L1", loss_pix.item(), global_step)
                writer.add_scalar("Loss/Perc_VGG", loss_perc.item(), global_step)
                writer.add_scalar("Loss/Edge", loss_edge.item(), global_step)
                writer.add_scalar("Loss/Freq", loss_freq.item(), global_step)
                writer.add_scalar("Metric/SSIM", val_ssim.item(), global_step)

                scaled_loss = total_loss / args.gradient_accumulation_steps

            scaler.scale(scaled_loss).backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            progress_bar.set_postfix(
                loss=total_loss.item(),
                ssim=val_ssim.item(),
                dice=loss_seg.item()
            )
            progress_bar.update(1)
            global_step += 1

        progress_bar.close()

        # Save Checkpoint
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{epoch + 1}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        pipeline = StableDiffusionPipeline.from_pretrained(args.model_id, unet=unet, text_encoder=text_encoder, tokenizer=tokenizer)
        pipeline.save_pretrained(checkpoint_dir)
        
        training_state = {
            'epoch': epoch + 1, 'global_step': global_step,
            'unet_state_dict': unet.state_dict(), 'text_encoder_state_dict': text_encoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        torch.save(training_state, os.path.join(checkpoint_dir, "training_state.pth"))
        print(f"Saved epoch {epoch + 1}")

    print("Saving final model...")
    pipeline.save_pretrained(args.output_dir)
    writer.close()

if __name__ == "__main__":
    main()
