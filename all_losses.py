# train_dreambooth_complete_loss.py

import os
import argparse
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
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

# --- NEW LOSS HELPER CLASSES ---

class PerceptualLoss(nn.Module):
    """VGG-based Perceptual Loss."""
    def __init__(self, device):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features
        # Extract features from specific layers (relu1_2, relu2_2, relu3_3, relu4_3)
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg[x])
        
        self.slice1.eval()
        self.slice2.eval()
        self.slice3.eval()
        self.slice4.eval()
        
        for param in self.parameters():
            param.requires_grad = False
            
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))

    def forward(self, input, target):
        # Input/Target are [0, 1]. Normalize for VGG.
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        
        h_x = input
        h_y = target
        loss = 0.0
        
        h_x = self.slice1(h_x)
        h_y = self.slice1(h_y)
        loss += F.mse_loss(h_x, h_y)
        
        h_x = self.slice2(h_x)
        h_y = self.slice2(h_y)
        loss += F.mse_loss(h_x, h_y)
        
        h_x = self.slice3(h_x)
        h_y = self.slice3(h_y)
        loss += F.mse_loss(h_x, h_y)
        
        h_x = self.slice4(h_x)
        h_y = self.slice4(h_y)
        loss += F.mse_loss(h_x, h_y)
        
        return loss

class EdgeLoss(nn.Module):
    """Sobel-based Edge/Gradient Loss."""
    def __init__(self, device):
        super().__init__()
        # Sobel kernels
        self.kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3).to(device)
        self.ky = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3).to(device)

    def forward(self, input, target):
        # Convert RGB to Grayscale for edge detection
        if input.shape[1] == 3:
            input_gray = 0.299 * input[:, 0, :, :] + 0.587 * input[:, 1, :, :] + 0.114 * input[:, 2, :, :]
            target_gray = 0.299 * target[:, 0, :, :] + 0.587 * target[:, 1, :, :] + 0.114 * target[:, 2, :, :]
            input_gray = input_gray.unsqueeze(1)
            target_gray = target_gray.unsqueeze(1)
        else:
            input_gray = input
            target_gray = target

        pred_gx = F.conv2d(input_gray, self.kx, padding=1)
        pred_gy = F.conv2d(input_gray, self.ky, padding=1)
        gt_gx = F.conv2d(target_gray, self.kx, padding=1)
        gt_gy = F.conv2d(target_gray, self.ky, padding=1)

        loss = F.l1_loss(pred_gx, gt_gx) + F.l1_loss(pred_gy, gt_gy)
        return loss

class FrequencyLoss(nn.Module):
    """Frequency/Spectrum Loss using FFT."""
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        # FFT2 returns complex tensor
        fft_input = torch.fft.rfft2(input, norm='ortho')
        fft_target = torch.fft.rfft2(target, norm='ortho')
        
        # Compare magnitude (spectrum)
        diff = torch.abs(fft_input) - torch.abs(fft_target)
        return torch.mean(diff ** 2)

class RadiomicsLoss(nn.Module):
    """Differentiable First-Order Radiomics (Statistics) Loss."""
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        # Calculate First Order Statistics: Mean, Variance, Skewness, Kurtosis
        # Input assumed [B, C, H, W]
        
        def get_moments(img):
            # Flatten spatial dims
            x = img.view(img.size(0), img.size(1), -1)
            mean = torch.mean(x, dim=2, keepdim=True)
            var = torch.var(x, dim=2, keepdim=True, unbiased=False)
            std = torch.sqrt(var + 1e-6)
            
            # Skewness
            centered = x - mean
            skew = torch.mean(centered ** 3, dim=2, keepdim=True) / (std ** 3 + 1e-6)
            
            # Kurtosis
            kurt = torch.mean(centered ** 4, dim=2, keepdim=True) / (std ** 4 + 1e-6)
            
            return mean.squeeze(), var.squeeze(), skew.squeeze(), kurt.squeeze()

        pred_mean, pred_var, pred_skew, pred_kurt = get_moments(input)
        gt_mean, gt_var, gt_skew, gt_kurt = get_moments(target)
        
        # Weighted sum of statistical differences
        loss_mean = F.mse_loss(pred_mean, gt_mean)
        loss_var = F.mse_loss(pred_var, gt_var)
        loss_skew = F.mse_loss(pred_skew, gt_skew)
        loss_kurt = F.mse_loss(pred_kurt, gt_kurt)
        
        return loss_mean + loss_var + 0.5 * loss_skew + 0.5 * loss_kurt

# -----------------------------

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="DreamBooth fine-tuning with a combined Multi-Objective loss.")
    
    # Core DreamBooth arguments
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-2-base", help="Pretrained model ID.")
    parser.add_argument("--instance_data_dir", type=str, required=True, help="Path to your training images.")
    parser.add_argument("--output_dir", type=str, default="./results_dreambooth_combined", help="Directory to save the model and logs.")
    parser.add_argument("--unique_token", type=str, default="<nih-xray>", help="Unique token for your concept.")
    
    # Prior Preservation arguments
    parser.add_argument("--class_token", type=str, default="x-ray", help="General class for prior preservation.")
    parser.add_argument("--class_data_dir", type=str, default="./class_images_xray", help="Directory to cache generated class images.")
    parser.add_argument("--num_class_images", type=int, default=200, help="Number of class images for prior preservation.")
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="Weight for prior preservation loss (cls).")
    
    # Training hyperparameters
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=2e-6, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of steps to accumulate gradients before updating.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")

    # Segmentation (Dice) Loss specific arguments
    parser.add_argument("--instance_mask_dir", type=str, required=True, help="Path to your training segmentation masks.")
    parser.add_argument("--seg_model_weights", type=str, required=True, help="Path to pre-trained segmentation model weights.")
    parser.add_argument("--seg_loss_weight", type=float, default=0.3, help="Weight for the Segmentation aware (Dice) loss [0.1-0.5].")
    
    # NEW LOSS WEIGHTS
    parser.add_argument("--pix_loss_weight", type=float, default=0.2, help="Weight for Pixel reconstruction loss [0.05-0.5].")
    parser.add_argument("--perc_loss_weight", type=float, default=0.1, help="Weight for Perceptual loss [0.05-0.3].")
    parser.add_argument("--ssim_loss_weight", type=float, default=0.05, help="Weight for SSIM loss [0.01-0.1].")
    parser.add_argument("--edge_loss_weight", type=float, default=0.1, help="Weight for Edge/Gradient loss.")
    parser.add_argument("--freq_loss_weight", type=float, default=0.1, help="Weight for Frequency/Spectrum loss.")
    parser.add_argument("--rad_loss_weight", type=float, default=0.1, help="Weight for Radiomics Feature loss.")
    
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
    
    # Initialize TensorBoard Writer
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))

    # --- 1. SETUP LOSS FUNCTIONS AND MODELS ---
    print("Initializing Multi-Objective Loss components...")
    
    # Segmentation Model (Dice)
    seg_model = smp.Unet(encoder_name="resnet34", in_channels=1, classes=1).to(device)
    seg_model.load_state_dict(torch.load(args.seg_model_weights, map_location=device))
    seg_model.eval()
    for param in seg_model.parameters():
        param.requires_grad = False
    dice_loss_fn = DiceLoss(sigmoid=True)
    
    # New Loss Modules
    perceptual_loss_fn = PerceptualLoss(device).to(device)
    edge_loss_fn = EdgeLoss(device).to(device)
    freq_loss_fn = FrequencyLoss().to(device)
    radiomics_loss_fn = RadiomicsLoss().to(device)
    
    print("Loss components initialized.")
    
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
        print("xFormers enabled for memory efficiency.")
    except ImportError:
        print("xFormers is not installed. For better memory efficiency, consider installing it.")

    # --- 3. GENERATE CLASS IMAGES IF NEEDED ---
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
                    epoch_num = int(re.search(r'checkpoint-(\d+)', d).group(1))
                    if epoch_num > latest_epoch:
                        state_file = os.path.join(args.output_dir, d, "training_state.pth")
                        if os.path.isfile(state_file):
                            latest_epoch = epoch_num
                            latest_checkpoint_dir = d
                except (AttributeError, ValueError):
                    continue
            
            if latest_checkpoint_dir:
                checkpoint_path = os.path.join(args.output_dir, latest_checkpoint_dir)
                training_state_path = os.path.join(checkpoint_path, "training_state.pth")
                
                print(f"Resuming training from checkpoint: {checkpoint_path}")
                
                try:
                    checkpoint = torch.load(training_state_path, map_location=device)
                    
                    unet.load_state_dict(checkpoint['unet_state_dict'])
                    text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
                    scaler.load_state_dict(checkpoint['scaler_state_dict'])
                    
                    initial_epoch = checkpoint['epoch']
                    global_step_resume = checkpoint['global_step']
                    
                    print(f"Successfully resumed from epoch {initial_epoch} at global step {global_step_resume}.")

                except Exception as e:
                    print(f"Could not load training state from {training_state_path}: {e}")
                    print("Starting training from scratch.")
                    initial_epoch = 0
                    global_step_resume = 0
            else:
                print("No valid 'training_state.pth' found in checkpoint directories. Starting from scratch.")

    # --- 5. TRAINING LOOP ---
    global_step = global_step_resume
    for epoch in range(initial_epoch, args.num_epochs):
        unet.train()
        text_encoder.train()
        progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}")

        for step, batch in enumerate(train_dataloader):
            with torch.no_grad():
                latents = vae.encode(batch["pixel_values"].to(device, dtype=torch.float32)).latent_dist.sample() * vae.config.scaling_factor

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            with torch.cuda.amp.autocast():
                # --- DDPM LOSS (Base) ---
                encoder_hidden_states = text_encoder(batch["input_ids"].to(device))[0]
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                
                noise_pred_instance, noise_pred_class = noise_pred.chunk(2, dim=0)
                noise_instance, noise_class = noise.chunk(2, dim=0)
                instance_loss = torch.nn.functional.mse_loss(noise_pred_instance.float(), noise_instance.float())
                class_loss = torch.nn.functional.mse_loss(noise_pred_class.float(), noise_class.float())
                
                # ddpm + cls
                dreambooth_loss = instance_loss + args.prior_loss_weight * class_loss

                # --- RECONSTRUCT X0 FOR IMAGE-SPACE LOSSES ---
                alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)
                sqrt_alpha_prod = alphas_cumprod[timesteps]**0.5
                sqrt_alpha_prod = sqrt_alpha_prod.flatten()
                while len(sqrt_alpha_prod.shape) < len(noisy_latents.shape):
                    sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

                sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps])**0.5
                sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
                while len(sqrt_one_minus_alpha_prod.shape) < len(noisy_latents.shape):
                    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
                
                pred_original_sample = (noisy_latents - sqrt_one_minus_alpha_prod * noise_pred) / sqrt_alpha_prod
                pred_instance_latents, _ = pred_original_sample.chunk(2, dim=0)

                pred_images = vae.decode(pred_instance_latents / vae.config.scaling_factor, return_dict=False)[0]

                # Prepare Images: GT vs Pred (Normalized to [0,1] for most losses)
                pixel_values_instance, _ = batch["pixel_values"].to(device).chunk(2, dim=0)
                pred_images_norm = (pred_images + 1) / 2
                pixel_values_instance_norm = (pixel_values_instance + 1) / 2
                
                # --- CALCULATE AUXILIARY LOSSES ---
                
                # 1. SSIM Loss
                ssim_val = ssim(pred_images_norm, pixel_values_instance_norm, data_range=1.0)
                ssim_loss = 1.0 - ssim_val

                # 2. Pixel Reconstruction Loss (pix)
                pix_loss = F.mse_loss(pred_images_norm, pixel_values_instance_norm)

                # 3. Perceptual Loss (perc)
                perc_loss = perceptual_loss_fn(pred_images_norm, pixel_values_instance_norm)

                # 4. Edge/Gradient Loss (edge)
                edge_loss = edge_loss_fn(pred_images_norm, pixel_values_instance_norm)

                # 5. Frequency/Spectrum Loss (freq)
                freq_loss = freq_loss_fn(pred_images_norm, pixel_values_instance_norm)

                # 6. Radiomics Feature Loss (rad)
                rad_loss = radiomics_loss_fn(pred_images_norm, pixel_values_instance_norm)

                # 7. Segmentation Aware Loss (seg)
                gt_masks = batch["masks"].to(device)
                pred_images_gray = transforms.functional.rgb_to_grayscale(pred_images)
                pred_mask_logits = seg_model(pred_images_gray)
                seg_loss = dice_loss_fn(pred_mask_logits, gt_masks)
                
                # --- COMBINE LOSSES ---
                total_loss = (
                    dreambooth_loss + 
                    args.ssim_loss_weight * ssim_loss + 
                    args.seg_loss_weight * seg_loss +
                    args.pix_loss_weight * pix_loss +
                    args.perc_loss_weight * perc_loss +
                    args.edge_loss_weight * edge_loss +
                    args.freq_loss_weight * freq_loss +
                    args.rad_loss_weight * rad_loss
                )
                
                # --- TENSORBOARD LOGGING ---
                writer.add_scalar("Loss/total", total_loss.detach().item(), global_step)
                writer.add_scalar("Loss/dreambooth_ddpm_cls", dreambooth_loss.detach().item(), global_step)
                writer.add_scalar("Loss/ssim", ssim_loss.detach().item(), global_step)
                writer.add_scalar("Loss/seg_dice", seg_loss.detach().item(), global_step)
                writer.add_scalar("Loss/pix", pix_loss.detach().item(), global_step)
                writer.add_scalar("Loss/perc", perc_loss.detach().item(), global_step)
                writer.add_scalar("Loss/edge", edge_loss.detach().item(), global_step)
                writer.add_scalar("Loss/freq", freq_loss.detach().item(), global_step)
                writer.add_scalar("Loss/rad", rad_loss.detach().item(), global_step)
                
                writer.add_scalar("Metric/SSIM_Score", ssim_val.detach().item(), global_step)
                writer.add_scalar("LearningRate", lr_scheduler.get_last_lr()[0], global_step)
                
                # Calculate and log Dice Score metric
                with torch.no_grad():
                    pred_masks_prob = torch.sigmoid(pred_mask_logits)
                    intersection = torch.sum(pred_masks_prob * gt_masks)
                    union = torch.sum(pred_masks_prob) + torch.sum(gt_masks)
                    dice_score = (2. * intersection) / (union + 1e-6)
                    writer.add_scalar("Metric/Dice_Score", dice_score.item(), global_step)
                # --- END OF LOGGING ---
                
                loss = total_loss / args.gradient_accumulation_steps

            scaler.scale(loss).backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            progress_bar.set_postfix(
                loss=loss.detach().item() * args.gradient_accumulation_steps,
                ssim=ssim_loss.detach().item(),
                seg=seg_loss.detach().item(),
                rad=rad_loss.detach().item()
            )
            progress_bar.update(1)
            global_step += 1

        progress_bar.close()

        # --- SAVE CHECKPOINT WITH FULL TRAINING STATE ---
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{epoch + 1}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save the inference pipeline
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.model_id, unet=unet, text_encoder=text_encoder, tokenizer=tokenizer,
        )
        pipeline.save_pretrained(checkpoint_dir)
        print(f"Saved inference pipeline for epoch {epoch + 1} at {checkpoint_dir}")
        
        # Save the complete training state
        training_state = {
            'epoch': epoch + 1,
            'global_step': global_step,
            'unet_state_dict': unet.state_dict(),
            'text_encoder_state_dict': text_encoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
        }
        torch.save(training_state, os.path.join(checkpoint_dir, "training_state.pth"))
        print(f"Saved complete training state for epoch {epoch + 1}")

    # --- 6. SAVE FINAL MODEL ---
    print("Saving final model...")
    pipeline = StableDiffusionPipeline.from_pretrained(
        args.model_id, unet=unet, text_encoder=text_encoder, tokenizer=tokenizer,
    )
    pipeline.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")
    
    writer.close()

if __name__ == "__main__":
    main()
