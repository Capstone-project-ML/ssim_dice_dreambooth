import os
import argparse
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from contextlib import nullcontext  # <--- FIXED: Added this import
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# --- LOSS HELPER CLASSES (OPTIMIZED) ---

class PerceptualLoss(nn.Module):
    """Optimized VGG-based perceptual loss with gradient checkpointing"""
    def __init__(self, device):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        
        # Use only 3 layers to save memory
        self.slice1 = vgg[:4].eval()
        self.slice2 = vgg[4:9].eval()
        self.slice3 = vgg[9:16].eval()
        
        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False
            
        # Move to device
        self.slice1 = self.slice1.to(device)
        self.slice2 = self.slice2.to(device)
        self.slice3 = self.slice3.to(device)
        
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target):
        # Ensure 3 channels for VGG
        if input.shape[1] == 1:
            input = input.repeat(1, 3, 1, 1)
        if target.shape[1] == 1:
            target = target.repeat(1, 3, 1, 1)
        
        # Normalize for VGG
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        
        loss = 0.0
        h_x, h_y = input, target
        
        # Use torch.no_grad for forward passes to save memory
        with torch.no_grad():
            h_x = self.slice1(h_x)
            h_y = self.slice1(h_y)
        loss += F.mse_loss(h_x, h_y)
        
        with torch.no_grad():
            h_x = self.slice2(h_x)
            h_y = self.slice2(h_y)
        loss += F.mse_loss(h_x, h_y)
        
        with torch.no_grad():
            h_x = self.slice3(h_x)
            h_y = self.slice3(h_y)
        loss += F.mse_loss(h_x, h_y)
        
        return loss

class EdgeLoss(nn.Module):
    """Memory-efficient Sobel edge loss"""
    def __init__(self, device):
        super().__init__()
        kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], 
                         dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], 
                         dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def forward(self, input, target):
        # Convert to grayscale if needed
        if input.shape[1] == 3:
            # Luma conversion using in-place operations
            input_gray = input[:, 0:1] * 0.299 + input[:, 1:2] * 0.587 + input[:, 2:3] * 0.114
            target_gray = target[:, 0:1] * 0.299 + target[:, 1:2] * 0.587 + target[:, 2:3] * 0.114
        else:
            input_gray = input
            target_gray = target
            
        # Edge detection
        pred_gx = F.conv2d(input_gray, self.kx, padding=1)
        pred_gy = F.conv2d(input_gray, self.ky, padding=1)
        gt_gx = F.conv2d(target_gray, self.kx, padding=1)
        gt_gy = F.conv2d(target_gray, self.ky, padding=1)
        
        # L1 loss for edges
        return F.l1_loss(pred_gx, gt_gx) + F.l1_loss(pred_gy, gt_gy)

class FrequencyLoss(nn.Module):
    """Stable frequency domain loss"""
    def forward(self, input, target):
        # Ensure float32 for FFT
        input_f32 = input.float()
        target_f32 = target.float()
        
        # Compute FFTs
        fft_input = torch.fft.rfft2(input_f32, norm='ortho')
        fft_target = torch.fft.rfft2(target_f32, norm='ortho')
        
        # Compute magnitude in log space for stability
        eps = 1e-8
        mag_input = torch.log1p(torch.abs(fft_input) + eps)
        mag_target = torch.log1p(torch.abs(fft_target) + eps)
        
        # MSE in log magnitude space
        diff = mag_input - mag_target
        loss = torch.mean(diff ** 2)
        
        # Handle potential NaN/Inf
        return torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

class StableRadiomicsLoss(nn.Module):
    """Numerically stable radiomics loss"""
    def forward(self, input, target):
        def get_stable_moments(img):
            # Flatten spatial dimensions
            B, C = img.shape[:2]
            x = img.reshape(B, C, -1).float()
            
            # Mean and variance
            mean = torch.mean(x, dim=2, keepdim=True)
            var = torch.var(x, dim=2, keepdim=True, unbiased=False)
            std = torch.sqrt(var + 1e-6)
            
            # Standardize
            standardized = (x - mean) / (std + 1e-6)
            
            # Use tanh to bound values before higher moments
            standardized = torch.tanh(standardized)
            
            # Calculate moments
            skew = torch.mean(standardized ** 3, dim=2, keepdim=True)
            kurt = torch.mean(standardized ** 4, dim=2, keepdim=True) - 3.0  # Excess kurtosis
            
            return mean.squeeze(), var.squeeze(), skew.squeeze(), kurt.squeeze()
        
        # Get moments
        p_m, p_v, p_s, p_k = get_stable_moments(input)
        g_m, g_v, g_s, g_k = get_stable_moments(target)
        
        # Weighted loss
        return (F.mse_loss(p_m, g_m) + 
                F.mse_loss(p_v, g_v) * 0.1 +
                F.mse_loss(p_s, g_s) * 0.01 + 
                F.mse_loss(p_k, g_k) * 0.001)

# --- DATASET & UTILS ---

def parse_args():
    parser = argparse.ArgumentParser()
    # Model & Data
    parser.add_argument("--model_id", type=str, default="Manojb/stable-diffusion-2-1-base")
    parser.add_argument("--instance_data_dir", type=str, required=True)
    parser.add_argument("--instance_mask_dir", type=str, required=True)
    parser.add_argument("--class_data_dir", type=str, default="./class_images")
    parser.add_argument("--output_dir", type=str, default="./results")
    
    # Training Tokens
    parser.add_argument("--unique_token", type=str, default="<nih-xray>")
    parser.add_argument("--class_token", type=str, default="x-ray")
    
    # Model Weights
    parser.add_argument("--num_class_images", type=int, default=200)
    parser.add_argument("--seg_model_weights", type=str, required=True)
    
    # Training Parameters
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=2e-6)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    
    # Loss Weights
    parser.add_argument("--prior_loss_weight", type=float, default=1.0)
    parser.add_argument("--seg_loss_weight", type=float, default=0.3)
    parser.add_argument("--pix_loss_weight", type=float, default=0.2)
    parser.add_argument("--perc_loss_weight", type=float, default=0.1)
    parser.add_argument("--ssim_loss_weight", type=float, default=0.05)
    parser.add_argument("--edge_loss_weight", type=float, default=0.1)
    parser.add_argument("--freq_loss_weight", type=float, default=0.05)
    parser.add_argument("--rad_loss_weight", type=float, default=0.05)
    
    # Optimization & Stability
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--validation_interval", type=int, default=100)
    parser.add_argument("--use_mixed_precision", action="store_true")
    # REMOVED LoRA ARGS
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing for UNet")
    parser.add_argument("--aux_loss_start_step", type=int, default=1000, help="Step to start auxiliary losses")
    parser.add_argument("--max_aux_timestep", type=int, default=300, help="Max timestep for auxiliary losses (0-1000)")
    parser.add_argument("--aux_loss_probability", type=float, default=0.5, help="Probability of applying auxiliary losses")
    
    return parser.parse_args()

class DreamBoothDataset(Dataset):
    """Dataset with proper handling of instance/class images and masks"""
    def __init__(self, instance_data_root, instance_mask_root, class_data_root, 
                 tokenizer, instance_prompt, class_prompt, size=512):
        self.tokenizer = tokenizer
        self.instance_prompt = instance_prompt
        self.class_prompt = class_prompt
        self.size = size
        
        # Instance images
        self.instance_images_path = []
        for f in os.listdir(instance_data_root):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.instance_images_path.append(os.path.join(instance_data_root, f))
        
        # Instance masks
        self.instance_masks_path = []
        for img_path in self.instance_images_path:
            mask_name = os.path.splitext(os.path.basename(img_path))[0] + ".png"
            mask_path = os.path.join(instance_mask_root, mask_name)
            if os.path.exists(mask_path):
                self.instance_masks_path.append(mask_path)
            else:
                self.instance_masks_path.append(None)
        
        # Class images
        self.has_class_images = False
        if os.path.exists(class_data_root) and len(os.listdir(class_data_root)) > 0:
            self.class_images_path = []
            for f in os.listdir(class_data_root):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.class_images_path.append(os.path.join(class_data_root, f))
            self.num_class_images = len(self.class_images_path)
            self.has_class_images = self.num_class_images > 0
        else:
            self.class_images_path = []
            self.num_class_images = 0
        
        self.num_instance_images = len(self.instance_images_path)
        
        # Dataset length (for epoch)
        if self.has_class_images:
            self._length = max(self.num_instance_images, self.num_class_images)
        else:
            self._length = self.num_instance_images
        
        # Transforms
        self.image_transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
        self.mask_transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
        ])
        
        print(f"Dataset initialized: {self.num_instance_images} instance images, "
              f"{self.num_class_images} class images")

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        
        # Instance data (always present)
        inst_idx = index % self.num_instance_images
        inst_img = Image.open(self.instance_images_path[inst_idx]).convert("RGB")
        example["instance_images"] = self.image_transforms(inst_img)
        
        # Instance mask
        mask_path = self.instance_masks_path[inst_idx]
        if mask_path is not None and os.path.exists(mask_path):
            mask_img = Image.open(mask_path).convert("L")
            example["instance_mask"] = self.mask_transforms(mask_img)
        else:
            # Create empty mask if not available
            example["instance_mask"] = torch.zeros((1, self.size, self.size))
        
        # Instance prompt
        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length
        ).input_ids
        
        # Class data (optional)
        if self.has_class_images:
            class_idx = index % self.num_class_images
            class_img = Image.open(self.class_images_path[class_idx]).convert("RGB")
            example["class_images"] = self.image_transforms(class_img)
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length
            ).input_ids
            # Dummy mask for class images
            example["class_mask"] = torch.zeros((1, self.size, self.size))
        else:
            example["class_images"] = None
            example["class_prompt_ids"] = None
            example["class_mask"] = None
            
        return example

def collate_fn(examples, pad_token_id):
    """Collate function that properly handles batch dimensions"""
    input_ids = []
    pixel_values = []
    masks = []
    
    for e in examples:
        # Instance data
        input_ids.append(e["instance_prompt_ids"])
        pixel_values.append(e["instance_images"])
        masks.append(e["instance_mask"])
        
        # Class data (if exists)
        if e["class_images"] is not None:
            input_ids.append(e["class_prompt_ids"])
            pixel_values.append(e["class_images"])
            masks.append(e["class_mask"])
    
    # Stack tensors
    pixel_values = torch.stack(pixel_values).contiguous().float()
    masks = torch.stack(masks).contiguous().float()
    
    # Pad input_ids
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(ids, dtype=torch.long) for ids in input_ids],
        batch_first=True,
        padding_value=pad_token_id
    )
    
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "masks": masks,
        "has_class": examples[0]["class_images"] is not None
    }

def save_checkpoint(unet, text_encoder, tokenizer, optimizer, scheduler, global_step, args, is_final=False):
    """Save model checkpoint"""
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    if is_final:
        # Save final pipeline
        save_path = os.path.join(args.output_dir, "final_model")
        
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.model_id,
            unet=unet,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipeline.save_pretrained(save_path)
        print(f"Saved final model to {save_path}")
    else:
        # Save training checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-{global_step}")
        
        # Save only trainable parameters
        save_dict = {
            'global_step': global_step,
            'unet_state_dict': unet.state_dict(),
            'text_encoder_state_dict': text_encoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'args': vars(args),
        }
        
        if scheduler is not None:
            save_dict['scheduler_state_dict'] = scheduler.state_dict()
        
        torch.save(save_dict, f"{checkpoint_path}.pt")
        print(f"Saved checkpoint to {checkpoint_path}.pt")

# --- MAIN TRAINING FUNCTION ---

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Arguments: {vars(args)}")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))
    
    # Save arguments
    with open(os.path.join(args.output_dir, "args.txt"), "w") as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
    
    # --- LOAD MODELS ---
    print("Loading models...")
    
    # Tokenizer and text encoder
    tokenizer = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.model_id, subfolder="text_encoder").to(device)
    
    # Add custom token
    num_added_tokens = tokenizer.add_tokens(args.unique_token)
    if num_added_tokens == 0:
        print(f"Warning: Token {args.unique_token} already exists in tokenizer")
    text_encoder.resize_token_embeddings(len(tokenizer))
    
    # VAE (frozen)
    vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae").to(device)
    vae.requires_grad_(False)
    vae.eval()
    
    # UNet
    unet = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet").to(device)
    
    # Apply gradient checkpointing if requested
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        print("Enabled gradient checkpointing for UNet")
    
    # Noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(args.model_id, subfolder="scheduler")
    
    # Segmentation model (frozen)
    print(f"Loading segmentation model from {args.seg_model_weights}")
    seg_model = smp.Unet(encoder_name="resnet34", in_channels=1, classes=1).to(device)
    seg_model.load_state_dict(torch.load(args.seg_model_weights, map_location=device))
    seg_model.requires_grad_(False)
    seg_model.eval()
    
    # Initialize loss functions
    print("Initializing loss functions...")
    # FIXED: Added .to(device) to ensure all buffers are on GPU
    perceptual_fn = PerceptualLoss(device).to(device)
    edge_fn = EdgeLoss(device).to(device)
    freq_fn = FrequencyLoss().to(device)
    rad_fn = StableRadiomicsLoss().to(device)
    dice_fn = DiceLoss(sigmoid=True).to(device)
    
    # --- DATASET AND DATALOADER ---
    print("Creating dataset and dataloader...")
    train_dataset = DreamBoothDataset(
        args.instance_data_dir,
        args.instance_mask_dir,
        args.class_data_dir,
        tokenizer,
        f"a photo of {args.unique_token}",
        f"a photo of {args.class_token}",
        size=512
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id),
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # --- OPTIMIZER AND SCHEDULER ---
    print("Setting up optimizer...")
    
    # Determine parameters to optimize - FULL FINE TUNING
    params_to_optimize = []
    
    # Optimize all UNet parameters
    params_to_optimize.append({"params": unet.parameters(), "lr": args.learning_rate})
    
    # Text encoder parameters (usually use lower LR)
    params_to_optimize.append({"params": text_encoder.parameters(), "lr": args.learning_rate * 0.5})
    
    optimizer = bnb.optim.AdamW8bit(params_to_optimize, lr=args.learning_rate)
    
    # Learning rate scheduler
    num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
    num_training_steps = args.num_epochs * num_update_steps_per_epoch
    num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Gradient scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler() if (torch.cuda.is_available() and args.use_mixed_precision) else None
    
    # --- TRAINING LOOP ---
    print("Starting training...")
    global_step = 0
    progress_bar = tqdm(total=num_training_steps, desc="Training")
    
    for epoch in range(args.num_epochs):
        unet.train()
        text_encoder.train()
        
        for step, batch in enumerate(train_dataloader):
            # Move batch to device
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            masks = batch["masks"].to(device)
            has_class = batch["has_class"]
            
            # Determine batch splits
            if has_class:
                # Split into instance and class
                instance_slice = slice(0, args.batch_size)
                class_slice = slice(args.batch_size, 2 * args.batch_size)
                instance_mask_slice = instance_slice
                class_mask_slice = class_slice
            else:
                # Only instance data
                instance_slice = slice(0, args.batch_size)
                class_slice = None
                instance_mask_slice = slice(0, args.batch_size)
                class_mask_slice = None
            
            # Mixed precision context
            use_amp = scaler is not None
            # FIXED: Used nullcontext instead of torch.no_grad() so standard training works!
            ctx = torch.cuda.amp.autocast() if use_amp else nullcontext()
            
            with ctx:
                # --- ENCODE TO LATENTS ---
                with torch.no_grad():  # VAE is frozen
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                
                # --- SAMPLE NOISE AND TIMESTEPS ---
                noise = torch.randn_like(latents)
                
                # Sample timesteps with different strategies for instance vs class
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, 
                    noise_scheduler.config.num_train_timesteps, 
                    (bsz,), 
                    device=device
                ).long()
                
                # --- ADD NOISE ---
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # --- GET TEXT EMBEDDINGS ---
                encoder_hidden_states = text_encoder(input_ids)[0]
                
                # --- PREDICT NOISE ---
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                
                # --- DREAMBOOTH LOSS ---
                if has_class:
                    # Split predictions and noise
                    noise_pred_instance = noise_pred[instance_slice]
                    noise_pred_class = noise_pred[class_slice]
                    noise_instance = noise[instance_slice]
                    noise_class = noise[class_slice]
                    
                    # Compute losses
                    instance_loss = F.mse_loss(noise_pred_instance, noise_instance)
                    class_loss = F.mse_loss(noise_pred_class, noise_class)
                    db_loss = instance_loss + args.prior_loss_weight * class_loss
                else:
                    db_loss = F.mse_loss(noise_pred, noise)
                
                # --- AUXILIARY LOSSES (Conditional) ---
                aux_loss = torch.tensor(0.0, device=device)
                
                # Apply auxiliary losses only when:
                # 1. Global step > aux_loss_start_step
                # 2. Random probability < aux_loss_probability
                # 3. Timestep < max_aux_timestep (for reconstruction quality)
                apply_aux = (
                    global_step > args.aux_loss_start_step and
                    torch.rand(1).item() < args.aux_loss_probability
                )
                
                if apply_aux:
                    # Get instance latents only
                    instance_latents = latents[instance_slice]
                    instance_noisy_latents = noisy_latents[instance_slice]
                    instance_noise_pred = noise_pred[instance_slice]
                    instance_timesteps = timesteps[instance_slice]
                    instance_gt_masks = masks[instance_mask_slice]
                    
                    # Filter by timestep for reconstruction quality
                    low_noise_mask = instance_timesteps < args.max_aux_timestep
                    
                    if low_noise_mask.any():
                        # Estimate x0 for low-noise timesteps only
                        alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)
                        sqrt_alpha_prod = alphas_cumprod[instance_timesteps].view(-1, 1, 1, 1) ** 0.5
                        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[instance_timesteps]).view(-1, 1, 1, 1) ** 0.5
                        
                        # Estimate clean latents (x0)
                        pred_x0_latents = (instance_noisy_latents - sqrt_one_minus_alpha_prod * instance_noise_pred) / sqrt_alpha_prod
                        
                        # Only decode images that need gradients
                        # We'll decode a subset to save memory
                        decode_indices = torch.where(low_noise_mask)[0]
                        if len(decode_indices) > 0:
                            # Take at most 2 images for aux losses to save memory
                            decode_indices = decode_indices[:min(2, len(decode_indices))]
                            
                            # Decode WITHOUT no_grad() - gradients flow through VAE to UNet
                            # Note: This increases memory but allows gradient flow
                            decoded = vae.decode(
                                pred_x0_latents[decode_indices] / vae.config.scaling_factor
                            ).sample
                            
                            # Convert to image space [0, 1]
                            pred_img = torch.clamp((decoded + 1) / 2, 0, 1)
                            
                            # Get corresponding ground truth
                            gt_img = torch.clamp((pixel_values[instance_slice][decode_indices] + 1) / 2, 0, 1)
                            gt_mask = instance_gt_masks[decode_indices]
                            
                            # --- COMPUTE AUXILIARY LOSSES ---
                            # 1. Pixel-wise loss
                            l_pix = F.mse_loss(pred_img, gt_img)
                            
                            # 2. SSIM loss
                            l_ssim = 1 - ssim(pred_img, gt_img, data_range=1.0)
                            
                            # 3. Perceptual loss
                            l_perc = perceptual_fn(pred_img, gt_img)
                            
                            # 4. Edge loss
                            l_edge = edge_fn(pred_img, gt_img)
                            
                            # 5. Frequency loss
                            l_freq = freq_fn(pred_img, gt_img)
                            
                            # 6. Radiomics loss
                            l_rad = rad_fn(pred_img, gt_img)
                            
                            # 7. Segmentation loss
                            pred_gray = 0.299 * pred_img[:, 0:1] + 0.587 * pred_img[:, 1:2] + 0.114 * pred_img[:, 2:3]
                            # Use with torch.no_grad() for seg model since it's frozen
                            with torch.no_grad():
                                seg_pred = seg_model(pred_gray)
                            l_seg = dice_fn(seg_pred, gt_mask)
                            
                            # Weighted aux loss
                            aux_loss = (
                                args.pix_loss_weight * l_pix +
                                args.ssim_loss_weight * l_ssim +
                                args.perc_loss_weight * l_perc +
                                args.edge_loss_weight * l_edge +
                                args.freq_loss_weight * l_freq +
                                args.rad_loss_weight * l_rad +
                                args.seg_loss_weight * l_seg
                            )
                            
                            # Scale aux loss by the fraction of valid timesteps
                            aux_loss = aux_loss * (len(decode_indices) / args.batch_size)
                
                # --- TOTAL LOSS ---
                total_loss = db_loss + aux_loss
            
            # --- BACKWARD PASS ---
            if use_amp:
                scaler.scale(total_loss).backward()
            else:
                total_loss.backward()
            
            # --- OPTIMIZATION STEP ---
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # Gradient clipping
                if use_amp:
                    scaler.unscale_(optimizer)
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(text_encoder.parameters(), 1.0)
                
                # Optimizer step
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                # Learning rate scheduler step
                scheduler.step()
                
                # Zero gradients
                optimizer.zero_grad()
            
            # --- LOGGING ---
            if global_step % 10 == 0:
                writer.add_scalar("Loss/Total", total_loss.item(), global_step)
                writer.add_scalar("Loss/DreamBooth", db_loss.item(), global_step)
                writer.add_scalar("Loss/Auxiliary", aux_loss.item(), global_step)
                writer.add_scalar("LR", optimizer.param_groups[0]['lr'], global_step)
                
                # Log individual aux losses if computed
                if apply_aux and aux_loss.item() > 0:
                    writer.add_scalar("Loss/SSIM", l_ssim.item(), global_step)
                    writer.add_scalar("Loss/Segmentation", l_seg.item(), global_step)
                    writer.add_scalar("Loss/Pixel", l_pix.item(), global_step)
                    # FIXED: Added missing metrics
                    writer.add_scalar("Loss/Perceptual", l_perc.item(), global_step)
                    writer.add_scalar("Loss/Edge", l_edge.item(), global_step)
                    writer.add_scalar("Loss/Frequency", l_freq.item(), global_step)
                    writer.add_scalar("Loss/Radiomics", l_rad.item(), global_step)
            
            # --- CHECKPOINTING ---
            if global_step % args.save_interval == 0 and global_step > 0:
                save_checkpoint(unet, text_encoder, tokenizer, optimizer, scheduler, global_step, args)
            
            # --- VALIDATION ---
            if global_step % args.validation_interval == 0 and global_step > 0:
                # Simple validation - just log some metrics
                unet.eval()
                text_encoder.eval()
                
                with torch.no_grad():
                    # Generate a sample image for visual inspection
                    sample_prompt = f"a photo of {args.unique_token}"
                    
                    # For now, just log the current loss components
                    writer.add_scalar("Validation/Loss_Components/DB", db_loss.item(), global_step)
                    writer.add_scalar("Validation/Loss_Components/Aux", aux_loss.item(), global_step)
                
                unet.train()
                text_encoder.train()
            
            # Update progress
            progress_bar.update(1)
            global_step += 1
            
            # Early stopping if we've reached total training steps
            if global_step >= num_training_steps:
                break
        
        # End of epoch
        print(f"Epoch {epoch+1}/{args.num_epochs} completed")
        
        if global_step >= num_training_steps:
            break
    
    # --- FINAL SAVE ---
    progress_bar.close()
    save_checkpoint(unet, text_encoder, tokenizer, optimizer, scheduler, global_step, args, is_final=True)
    
    # Close tensorboard writer
    writer.close()
    print(f"Training completed. Final model saved to {args.output_dir}")
    
    # Print final statistics
    print("\n" + "="*50)
    print("TRAINING COMPLETED")
    print(f"Total steps: {global_step}")
    print(f"Final loss: {total_loss.item():.6f}")
    print("="*50)

if __name__ == "__main__":
    main()
