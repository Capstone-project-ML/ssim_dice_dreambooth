import os
import argparse
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DDPMScheduler, AutoencoderKL
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.optimization import get_cosine_schedule_with_warmup
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
from tqdm.auto import tqdm
from torchmetrics.functional import structural_similarity_index_measure as ssim
import bitsandbytes as bnb
from monai.losses import DiceLoss
import segmentation_models_pytorch as smp
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# --- LOSS HELPER CLASSES (FIXED GRADIENT FLOW) ---

class PerceptualLoss(nn.Module):
    """Perceptual loss with proper gradient flow"""
    def __init__(self, device):
        super().__init__()
        # Load VGG16 and extract features
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        
        # Take first 16 layers (up to relu3_3) - sufficient for perceptual similarity
        layers = []
        for i, layer in enumerate(vgg):
            layers.append(layer)
            if i == 15:  # Stop at relu3_3
                break
        
        self.model = nn.Sequential(*layers)
        self.model = self.model.to(device)
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        
        # Normalization for VGG
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))

    def forward(self, input, target):
        # Ensure 3 channels
        if input.shape[1] == 1:
            input = input.repeat(1, 3, 1, 1)
        if target.shape[1] == 1:
            target = target.repeat(1, 3, 1, 1)
        
        # Normalize
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        
        # Forward through VGG - NO torch.no_grad() here!
        # The model is frozen, but gradients flow through the computation graph
        features_input = self.model(input)
        features_target = self.model(target)
        
        # MSE between features
        return F.mse_loss(features_input, features_target)

class MemoryEfficientEdgeLoss(nn.Module):
    """Sobel edge loss with memory efficiency"""
    def __init__(self, device):
        super().__init__()
        # Sobel kernels
        kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], 
                         dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], 
                         dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)
        
    def rgb_to_grayscale(self, x):
        """Convert RGB to grayscale efficiently"""
        if x.shape[1] == 3:
            return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        return x
    
    def forward(self, input, target):
        # Convert to grayscale
        input_gray = self.rgb_to_grayscale(input)
        target_gray = self.rgb_to_grayscale(target)
        
        # Compute gradients
        gx_input = F.conv2d(input_gray, self.kx, padding=1)
        gy_input = F.conv2d(input_gray, self.ky, padding=1)
        gx_target = F.conv2d(target_gray, self.kx, padding=1)
        gy_target = F.conv2d(target_gray, self.ky, padding=1)
        
        # L1 loss on gradients
        return F.l1_loss(gx_input, gx_target) + F.l1_loss(gy_input, gy_target)

class StableFrequencyLoss(nn.Module):
    """Numerically stable frequency domain loss"""
    def forward(self, input, target):
        # Ensure contiguous float32
        input_f32 = input.contiguous().float()
        target_f32 = target.contiguous().float()
        
        # Compute 2D FFT
        fft_input = torch.fft.fft2(input_f32, dim=(-2, -1))
        fft_target = torch.fft.fft2(target_f32, dim=(-2, -1))
        
        # Compute magnitude spectrum in log space
        eps = 1e-8
        mag_input = torch.log(torch.abs(fft_input) + eps)
        mag_target = torch.log(torch.abs(fft_target) + eps)
        
        # MSE in log magnitude space
        loss = F.mse_loss(mag_input, mag_target)
        
        # Safety check
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=input.device)
        return loss

class SafeRadiomicsLoss(nn.Module):
    """Ultra-stable radiomics loss with gradient protection"""
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
        
    def forward(self, input, target):
        B, C, H, W = input.shape
        
        # Flatten spatial dimensions
        x_input = input.reshape(B, C, -1)
        x_target = target.reshape(B, C, -1)
        
        # Compute mean with stability
        mean_input = x_input.mean(dim=2, keepdim=True)
        mean_target = x_target.mean(dim=2, keepdim=True)
        
        # Compute variance with stability
        var_input = ((x_input - mean_input) ** 2).mean(dim=2, keepdim=True)
        var_target = ((x_target - mean_target) ** 2).mean(dim=2, keepdim=True)
        
        # Compute standard deviation with protection
        std_input = torch.sqrt(var_input + self.eps)
        std_target = torch.sqrt(var_target + self.eps)
        
        # Standardize with protection against division by zero
        z_input = (x_input - mean_input) / (std_input + self.eps)
        z_target = (x_target - mean_target) / (std_target + self.eps)
        
        # Clamp to prevent extreme values in higher moments
        z_input = torch.clamp(z_input, -5, 5)
        z_target = torch.clamp(z_target, -5, 5)
        
        # Compute skewness (3rd moment)
        skew_input = (z_input ** 3).mean(dim=2)
        skew_target = (z_target ** 3).mean(dim=2)
        
        # Compute kurtosis (4th moment) - subtract 3 for excess kurtosis
        kurt_input = (z_input ** 4).mean(dim=2) - 3.0
        kurt_target = (z_target ** 4).mean(dim=2) - 3.0
        
        # Weighted loss components
        loss_mean = F.mse_loss(mean_input.squeeze(), mean_target.squeeze())
        loss_std = F.mse_loss(std_input.squeeze(), std_target.squeeze())
        loss_skew = 0.01 * F.mse_loss(skew_input, skew_target)
        loss_kurt = 0.001 * F.mse_loss(kurt_input, kurt_target)
        
        return loss_mean + 0.1 * loss_std + loss_skew + loss_kurt

# --- DATASET & UTILS ---

def parse_args():
    parser = argparse.ArgumentParser()
    # Model & Data
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-2-base")
    parser.add_argument("--instance_data_dir", type=str, required=True)
    parser.add_argument("--instance_mask_dir", type=str, required=True)
    parser.add_argument("--class_data_dir", type=str, default="./class_images")
    parser.add_argument("--output_dir", type=str, default="./results")
    
    # Training Tokens
    parser.add_argument("--unique_token", type=str, default="<nih-xray>")
    parser.add_argument("--class_token", type=str, default="xray")
    
    # Model Weights
    parser.add_argument("--num_class_images", type=int, default=200)
    parser.add_argument("--seg_model_weights", type=str, required=True)
    
    # Training Parameters
    parser.add_argument("--num_epochs", type=int, default=100)  # More epochs for steps-based training
    parser.add_argument("--max_train_steps", type=int, default=1200, help="Total training steps")
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    
    # Loss Weights
    parser.add_argument("--prior_loss_weight", type=float, default=1.0)
    parser.add_argument("--seg_loss_weight", type=float, default=0.1)
    parser.add_argument("--pix_loss_weight", type=float, default=0.05)
    parser.add_argument("--perc_loss_weight", type=float, default=0.02)
    parser.add_argument("--ssim_loss_weight", type=float, default=0.01)
    parser.add_argument("--edge_loss_weight", type=float, default=0.02)
    parser.add_argument("--freq_loss_weight", type=float, default=0.005)
    parser.add_argument("--rad_loss_weight", type=float, default=0.001)
    
    # Optimization & Stability
    parser.add_argument("--save_interval", type=int, default=200)
    parser.add_argument("--validation_interval", type=int, default=100)
    parser.add_argument("--use_mixed_precision", action="store_true", default=True)
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--aux_loss_start_step", type=int, default=300, help="Step to start auxiliary losses")
    parser.add_argument("--max_aux_timestep", type=int, default=200, help="Max timestep for auxiliary losses (0-1000)")
    parser.add_argument("--aux_loss_probability", type=float, default=0.3, help="Probability of applying auxiliary losses")
    parser.add_argument("--aux_batch_reduction", type=int, default=4, help="Process 1/N images for aux losses")
    parser.add_argument("--warmup_steps", type=int, default=100, help="LR warmup steps")
    
    # Token initialization
    parser.add_argument("--init_token_with_class", action="store_true", default=True, 
                       help="Initialize new token with class token embedding")
    
    return parser.parse_args()

class DreamBoothDataset(Dataset):
    """Memory-efficient dataset for DreamBooth"""
    def __init__(self, instance_data_root, instance_mask_root, class_data_root, 
                 tokenizer, instance_prompt, class_prompt, size=512):
        self.tokenizer = tokenizer
        self.instance_prompt = instance_prompt
        self.class_prompt = class_prompt
        self.size = size
        
        # Load instance images
        self.instance_image_paths = []
        self.instance_mask_paths = []
        
        valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        for fname in sorted(os.listdir(instance_data_root)):
            if os.path.splitext(fname.lower())[1] in valid_extensions:
                img_path = os.path.join(instance_data_root, fname)
                self.instance_image_paths.append(img_path)
                
                # Find corresponding mask
                mask_name = f"{os.path.splitext(fname)[0]}.png"
                mask_path = os.path.join(instance_mask_root, mask_name)
                if os.path.exists(mask_path):
                    self.instance_mask_paths.append(mask_path)
                else:
                    # Try other extensions
                    for ext in ['.jpg', '.jpeg', '.bmp', '.tiff']:
                        mask_path = os.path.join(instance_mask_root, f"{os.path.splitext(fname)[0]}{ext}")
                        if os.path.exists(mask_path):
                            self.instance_mask_paths.append(mask_path)
                            break
                    else:
                        # No mask found
                        self.instance_mask_paths.append(None)
        
        # Load class images
        self.class_image_paths = []
        if os.path.exists(class_data_root) and os.listdir(class_data_root):
            for fname in sorted(os.listdir(class_data_root)):
                if os.path.splitext(fname.lower())[1] in valid_extensions:
                    self.class_image_paths.append(os.path.join(class_data_root, fname))
        
        self.has_class_images = len(self.class_image_paths) > 0
        self.num_instance = len(self.instance_image_paths)
        self.num_class = len(self.class_image_paths)
        
        print(f"Dataset: {self.num_instance} instance images, {self.num_class} class images")
        
        # Transforms
        self.image_transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        if self.has_class_images:
            return max(self.num_instance, self.num_class)
        return self.num_instance

    def __getitem__(self, idx):
        # Instance data
        instance_idx = idx % self.num_instance
        instance_image = Image.open(self.instance_image_paths[instance_idx]).convert("RGB")
        
        # Instance mask
        mask_path = self.instance_mask_paths[instance_idx]
        if mask_path and os.path.exists(mask_path):
            instance_mask = Image.open(mask_path).convert("L")
        else:
            # Create empty mask
            instance_mask = Image.new("L", (self.size, self.size), 0)
        
        # Class data
        if self.has_class_images:
            class_idx = idx % self.num_class
            class_image = Image.open(self.class_image_paths[class_idx]).convert("RGB")
            class_mask = Image.new("L", (self.size, self.size), 0)  # Dummy mask
        else:
            class_image = None
            class_mask = None
        
        # Apply transforms
        instance_image_tensor = self.image_transform(instance_image)
        instance_mask_tensor = self.mask_transform(instance_mask)
        
        # Tokenize prompts
        instance_prompt_ids = self.tokenizer(
            self.instance_prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt"
        ).input_ids[0]
        
        if class_image is not None:
            class_image_tensor = self.image_transform(class_image)
            class_mask_tensor = self.mask_transform(class_mask)
            class_prompt_ids = self.tokenizer(
                self.class_prompt,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt"
            ).input_ids[0]
        else:
            class_image_tensor = None
            class_mask_tensor = None
            class_prompt_ids = None
        
        return {
            "instance_image": instance_image_tensor,
            "instance_mask": instance_mask_tensor,
            "instance_prompt_ids": instance_prompt_ids,
            "class_image": class_image_tensor,
            "class_mask": class_mask_tensor,
            "class_prompt_ids": class_prompt_ids,
            "has_class": class_image is not None
        }

def collate_fn(batch, tokenizer):
    """Proper collation with batch dimension handling"""
    has_class = batch[0]["has_class"]
    
    # Collect all tensors
    instance_images = []
    instance_masks = []
    instance_prompts = []
    
    class_images = []
    class_masks = []
    class_prompts = []
    
    for item in batch:
        instance_images.append(item["instance_image"])
        instance_masks.append(item["instance_mask"])
        instance_prompts.append(item["instance_prompt_ids"])
        
        if has_class:
            class_images.append(item["class_image"])
            class_masks.append(item["class_mask"])
            class_prompts.append(item["class_prompt_ids"])
    
    # Stack tensors
    instance_images = torch.stack(instance_images)
    instance_masks = torch.stack(instance_masks)
    instance_prompts = torch.stack(instance_prompts)
    
    if has_class:
        class_images = torch.stack(class_images)
        class_masks = torch.stack(class_masks)
        class_prompts = torch.stack(class_prompts)
        
        # Concatenate instance and class
        images = torch.cat([instance_images, class_images], dim=0)
        masks = torch.cat([instance_masks, class_masks], dim=0)
        prompts = torch.cat([instance_prompts, class_prompts], dim=0)
    else:
        images = instance_images
        masks = instance_masks
        prompts = instance_prompts
    
    return {
        "pixel_values": images,
        "masks": masks,
        "input_ids": prompts,
        "has_class": has_class,
        "batch_size": len(batch)
    }

def initialize_token_embedding(tokenizer, text_encoder, unique_token, class_token):
    """Initialize new token with class token embedding for better convergence"""
    # Get token IDs
    unique_token_id = tokenizer.convert_tokens_to_ids(unique_token)
    class_token_id = tokenizer.convert_tokens_to_ids(class_token)
    
    if unique_token_id == tokenizer.unk_token_id:
        print(f"Warning: Unique token '{unique_token}' not found in tokenizer")
        return
    
    if class_token_id == tokenizer.unk_token_id:
        print(f"Warning: Class token '{class_token}' not found in tokenizer")
        # Initialize with random but reasonable values
        with torch.no_grad():
            embeddings = text_encoder.get_input_embeddings().weight.data
            # Use average of all embeddings
            embeddings[unique_token_id] = embeddings.mean(dim=0)
        return
    
    # Initialize unique token with class token embedding
    with torch.no_grad():
        embeddings = text_encoder.get_input_embeddings().weight.data
        embeddings[unique_token_id] = embeddings[class_token_id].clone()
    
    print(f"Initialized '{unique_token}' embedding with '{class_token}' embedding")

def apply_lora(unet, rank=4):
    """Apply LoRA to UNet attention layers"""
    lora_attn_procs = {}
    
    for name, attn_processor in unet.attn_processors.items():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        else:
            continue
        
        lora_attn_procs[name] = LoRAAttnProcessor(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=rank,
        )
    
    unet.set_attn_processor(lora_attn_procs)
    return unet

def save_checkpoint(unet, text_encoder, tokenizer, optimizer, scheduler, global_step, args, is_final=False):
    """Save checkpoint with proper handling"""
    os.makedirs(args.output_dir, exist_ok=True)
    
    if is_final:
        # Save final pipeline
        save_path = os.path.join(args.output_dir, "final_model")
        
        # Save pipeline
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.model_id,
            unet=unet,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipeline.save_pretrained(save_path)
        print(f"✓ Saved final model to {save_path}")
    else:
        # Save training checkpoint
        checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'global_step': global_step,
            'unet_state_dict': unet.state_dict(),
            'text_encoder_state_dict': text_encoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'args': vars(args),
        }
        
        checkpoint_path = os.path.join(checkpoint_dir, f"step_{global_step:06d}.pt")
        torch.save(checkpoint, checkpoint_path)
        
        # Keep only last 5 checkpoints
        checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')])
        for old_checkpoint in checkpoints[:-5]:
            os.remove(os.path.join(checkpoint_dir, old_checkpoint))

# --- MAIN TRAINING FUNCTION ---

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Arguments:", vars(args))
    
    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))
    
    # Save arguments
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        import json
        json.dump(vars(args), f, indent=2)
    
    # --- LOAD MODELS ---
    print("\n1. Loading models...")
    
    # Tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(
        args.model_id, 
        subfolder="tokenizer",
        model_max_length=77
    )
    
    # Text encoder
    text_encoder = CLIPTextModel.from_pretrained(
        args.model_id, 
        subfolder="text_encoder"
    ).to(device)
    
    # Add and initialize custom token
    if args.unique_token not in tokenizer.get_vocab():
        num_added = tokenizer.add_tokens([args.unique_token])
        print(f"Added {num_added} new token: {args.unique_token}")
    
    text_encoder.resize_token_embeddings(len(tokenizer))
    
    # Initialize token embedding
    if args.init_token_with_class:
        initialize_token_embedding(tokenizer, text_encoder, args.unique_token, args.class_token)
    
    # VAE (frozen)
    vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae").to(device)
    vae.requires_grad_(False)
    vae.eval()
    
    # UNet
    unet = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet").to(device)
    
    # Apply gradient checkpointing
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        print("✓ Enabled gradient checkpointing")
    
    # Apply LoRA
    if args.use_lora:
        unet = apply_lora(unet, rank=args.lora_rank)
        print(f"✓ Applied LoRA (rank={args.lora_rank})")
    
    # Noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(args.model_id, subfolder="scheduler")
    
    # Segmentation model (frozen)
    print(f"Loading segmentation model from {args.seg_model_weights}")
    seg_model = smp.Unet(encoder_name="resnet34", in_channels=1, classes=1).to(device)
    seg_model.load_state_dict(torch.load(args.seg_model_weights, map_location=device))
    seg_model.requires_grad_(False)
    seg_model.eval()
    
    # --- INITIALIZE LOSS FUNCTIONS ---
    print("\n2. Initializing loss functions...")
    perceptual_loss = PerceptualLoss(device)
    edge_loss = MemoryEfficientEdgeLoss(device)
    freq_loss = StableFrequencyLoss().to(device)
    rad_loss = SafeRadiomicsLoss().to(device)
    dice_loss = DiceLoss(sigmoid=True, reduction='mean').to(device)
    
    # --- DATASET AND DATALOADER ---
    print("\n3. Creating dataset...")
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
        collate_fn=lambda x: collate_fn(x, tokenizer),
        num_workers=2,
        pin_memory=True,
        drop_last=True  # Important for gradient accumulation
    )
    
    # --- OPTIMIZER AND SCHEDULER ---
    print("\n4. Setting up optimizer...")
    
    # Collect trainable parameters
    params_to_optimize = []
    
    # UNet parameters
    if args.use_lora:
        # Only LoRA parameters
        lora_params = []
        for name, param in unet.named_parameters():
            if "lora" in name and param.requires_grad:
                lora_params.append(param)
        params_to_optimize.append({"params": lora_params, "lr": args.learning_rate})
    else:
        params_to_optimize.append({"params": unet.parameters(), "lr": args.learning_rate})
    
    # Text encoder parameters (lower learning rate)
    text_encoder_params = list(text_encoder.text_model.encoder.parameters()) + \
                         list(text_encoder.text_model.final_layer_norm.parameters()) + \
                         [text_encoder.text_model.embeddings.token_embedding.weight]
    params_to_optimize.append({"params": text_encoder_params, "lr": args.learning_rate * 0.5})
    
    # Create optimizer
    optimizer = bnb.optim.AdamW8bit(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2
    )
    
    # Calculate training steps
    if args.max_train_steps > 0:
        total_steps = args.max_train_steps
        num_epochs = (total_steps * args.gradient_accumulation_steps) // len(train_dataloader) + 1
        args.num_epochs = num_epochs
    else:
        total_steps = args.num_epochs * len(train_dataloader) // args.gradient_accumulation_steps
    
    # Learning rate scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )
    
    print(f"Total steps: {total_steps}, Epochs: {args.num_epochs}")
    print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    
    # Gradient scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler() if (torch.cuda.is_available() and args.use_mixed_precision) else None
    
    # --- TRAINING LOOP ---
    print("\n5. Starting training...")
    global_step = 0
    progress_bar = tqdm(total=total_steps, desc="Training")
    
    for epoch in range(args.num_epochs):
        unet.train()
        text_encoder.train()
        
        for batch_idx, batch in enumerate(train_dataloader):
            # Unpack batch
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            masks = batch["masks"].to(device)
            has_class = batch["has_class"]
            batch_size = batch["batch_size"]
            
            # Determine slices
            if has_class:
                instance_slice = slice(0, batch_size)
                class_slice = slice(batch_size, 2 * batch_size)
            else:
                instance_slice = slice(0, batch_size)
                class_slice = None
            
            # Mixed precision context
            use_amp = scaler is not None
            autocast = torch.cuda.amp.autocast if use_amp else (lambda: contextlib.nullcontext())
            
            with autocast():
                # --- ENCODE TO LATENTS ---
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                
                # --- SAMPLE NOISE AND TIMESTEPS ---
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, 
                    noise_scheduler.config.num_train_timesteps, 
                    (latents.shape[0],), 
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
                    # Split for instance and class
                    noise_pred_inst = noise_pred[instance_slice]
                    noise_pred_class = noise_pred[class_slice]
                    noise_inst = noise[instance_slice]
                    noise_class = noise[class_slice]
                    
                    loss_inst = F.mse_loss(noise_pred_inst, noise_inst)
                    loss_class = F.mse_loss(noise_pred_class, noise_class)
                    loss_db = loss_inst + args.prior_loss_weight * loss_class
                else:
                    loss_db = F.mse_loss(noise_pred, noise)
                
                # --- AUXILIARY LOSSES ---
                loss_aux = torch.tensor(0.0, device=device)
                aux_components = {}
                
                # Conditions for auxiliary losses:
                # 1. Past warmup phase
                # 2. Random chance
                # 3. Only for low-noise timesteps
                apply_aux = (
                    global_step >= args.aux_loss_start_step and
                    torch.rand(1).item() < args.aux_loss_probability
                )
                
                if apply_aux:
                    # Get instance data only
                    instance_latents = latents[instance_slice]
                    instance_noisy = noisy_latents[instance_slice]
                    instance_noise_pred = noise_pred[instance_slice]
                    instance_timesteps = timesteps[instance_slice]
                    instance_masks = masks[instance_slice]
                    
                    # Only apply aux loss for low-noise timesteps
                    low_noise_mask = instance_timesteps < args.max_aux_timestep
                    
                    if low_noise_mask.any() and low_noise_mask.sum() > 0:
                        # Select subset of images for aux loss (memory efficient)
                        n_aux = max(1, batch_size // args.aux_batch_reduction)
                        aux_indices = torch.where(low_noise_mask)[0][:n_aux]
                        
                        if len(aux_indices) > 0:
                            # Get scheduler parameters
                            alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)
                            sqrt_alpha = alphas_cumprod[instance_timesteps].view(-1, 1, 1, 1) ** 0.5
                            sqrt_one_minus_alpha = (1 - alphas_cumprod[instance_timesteps]).view(-1, 1, 1, 1) ** 0.5
                            
                            # Estimate x0 (clean latents) - gradients flow through this!
                            pred_x0 = (instance_noisy - sqrt_one_minus_alpha * instance_noise_pred) / sqrt_alpha
                            
                            # Decode selected latents to image space
                            # IMPORTANT: No torch.no_grad() here - gradients must flow through VAE
                            pred_x0_selected = pred_x0[aux_indices]
                            decoded = vae.decode(pred_x0_selected / vae.config.scaling_factor).sample
                            
                            # Normalize to [0, 1]
                            pred_images = torch.clamp((decoded + 1) / 2, 0, 1)
                            
                            # Get ground truth
                            gt_images = torch.clamp((pixel_values[instance_slice][aux_indices] + 1) / 2, 0, 1)
                            gt_masks = instance_masks[aux_indices]
                            
                            # --- COMPUTE AUXILIARY LOSSES ---
                            # Pixel loss
                            aux_components['pixel'] = F.mse_loss(pred_images, gt_images)
                            
                            # SSIM loss
                            aux_components['ssim'] = 1 - ssim(pred_images, gt_images, data_range=1.0)
                            
                            # Perceptual loss (gradients flow properly now)
                            aux_components['perceptual'] = perceptual_loss(pred_images, gt_images)
                            
                            # Edge loss
                            aux_components['edge'] = edge_loss(pred_images, gt_images)
                            
                            # Frequency loss
                            aux_components['frequency'] = freq_loss(pred_images, gt_images)
                            
                            # Radiomics loss
                            aux_components['radiomics'] = rad_loss(pred_images, gt_images)
                            
                            # Segmentation loss
                            pred_gray = 0.299 * pred_images[:, 0:1] + 0.587 * pred_images[:, 1:2] + 0.114 * pred_images[:, 2:3]
                            # Segmentation model is frozen, but gradients flow through input
                            with torch.no_grad():
                                seg_pred = seg_model(pred_gray)
                            aux_components['segmentation'] = dice_loss(seg_pred, gt_masks)
                            
                            # Weighted auxiliary loss
                            loss_aux = (
                                args.pix_loss_weight * aux_components['pixel'] +
                                args.ssim_loss_weight * aux_components['ssim'] +
                                args.perc_loss_weight * aux_components['perceptual'] +
                                args.edge_loss_weight * aux_components['edge'] +
                                args.freq_loss_weight * aux_components['frequency'] +
                                args.rad_loss_weight * aux_components['radiomics'] +
                                args.seg_loss_weight * aux_components['segmentation']
                            )
                            
                            # Scale by the fraction of processed samples
                            loss_aux = loss_aux * (len(aux_indices) / batch_size)
                
                # Total loss
                loss_total = loss_db + loss_aux
            
            # --- BACKWARD PASS ---
            if use_amp:
                scaler.scale(loss_total).backward()
            else:
                loss_total.backward()
            
            # --- OPTIMIZATION STEP ---
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                # Gradient clipping
                if use_amp:
                    scaler.unscale_(optimizer)
                
                torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(text_encoder.parameters(), max_norm=1.0)
                
                # Optimizer step
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                # Scheduler step
                scheduler.step()
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Update global step
                global_step += 1
                progress_bar.update(1)
            
            # --- LOGGING ---
            if global_step % 10 == 0 and (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                writer.add_scalar("train/loss_total", loss_total.item(), global_step)
                writer.add_scalar("train/loss_db", loss_db.item(), global_step)
                writer.add_scalar("train/loss_aux", loss_aux.item(), global_step)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
                
                if apply_aux and aux_components:
                    for name, value in aux_components.items():
                        writer.add_scalar(f"train/aux_{name}", value.item(), global_step)
            
            # --- CHECKPOINTING ---
            if global_step % args.save_interval == 0 and global_step > 0:
                save_checkpoint(unet, text_encoder, tokenizer, optimizer, scheduler, global_step, args)
                print(f"\nCheckpoint saved at step {global_step}")
            
            # --- VALIDATION ---
            if global_step % args.validation_interval == 0 and global_step > 0:
                # Simple validation logging
                unet.eval()
                text_encoder.eval()
                
                with torch.no_grad():
                    # Log some metrics
                    writer.add_scalar("val/loss_db", loss_db.item(), global_step)
                    
                    if apply_aux:
                        writer.add_scalar("val/loss_aux", loss_aux.item(), global_step)
                
                unet.train()
                text_encoder.train()
            
            # Early stopping if max steps reached
            if global_step >= total_steps:
                break
        
        # End of epoch
        if global_step >= total_steps:
            break
    
    # --- FINAL SAVE ---
    progress_bar.close()
    save_checkpoint(unet, text_encoder, tokenizer, optimizer, scheduler, global_step, args, is_final=True)
    
    # Close writer
    writer.close()
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print(f"Total steps: {global_step}")
    print(f"Final model saved to: {args.output_dir}/final_model")
    print("="*50)

if __name__ == "__main__":
    import contextlib
    main()
