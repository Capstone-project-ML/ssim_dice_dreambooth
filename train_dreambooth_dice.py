# train_dreambooth_dice.py
import os
import argparse
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DDPMScheduler, AutoencoderKL
from diffusers.optimization import get_cosine_schedule_with_warmup
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import random
from monai.losses import DiceLoss
import segmentation_models_pytorch as smp

def parse_args():
    parser = argparse.ArgumentParser(description="DreamBooth fine-tuning with Dice loss.")
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-2-base", help="Pretrained model ID.")
    parser.add_argument("--instance_data_dir", type=str, required=True, help="Path to your training images.")
    parser.add_argument("--instance_mask_dir", type=str, required=True, help="Path to your training segmentation masks.")
    parser.add_argument("--output_dir", type=str, default="./results_dreambooth_dice", help="Directory to save the model and logs.")
    parser.add_argument("--seg_model_weights", type=str, required=True, help="Path to pre-trained segmentation model weights.")
    parser.add_argument("--unique_token", type=str, default="<nih-xray>", help="Unique token for your concept.")
    parser.add_argument("--class_token", type=str, default="chest x-ray", help="General class for prior preservation.")
    parser.add_argument("--class_data_dir", type=str, default="./class_images_xray", help="Directory to cache generated class images.")
    parser.add_argument("--num_class_images", type=int, default=200, help="Number of class images for prior preservation.")
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="Weight for prior preservation loss.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=2e-6, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--dice_loss_weight", type=float, default=0.1, help="Weight for the Dice loss.")
    return parser.parse_args()

class DreamBoothDataset(Dataset):
    def __init__(self, instance_data_root, instance_mask_root, class_data_root, tokenizer, instance_prompt, class_prompt, size=512):
        self.tokenizer = tokenizer
        self.instance_prompt = instance_prompt
        self.class_prompt = class_prompt
        self.size = size
        self.instance_images_path = [os.path.join(instance_data_root, f) for f in os.listdir(instance_data_root) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.instance_masks_path = [os.path.join(instance_mask_root, f) for f in os.listdir(instance_data_root) if f.endswith(('.png', '.jpg', '.jpeg'))]
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
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images]).convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_prompt_ids"] = self.tokenizer(self.instance_prompt, padding="do_not_pad", truncation=True, max_length=self.tokenizer.model_max_length).input_ids
        
        instance_mask = Image.open(self.instance_masks_path[index % self.num_instance_images]).convert("L")
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
    masks = torch.stack(masks)
    input_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(ids, dtype=torch.long) for ids in input_ids], batch_first=True, padding_value=49407)
    
    return {"pixel_values": pixel_values, "input_ids": input_ids, "masks": masks}


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.class_data_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    seg_model = smp.Unet(encoder_name="resnet34", in_channels=1, classes=1).to(device)
    seg_model.load_state_dict(torch.load(args.seg_model_weights))
    seg_model.eval()
    for param in seg_model.parameters():
        param.requires_grad = False
    print("Loaded and froze pre-trained segmentation model.")

    dice_loss_fn = DiceLoss(sigmoid=True)
    instance_prompt = f"a photo of {args.unique_token}"
    class_prompt = f"a photo of a {args.class_token}"

    tokenizer = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.model_id, subfolder="text_encoder").to(device)

    # Token setup... (omitted for brevity, same as before)
    # ...

    unet = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet").to(device)
    vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae").to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(args.model_id, subfolder="scheduler")

    # Class image generation... (omitted for brevity, same as before)
    # ...

    train_dataset = DreamBoothDataset(args.instance_data_dir, args.instance_mask_dir, args.class_data_dir, tokenizer, instance_prompt, class_prompt)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda examples: collate_fn(examples))
    
    # Optimizer setup... (omitted for brevity, same as before)
    # ...
    
    # --- TRAINING LOOP ---
    for epoch in range(args.num_epochs):
        # ... (same loop structure as original script)
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            with torch.no_grad():
                latents = vae.encode(batch["pixel_values"].to(device)).latent_dist.sample() * vae.config.scaling_factor
            
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            with torch.cuda.amp.autocast():
                encoder_hidden_states = text_encoder(batch["input_ids"].to(device))[0]
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                
                noise_pred_instance, noise_pred_class = noise_pred.chunk(2, dim=0)
                noise_instance, noise_class = noise.chunk(2, dim=0)
                
                instance_loss = torch.nn.functional.mse_loss(noise_pred_instance.float(), noise_instance.float())
                class_loss = torch.nn.functional.mse_loss(noise_pred_class.float(), noise_class.float())
                dreambooth_loss = instance_loss + args.prior_loss_weight * class_loss

                # --- DICE LOSS CALCULATION ---
                pred_original_sample = noise_scheduler.step(noise_pred, timesteps, noisy_latents).pred_original_sample
                pred_instance_latents, _ = pred_original_sample.chunk(2, dim=0)
                
                with torch.no_grad():
                    pred_images = vae.decode(pred_instance_latents / vae.config.scaling_factor, return_dict=False)[0]

                gt_masks = batch["masks"].to(device)
                pred_images_gray = transforms.functional.rgb_to_grayscale(pred_images)
                pred_mask_logits = seg_model(pred_images_gray)
                dice_loss = dice_loss_fn(pred_mask_logits, gt_masks)
                
                loss = dreambooth_loss + args.dice_loss_weight * dice_loss
            
            # ... (scaler, progress bar, metrics, etc. same as original script, but logging dice_loss)
    
    # --- (Checkpointing, final saving, and plotting are the same, just update plot labels for Dice) ---

if __name__ == "__main__":

    main()
