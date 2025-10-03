# evaluate_plus.py
#
# An extended script to calculate a comprehensive set of quantitative metrics for
# fine-tuned Stable Diffusion models.
#
# Includes: FID, CLIP Score, Inception Score (IS), Kernel Inception Distance (KID),
#           Dice Score (for segmentation), SSIM, PSNR, and Aesthetic Score.

import os
import argparse
import torch
from pathlib import Path
from diffusers import StableDiffusionPipeline
from PIL import Image
from tqdm.auto import tqdm
import numpy as np

# --- New Imports for Extended Metrics ---
# Make sure to install these packages:
# pip install torchmetrics segmentation-models-pytorch monai timm transformers
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.multimodal.clip_score import CLIPScore
from torchmetrics.image.inception import InceptionScore, KernelInceptionDistance
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics import Dice
import segmentation_models_pytorch as smp
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel

# --- Helper Functions ---

def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert PIL image to torch tensor normalized to [0, 1]."""
    return transforms.ToTensor()(img)

def parse_args():
    """Parses command-line arguments for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Stable Diffusion models with extended metrics.")
    # Core arguments
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model directory.")
    parser.add_argument("--real_images_path", type=str, required=True, help="Path to the directory of real images for comparison.")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to use for generating images.")
    parser.add_argument("--output_path", type=str, default="./generated_images_for_eval", help="Directory to save generated images.")
    
    # Generation arguments
    parser.add_argument("--num_samples", type=int, default=100, help="Number of images to generate for evaluation. A lower number for a quick test, 1k+ for more reliable metrics.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for image generation.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible generation.")

    # Extended metrics arguments
    parser.add_argument("--calculate_extended_metrics", action="store_true", help="Flag to enable calculation of IS, KID, SSIM, PSNR, Dice, and Aesthetic score.")
    parser.add_argument("--seg_model_weights", type=str, default=None, help="[Required for Dice] Path to pre-trained segmentation model weights (.pth).")
    parser.add_argument("--real_masks_path", type=str, default=None, help="[Required for Dice] Path to the directory of real segmentation masks.")
    parser.add_argument("--aesthetic_model_path", type=str, default="christophschuhmann/improved-aesthetic-predictor", help="HF model name or path for the aesthetic predictor.")
    
    args = parser.parse_args()

    if args.calculate_extended_metrics and (not args.seg_model_weights or not args.real_masks_path):
        print("Warning: To calculate the Dice Score, both --seg_model_weights and --real_masks_path must be provided. Skipping Dice Score calculation.")
        args.seg_model_weights = None # Disable Dice calculation

    return args

class AestheticPredictor(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(768, 1024),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(1024, 128),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(64, 16),
            torch.nn.Linear(16, 1)
        )
        state_dict = torch.hub.load_state_dict_from_url(
            f"https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/main/{os.path.basename(model_path)}?raw=true"
        )
        self.mlp.load_state_dict(state_dict)

    def forward(self, x):
        inputs = self.processor(images=x, return_tensors="pt")
        inputs = {k: v.to(self.clip_model.device) for k, v in inputs.items()}
        with torch.no_grad():
            embed = self.clip_model.get_image_features(**inputs)
            embed = embed / torch.linalg.norm(embed, dim=-1, keepdim=True)
        return self.mlp(embed.float())


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- 1. Generate Images (if they don't exist) ---
    generated_images_path = Path(args.output_path)
    generated_images_path.mkdir(parents=True, exist_ok=True)
    generated_image_files = list(generated_images_path.glob("*.png"))

    if len(generated_image_files) < args.num_samples:
        print(f"Found {len(generated_image_files)} images, but {args.num_samples} are required. Generating missing images...")
        pipeline = StableDiffusionPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16).to(device)
        generator = torch.Generator(device=device).manual_seed(args.seed)
        
        num_generated = len(generated_image_files)
        with tqdm(total=args.num_samples, initial=num_generated) as pbar:
            while num_generated < args.num_samples:
                current_batch_size = min(args.batch_size, args.num_samples - num_generated)
                if current_batch_size <= 0: break
                
                with torch.no_grad():
                    images = pipeline(prompt=[args.prompt] * current_batch_size, generator=generator, num_inference_steps=50).images
                
                for img in images:
                    img.save(generated_images_path / f"{num_generated:05d}.png")
                    num_generated += 1
                pbar.update(len(images))

        print(f"Successfully generated images. Total count: {num_generated}")
        del pipeline
        torch.cuda.empty_cache()
    else:
        print(f"Found {len(generated_image_files)} existing images. Skipping generation.")

    generated_image_files = sorted(list(generated_images_path.glob("*.png")))[:args.num_samples]
    real_image_files = sorted(list(Path(args.real_images_path).rglob("*.png")))[:args.num_samples]

    # --- 2. Initialize Metrics ---
    print("\nInitializing metrics...")
    # Standard metrics
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device) # Use normalize=True for uint8 images
    clip_score = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(device)
    
    # Extended metrics
    if args.calculate_extended_metrics:
        inception_score = InceptionScore(normalize=True).to(device)
        kid = KernelInceptionDistance(subset_size=50, normalize=True).to(device) # subset_size should be smaller than num_samples
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
        aesthetic_scorer = AestheticPredictor(args.aesthetic_model_path).to(device).eval()
        total_aesthetic_score = 0.0

    # Dice/Segmentation metrics
    if args.seg_model_weights:
        seg_model = smp.Unet(encoder_name="resnet34", in_channels=1, classes=1).to(device)
        seg_model.load_state_dict(torch.load(args.seg_model_weights, map_location=device))
        seg_model.eval()
        dice = Dice(average='micro').to(device)
        real_mask_files = sorted(list(Path(args.real_masks_path).rglob("*.png")))[:args.num_samples]
        if len(real_mask_files) != len(real_image_files):
            raise ValueError("Number of real images and real masks must be the same for Dice score calculation.")

    # --- 3. Process Images and Update Metrics ---
    print("Processing real images...")
    real_images_tensors = []
    for file_path in tqdm(real_image_files):
        img = Image.open(file_path).convert("RGB")
        img_tensor = _pil_to_tensor(img).to(device) # Shape: [C, H, W]
        # Torchmetrics FID/IS/KID expect [N, C, H, W] and uint8 [0, 255]
        fid.update((img_tensor.unsqueeze(0) * 255).to(torch.uint8), real=True)
        if args.calculate_extended_metrics:
            kid.update((img_tensor.unsqueeze(0) * 255).to(torch.uint8), real=True)
            real_images_tensors.append(img_tensor)

    print("Processing generated images...")
    for i, file_path in enumerate(tqdm(generated_image_files)):
        img = Image.open(file_path).convert("RGB")
        img_tensor = _pil_to_tensor(img).to(device) # Shape: [C, H, W]
        
        # Update standard metrics
        fid.update((img_tensor.unsqueeze(0) * 255).to(torch.uint8), real=False)
        clip_score.update((img_tensor.unsqueeze(0) * 255).to(torch.uint8), args.prompt)

        # Update extended metrics
        if args.calculate_extended_metrics:
            inception_score.update((img_tensor.unsqueeze(0) * 255).to(torch.uint8))
            kid.update((img_tensor.unsqueeze(0) * 255).to(torch.uint8), real=False)
            
            # For SSIM/PSNR, compare generated image to the corresponding real image
            ssim.update(img_tensor.unsqueeze(0), real_images_tensors[i].unsqueeze(0))
            psnr.update(img_tensor.unsqueeze(0), real_images_tensors[i].unsqueeze(0))

            # Update aesthetic score
            with torch.no_grad():
                total_aesthetic_score += aesthetic_scorer(img).item()

        # Update Dice score
        if args.seg_model_weights:
            # Get predicted mask from generated image
            img_gray_tensor = transforms.functional.rgb_to_grayscale(img_tensor.unsqueeze(0))
            with torch.no_grad():
                pred_mask_logits = seg_model(img_gray_tensor)
                pred_mask = (torch.sigmoid(pred_mask_logits) > 0.5).int()

            # Load ground truth mask
            gt_mask_img = Image.open(real_mask_files[i]).convert("L")
            gt_mask_tensor = (transforms.ToTensor()(gt_mask_img) > 0.5).int().to(device)
            dice.update(pred_mask, gt_mask_tensor)

    # --- 4. Compute and Display Final Scores ---
    print("\n--- Evaluation Results ---")
    
    fid_score = fid.compute()
    print(f"Fréchet Inception Distance (FID): {fid_score.item():.4f} (Lower is better)")

    final_clip_score = clip_score.compute()
    print(f"CLIP Score: {final_clip_score.item():.4f} (Higher is better)")

    if args.calculate_extended_metrics:
        is_mean, is_std = inception_score.compute()
        print(f"Inception Score (IS): {is_mean.item():.4f} ± {is_std.item():.4f} (Higher is better)")
        
        kid_mean, kid_std = kid.compute()
        print(f"Kernel Inception Distance (KID): {kid_mean.item():.4f} ± {kid_std.item():.4f} (Lower is better)")

        final_ssim = ssim.compute()
        print(f"Structural Similarity Index (SSIM): {final_ssim.item():.4f} (Higher is better)")

        final_psnr = psnr.compute()
        print(f"Peak Signal-to-Noise Ratio (PSNR): {final_psnr.item():.4f} (Higher is better)")

        avg_aesthetic_score = total_aesthetic_score / args.num_samples
        print(f"Average Aesthetic Score: {avg_aesthetic_score:.4f} (Higher is better, typically 1-10)")

    if args.seg_model_weights:
        final_dice = dice.compute()
        print(f"Dice Score (Segmentation Accuracy): {final_dice.item():.4f} (Higher is better)")
        
    print("--------------------------\n")

if __name__ == "__main__":
    main()