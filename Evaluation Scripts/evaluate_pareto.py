# evaluate.py
#
# An extended script to calculate a comprehensive set of quantitative metrics for
# fine-tuned Stable Diffusion models.
#
# Now supports: CLIP-FID Pareto Curve plotting via --guidance_scales sweep.

import os
import argparse
import torch
from pathlib import Path
from diffusers import StableDiffusionPipeline
from PIL import Image
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt  # --- NEW: For plotting ---

# --- Imports for Metrics ---
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.multimodal.clip_score import CLIPScore
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image import (
    PeakSignalNoiseRatio, 
    StructuralSimilarityIndexMeasure, 
    MultiScaleStructuralSimilarityIndexMeasure
)
from torchmetrics.segmentation import dice 
import segmentation_models_pytorch as smp
from torchvision import transforms
import torchvision.models as models 

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
    parser.add_argument("--num_samples", type=int, default=100, help="Number of images to generate for evaluation.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for image generation.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible generation.")
    
    # --- NEW: Argument for Pareto Curve ---
    parser.add_argument("--guidance_scales", type=float, nargs="+", default=[7.5], 
                        help="List of guidance scales to test. If more than one is provided, a Pareto curve will be plotted.")

    # Extended metrics arguments
    parser.add_argument("--calculate_extended_metrics", action="store_true", help="Flag to enable calculation of IS, KID, SSIM, and PSNR.")
    parser.add_argument("--calculate_frd", action="store_true", help="Flag to enable calculation of Fréchet ResNet Distance (FRD).")
    parser.add_argument("--calculate_ms_ssim", action="store_true", help="Flag to enable calculation of Multi-Scale SSIM (MS-SSIM).")
    
    # Dice/Segmentation arguments
    parser.add_argument("--seg_model_weights", type=str, default=None, help="[Required for Dice] Path to pre-trained segmentation model weights (.pth).")
    parser.add_argument("--real_masks_path", type=str, default=None, help="[Required for Dice] Path to the directory of real segmentation masks.")
    
    args = parser.parse_args()

    # Logic to disable Dice if args are missing
    if args.calculate_extended_metrics and args.seg_model_weights is None:
        print("Warning: --seg_model_weights not provided. Skipping Dice Score calculation.")
    elif args.calculate_extended_metrics and args.real_masks_path is None:
         print("Warning: --real_masks_path not provided. Skipping Dice Score calculation.")
    elif args.seg_model_weights and not args.real_masks_path:
        print("Warning: --seg_model_weights was provided, but --real_masks_path was not. Skipping Dice Score calculation.")
        args.seg_model_weights = None 
    elif not args.seg_model_weights and args.real_masks_path:
        print("Warning: --real_masks_path was provided, but --seg_model_weights was not. Skipping Dice Score calculation.")
        args.seg_model_weights = None 

    return args

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Transform for real images (512x512)
    image_transform = transforms.Compose([
        transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor()
    ])

    # Base output path
    base_output_path = Path(args.output_path)
    base_output_path.mkdir(parents=True, exist_ok=True)
    
    # List to store results for plotting
    pareto_results = []
    
    # --- 1. Loop through Guidance Scales ---
    print(f"\n--- Starting Evaluation for Scales: {args.guidance_scales} ---")

    for scale in args.guidance_scales:
        print(f"\n[Processing Guidance Scale: {scale}]")
        
        # Create sub-directory for this scale
        scale_output_dir = base_output_path / f"scale_{scale}"
        scale_output_dir.mkdir(parents=True, exist_ok=True)
        
        # --- Generate Images ---
        generated_image_files = list(scale_output_dir.glob("*.png"))
        
        if len(generated_image_files) < args.num_samples:
            print(f"  Generating {args.num_samples - len(generated_image_files)} missing images for scale {scale}...")
            
            # Initialize pipeline only when needed to save memory
            pipeline = StableDiffusionPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16).to(device)
            generator = torch.Generator(device=device).manual_seed(args.seed)
            
            num_generated = len(generated_image_files)
            with tqdm(total=args.num_samples, initial=num_generated, desc=f"Gen Scale {scale}") as pbar:
                while num_generated < args.num_samples:
                    current_batch_size = min(args.batch_size, args.num_samples - num_generated)
                    if current_batch_size <= 0: break
                    
                    with torch.no_grad():
                        # --- MODIFIED: Pass guidance_scale ---
                        images = pipeline(
                            prompt=[args.prompt] * current_batch_size, 
                            generator=generator, 
                            num_inference_steps=50,
                            guidance_scale=scale
                        ).images
                    
                    for img in images:
                        img.save(scale_output_dir / f"{num_generated:05d}.png")
                        num_generated += 1
                    pbar.update(len(images))
            
            # Clean up pipeline to free VRAM for metrics
            del pipeline
            torch.cuda.empty_cache()
        else:
            print(f"  Images already exist for scale {scale}. Skipping generation.")

        # --- Initialize/Reset Metrics for this Scale ---
        # We re-initialize or reset metrics here to ensure no data leakage between scales
        print(f"  Initializing metrics for scale {scale}...")
        
        fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
        clip_score = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(device)
        
        # Extended metrics
        inception_score = None
        kid = None
        ssim = None
        psnr = None
        if args.calculate_extended_metrics:
            inception_score = InceptionScore(normalize=True).to(device)
            safe_subset_size = max(1, min(args.num_samples, 50))
            kid = KernelInceptionDistance(subset_size=safe_subset_size, normalize=True).to(device) 
            ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
            psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)

        frd = None
        if args.calculate_frd:
            resnet_features = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            resnet_features.fc = torch.nn.Identity()
            resnet_features.eval()
            frd = FrechetInceptionDistance(feature=resnet_features, normalize=True).to(device)

        ms_ssim = None
        if args.calculate_ms_ssim:
            ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(device)

        seg_model = None
        dice_metric = None
        if args.seg_model_weights:
            seg_model = smp.Unet(encoder_name="resnet34", in_channels=1, classes=1).to(device)
            seg_model.load_state_dict(torch.load(args.seg_model_weights, map_location=device))
            seg_model.eval()
            dice_metric = dice.DiceScore(num_classes=2, average='micro').to(device)

        # --- Process Images ---
        # 1. Update Real Images (Must be done every loop iteration since metrics were reset)
        real_image_files = sorted(list(Path(args.real_images_path).rglob("*.png")))[:args.num_samples]
        real_mask_files = []
        if args.real_masks_path:
             real_mask_files = sorted(list(Path(args.real_masks_path).rglob("*.png")))[:args.num_samples]

        real_images_tensors_cache = [] # Cache for SSIM/PSNR to avoid re-reading

        print(f"  Processing real images for metrics (Scale {scale})...")
        for file_path in tqdm(real_image_files, desc="Real Imgs", leave=False):
            img = Image.open(file_path).convert("RGB")
            img_tensor = image_transform(img).to(device)
            
            fid.update((img_tensor.unsqueeze(0) * 255).to(torch.uint8), real=True)
            
            if args.calculate_extended_metrics:
                kid.update((img_tensor.unsqueeze(0) * 255).to(torch.uint8), real=True)
                real_images_tensors_cache.append(img_tensor) # Store for SSIM/PSNR comparison
            elif args.calculate_ms_ssim:
                real_images_tensors_cache.append(img_tensor)
            
            if args.calculate_frd:
                frd.update(img_tensor.unsqueeze(0), real=True)

        # 2. Update Generated Images
        generated_files_current = sorted(list(scale_output_dir.glob("*.png")))[:args.num_samples]
        print(f"  Processing generated images for metrics (Scale {scale})...")
        
        for i, file_path in enumerate(tqdm(generated_files_current, desc="Gen Imgs", leave=False)):
            img = Image.open(file_path).convert("RGB")
            img_tensor = _pil_to_tensor(img).to(device)
            
            fid.update((img_tensor.unsqueeze(0) * 255).to(torch.uint8), real=False)
            clip_score.update((img_tensor.unsqueeze(0) * 255).to(torch.uint8), args.prompt)

            if args.calculate_frd:
                frd.update(img_tensor.unsqueeze(0), real=False)

            if args.calculate_extended_metrics:
                inception_score.update((img_tensor.unsqueeze(0) * 255).to(torch.uint8))
                kid.update((img_tensor.unsqueeze(0) * 255).to(torch.uint8), real=False)
                # Compare to corresponding real image
                ssim.update(img_tensor.unsqueeze(0), real_images_tensors_cache[i].unsqueeze(0))
                psnr.update(img_tensor.unsqueeze(0), real_images_tensors_cache[i].unsqueeze(0))

            if args.calculate_ms_ssim:
                ms_ssim.update(img_tensor.unsqueeze(0), real_images_tensors_cache[i].unsqueeze(0))

            if seg_model and i < len(real_mask_files):
                img_gray_tensor = transforms.functional.rgb_to_grayscale(img_tensor.unsqueeze(0))
                with torch.no_grad():
                    pred_mask = (torch.sigmoid(seg_model(img_gray_tensor)) > 0.5).int()
                
                gt_mask_img = Image.open(real_mask_files[i]).convert("L")
                mask_transform = transforms.Compose([
                    transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST),
                    transforms.ToTensor()
                ])
                gt_mask_tensor = (mask_transform(gt_mask_img) > 0.5).int().to(device)
                dice_metric.update(pred_mask.squeeze(1), gt_mask_tensor)

        # --- Compute Scores for this Scale ---
        current_results = {"scale": scale}
        
        fid_val = fid.compute().item()
        clip_val = clip_score.compute().item()
        current_results["fid"] = fid_val
        current_results["clip"] = clip_val
        
        print(f"  -> FID: {fid_val:.4f} | CLIP: {clip_val:.4f}")
        
        if args.calculate_frd:
            current_results["frd"] = frd.compute().item()
        
        if args.calculate_extended_metrics:
            is_mean, is_std = inception_score.compute()
            current_results["is"] = is_mean.item()
            kid_mean, kid_std = kid.compute()
            current_results["kid"] = kid_mean.item()
            current_results["ssim"] = ssim.compute().item()
            current_results["psnr"] = psnr.compute().item()
            
        if args.calculate_ms_ssim:
            current_results["ms_ssim"] = ms_ssim.compute().item()

        if dice_metric:
            current_results["dice"] = dice_metric.compute().item()

        pareto_results.append(current_results)
        
        # Cleanup to prevent VRAM accumulation
        del fid, clip_score, frd, inception_score, kid, ssim, psnr, ms_ssim, seg_model, dice_metric
        torch.cuda.empty_cache()

    # --- 2. Plotting (Only if multiple scales provided) ---
    print("\n--- Final Results ---")
    for res in pareto_results:
        print(f"Scale {res['scale']}: FID={res['fid']:.4f}, CLIP={res['clip']:.4f}")

    if len(pareto_results) > 1:
        print("\nPlotting Pareto Curve...")
        
        scales = [r['scale'] for r in pareto_results]
        fids = [r['fid'] for r in pareto_results]
        clips = [r['clip'] for r in pareto_results]
        
        plt.figure(figsize=(10, 6))
        
        # Plot line
        plt.plot(clips, fids, marker='o', linestyle='-', color='b', label='Pareto Curve')
        
        # Annotate points with scale
        for i, scale in enumerate(scales):
            plt.annotate(f"CFG {scale}", 
                         (clips[i], fids[i]), 
                         textcoords="offset points", 
                         xytext=(0,10), 
                         ha='center',
                         fontsize=9)
            
        plt.title(f"CLIP-FID Pareto Curve\nPrompt: {args.prompt[:50]}...")
        plt.xlabel("CLIP Score (Prompt Alignment) -> Higher is better")
        plt.ylabel("FID (Image Quality) -> Lower is better")
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        plt.legend()
        
        # Save plot
        plot_path = base_output_path / "clip_fid_pareto_curve.png"
        plt.savefig(plot_path, dpi=300)
        print(f"Pareto curve saved to: {plot_path}")

    print("\nEvaluation Complete.")

if __name__ == "__main__":
    main()
