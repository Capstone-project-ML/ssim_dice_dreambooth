import os
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from diffusers import StableDiffusionPipeline
from PIL import Image
from tqdm.auto import tqdm
import numpy as np
import timm
from transformers import CLIPProcessor, CLIPModel

# Torchmetrics for Distribution Metrics (Sanity Checks)
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchvision import transforms

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DreamBooth models with Subject & Prompt Fidelity metrics.")
    
    # Paths
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned model.")
    parser.add_argument("--real_images_path", type=str, required=True, help="Path to real reference images (subject).")
    parser.add_argument("--prompt", type=str, required=True, help="The prompt used for generation (e.g., 'a [V] dog in the jungle').")
    parser.add_argument("--output_path", type=str, default="./generated_eval", help="Where to save generated images.")
    
    # Generation Settings
    parser.add_argument("--num_samples", type=int, default=100, help="Total images to evaluate.")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    
    # Metrics Toggle
    parser.add_argument("--skip_distribution_metrics", action="store_true", help="Skip FID/KID (faster).")
    
    return parser.parse_args()

class DreamBoothEvaluator:
    def __init__(self, device):
        self.device = device
        print(f"Initializing Metrics on {device}...")

        # 1. DINO (Subject Fidelity) - The Gold Standard per DreamBooth Paper
        # We use ViT-S/16 DINO as specified in the paper
        print("Loading DINO (ViT-S/16)...")
        self.dino_model = timm.create_model('vit_small_patch16_224.dino', pretrained=True).eval().to(device)
        self.dino_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        # 2. CLIP (Prompt Fidelity & CLIP-I)
        # We use standard CLIP ViT-B/32 or similar for text-image alignment
        print("Loading CLIP...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval().to(device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # 3. Distribution Metrics (FID/KID) - Sanity checks for realism
        self.fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
        self.kid = KernelInceptionDistance(subset_size=50, normalize=True).to(device)

    def get_dino_features(self, images):
        """Extracts DINO CLS token features."""
        # images: List of PIL Images
        tensors = torch.stack([self.dino_transform(img) for img in images]).to(self.device)
        with torch.no_grad():
            features = self.dino_model(tensors) # Shape: [B, 384]
        return F.normalize(features, dim=-1)

    def get_clip_features(self, images=None, text=None):
        """Extracts CLIP image or text features."""
        inputs = self.clip_processor(text=text, images=images, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            if text:
                features = self.clip_model.get_text_features(**inputs)
            else:
                features = self.clip_model.get_image_features(**inputs)
        return F.normalize(features, dim=-1)

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- 1. Generate or Load Images ---
    generated_path = Path(args.output_path)
    generated_path.mkdir(parents=True, exist_ok=True)
    
    # Check existing images
    existing_files = sorted(list(generated_path.glob("*.png")))
    
    if len(existing_files) < args.num_samples:
        print(f"Generating {args.num_samples - len(existing_files)} images...")
        pipeline = StableDiffusionPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16).to(device)
        pipeline.set_progress_bar_config(disable=True)
        generator = torch.Generator(device=device).manual_seed(args.seed)

        needed = args.num_samples - len(existing_files)
        # Simple batch generation loop
        for i in tqdm(range(0, needed, args.batch_size)):
            batch_size = min(args.batch_size, needed - i)
            with torch.no_grad():
                imgs = pipeline([args.prompt]*batch_size, generator=generator, num_inference_steps=50).images
            
            for j, img in enumerate(imgs):
                img.save(generated_path / f"gen_{len(existing_files) + i + j:04d}.png")
        
        del pipeline, imgs
        torch.cuda.empty_cache()
    
    # Reload all files to ensure order
    gen_files = sorted(list(generated_path.glob("*.png")))[:args.num_samples]
    real_files = sorted(list(Path(args.real_images_path).glob("*.png"))) # or *.jpg
    
    if not real_files:
        raise ValueError(f"No images found in {args.real_images_path}")

    # --- 2. Initialize Evaluator ---
    evaluator = DreamBoothEvaluator(device)

    # --- 3. Pre-compute Real Image Embeddings (Reference) ---
    print("Computing embeddings for Real Images...")
    real_pil = [Image.open(f).convert("RGB") for f in real_files]
    
    # DINO Embeddings for Real Images
    real_dino_feats = evaluator.get_dino_features(real_pil) # [M, 384]
    
    # CLIP Image Embeddings for Real Images (for CLIP-I)
    real_clip_feats = evaluator.get_clip_features(images=real_pil) # [M, 512]

    # Pre-compute Prompt Embedding (for CLIP-T)
    prompt_feat = evaluator.get_clip_features(text=[args.prompt]) # [1, 512]

    # FID/KID Update for Real
    if not args.skip_distribution_metrics:
        # Resize for FID (FID expects uint8 tensor [0, 255] or float normalized)
        # Torchmetrics handles normalization if normalize=True. We just need to load to tensor.
        to_tensor = transforms.ToTensor()
        real_tensors = torch.stack([to_tensor(img) for img in real_pil]).to(device)
        
        # Convert to uint8 [0, 255] for standard FID usage in torchmetrics
        real_uint8 = (real_tensors * 255).byte()
        evaluator.fid.update(real_uint8, real=True)
        evaluator.kid.update(real_uint8, real=True)

    # --- 4. Evaluate Generated Images ---
    print("Evaluating Generated Images...")
    
    dino_scores = []
    clip_i_scores = []
    clip_t_scores = []
    
    batch_size = 16
    for i in tqdm(range(0, len(gen_files), batch_size)):
        batch_files = gen_files[i:i+batch_size]
        batch_pil = [Image.open(f).convert("RGB") for f in batch_files]
        
        # A. DINO Score Calculation (Subject Fidelity)
        # Compare batch generated images vs ALL real images
        gen_dino = evaluator.get_dino_features(batch_pil) # [B, 384]
        # Pairwise cosine similarity: [B, M]
        sim_matrix = torch.mm(gen_dino, real_dino_feats.transpose(0, 1)) 
        # Average max similarity? Or average pairwise? 
        # The paper says "Average pairwise cosine similarity".
        # We take the mean of the similarity matrix.
        dino_scores.append(sim_matrix.mean(dim=1).cpu()) 

        # B. CLIP-I Calculation (Subject Fidelity - Baseline Metric)
        gen_clip = evaluator.get_clip_features(images=batch_pil)
        sim_matrix_clip = torch.mm(gen_clip, real_clip_feats.transpose(0, 1))
        clip_i_scores.append(sim_matrix_clip.mean(dim=1).cpu())

        # C. CLIP-T Calculation (Prompt Fidelity)
        # Cosine sim between Gen Image and Prompt Text
        # gen_clip: [B, 512], prompt_feat: [1, 512]
        prompt_sim = F.cosine_similarity(gen_clip, prompt_feat)
        clip_t_scores.append(prompt_sim.cpu())

        # D. FID/KID Update
        if not args.skip_distribution_metrics:
            to_tensor = transforms.ToTensor()
            gen_tensors = torch.stack([to_tensor(img) for img in batch_pil]).to(device)
            gen_uint8 = (gen_tensors * 255).byte()
            evaluator.fid.update(gen_uint8, real=False)
            evaluator.kid.update(gen_uint8, real=False)

    # --- 5. Aggregate Results ---
    avg_dino = torch.cat(dino_scores).mean().item()
    avg_clip_i = torch.cat(clip_i_scores).mean().item()
    avg_clip_t = torch.cat(clip_t_scores).mean().item()
    
    print("\n" + "="*40)
    print("   DREAMBOOTH EVALUATION RESULTS")
    print("="*40)
    print(f"Subject Fidelity (DINO):  {avg_dino:.4f}  <-- PRIMARY METRIC (Target > 0.6)")
    print(f"Subject Fidelity (CLIP-I):{avg_clip_i:.4f}")
    print(f"Prompt Fidelity (CLIP-T): {avg_clip_t:.4f}  (Target > 0.25)")
    
    if not args.skip_distribution_metrics:
        fid_score = evaluator.fid.compute().item()
        kid_score = evaluator.kid.compute()[0].item()
        print("-" * 20)
        print(f"FID Score: {fid_score:.4f} (Lower is better)")
        print(f"KID Score: {kid_score:.4f} (Lower is better)")
    
    print("="*40)
    print("Note: DINO is the preferred metric for subject identity per Ruiz et al. (2023).")

if __name__ == "__main__":
    main()
