import os
import torch
import pandas as pd
import torch.nn.functional as F
from pathlib import Path
from diffusers import StableDiffusionPipeline
from tqdm.auto import tqdm
from PIL import Image
import timm
from transformers import CLIPProcessor, CLIPModel
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms

# --- CONFIGURATION ---
MODEL_PATH = "./all_losses_model_18/exp17_long_run/final_model" # Update this to your path
OUTPUT_ROOT = "./benchmark_results"
UNIQUE_TOKEN = "<nih-xray>" 
NUM_IMAGES_PER_CLASS = 1000 
BATCH_SIZE = 4       # For Generation (Keep small)
EVAL_BATCH_SIZE = 32 # For Metrics (New setting to prevent OOM)

# --- CSV CONFIGURATION ---
METADATA_CSV_PATH = "./sample/sample_labels.csv" # Update to your CSV path
REAL_IMAGES_ROOT = "./sample/images"        # Update to your images path

# Define Pathologies
PATHOLOGIES = {
    "Atelectasis":  f"A photo of {UNIQUE_TOKEN} showing Atelectasis",
    "Cardiomegaly": f"A photo of {UNIQUE_TOKEN} showing Cardiomegaly",
    "Effusion":     f"A photo of {UNIQUE_TOKEN} showing Effusion",
    "Infiltration": f"A photo of {UNIQUE_TOKEN} showing Infiltration",
    "Mass":         f"A photo of {UNIQUE_TOKEN} showing a Mass",
    "Nodule":       f"A photo of {UNIQUE_TOKEN} showing a Nodule",
    "Pneumonia":    f"A photo of {UNIQUE_TOKEN} showing Pneumonia",
    "Pneumothorax": f"A photo of {UNIQUE_TOKEN} showing Pneumothorax",
    "Consolidation": f"A photo of {UNIQUE_TOKEN} showing Consolidation",
    "No Finding":   f"A photo of {UNIQUE_TOKEN} with No Findings"
}

# --- EVALUATOR CLASS (MEMORY OPTIMIZED) ---
class MedicalEvaluator:
    def __init__(self, device):
        self.device = device
        print(f"Initializing Metrics on {device}...")
        
        # DINO
        self.dino_model = timm.create_model('vit_small_patch16_224.dino', pretrained=True).eval().to(device)
        self.dino_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        
        # CLIP
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval().to(device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # FID
        self.fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    def get_features_batched(self, image_paths, model_type="dino"):
        """Process features in batches to save memory"""
        all_features = []
        
        for i in range(0, len(image_paths), EVAL_BATCH_SIZE):
            batch_paths = image_paths[i:i + EVAL_BATCH_SIZE]
            batch_pil = [Image.open(p).convert("RGB") for p in batch_paths]
            
            with torch.no_grad():
                if model_type == "dino":
                    tensors = torch.stack([self.dino_transform(img) for img in batch_pil]).to(self.device)
                    feats = self.dino_model(tensors)
                elif model_type == "clip":
                    inputs = self.clip_processor(images=batch_pil, return_tensors="pt", padding=True).to(self.device)
                    feats = self.clip_model.get_image_features(**inputs)
            
            all_features.append(feats.cpu()) # Move to CPU immediately to free VRAM
            del batch_pil, batch_paths
            torch.cuda.empty_cache()
            
        return F.normalize(torch.cat(all_features).to(self.device), dim=-1) # Move back to GPU for matmul

    def update_fid_batched(self, image_paths, is_real):
        """Update FID stats in batches"""
        to_tensor = transforms.ToTensor()
        
        for i in range(0, len(image_paths), EVAL_BATCH_SIZE):
            batch_paths = image_paths[i:i + EVAL_BATCH_SIZE]
            batch_pil = [Image.open(p).convert("RGB") for p in batch_paths]
            # FID expects uint8 [0, 255]
            batch_tensors = torch.stack([to_tensor(img) for img in batch_pil]).to(self.device)
            batch_uint8 = (batch_tensors * 255).byte()
            
            self.fid.update(batch_uint8, real=is_real)
            
            del batch_tensors, batch_uint8, batch_pil
            torch.cuda.empty_cache()

    def compute_metrics(self, gen_paths, real_paths, prompt):
        # 1. DINO Score
        print("    Computing DINO...")
        real_dino = self.get_features_batched(real_paths, "dino")
        gen_dino = self.get_features_batched(gen_paths, "dino")
        
        # Calculate Similarity Matrix (GPU is fine here, matrices are small)
        sim_matrix = torch.mm(gen_dino, real_dino.transpose(0, 1))
        dino_score = sim_matrix.mean().item()
        
        # Cleanup DINO features
        del real_dino, gen_dino, sim_matrix
        torch.cuda.empty_cache()
        
        # 2. CLIP-T Score
        print("    Computing CLIP-T...")
        gen_clip = self.get_features_batched(gen_paths, "clip")
        
        with torch.no_grad():
            inputs = self.clip_processor(text=[prompt], return_tensors="pt", padding=True).to(self.device)
            prompt_feat = F.normalize(self.clip_model.get_text_features(**inputs), dim=-1)
        
        clip_t_score = F.cosine_similarity(gen_clip, prompt_feat).mean().item()
        
        del gen_clip, prompt_feat
        torch.cuda.empty_cache()
        
        # 3. FID Score
        print("    Computing FID...")
        self.fid.reset()
        self.update_fid_batched(real_paths, is_real=True)
        self.update_fid_batched(gen_paths, is_real=False)
        fid_score = self.fid.compute().item()
        
        return {"DINO": dino_score, "FID": fid_score, "CLIP-T": clip_t_score}

# --- HELPER: FIND IMAGES BY CSV ---
def get_real_images_from_csv(df, pathology, root_dir):
    subset = df[df['Finding Labels'].str.contains(pathology, na=False)]
    target_files = set(subset['Image Index'].tolist())
    found_paths = []
    
    # Optimization: Scan directory once if you want, but for simplicity:
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file in target_files:
                found_paths.append(os.path.join(root, file))
                if len(found_paths) >= NUM_IMAGES_PER_CLASS:
                    return found_paths
    return found_paths

# --- MAIN EXECUTION ---
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    # 1. Load CSV
    print(f"Loading Metadata from {METADATA_CSV_PATH}...")
    df = pd.read_csv(METADATA_CSV_PATH)
    
    # 2. Load Pipeline
    print(f"Loading Model from {MODEL_PATH}...")
    # Add low_cpu_mem_usage=False to match your environment warning
    pipeline = StableDiffusionPipeline.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, low_cpu_mem_usage=False).to(device)
    pipeline.set_progress_bar_config(disable=True)
    
    evaluator = MedicalEvaluator(device)
    results = []
    
    # 3. Loop Pathologies
    for pathology, prompt in PATHOLOGIES.items():
        print(f"\n--- Processing: {pathology} ---")
        class_out_dir = Path(OUTPUT_ROOT) / pathology
        class_out_dir.mkdir(parents=True, exist_ok=True)
        
        # A. FIND REAL IMAGES
        real_paths = get_real_images_from_csv(df, pathology, REAL_IMAGES_ROOT)
        if len(real_paths) < 10:
            print(f"SKIPPING {pathology}: Found only {len(real_paths)} real images.")
            continue
        print(f"  > Found {len(real_paths)} real images for reference.")

        # B. GENERATION
        existing_gen = list(class_out_dir.glob("*.png"))
        images_needed = NUM_IMAGES_PER_CLASS - len(existing_gen)
        
        if images_needed > 0:
            print(f"  > Generating {images_needed} images...")
            generator = torch.Generator(device=device).manual_seed(42)
            for i in tqdm(range(0, images_needed, BATCH_SIZE)):
                curr_batch = min(BATCH_SIZE, images_needed - i)
                with torch.no_grad():
                    imgs = pipeline([prompt]*curr_batch, generator=generator, num_inference_steps=50).images
                for j, img in enumerate(imgs):
                    img.save(class_out_dir / f"{len(existing_gen) + i + j:05d}.png")
        
        # C. EVALUATION
        gen_files = sorted(list(class_out_dir.glob("*.png")))[:NUM_IMAGES_PER_CLASS]
        print(f"  > Calculating Metrics (Batched)...")
        metrics = evaluator.compute_metrics(gen_files, real_paths, prompt)
        metrics["Pathology"] = pathology
        results.append(metrics)
        print(f"  > {pathology}: DINO={metrics['DINO']:.4f}, FID={metrics['FID']:.4f}")

    # 4. RESULTS
    print("\n" + "="*50)
    print("FINAL BENCHMARK RESULTS")
    print("="*50)
    if results:
        df_res = pd.DataFrame(results)[["Pathology", "DINO", "FID", "CLIP-T"]]
        print(df_res.to_string(index=False))
        print("-" * 50)
        print("AVERAGE:")
        print(df_res.mean(numeric_only=True).to_string())
        df_res.to_csv(Path(OUTPUT_ROOT) / "final_metrics.csv", index=False)
    else:
        print("No results computed.")

if __name__ == "__main__":
    main()
