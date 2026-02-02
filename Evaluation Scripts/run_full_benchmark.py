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
MODEL_PATH = "./results"  # Path to your fine-tuned model
OUTPUT_ROOT = "./benchmark_results"
UNIQUE_TOKEN = "<nih-xray>" 
NUM_IMAGES_PER_CLASS = 1000 
BATCH_SIZE = 4 

# --- NEW: DATASET CONFIGURATION ---
# Path to the standard NIH CSV file (Data_Entry_2017.csv)
METADATA_CSV_PATH = "./nih_dataset/Data_Entry_2017.csv" 

# Path to the folder containing ALL your real images (mixed together is fine)
# If your images are split into images_001, images_002, etc., put the parent folder here.
REAL_IMAGES_ROOT = "./nih_dataset/images" 

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

# --- EVALUATOR CLASS ---
class MedicalEvaluator:
    def __init__(self, device):
        self.device = device
        print(f"Initializing Metrics on {device}...")
        self.dino_model = timm.create_model('vit_small_patch16_224.dino', pretrained=True).eval().to(device)
        self.dino_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval().to(device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    def get_dino_features(self, images):
        tensors = torch.stack([self.dino_transform(img) for img in images]).to(self.device)
        with torch.no_grad():
            features = self.dino_model(tensors)
        return F.normalize(features, dim=-1)

    def get_clip_features(self, images=None, text=None):
        inputs = self.clip_processor(text=text, images=images, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            if text: features = self.clip_model.get_text_features(**inputs)
            else: features = self.clip_model.get_image_features(**inputs)
        return F.normalize(features, dim=-1)

    def compute_metrics(self, gen_paths, real_paths, prompt):
        # Load Images
        real_pil = [Image.open(p).convert("RGB") for p in real_paths[:len(gen_paths)]]
        gen_pil = [Image.open(p).convert("RGB") for p in gen_paths]
        
        # DINO
        real_dino = self.get_dino_features(real_pil)
        gen_dino = self.get_dino_features(gen_pil)
        sim_matrix = torch.mm(gen_dino, real_dino.transpose(0, 1))
        dino_score = sim_matrix.mean().item()
        
        # CLIP-T
        prompt_feat = self.get_clip_features(text=[prompt])
        gen_clip = self.get_clip_features(images=gen_pil)
        clip_t_score = F.cosine_similarity(gen_clip, prompt_feat).mean().item()
        
        # FID
        self.fid.reset()
        real_tensors = torch.stack([transforms.ToTensor()(img) for img in real_pil]).to(self.device)
        self.fid.update((real_tensors * 255).byte(), real=True)
        gen_tensors = torch.stack([transforms.ToTensor()(img) for img in gen_pil]).to(self.device)
        self.fid.update((gen_tensors * 255).byte(), real=False)
        fid_score = self.fid.compute().item()
        
        return {"DINO": dino_score, "FID": fid_score, "CLIP-T": clip_t_score}

# --- HELPER: FIND IMAGES BY CSV ---
def get_real_images_from_csv(df, pathology, root_dir):
    # Filter DF for rows containing the pathology label
    # NIH dataset uses "Pneumonia|Infiltration", so we use str.contains
    subset = df[df['Finding Labels'].str.contains(pathology, na=False)]
    
    found_paths = []
    # Search recursively for the files because NIH data is often in subfolders (images_001, etc.)
    # Optimization: If you know it's flat, remove rglob and just join path.
    # We pre-scan the directory to map filenames to full paths for speed.
    print(f"  > Scanning real image directory for {pathology} samples...")
    
    # 1. Get list of target filenames from CSV
    target_files = set(subset['Image Index'].tolist())
    
    # 2. Walk directory to find matches (Efficiently)
    count = 0
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file in target_files:
                found_paths.append(os.path.join(root, file))
                count += 1
                if count >= NUM_IMAGES_PER_CLASS: # Stop once we have enough
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
    pipeline = StableDiffusionPipeline.from_pretrained(MODEL_PATH, torch_dtype=torch.float16).to(device)
    pipeline.set_progress_bar_config(disable=True)
    
    evaluator = MedicalEvaluator(device)
    results = []
    
    # 3. Loop Pathologies
    for pathology, prompt in PATHOLOGIES.items():
        print(f"\n--- Processing: {pathology} ---")
        class_out_dir = Path(OUTPUT_ROOT) / pathology
        class_out_dir.mkdir(parents=True, exist_ok=True)
        
        # A. FIND REAL IMAGES via CSV
        real_paths = get_real_images_from_csv(df, pathology, REAL_IMAGES_ROOT)
        
        if len(real_paths) < 10:
            print(f"SKIPPING {pathology}: Found only {len(real_paths)} real images (Need at least 10).")
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
        print(f"  > Calculating Metrics...")
        metrics = evaluator.compute_metrics(gen_files, real_paths, prompt)
        metrics["Pathology"] = pathology
        results.append(metrics)
        print(f"  > {pathology}: DINO={metrics['DINO']:.4f}, FID={metrics['FID']:.4f}")

    # 4. RESULTS
    print("\n" + "="*50)
    print("FINAL BENCHMARK RESULTS")
    print("="*50)
    df_res = pd.DataFrame(results)[["Pathology", "DINO", "FID", "CLIP-T"]]
    print(df_res.to_string(index=False))
    print("-" * 50)
    print("AVERAGE:")
    print(df_res.mean(numeric_only=True).to_string())
    df_res.to_csv(Path(OUTPUT_ROOT) / "final_metrics.csv", index=False)

if __name__ == "__main__":
    main()
