import os
import torch
import pandas as pd
import torch.nn.functional as F
from pathlib import Path
from diffusers import StableDiffusionPipeline
from tqdm.auto import tqdm
from PIL import Image
import timm
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms

# --- CONFIGURATION ---
# Using the verified local subfolder path
MODEL_PATH = "/home/mluser/.cache/huggingface/hub/models--vanillacoke--all_losses_model_1/snapshots/d1a724c741dffc7d7a67ec2f54e0fb50a7352662/final_model"
OUTPUT_DIR = "./eval_output/leakage_fixed_overall"
UNIQUE_TOKEN = "<nih-xray>" 

# Data paths
METADATA_CSV_PATH = "./Data_Entry_2017.csv" 
REAL_IMAGES_ROOT = "./eval_unseen_data/all_images" 
TRAIN_SAMPLE_CSV = "./sample/sample_labels.csv"

NUM_TEST_SAMPLES = 1000 
BATCH_SIZE = 4       
EVAL_BATCH_SIZE = 32 

class MedicalEvaluator:
    def __init__(self, device):
        self.device = device
        print(f"Initializing Evaluation Models on {device}...")
        
        # DINO for structural similarity
        self.dino_model = timm.create_model('vit_small_patch16_224.dino', pretrained=True).eval().to(device)
        self.dino_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        
        # FID for distribution quality
        self.fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    def get_dino_features(self, image_paths):
        all_features = []
        for i in range(0, len(image_paths), EVAL_BATCH_SIZE):
            batch_paths = image_paths[i:i + EVAL_BATCH_SIZE]
            batch_pil = [Image.open(p).convert("RGB") for p in batch_paths]
            with torch.no_grad():
                tensors = torch.stack([self.dino_transform(img) for img in batch_pil]).to(self.device)
                feats = self.dino_model(tensors)
            all_features.append(feats.cpu())
            torch.cuda.empty_cache()
        return F.normalize(torch.cat(all_features).to(self.device), dim=-1)

    def update_fid(self, image_paths, is_real):
        to_tensor = transforms.ToTensor()
        for i in range(0, len(image_paths), EVAL_BATCH_SIZE):
            batch_paths = image_paths[i:i + EVAL_BATCH_SIZE]
            batch_pil = [Image.open(p).convert("RGB") for p in batch_paths]
            batch_tensors = torch.stack([to_tensor(img) for img in batch_pil]).to(self.device)
            # FID expects uint8 [0, 255]
            self.fid.update((batch_tensors * 255).byte(), real=is_real)
            torch.cuda.empty_cache()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. IDENTIFY UNSEEN IMAGES (DATA LEAKAGE CHECK)
    print("Checking for data leakage...")
    train_df = pd.read_csv(TRAIN_SAMPLE_CSV)
    train_ids = set(train_df['Image Index'].astype(str).tolist())
    
    if not os.path.exists(REAL_IMAGES_ROOT) or not os.listdir(REAL_IMAGES_ROOT):
        print(f"CRITICAL ERROR: {REAL_IMAGES_ROOT} is empty. Please copy your images first.")
        return

    all_files = [f for f in os.listdir(REAL_IMAGES_ROOT) if f.lower().endswith(('.png', '.jpg'))]
    all_unseen_files = [f for f in all_files if f not in train_ids]
    
    if not all_unseen_files:
        print("CRITICAL ERROR: 0 unseen images found. All images in folder were used for training.")
        return

    real_paths = [os.path.join(REAL_IMAGES_ROOT, f) for f in all_unseen_files[:NUM_TEST_SAMPLES]]
    print(f"Verified: {len(all_unseen_files)} images are strictly unseen.")
    print(f"Using {len(real_paths)} reference images for this benchmark.")

    # 2. LOAD PIPELINE
    print(f"Loading local pipeline from: {MODEL_PATH}")
    try:
        pipeline = StableDiffusionPipeline.from_pretrained(
            MODEL_PATH, 
            torch_dtype=torch.float16,
            local_files_only=True
        ).to(device)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    # 3. GENERATE SYNTHETIC SAMPLES
    gen_dir = Path(OUTPUT_DIR) / "generated_samples"
    gen_dir.mkdir(exist_ok=True)
    
    print(f"Generating {len(real_paths)} synthetic images...")
    prompt = f"a photo of {UNIQUE_TOKEN}"
    generator = torch.manual_seed(42)
    
    for i in tqdm(range(0, len(real_paths), BATCH_SIZE)):
        curr_batch = min(BATCH_SIZE, len(real_paths) - i)
        with torch.no_grad():
            imgs = pipeline([prompt]*curr_batch, generator=generator).images
        for j, img in enumerate(imgs):
            img.save(gen_dir / f"gen_{i+j:05d}.png")

    # 4. CALCULATE BENCHMARK METRICS
    evaluator = MedicalEvaluator(device)
    gen_paths = sorted(list(gen_dir.glob("*.png")))
    
    print("Computing metrics (FID and DINO)...")
    evaluator.update_fid(real_paths, is_real=True)
    evaluator.update_fid(gen_paths, is_real=False)
    fid_val = evaluator.fid.compute().item()
    
    real_feats = evaluator.get_dino_features(real_paths)
    gen_feats = evaluator.get_dino_features(gen_paths)
    dino_val = torch.mm(gen_feats, real_feats.transpose(0, 1)).mean().item()

    # 5. SAVE RESULTS
    results_file = os.path.join(OUTPUT_DIR, "overall_results.txt")
    with open(results_file, "w") as f:
        f.write(f"Benchmark Results\n")
        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Reference Images: {len(real_paths)}\n")
        f.write(f"FID Score: {fid_val:.4f}\n")
        f.write(f"DINO Similarity: {dino_val:.4f}\n")

    print(f"\n--- Benchmark Results ---")
    print(f"FID: {fid_val:.4f}")
    print(f"DINO: {dino_val:.4f}")
    print(f"Full report saved to: {results_file}")

if __name__ == "__main__":
    main()
