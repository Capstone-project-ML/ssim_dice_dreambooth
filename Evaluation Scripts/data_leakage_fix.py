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
MODEL_PATH = "./results_seed42/final_model" 
OUTPUT_DIR = "./eval_output/leakage_fixed_overall"
UNIQUE_TOKEN = "<nih-xray>" 

# Data paths for unseen data
METADATA_CSV_PATH = "./Data_Entry_2017.csv" 
REAL_IMAGES_ROOT = "./eval_unseen_data/all_images" 
TRAIN_SAMPLE_CSV = "./sample/sample_labels.csv"

NUM_TEST_SAMPLES = 1000 # Standard for a robust FID score
BATCH_SIZE = 4       
EVAL_BATCH_SIZE = 32 

class MedicalEvaluator:
    def __init__(self, device):
        self.device = device
        # DINO for feature similarity
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
            self.fid.update((batch_tensors * 255).byte(), real=is_real)
            torch.cuda.empty_cache()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. IDENTIFY UNSEEN IMAGES
    full_df = pd.read_csv(METADATA_CSV_PATH)
    train_df = pd.read_csv(TRAIN_SAMPLE_CSV)
    train_ids = set(train_df['Image Index'].tolist())
    
    # Find images in the new folder that were NOT in training
    all_unseen_files = [f for f in os.listdir(REAL_IMAGES_ROOT) if f not in train_ids]
    real_paths = [os.path.join(REAL_IMAGES_ROOT, f) for f in all_unseen_files[:NUM_TEST_SAMPLES]]
    
    print(f"Total Unseen images identified: {len(all_unseen_files)}")
    print(f"Using {len(real_paths)} images for reference.")

    # 2. LOAD MODEL
    pipeline = StableDiffusionPipeline.from_pretrained(MODEL_PATH, torch_dtype=torch.float16).to(device)
    
    # 3. GENERATE SAMPLES
    gen_dir = Path(OUTPUT_DIR) / "generated_samples"
    gen_dir.mkdir(exist_ok=True)
    
    print(f"Generating {NUM_TEST_SAMPLES} synthetic images...")
    prompt = f"a photo of {UNIQUE_TOKEN}"
    generator = torch.manual_seed(42)
    
    for i in tqdm(range(0, NUM_TEST_SAMPLES, BATCH_SIZE)):
        curr_batch = min(BATCH_SIZE, NUM_TEST_SAMPLES - i)
        imgs = pipeline([prompt]*curr_batch, generator=generator).images
        for j, img in enumerate(imgs):
            img.save(gen_dir / f"gen_{i+j:05d}.png")

    # 4. CALCULATE METRICS
    evaluator = MedicalEvaluator(device)
    gen_paths = sorted(list(gen_dir.glob("*.png")))
    
    print("Calculating overall FID...")
    evaluator.update_fid(real_paths, is_real=True)
    evaluator.update_fid(gen_paths, is_real=False)
    fid_val = evaluator.fid.compute().item()
    
    print("Calculating overall DINO Similarity...")
    real_feats = evaluator.get_dino_features(real_paths)
    gen_feats = evaluator.get_dino_features(gen_paths)
    dino_val = torch.mm(gen_feats, real_feats.transpose(0, 1)).mean().item()

    # 5. SAVE RESULTS
    with open(os.path.join(OUTPUT_DIR, "overall_results.txt"), "w") as f:
        f.write(f"Overall Evaluation (Data Leakage Fixed)\n")
        f.write(f"Reference Images: {len(real_paths)} (Strictly Unseen)\n")
        f.write(f"Generated Images: {len(gen_paths)}\n")
        f.write(f"FID: {fid_val:.4f}\n")
        f.write(f"DINO Similarity: {dino_val:.4f}\n")
    
    print(f"\nFinal Results:\nFID: {fid_val:.4f}\nDINO: {dino_val:.4f}")

if __name__ == "__main__":
    main()
