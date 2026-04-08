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
# 1. Update these to your Ubuntu server paths
MODEL_PATH = "./results_seed42/final_model" 
OUTPUT_ROOT = "./eval_output/per_class_results"
UNIQUE_TOKEN = "<nih-xray>" 

# 2. Update to the NEW unseen data paths (from Phase 1 & 2)
METADATA_CSV_PATH = "./Data_Entry_2017.csv" 
REAL_IMAGES_ROOT = "./eval_unseen_data/all_images" 
TRAIN_SAMPLE_CSV = "./sample/sample_labels.csv" # To prevent leakage

NUM_IMAGES_PER_CLASS = 100 # Adjusted for per-class consistency
BATCH_SIZE = 4       
EVAL_BATCH_SIZE = 32 

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

class MedicalEvaluator:
    def __init__(self, device):
        self.device = device
        print(f"Initializing Metrics on {device}...")
        
        # DINO - Excellent for medical image feature similarity
        self.dino_model = timm.create_model('vit_small_patch16_224.dino', pretrained=True).eval().to(device)
        self.dino_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        
        # CLIP - Measures how well the pathology matches the prompt
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval().to(device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # FID - Standard metric for image quality/diversity
        self.fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    def get_features_batched(self, image_paths, model_type="dino"):
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
            
            all_features.append(feats.cpu())
            torch.cuda.empty_cache()
            
        return F.normalize(torch.cat(all_features).to(self.device), dim=-1)

    def update_fid_batched(self, image_paths, is_real):
        to_tensor = transforms.ToTensor()
        for i in range(0, len(image_paths), EVAL_BATCH_SIZE):
            batch_paths = image_paths[i:i + EVAL_BATCH_SIZE]
            batch_pil = [Image.open(p).convert("RGB") for p in batch_paths]
            batch_tensors = torch.stack([to_tensor(img) for img in batch_pil]).to(self.device)
            batch_uint8 = (batch_tensors * 255).byte()
            self.fid.update(batch_uint8, real=is_real)
            torch.cuda.empty_cache()

    def compute_metrics(self, gen_paths, real_paths, prompt):
        print("    Computing DINO...")
        real_dino = self.get_features_batched(real_paths, "dino")
        gen_dino = self.get_features_batched(gen_paths, "dino")
        sim_matrix = torch.mm(gen_dino, real_dino.transpose(0, 1))
        dino_score = sim_matrix.mean().item()
        
        print("    Computing CLIP-T...")
        gen_clip = self.get_features_batched(gen_paths, "clip")
        with torch.no_grad():
            inputs = self.clip_processor(text=[prompt], return_tensors="pt", padding=True).to(self.device)
            prompt_feat = F.normalize(self.clip_model.get_text_features(**inputs), dim=-1)
        clip_t_score = F.cosine_similarity(gen_clip, prompt_feat).mean().item()
        
        print("    Computing FID...")
        self.fid.reset()
        self.update_fid_batched(real_paths, is_real=True)
        self.update_fid_batched(gen_paths, is_real=False)
        fid_score = self.fid.compute().item()
        
        return {"DINO": dino_score, "FID": fid_score, "CLIP-T": clip_t_score}

def get_clean_real_images(df, training_df, pathology, root_dir):
    """Crucial: Filters out images used during training to fix leakage"""
    training_ids = set(training_df['Image Index'].tolist())
    # Exclude training IDs
    unseen_df = df[~df['Image Index'].isin(training_ids)]
    # Filter for pathology
    subset = unseen_df[unseen_df['Finding Labels'].str.contains(pathology, na=False)]
    
    found_paths = []
    for fname in subset['Image Index']:
        path = os.path.join(root_dir, fname)
        if os.path.exists(path):
            found_paths.append(path)
            if len(found_paths) >= NUM_IMAGES_PER_CLASS:
                break
    return found_paths

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    print("Loading Metadata and Training Logs (for Leakage Protection)...")
    df = pd.read_csv(METADATA_CSV_PATH)
    train_df = pd.read_csv(TRAIN_SAMPLE_CSV)
    
    print(f"Loading Model...")
    pipeline = StableDiffusionPipeline.from_pretrained(MODEL_PATH, torch_dtype=torch.float16).to(device)
    pipeline.set_progress_bar_config(disable=True)
    
    evaluator = MedicalEvaluator(device)
    results = []
    
    for pathology, prompt in PATHOLOGIES.items():
        print(f"\n--- Class: {pathology} ---")
        class_out_dir = Path(OUTPUT_ROOT) / pathology
        class_out_dir.mkdir(parents=True, exist_ok=True)
        
        real_paths = get_clean_real_images(df, train_df, pathology, REAL_IMAGES_ROOT)
        if len(real_paths) < 10:
            print(f"Skipping {pathology}: Insufficient unseen images ({len(real_paths)})")
            continue

        # Generation
        existing_gen = list(class_out_dir.glob("*.png"))
        needed = NUM_IMAGES_PER_CLASS - len(existing_gen)
        if needed > 0:
            print(f"  Generating {needed} images...")
            generator = torch.manual_seed(42)
            for i in tqdm(range(0, needed, BATCH_SIZE)):
                curr_batch = min(BATCH_SIZE, needed - i)
                imgs = pipeline([prompt]*curr_batch, generator=generator).images
                for j, img in enumerate(imgs):
                    img.save(class_out_dir / f"gen_{len(existing_gen) + i + j:05d}.png")
        
        # Metrics
        gen_files = sorted(list(class_out_dir.glob("*.png")))[:NUM_IMAGES_PER_CLASS]
        metrics = evaluator.compute_metrics(gen_files, real_paths, prompt)
        metrics["Pathology"] = pathology
        results.append(metrics)
        
    # Save Final Table
    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(OUTPUT_ROOT, "final_metrics.csv"), index=False)
    print("\nEvaluation Complete. Results saved to final_metrics.csv")

if __name__ == "__main__":
    main()
