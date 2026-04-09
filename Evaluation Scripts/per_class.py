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
from torchmetrics.image.kid import KernelInceptionDistance
from torchvision import transforms

# --- CONFIGURATION ---
MODEL_PATH = "/home/mluser/.cache/huggingface/hub/models--vanillacoke--all_losses_model_1/snapshots/d1a724c741dffc7d7a67ec2f54e0fb50a7352662/final_model"
OUTPUT_ROOT = "./eval_output/per_class_results"
UNIQUE_TOKEN = "<nih-xray>" 

METADATA_CSV_PATH = "./Data_Entry_2017.csv" 
TRAIN_SAMPLE_CSV = "./sample/sample_labels.csv" 
REAL_IMAGES_ROOT = "./eval_unseen_data/all_images"

NUM_IMAGES_PER_CLASS = 50  
BATCH_SIZE = 4       
EVAL_BATCH_SIZE = 32 

PATHOLOGIES = {
    "Atelectasis":  f"a photo of {UNIQUE_TOKEN} showing Atelectasis",
    "Cardiomegaly": f"a photo of {UNIQUE_TOKEN} showing Cardiomegaly",
    "Effusion":     f"a photo of {UNIQUE_TOKEN} showing Effusion",
    "Infiltration": f"a photo of {UNIQUE_TOKEN} showing Infiltration",
    "Mass":         f"a photo of {UNIQUE_TOKEN} showing a Mass",
    "Nodule":       f"a photo of {UNIQUE_TOKEN} showing a Nodule",
    "Pneumonia":    f"a photo of {UNIQUE_TOKEN} showing Pneumonia",
    "Pneumothorax": f"a photo of {UNIQUE_TOKEN} showing Pneumothorax",
    "Consolidation": f"a photo of {UNIQUE_TOKEN} showing Consolidation",
    "Edema":        f"a photo of {UNIQUE_TOKEN} showing Edema",
    "Emphysema":    f"a photo of {UNIQUE_TOKEN} showing Emphysema",
    "Fibrosis":     f"a photo of {UNIQUE_TOKEN} showing Fibrosis",
    "Pleural_Thickening": f"a photo of {UNIQUE_TOKEN} showing Pleural Thickening",
    "Hernia":       f"a photo of {UNIQUE_TOKEN} showing Hernia",
    "No Finding":   f"a photo of {UNIQUE_TOKEN} with No Findings"
}

class MedicalEvaluator:
    def __init__(self, device):
        self.device = device
        print(f"Initializing Evaluation Tools on {device}...")
        
        # DINO (Visual Similarity)
        self.dino_model = timm.create_model('vit_small_patch16_224.dino', pretrained=True).eval().to(device)
        self.dino_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        
        # CLIP (Text Alignment)
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval().to(device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # FID
        self.fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    def _to_tensor(self, output):
        if isinstance(output, torch.Tensor): return output
        for attr in ['image_embeds', 'text_embeds', 'pooler_output']:
            if hasattr(output, attr):
                val = getattr(output, attr)
                if isinstance(val, torch.Tensor): return val
        return output[0]

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
                    feats = self._to_tensor(self.clip_model.get_image_features(**inputs))
            all_features.append(feats.cpu())
        return F.normalize(torch.cat(all_features).to(self.device), dim=-1)

    def update_dist_metrics(self, image_paths, is_real, kid_obj):
        to_tensor = transforms.ToTensor()
        for i in range(0, len(image_paths), EVAL_BATCH_SIZE):
            batch_paths = image_paths[i:i + EVAL_BATCH_SIZE]
            batch_pil = [Image.open(p).convert("RGB") for p in batch_paths]
            batch_tensors = torch.stack([to_tensor(img) for img in batch_pil]).to(self.device)
            batch_uint8 = (batch_tensors * 255).byte()
            self.fid.update(batch_uint8, real=is_real)
            kid_obj.update(batch_uint8, real=is_real)

    def compute_metrics(self, gen_paths, real_paths, prompt):
        # Dynamic KID setup to prevent subset_size error
        n_samples = min(len(gen_paths), len(real_paths))
        dynamic_subset = max(2, n_samples - 2) 
        kid = KernelInceptionDistance(subset_size=dynamic_subset, normalize=True).to(self.device)
        
        # DINO Score
        real_dino = self.get_features_batched(real_paths, "dino")
        gen_dino = self.get_features_batched(gen_paths, "dino")
        dino_score = torch.mm(gen_dino, real_dino.transpose(0, 1)).mean().item()
        
        # CLIP-T Score
        gen_clip = self.get_features_batched(gen_paths, "clip")
        with torch.no_grad():
            inputs = self.clip_processor(text=[prompt], return_tensors="pt", padding=True).to(self.device)
            prompt_feat = F.normalize(self._to_tensor(self.clip_model.get_text_features(**inputs)), dim=-1)
        clip_t_score = (gen_clip @ prompt_feat.T).mean().item()
        
        # Distribution metrics
        self.fid.reset()
        self.update_dist_metrics(real_paths, is_real=True, kid_obj=kid)
        self.update_dist_metrics(gen_paths, is_real=False, kid_obj=kid)
        
        fid_score = self.fid.compute().item()
        kid_score, _ = kid.compute()
        
        return {
            "DINO": dino_score, 
            "FID": fid_score, 
            "KID": kid_score.item(), 
            "CLIP-T": clip_t_score,
            "Ref_Images_Found": len(real_paths)
        }

def get_clean_real_images(df, training_ids, pathology):
    subset = df[df['Finding Labels'].str.contains(pathology, na=False)]
    unseen_subset = subset[~subset['Image Index'].isin(training_ids)]
    
    found_paths = []
    for fname in unseen_subset['Image Index']:
        path = os.path.join(REAL_IMAGES_ROOT, fname)
        if os.path.exists(path):
            found_paths.append(path)
        if len(found_paths) >= NUM_IMAGES_PER_CLASS:
            break
    return found_paths

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    df = pd.read_csv(METADATA_CSV_PATH)
    train_df = pd.read_csv(TRAIN_SAMPLE_CSV)
    train_ids = set(train_df['Image Index'].tolist())
    
    print(f"Loading local pipeline from: {MODEL_PATH}")
    pipeline = StableDiffusionPipeline.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, local_files_only=True
    ).to(device)
    pipeline.set_progress_bar_config(disable=True)
    
    evaluator = MedicalEvaluator(device)
    results = []

    for pathology, prompt in PATHOLOGIES.items():
        print(f"\n--- Pathology: {pathology} ---")
        class_out_dir = Path(OUTPUT_ROOT) / pathology.replace(" ", "_")
        class_out_dir.mkdir(parents=True, exist_ok=True)
        
        real_paths = get_clean_real_images(df, train_ids, pathology)
        if len(real_paths) < 5:
            print(f"  Skipping: Only {len(real_paths)} reference images found.")
            continue
            
        gen_files = list(class_out_dir.glob("gen_*.png"))
        if len(gen_files) < NUM_IMAGES_PER_CLASS:
            needed = NUM_IMAGES_PER_CLASS - len(gen_files)
            print(f"  Generating {needed} images...")
            generator = torch.manual_seed(42)
            for i in tqdm(range(0, needed, BATCH_SIZE)):
                curr_batch = min(BATCH_SIZE, needed - i)
                with torch.no_grad():
                    imgs = pipeline([prompt]*curr_batch, generator=generator).images
                for j, img in enumerate(imgs):
                    img.save(class_out_dir / f"gen_{len(gen_files) + i + j:05d}.png")
        
        gen_paths = sorted([str(p) for p in class_out_dir.glob("gen_*.png")])[:NUM_IMAGES_PER_CLASS]
        
        print("  Computing Metrics...")
        metrics = evaluator.compute_metrics(gen_paths, real_paths, prompt)
        metrics["Pathology"] = pathology
        results.append(metrics)
        print(f"  FID: {metrics['FID']:.2f} | DINO: {metrics['DINO']:.4f} | CLIP-T: {metrics['CLIP-T']:.4f}")

    if results:
        df_res = pd.DataFrame(results)
        df_res.to_csv(os.path.join(OUTPUT_ROOT, "per_class_metrics.csv"), index=False)
        print(f"\nEvaluation Complete! Table saved to {OUTPUT_ROOT}/per_class_metrics.csv")

if __name__ == "__main__":
    main()
