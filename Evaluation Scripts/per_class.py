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
MODEL_PATH = "./results_seed123/final_model" 
OUTPUT_ROOT = "./eval_output/per_class_results"
UNIQUE_TOKEN = "<nih-xray>" 

METADATA_CSV_PATH = "./Data_Entry_2017.csv" 
TRAIN_SAMPLE_CSV = "./sample/sample_labels.csv" 
REAL_IMAGES_PACKS = ["./eval_temp/images_001/images", "./eval_temp/images_002/images"]

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
        self.kid = KernelInceptionDistance(subset_size=max(10, NUM_IMAGES_PER_CLASS//2), normalize=True).to(device)

    def _to_tensor(self, output):
        """Forces any output type from CLIP into a raw torch Tensor."""
        if isinstance(output, torch.Tensor):
            return output
        # Check for common attribute names used in BaseModelOutput objects
        for attr in ['image_embeds', 'text_embeds', 'pooler_output', 'last_hidden_state']:
            if hasattr(output, attr):
                val = getattr(output, attr)
                if isinstance(val, torch.Tensor):
                    return val
        # Final fallback: try indexing if it acts like a tuple/list
        try:
            return output[0]
        except:
            return output

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
                    # Force result to be a tensor
                    feats = self._to_tensor(self.clip_model.get_image_features(**inputs))
            
            all_features.append(feats.cpu())
            torch.cuda.empty_cache()
            
        return F.normalize(torch.cat(all_features).to(self.device), dim=-1)

    def update_dist_metrics(self, image_paths, is_real):
        to_tensor = transforms.ToTensor()
        for i in range(0, len(image_paths), EVAL_BATCH_SIZE):
            batch_paths = image_paths[i:i + EVAL_BATCH_SIZE]
            batch_pil = [Image.open(p).convert("RGB") for p in batch_paths]
            batch_tensors = torch.stack([to_tensor(img) for img in batch_pil]).to(self.device)
            batch_uint8 = (batch_tensors * 255).byte()
            self.fid.update(batch_uint8, real=is_real)
            self.kid.update(batch_uint8, real=is_real)
            torch.cuda.empty_cache()

    def compute_metrics(self, gen_paths, real_paths, prompt):
        print("    Computing DINO...")
        real_dino = self.get_features_batched(real_paths, "dino")
        gen_dino = self.get_features_batched(gen_paths, "dino")
        dino_score = torch.mm(gen_dino, real_dino.transpose(0, 1)).mean().item()
        
        print("    Computing CLIP-T...")
        gen_clip = self.get_features_batched(gen_paths, "clip") # This returns (Batch, 512)
        with torch.no_grad():
            inputs = self.clip_processor(text=[prompt], return_tensors="pt", padding=True).to(self.device)
            # Force result to be a tensor and normalize
            prompt_feat = self._to_tensor(self.clip_model.get_text_features(**inputs))
            prompt_feat = F.normalize(prompt_feat, dim=-1) # This is (1, 512)
            
        # Matrix multiplication is safer than cosine_similarity when shapes are different
        clip_t_score = (gen_clip @ prompt_feat.T).mean().item()
        
        print("    Computing FID/KID...")
        self.fid.reset()
        self.kid.reset()
        self.update_dist_metrics(real_paths, is_real=True)
        self.update_dist_metrics(gen_paths, is_real=False)
        fid_score = self.fid.compute().item()
        kid_score, _ = self.kid.compute()
        
        return {"DINO": dino_score, "FID": fid_score, "KID": kid_score.item(), "CLIP-T": clip_t_score}

def get_clean_real_images(df, training_df, pathology, pack_dirs):
    training_ids = set(training_df['Image Index'].tolist())
    unseen_df = df[~df['Image Index'].isin(training_ids)]
    subset = unseen_df[unseen_df['Finding Labels'].str.contains(pathology, na=False)]
    
    found_paths = []
    for fname in subset['Image Index']:
        for pack in pack_dirs:
            path = os.path.join(pack, fname)
            if os.path.exists(path):
                found_paths.append(path)
                break
        if len(found_paths) >= NUM_IMAGES_PER_CLASS:
            break
    return found_paths

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    df = pd.read_csv(METADATA_CSV_PATH)
    train_df = pd.read_csv(TRAIN_SAMPLE_CSV)
    pipeline = StableDiffusionPipeline.from_pretrained(MODEL_PATH, torch_dtype=torch.float16).to(device)
    pipeline.set_progress_bar_config(disable=True)
    evaluator = MedicalEvaluator(device)
    results = []
    for pathology, prompt in PATHOLOGIES.items():
        print(f"\n--- Pathology: {pathology} ---")
        class_out_dir = Path(OUTPUT_ROOT) / pathology
        class_out_dir.mkdir(parents=True, exist_ok=True)
        real_paths = get_clean_real_images(df, train_df, pathology, REAL_IMAGES_PACKS)
        if len(real_paths) < 10:
            print(f"  Skipping {pathology}: Only {len(real_paths)} unseen images found.")
            continue
        existing_gen = list(class_out_dir.glob("gen_*.png"))
        needed = NUM_IMAGES_PER_CLASS - len(existing_gen)
        if needed > 0:
            print(f"  Generating {needed} images...")
            generator = torch.manual_seed(42)
            for i in tqdm(range(0, needed, BATCH_SIZE)):
                curr_batch = min(BATCH_SIZE, needed - i)
                with torch.no_grad():
                    imgs = pipeline([prompt]*curr_batch, generator=generator).images
                for j, img in enumerate(imgs):
                    img.save(class_out_dir / f"gen_{len(existing_gen) + i + j:05d}.png")
        gen_files = sorted([str(p) for p in class_out_dir.glob("gen_*.png")])[:NUM_IMAGES_PER_CLASS]
        metrics = evaluator.compute_metrics(gen_files, real_paths, prompt)
        metrics["Pathology"] = pathology
        results.append(metrics)
        print(f"  Results: FID={metrics['FID']:.2f}, DINO={metrics['DINO']:.4f}, CLIP-T={metrics['CLIP-T']:.4f}")
    if results:
        df_res = pd.DataFrame(results)
        df_res.to_csv(os.path.join(OUTPUT_ROOT, "per_class_metrics.csv"), index=False)
        print(f"\nBenchmark finished! Results in {OUTPUT_ROOT}/per_class_metrics.csv")

if __name__ == "__main__":
    main()
