import os
import torch
import pandas as pd
import torch.nn.functional as F
from pathlib import Path
from diffusers import StableDiffusionPipeline
from tqdm.auto import tqdm
from PIL import Image
import timm
from transformers import CLIPModel, CLIPProcessor
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchvision import transforms

# --- CONFIGURATION ---
MODEL_PATH = "/home/mluser/.cache/huggingface/hub/models--vanillacoke--all_losses_model_1/snapshots/d1a724c741dffc7d7a67ec2f54e0fb50a7352662/final_model"
OUTPUT_DIR = "./eval_output/leakage_fixed_overall"
UNIQUE_TOKEN = "<nih-xray>"

METADATA_CSV_PATH = "./Data_Entry_2017.csv"
REAL_IMAGES_ROOT = "./eval_unseen_data/all_images"
TRAIN_SAMPLE_CSV = "./sample/sample_labels.csv"

NUM_TEST_SAMPLES = 1000
BATCH_SIZE = 4
EVAL_BATCH_SIZE = 32

PROMPT = f"a photo of {UNIQUE_TOKEN}"


class MedicalEvaluator:
    def __init__(self, device):
        self.device = device
        print(f"Initializing evaluation models on {device}...")

        # DINO (structural similarity)
        self.dino_model = timm.create_model(
            'vit_small_patch16_224.dino', pretrained=True
        ).eval().to(device)
        self.dino_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        # FID
        self.fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

        # KID — subset_size set after we know sample count; initialised in compute()
        self.kid = None

        # IS (generated images only)
        self.inception = InceptionScore(normalize=True).to(device)

        # CLIP — using transformers directly to avoid torchmetrics version issues
        self.clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        ).eval().to(device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_scores = []  # accumulate per-batch scores, average at end

    def get_dino_features(self, image_paths):
        all_features = []
        for i in range(0, len(image_paths), EVAL_BATCH_SIZE):
            batch_pil = [Image.open(p).convert("RGB") for p in image_paths[i:i + EVAL_BATCH_SIZE]]
            with torch.no_grad():
                tensors = torch.stack(
                    [self.dino_transform(img) for img in batch_pil]
                ).to(self.device)
                feats = self.dino_model(tensors)
            all_features.append(feats.cpu())
            torch.cuda.empty_cache()
        return F.normalize(torch.cat(all_features).to(self.device), dim=-1)

    def update_dist_metrics(self, image_paths, is_real, prompt=None):
        """Feed images into FID, KID, IS (generated only), and CLIP (generated only)."""
        to_tensor = transforms.ToTensor()
        for i in range(0, len(image_paths), EVAL_BATCH_SIZE):
            batch_pil = [Image.open(p).convert("RGB") for p in image_paths[i:i + EVAL_BATCH_SIZE]]
            batch_t = torch.stack([to_tensor(img) for img in batch_pil]).to(self.device)
            batch_uint8 = (batch_t * 255).byte()

            self.fid.update(batch_uint8, real=is_real)
            self.kid.update(batch_uint8, real=is_real)

            if not is_real:
                self.inception.update(batch_uint8)
                # Compute CLIP score directly via transformers
                with torch.no_grad():
                    inputs = self.clip_processor(
                        images=batch_pil, text=[prompt] * len(batch_pil),
                        return_tensors="pt", padding=True
                    ).to(self.device)
                    outputs = self.clip_model(**inputs)
                    img_feats  = F.normalize(outputs.image_embeds, dim=-1)
                    text_feats = F.normalize(outputs.text_embeds, dim=-1)
                    scores = (img_feats * text_feats).sum(dim=-1)
                self.clip_scores.append(scores.cpu())

            torch.cuda.empty_cache()

    def compute_all(self, real_paths, gen_paths, prompt):
        n = min(len(real_paths), len(gen_paths))
        safe_subset = max(2, n - 2)
        self.kid = KernelInceptionDistance(
            subset_size=safe_subset, normalize=True
        ).to(self.device)

        print("  Feeding real images into distribution metrics...")
        self.update_dist_metrics(real_paths, is_real=True)

        print("  Feeding generated images into all metrics...")
        self.update_dist_metrics(gen_paths, is_real=False, prompt=prompt)

        fid_val  = self.fid.compute().item()
        kid_mean, kid_std = self.kid.compute()
        is_mean,  is_std  = self.inception.compute()
        clip_val = torch.cat(self.clip_scores).mean().item()

        print("  Computing DINO similarity...")
        real_feats = self.get_dino_features(real_paths)
        gen_feats  = self.get_dino_features(gen_paths)
        dino_val   = torch.mm(gen_feats, real_feats.T).mean().item()

        return {
            "FID":      round(fid_val, 4),
            "KID_mean": round(kid_mean.item(), 4),
            "KID_std":  round(kid_std.item(), 4),
            "IS_mean":  round(is_mean.item(), 4),
            "IS_std":   round(is_std.item(), 4),
            "CLIP":     round(clip_val, 4),
            "DINO":     round(dino_val, 4),
            "N_real":   len(real_paths),
            "N_gen":    len(gen_paths),
        }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 1. Identify unseen reference images (no data leakage) ---
    print("Checking for data leakage...")
    train_df = pd.read_csv(TRAIN_SAMPLE_CSV)
    train_ids = set(train_df['Image Index'].astype(str).tolist())

    if not os.path.exists(REAL_IMAGES_ROOT) or not os.listdir(REAL_IMAGES_ROOT):
        print(f"CRITICAL ERROR: {REAL_IMAGES_ROOT} is empty.")
        return

    all_files = [
        f for f in os.listdir(REAL_IMAGES_ROOT)
        if f.lower().endswith(('.png', '.jpg'))
    ]
    unseen = [f for f in all_files if f not in train_ids]

    if not unseen:
        print("CRITICAL ERROR: 0 unseen images found.")
        return

    real_paths = [
        os.path.join(REAL_IMAGES_ROOT, f) for f in unseen[:NUM_TEST_SAMPLES]
    ]
    print(f"Using {len(real_paths)} strictly unseen reference images.")

    # --- 2. Generate synthetic samples ---
    gen_dir = Path(OUTPUT_DIR) / "generated_samples"
    gen_dir.mkdir(exist_ok=True)

    existing_gen = sorted(gen_dir.glob("gen_*.png"))
    if len(existing_gen) < len(real_paths):
        needed = len(real_paths) - len(existing_gen)
        print(f"Generating {needed} synthetic images...")
        pipeline = StableDiffusionPipeline.from_pretrained(
            MODEL_PATH, torch_dtype=torch.float16, local_files_only=True
        ).to(device)
        pipeline.set_progress_bar_config(disable=True)
        generator = torch.manual_seed(42)
        idx = len(existing_gen)
        for i in tqdm(range(0, needed, BATCH_SIZE)):
            curr = min(BATCH_SIZE, needed - i)
            with torch.no_grad():
                imgs = pipeline([PROMPT] * curr, generator=generator).images
            for img in imgs:
                img.save(gen_dir / f"gen_{idx:05d}.png")
                idx += 1
        del pipeline
        torch.cuda.empty_cache()
    else:
        print(f"Found {len(existing_gen)} existing generated images. Skipping generation.")

    gen_paths = sorted([str(p) for p in gen_dir.glob("gen_*.png")])[:len(real_paths)]

    # --- 3. Compute all metrics ---
    print("\nComputing metrics: FID, KID, IS, CLIP, DINO...")
    evaluator = MedicalEvaluator(device)
    metrics = evaluator.compute_all(real_paths, gen_paths, PROMPT)

    # --- 4. Save and print results ---
    results_file = os.path.join(OUTPUT_DIR, "overall_results.txt")
    with open(results_file, "w") as f:
        f.write("Overall Benchmark Results\n")
        f.write(f"Model:          {MODEL_PATH}\n")
        f.write(f"Reference imgs: {metrics['N_real']}\n")
        f.write(f"Generated imgs: {metrics['N_gen']}\n")
        f.write("-" * 40 + "\n")
        f.write(f"FID:            {metrics['FID']:.4f}  (lower is better)\n")
        f.write(f"KID:            {metrics['KID_mean']:.4f} ± {metrics['KID_std']:.4f}  (lower is better)\n")
        f.write(f"IS:             {metrics['IS_mean']:.4f} ± {metrics['IS_std']:.4f}  (higher is better)\n")
        f.write(f"CLIP Score:     {metrics['CLIP']:.4f}  (higher is better)\n")
        f.write(f"DINO Similarity:{metrics['DINO']:.4f}  (higher is better)\n")

    print(f"\n--- Overall Benchmark Results ---")
    print(f"FID:            {metrics['FID']:.4f}")
    print(f"KID:            {metrics['KID_mean']:.4f} ± {metrics['KID_std']:.4f}")
    print(f"IS:             {metrics['IS_mean']:.4f} ± {metrics['IS_std']:.4f}")
    print(f"CLIP Score:     {metrics['CLIP']:.4f}")
    print(f"DINO Similarity:{metrics['DINO']:.4f}")
    print(f"\nFull report saved to: {results_file}")


if __name__ == "__main__":
    main()
