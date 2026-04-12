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
from torchmetrics.image.inception import InceptionScore
from torchvision import transforms

# --- CONFIGURATION ---
MODEL_PATH = "/home/mluser/.cache/huggingface/hub/models--vanillacoke--all_losses_model_1/snapshots/d1a724c741dffc7d7a67ec2f54e0fb50a7352662/final_model"
OUTPUT_ROOT = "./eval_output/per_class_results"
UNIQUE_TOKEN = "<nih-xray>"

METADATA_CSV_PATH = "./Data_Entry_2017.csv"
TRAIN_SAMPLE_CSV = "./sample/sample_labels.csv"
REAL_IMAGES_ROOT = "./eval_unseen_data/all_images"

TOTAL_IMAGE_BUDGET = 1000   # Total generated images across all classes
MIN_PER_CLASS = 10          # Floor so rare classes (e.g. Hernia) still get evaluated
BATCH_SIZE = 4
EVAL_BATCH_SIZE = 32

# NIH ChestX-ray14 approximate training-set frequencies (%) — used for proportional allocation
CLASS_FREQUENCIES = {
    "Atelectasis":        10.3,
    "Cardiomegaly":        2.5,
    "Effusion":            7.6,
    "Infiltration":       17.7,
    "Mass":                5.1,
    "Nodule":              5.6,
    "Pneumonia":           1.3,
    "Pneumothorax":        4.8,
    "Consolidation":       4.2,
    "Edema":               2.4,
    "Emphysema":           2.2,
    "Fibrosis":            1.5,
    "Pleural_Thickening":  3.0,
    "Hernia":              0.2,
    "No Finding":         31.6,
}

PATHOLOGY_PROMPTS = {
    "Atelectasis":        f"a photo of {UNIQUE_TOKEN} showing Atelectasis",
    "Cardiomegaly":       f"a photo of {UNIQUE_TOKEN} showing Cardiomegaly",
    "Effusion":           f"a photo of {UNIQUE_TOKEN} showing Effusion",
    "Infiltration":       f"a photo of {UNIQUE_TOKEN} showing Infiltration",
    "Mass":               f"a photo of {UNIQUE_TOKEN} showing a Mass",
    "Nodule":             f"a photo of {UNIQUE_TOKEN} showing a Nodule",
    "Pneumonia":          f"a photo of {UNIQUE_TOKEN} showing Pneumonia",
    "Pneumothorax":       f"a photo of {UNIQUE_TOKEN} showing Pneumothorax",
    "Consolidation":      f"a photo of {UNIQUE_TOKEN} showing Consolidation",
    "Edema":              f"a photo of {UNIQUE_TOKEN} showing Edema",
    "Emphysema":          f"a photo of {UNIQUE_TOKEN} showing Emphysema",
    "Fibrosis":           f"a photo of {UNIQUE_TOKEN} showing Fibrosis",
    "Pleural_Thickening": f"a photo of {UNIQUE_TOKEN} showing Pleural Thickening",
    "Hernia":             f"a photo of {UNIQUE_TOKEN} showing Hernia",
    "No Finding":         f"a photo of {UNIQUE_TOKEN} with No Findings",
}


def compute_class_budgets(total_budget, frequencies, min_per_class):
    """
    Allocate total_budget images across classes proportional to training-set
    frequency, with a per-class floor of min_per_class.
    """
    total_freq = sum(frequencies.values())
    raw = {cls: (freq / total_freq) * total_budget for cls, freq in frequencies.items()}

    # Apply floor
    budgets = {cls: max(min_per_class, int(n)) for cls, n in raw.items()}

    # Trim back down to total_budget if flooring pushed us over
    overage = sum(budgets.values()) - total_budget
    if overage > 0:
        # Reduce the largest classes first
        for cls in sorted(budgets, key=lambda c: -budgets[c]):
            if overage <= 0:
                break
            reducible = budgets[cls] - min_per_class
            cut = min(reducible, overage)
            budgets[cls] -= cut
            overage -= cut

    return budgets


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

        # CLIP (text alignment)
        self.clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        ).eval().to(device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def _to_tensor(self, output):
        if isinstance(output, torch.Tensor):
            return output
        for attr in ['image_embeds', 'text_embeds', 'pooler_output']:
            if hasattr(output, attr):
                val = getattr(output, attr)
                if isinstance(val, torch.Tensor):
                    return val
        return output[0]

    def get_features_batched(self, image_paths, model_type="dino"):
        all_features = []
        for i in range(0, len(image_paths), EVAL_BATCH_SIZE):
            batch_pil = [Image.open(p).convert("RGB") for p in image_paths[i:i + EVAL_BATCH_SIZE]]
            with torch.no_grad():
                if model_type == "dino":
                    tensors = torch.stack([self.dino_transform(img) for img in batch_pil]).to(self.device)
                    feats = self.dino_model(tensors)
                else:  # clip
                    inputs = self.clip_processor(
                        images=batch_pil, return_tensors="pt", padding=True
                    ).to(self.device)
                    feats = self._to_tensor(self.clip_model.get_image_features(**inputs))
            all_features.append(feats.cpu())
        return F.normalize(torch.cat(all_features).to(self.device), dim=-1)

    def compute_metrics(self, gen_paths, real_paths, prompt, n_gen):
        """
        Compute FID, KID, IS, CLIP-T, and DINO for one class.
        n_gen is the allocated budget for this class (used for KID subset_size).
        """
        # --- FID ---
        fid = FrechetInceptionDistance(feature=2048, normalize=True).to(self.device)

        # --- KID: subset_size must be <= min(n_real, n_gen) ---
        n_real = len(real_paths)
        safe_subset = max(2, min(n_gen, n_real) - 2)
        kid = KernelInceptionDistance(subset_size=safe_subset, normalize=True).to(self.device)

        # --- IS (only on generated images) ---
        inception = InceptionScore(normalize=True).to(self.device)

        to_tensor = transforms.ToTensor()

        # Feed real images into FID + KID
        for i in range(0, n_real, EVAL_BATCH_SIZE):
            batch = [Image.open(p).convert("RGB") for p in real_paths[i:i + EVAL_BATCH_SIZE]]
            t = (torch.stack([to_tensor(img) for img in batch]) * 255).byte().to(self.device)
            fid.update(t, real=True)
            kid.update(t, real=True)

        # Feed generated images into FID + KID + IS
        for i in range(0, len(gen_paths), EVAL_BATCH_SIZE):
            batch = [Image.open(p).convert("RGB") for p in gen_paths[i:i + EVAL_BATCH_SIZE]]
            t = (torch.stack([to_tensor(img) for img in batch]) * 255).byte().to(self.device)
            fid.update(t, real=False)
            kid.update(t, real=False)
            inception.update(t)

        fid_score = fid.compute().item()
        kid_mean, _ = kid.compute()
        is_mean, is_std = inception.compute()

        # --- CLIP-T score ---
        gen_clip = self.get_features_batched(gen_paths, "clip")
        with torch.no_grad():
            inputs = self.clip_processor(
                text=[prompt], return_tensors="pt", padding=True
            ).to(self.device)
            prompt_feat = F.normalize(
                self._to_tensor(self.clip_model.get_text_features(**inputs)), dim=-1
            )
        clip_t = (gen_clip @ prompt_feat.T).mean().item()

        # --- DINO similarity ---
        real_dino = self.get_features_batched(real_paths, "dino")
        gen_dino  = self.get_features_batched(gen_paths,  "dino")
        dino = torch.mm(gen_dino, real_dino.T).mean().item()

        return {
            "FID":    round(fid_score, 4),
            "KID":    round(kid_mean.item(), 4),
            "IS":     round(is_mean.item(), 4),
            "CLIP-T": round(clip_t, 4),
            "DINO":   round(dino, 4),
            "N_gen":  len(gen_paths),
            "N_real": n_real,
        }


def get_unseen_real_images(df, training_ids, pathology, n_needed):
    """Return up to n_needed real images for a class, excluding training images."""
    label = "No Finding" if pathology == "No Finding" else pathology
    subset = df[df['Finding Labels'].str.contains(label, na=False)]
    unseen = subset[~subset['Image Index'].isin(training_ids)]

    found = []
    for fname in unseen['Image Index']:
        path = os.path.join(REAL_IMAGES_ROOT, fname)
        if os.path.exists(path):
            found.append(path)
        if len(found) >= n_needed:
            break
    return found


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # --- Load metadata ---
    df = pd.read_csv(METADATA_CSV_PATH)
    train_df = pd.read_csv(TRAIN_SAMPLE_CSV)
    train_ids = set(train_df['Image Index'].tolist())

    # --- Compute per-class image budgets ---
    budgets = compute_class_budgets(TOTAL_IMAGE_BUDGET, CLASS_FREQUENCIES, MIN_PER_CLASS)
    print("\nPer-class image budgets (proportional to training frequency):")
    for cls, n in budgets.items():
        print(f"  {cls:<22} → {n} images")
    print(f"  Total: {sum(budgets.values())} images\n")

    # --- Load pipeline ---
    print(f"Loading model from: {MODEL_PATH}")
    pipeline = StableDiffusionPipeline.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, local_files_only=True
    ).to(device)
    pipeline.set_progress_bar_config(disable=True)

    evaluator = MedicalEvaluator(device)
    results = []

    for pathology, prompt in PATHOLOGY_PROMPTS.items():
        n_gen = budgets[pathology]
        print(f"\n--- {pathology} (budget: {n_gen} images) ---")

        class_out_dir = Path(OUTPUT_ROOT) / pathology.replace(" ", "_")
        class_out_dir.mkdir(parents=True, exist_ok=True)

        # Get real reference images (match budget size)
        real_paths = get_unseen_real_images(df, train_ids, pathology, n_gen)
        if len(real_paths) < MIN_PER_CLASS:
            print(f"  Skipping: only {len(real_paths)} unseen reference images found.")
            continue

        # Generate synthetic images if not already present
        existing = sorted(class_out_dir.glob("gen_*.png"))
        if len(existing) < n_gen:
            needed = n_gen - len(existing)
            print(f"  Generating {needed} images...")
            generator = torch.manual_seed(42)
            idx = len(existing)
            for i in tqdm(range(0, needed, BATCH_SIZE)):
                curr = min(BATCH_SIZE, needed - i)
                with torch.no_grad():
                    imgs = pipeline([prompt] * curr, generator=generator).images
                for img in imgs:
                    img.save(class_out_dir / f"gen_{idx:05d}.png")
                    idx += 1

        gen_paths = sorted([str(p) for p in class_out_dir.glob("gen_*.png")])[:n_gen]

        # Compute all metrics
        print("  Computing metrics...")
        metrics = evaluator.compute_metrics(gen_paths, real_paths, prompt, n_gen)
        metrics["Pathology"] = pathology
        metrics["Budget"] = n_gen
        results.append(metrics)

        print(
            f"  FID: {metrics['FID']:.2f} | KID: {metrics['KID']:.4f} | "
            f"IS: {metrics['IS']:.4f} | CLIP-T: {metrics['CLIP-T']:.4f} | "
            f"DINO: {metrics['DINO']:.4f}"
        )

    # --- Save results table ---
    if results:
        cols = ["Pathology", "Budget", "N_gen", "N_real", "FID", "KID", "IS", "CLIP-T", "DINO"]
        df_res = pd.DataFrame(results)[cols]

        # Append mean row
        numeric_cols = ["FID", "KID", "IS", "CLIP-T", "DINO"]
        mean_row = {c: round(df_res[c].mean(), 4) for c in numeric_cols}
        mean_row["Pathology"] = "Mean"
        mean_row["Budget"] = sum(budgets.values())
        mean_row["N_gen"] = df_res["N_gen"].sum()
        mean_row["N_real"] = df_res["N_real"].sum()
        df_res = pd.concat([df_res, pd.DataFrame([mean_row])], ignore_index=True)

        out_csv = os.path.join(OUTPUT_ROOT, "per_class_metrics.csv")
        df_res.to_csv(out_csv, index=False)
        print(f"\nEvaluation complete. Results saved to: {out_csv}")
        print(df_res.to_string(index=False))


if __name__ == "__main__":
    main()
