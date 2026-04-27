import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score
import torchxrayvision as xrv

# -----------------------------
# CONFIG
# -----------------------------
DATA_DIR = "generated_data" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = 0.5

CLASS_NAMES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]

# -----------------------------
# LOAD MODEL
# -----------------------------
print("Loading XRV DenseNet121...")
model = xrv.models.DenseNet(weights="densenet121-res224-all")
model = model.to(DEVICE)
model.eval()

xrv_classes = model.pathologies
print(f"Model loaded. Targeting {len(xrv_classes)} pathologies.")

# -----------------------------
# TRANSFORM
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -----------------------------
# STORAGE
# -----------------------------
y_true_all = []
y_scores_all = []
all_ranks = []
gt_confidences = []
per_class_data = {} 

# -----------------------------
# INFERENCE
# -----------------------------
print("\nRunning inference...")

for cname in CLASS_NAMES:
    folder = os.path.join(DATA_DIR, cname)
    if not os.path.exists(folder):
        continue

    # Robust mapping for naming variations (space vs underscore)
    mapped_name = None
    for candidate in [cname, cname.replace("_", " "), cname.replace(" ", "_")]:
        if candidate in xrv_classes:
            mapped_name = candidate
            break
    
    if mapped_name is None:
        print(f"Skipping {cname}: Not found in model vocabulary")
        continue

    idx = xrv_classes.index(mapped_name)
    per_class_data[mapped_name] = {"probs": [], "ranks": []}

    for img_name in tqdm(os.listdir(folder), desc=f"Evaluating {cname}"):
        path = os.path.join(folder, img_name)
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        try:
            img = Image.open(path).convert("L")
            img = transform(img)
            
            # ✅ XRV Normalization [-1024, 1024]
            img = (img * 2048) - 1024 
            img = img.unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = model(img)
                probs = torch.sigmoid(logits).cpu().numpy()[0]

            # Global Storage
            gt_vector = np.zeros(len(xrv_classes))
            gt_vector[idx] = 1
            y_true_all.append(gt_vector)
            y_scores_all.append(probs)

            # Metrics Tracking
            conf = probs[idx]
            gt_confidences.append(conf)
            
            sorted_indices = np.argsort(probs)[::-1]
            rank = np.where(sorted_indices == idx)[0][0] + 1
            all_ranks.append(rank)
            
            # Per-class Tracking
            per_class_data[mapped_name]["probs"].append(conf)
            per_class_data[mapped_name]["ranks"].append(rank)

        except Exception as e:
            print(f"Error processing {img_name}: {e}")

# -----------------------------
# FINAL METRICS CALCULATION
# -----------------------------
if len(y_true_all) > 0:
    y_true_all = np.array(y_true_all)
    y_scores_all = np.array(y_scores_all)
    all_aurocs = []

    print("\n" + "="*85)
    print(f"{'Pathology':<25} | {'AUROC':<8} | {'Mean Prob':<10} | {'Mean Rank':<10} | {'Acc @ 0.5'}")
    print("-" * 85)

    for i, class_label in enumerate(xrv_classes):
        if class_label in per_class_data and np.sum(y_true_all[:, i]) > 0:
            # Per-class AUROC
            auroc = roc_auc_score(y_true_all[:, i], y_scores_all[:, i])
            all_aurocs.append(auroc)
            
            p_list = per_class_data[class_label]["probs"]
            r_list = per_class_data[class_label]["ranks"]
            
            mean_p = np.mean(p_list)
            mean_r = np.mean(r_list)
            acc_05 = sum(1 for p in p_list if p > THRESHOLD) / len(p_list)
            
            print(f"{class_label:<25} | {auroc:.4f} | {mean_p:.4f}    | {mean_r:<10.2f} | {acc_05:.4f}")

    # Top-K Accuracies
    total = len(all_ranks)
    top1 = sum(1 for r in all_ranks if r == 1) / total
    top3 = sum(1 for r in all_ranks if r <= 3) / total
    top5 = sum(1 for r in all_ranks if r <= 5) / total

    print("="*85)
    print(f"SUMMARY METRICS (N={total})")
    print("-" * 30)
    print(f"Macro AUROC:         {np.mean(all_aurocs):.4f}")
    print(f"Mean GT Probability: {np.mean(gt_confidences):.4f}")
    print(f"Accuracy @ {THRESHOLD}:    {sum(1 for c in gt_confidences if c > THRESHOLD)/total:.4f}")
    print(f"Overall Mean Rank:   {np.mean(all_ranks):.2f} (out of {len(xrv_classes)})")
    print(f"Top-1 Accuracy:      {top1:.4f}")
    print(f"Top-3 Accuracy:      {top3:.4f}")
    print(f"Top-5 Accuracy:      {top5:.4f}")
    print("="*85)
else:
    print("\nNo data processed.")
