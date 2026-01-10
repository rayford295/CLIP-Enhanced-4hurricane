# ==============================================================================
# 1. Library Imports and Global Settings
# ==============================================================================
import os
import random
import gc
import traceback
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats import entropy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

import timm
import torchvision.transforms as T

# —— Set Random Seeds for Reproducibility ——
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device in use: {device}")

# ==============================================================================
# 2. Configuration and Data Loading
# ==============================================================================
# --- Paths and Column Configurations ---
excel_path = "/content/0311_POST_cleaned.xlsx"
img_root   = "/content/0310_post_folder/0310_post_folder"
label_col  = "human_damage_perception"  # mild/moderate/severe
root_col   = "root"                     # subdirectory name
class_names = ["mild", "moderate", "severe"]
label_map = {c: i for i, c in enumerate(class_names)}

# --- Load and Clean DataFrame ---
df = pd.read_excel(excel_path)
df = df.dropna(subset=[root_col, label_col]).copy()
df["label"] = df[label_col].astype(str).str.lower().map(label_map)
df = df.dropna(subset=["label"]).copy()
df["label"] = df["label"].astype(int)

print(f"Total samples: {len(df)}")
print("Label distribution:\n", df["label"].value_counts().sort_index())

# --- Image Transformations ---
IM_SIZE = 224
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
train_tf = T.Compose([
    T.Resize((IM_SIZE, IM_SIZE)),
    T.RandomHorizontalFlip(p=0.5),
    T.ToTensor(),
    T.Normalize(mean, std),
])
eval_tf = T.Compose([
    T.Resize((IM_SIZE, IM_SIZE)),
    T.ToTensor(),
    T.Normalize(mean, std),
])

# ==============================================================================
# 3. Dataset and Model Definition
# ==============================================================================
class ImageDataset(Dataset):
    def __init__(self, df, img_root, root_col, label_col, transform):
        self.df = df.reset_index(drop=True)
        self.img_root = img_root
        self.root_col = root_col
        self.label_col = label_col
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        folder = os.path.join(self.img_root, str(row[self.root_col]))
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Folder not found: {folder}")
        files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if not files:
            raise FileNotFoundError(f"No image files found in folder: {folder}")

        img_path = os.path.join(folder, files[0])
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Unable to open image: {img_path}, Error: {e}")
            return self.__getitem__((i + 1) % len(self))

        img = self.transform(img)
        return {
            "pixel_values": img,
            "label": torch.tensor(row[self.label_col], dtype=torch.long),
            "path": img_path
        }

def create_vit_b16(num_classes=3, pretrained=True):
    model = timm.create_model("vit_base_patch16_224", pretrained=pretrained, num_classes=num_classes)
    return model

# ==============================================================================
# 4. Core Training and Evaluation Functions
# ==============================================================================
def train_one_fold(train_df, val_df, epochs=8, bsz=32, lr=1e-4, weight_decay=1e-4, fold_info=""):
    """
    Train a single fold and return the model with the best validation accuracy.
    """
    model = create_vit_b16(num_classes=len(class_names), pretrained=True).to(device)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss()

    dl_tr = DataLoader(ImageDataset(train_df, img_root, root_col, "label", train_tf),
                       batch_size=bsz, shuffle=True, num_workers=2, pin_memory=True)
    dl_va = DataLoader(ImageDataset(val_df, img_root, root_col, "label", eval_tf),
                       batch_size=bsz*2, shuffle=False, num_workers=2, pin_memory=True)

    best_acc, best_w = -1, None
    for ep in range(1, epochs + 1):
        model.train()
        tr_loss, n = 0.0, 0
        for batch in dl_tr:
            x = batch["pixel_values"].to(device)
            y = batch["label"].to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * y.size(0)
            n += y.size(0)

        model.eval()
        all_p, all_y = [], []
        with torch.no_grad():
            for batch in dl_va:
                x = batch["pixel_values"].to(device)
                y = batch["label"].cpu().numpy()
                logits = model(x).cpu()
                pred = logits.argmax(1).numpy()
                all_p.extend(pred)
                all_y.extend(y)

        if not all_y:
            print(f"{fold_info} Epoch {ep}/{epochs} - Validation set empty, skipping validation.")
            continue

        acc = accuracy_score(all_y, all_p)
        if acc > best_acc:
            best_acc = acc
            best_w = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"{fold_info} Epoch {ep}/{epochs} \t train_loss={tr_loss/n:.4f} \t val_acc={acc:.4f}")

    if best_w:
        model.load_state_dict(best_w)
    print(f"{fold_info} Training completed. Best validation accuracy: {best_acc:.4f}")
    return model

def run_cross_validation(df, folds=3, epochs=8, bsz=32, lr=1e-4):
    """
    Run cross-validation and generate misclassification analysis files.
    """
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    oof_rows = []

    for k, (tr_idx, va_idx) in enumerate(skf.split(df, df["label"])):
        fold_info = f"[Fold {k+1}/{folds}]"
        print(f"\n==== {fold_info} ====")
        tr_df, va_df = df.iloc[tr_idx], df.iloc[va_idx]

        model = train_one_fold(tr_df, va_df, epochs=epochs, bsz=bsz, lr=lr, fold_info=fold_info)

        dl_va = DataLoader(ImageDataset(va_df, img_root, root_col, "label", eval_tf),
                           batch_size=bsz*2, shuffle=False, num_workers=2, pin_memory=True)
        model.to(device).eval()
        with torch.no_grad():
            for batch in dl_va:
                x, y, paths = batch["pixel_values"].to(device), batch["label"].numpy(), batch["path"]
                logits = model(x).cpu()
                probs = F.softmax(logits, dim=1).numpy()
                preds = probs.argmax(1)
                for i in range(len(y)):
                    oof_rows.append({
                        "fold": k + 1, "path": paths[i], "true": class_names[y[i]], "pred": class_names[preds[i]],
                        "p_true": float(probs[i, y[i]]), "p_pred": float(probs[i, preds[i]]),
                        "confidence": float(probs[i, preds[i]]) - float(probs[i, y[i]]),
                        "entropy": float(entropy(probs[i], base=2)), "is_correct": int(y[i] == preds[i]),
                    })
        del model; gc.collect(); torch.cuda.empty_cache()

    oof = pd.DataFrame(oof_rows)
    oof.to_csv("/content/oof_vitb16_all.csv", index=False)
    print("\n[Report] Saved OOF predictions to /content/oof_vitb16_all.csv")

    mis = oof[oof["is_correct"] == 0].copy()
    overconf_th, ambiguous_th = 0.40, 0.10
    mis["error_type"] = np.where(mis["confidence"] >= overconf_th, "overconfident",
                                 np.where(mis["confidence"] <= ambiguous_th, "ambiguous", "medium"))
    mis.to_csv("/content/oof_vitb16_miscls.csv", index=False)
    print("[Report] Saved misclassification analysis to /content/oof_vitb16_miscls.csv")
    print("\n[Report] Misclassification Type Distribution:")
    print(mis["error_type"].value_counts())

# ==============================================================================
# 5. Execution
# ==============================================================================
print("--- Starting cross-validation for performance evaluation and error analysis ---")
run_cross_validation(df, folds=3, epochs=8, bsz=32, lr=1e-4)
print("\n--- Cross-validation and analysis completed ---\n")

print("--- Training the final model for inference ---")
final_train_df, final_val_df = train_test_split(
    df, test_size=0.1, random_state=42, stratify=df['label']
)
print(f"Final training set size: {len(final_train_df)}, Validation set size: {len(final_val_df)}")

final_model = train_one_fold(
    final_train_df,
    final_val_df,
    epochs=8,
    bsz=32,
    lr=1e-4,
    fold_info="[Final Model]"
)

save_path = "/content/final_vit_model.pth"
torch.save(final_model.state_dict(), save_path)
print(f"\n✅ Final model training completed. Weights saved to: {save_path}")
