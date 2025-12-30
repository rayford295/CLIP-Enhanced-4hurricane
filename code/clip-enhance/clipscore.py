# ============================================================
# Compute CLIPScore (image ↔ text) for one or multiple CSV files
# CLIPScore here = 100 * cosine(clip_image_feature, clip_text_feature)
# - Uses pretrained CLIP only (NO fine-tuning)
# - Text is truncated/padded to max_length=77 (same as standard CLIP)
# ============================================================

import os
import gc
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor

# -------------------------
# 1) CONFIG (edit these)
# -------------------------
MODEL_ID = "openai/clip-vit-base-patch16"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Your CSVs (edit)
CSV_LIST = [
     ("Gemini", "/content/drive/MyDrive/2nd_disasterCLIP (revise)/datatset/12.28 llm label/post_image_gemini_labels.csv"),
     ("GPT",    "/content/drive/MyDrive/2nd_disasterCLIP (revise)/datatset/12.28 llm label/post_image_gpt_labels.csv"),
]

# Base dir for relative image paths (edit if needed)
BASE_DIR = "/content/drive/MyDrive/2nd_disasterCLIP (revise)/dataset/0310_dataset/0310_post_folder"

# Column names (edit to match your CSV)
TEXT_COL = "long_caption"
PATH_COL = "image_path"

# Optional label column for per-class mean CLIPScore (set None to disable)
LABEL_COL = "folder_tag"
LABEL_MAP = {"mild": 0, "moderate": 1, "severe": 2}  # used only if LABEL_COL is provided

# Dataloader / tokenization
BATCH_SIZE = 64
NUM_WORKERS = 0
MAX_LENGTH = 77

# If True: print a few sample rows with resolved image path and truncated token length
DEBUG_SHOW_SAMPLES = False
DEBUG_N = 3


# -------------------------
# 2) Dataset
# -------------------------
class ClipScoreDataset(Dataset):
    def __init__(self, df, processor, base_dir, text_col, path_col,
                 label_col=None, label_map=None, max_length=77):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.base_dir = base_dir
        self.text_col = text_col
        self.path_col = path_col
        self.label_col = label_col
        self.label_map = label_map or {}
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def _resolve_path(self, p):
        p = str(p)
        if not os.path.isabs(p):
            p = os.path.join(self.base_dir, p)
        return p

    def _get_label_idx(self, row):
        if self.label_col is None or self.label_col not in row:
            return -1
        v = row[self.label_col]
        if pd.isna(v):
            return -1
        if isinstance(v, str):
            return int(self.label_map.get(v.strip().lower(), -1))
        try:
            return int(v)
        except Exception:
            return -1

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self._resolve_path(row[self.path_col])

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            # keep consistent with your training fallback
            image = Image.new("RGB", (224, 224), color="black")

        text = "" if pd.isna(row[self.text_col]) else str(row[self.text_col])
        label_idx = self._get_label_idx(row)

        inputs = self.processor(
            text=[text],
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "label_idx": torch.tensor(label_idx, dtype=torch.long),
            "img_path": img_path,
        }


# -------------------------
# 3) Core CLIPScore compute
# -------------------------
@torch.no_grad()
def compute_clipscore(csv_path, name="RUN"):
    df = pd.read_csv(csv_path).copy()
    df = df.dropna(subset=[TEXT_COL, PATH_COL]).copy()

    if len(df) == 0:
        raise ValueError(f"[{name}] No valid rows after dropping NaNs for {TEXT_COL}/{PATH_COL}")

    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    model = CLIPModel.from_pretrained(MODEL_ID).to(DEVICE).eval()

    ds = ClipScoreDataset(
        df=df,
        processor=processor,
        base_dir=BASE_DIR,
        text_col=TEXT_COL,
        path_col=PATH_COL,
        label_col=LABEL_COL,
        label_map=LABEL_MAP,
        max_length=MAX_LENGTH,
    )

    if DEBUG_SHOW_SAMPLES:
        print(f"\n[{name}] Debug samples:")
        for i in range(min(DEBUG_N, len(ds))):
            item = ds[i]
            # token count (non-pad) roughly by attention mask sum
            token_len = int(item["attention_mask"].sum().item())
            print(f"  - img: {item['img_path']} | token_len(non-pad)={token_len}")

    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    scores = []
    per_class = defaultdict(list)

    for batch in tqdm(dl, desc=f"CLIPScore [{name}]"):
        pixel_values = batch["pixel_values"].to(DEVICE)
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        label_idx = batch["label_idx"].cpu().numpy()

        img_feat = model.get_image_features(pixel_values=pixel_values)
        txt_feat = model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)

        img_feat = F.normalize(img_feat, dim=-1)
        txt_feat = F.normalize(txt_feat, dim=-1)

        # cosine per sample (same image with its own text)
        sim = torch.sum(img_feat * txt_feat, dim=-1)          # [B]
        clipscore = (100.0 * sim).detach().cpu().numpy()      # [B]

        scores.extend(clipscore.tolist())
        for s, y in zip(clipscore, label_idx):
            if int(y) != -1:
                per_class[int(y)].append(float(s))

    scores_np = np.array(scores, dtype=np.float32)
    summary = {
        "n": int(scores_np.size),
        "mean": float(scores_np.mean()) if scores_np.size else 0.0,
        "median": float(np.median(scores_np)) if scores_np.size else 0.0,
        "std": float(scores_np.std()) if scores_np.size else 0.0,
        "min": float(scores_np.min()) if scores_np.size else 0.0,
        "max": float(scores_np.max()) if scores_np.size else 0.0,
    }

    per_class_mean = {}
    if len(per_class) > 0:
        for k, v in per_class.items():
            if len(v) > 0:
                per_class_mean[k] = float(np.mean(np.array(v, dtype=np.float32)))

    # cleanup
    del model, processor, ds, dl
    torch.cuda.empty_cache()
    gc.collect()

    return summary, per_class_mean


# -------------------------
# 4) Run
# -------------------------
def main():
    if not CSV_LIST:
        raise ValueError("CSV_LIST is empty. Please add your CSV paths in CSV_LIST.")

    all_rows = []
    print(f"Using model: {MODEL_ID} | device: {DEVICE}")
    print(f"Text col: {TEXT_COL} | Path col: {PATH_COL} | Max length: {MAX_LENGTH}")

    for name, csv_path in CSV_LIST:
        print(f"\n=== {name} ===")
        summary, per_class = compute_clipscore(csv_path, name=name)

        print("CLIPScore summary (100*cosine):")
        print({k: round(v, 4) for k, v in summary.items()})

        if LABEL_COL is not None and per_class:
            # map idx back to name if possible
            inv_map = {v: k for k, v in LABEL_MAP.items()}
            pretty = {inv_map.get(k, str(k)): round(v, 4) for k, v in per_class.items()}
            print(f"Per-class mean (by {LABEL_COL}):")
            print(pretty)

        row = {"Method": name, **summary}
        if per_class:
            for k, v in per_class.items():
                row[f"class_mean_{k}"] = v
        all_rows.append(row)

    out_df = pd.DataFrame(all_rows)
    print("\n=== Final Table ===")
    print(out_df.round(4))

    # Optional: save
    out_path = "clipscore_summary.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
