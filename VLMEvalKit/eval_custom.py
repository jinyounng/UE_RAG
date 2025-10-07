#!/usr/bin/env python3
import pandas as pd
import re
from pathlib import Path


def normalize(text: str):
    """Extract a comparable answer token from model output."""
    if text is None:
        return ""

    t = str(text).strip().lower()

    # 1) yes/no quick check
    if "yes" in t:
        return "yes"
    if "no" in t:
        return "no"

    # 2) look for options like 'option a', '(a)', '[A]', etc.
    match = re.search(r"\b([abcd])\b", t)
    if match:
        return match.group(1)

    match = re.search(r"option\s*([abcd])", t)
    if match:
        return match.group(1)

    # 3) fallback: strip punctuation/extra whitespace
    return re.sub(r"[^a-z0-9]", "", t)


def compute_accuracy(gt_tsv: Path, pred_xlsx: Path):
    gt_df = pd.read_csv(gt_tsv, sep="\t")
    pred_df = pd.read_excel(pred_xlsx)

    merged = gt_df.merge(
        pred_df[["index", "prediction"]],
        on="index",
        how="inner",
        suffixes=("_gt", "_pred"),
    )

    merged["norm_gt"] = merged["answer"].apply(normalize)
    merged["norm_pred"] = merged["prediction"].apply(normalize)
    merged["correct"] = merged["norm_gt"] == merged["norm_pred"]

    accuracy = merged["correct"].mean() * 100
    print(f"Samples compared: {len(merged)}")
    print(f"Accuracy: {accuracy:.2f}%")

    return merged  # 필요하면 breakdown용으로 리턴


if __name__ == "__main__":
    gt_path = Path("data/Spatial457_1k.tsv")
    pred_path = Path("outputs/llf_ragseq/LLF-Qwen2.5VL-3B-RAGSeq_Spatial457_1k.xlsx")

    compute_accuracy(gt_path, pred_path)
