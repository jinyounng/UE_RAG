import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description="Cut the first N rows from a TSV.")
    ap.add_argument("--src", required=True, help="원본 TSV 경로")
    ap.add_argument("--dst", required=True, help="저장할 TSV 경로")
    ap.add_argument("--n", type=int, required=True, help="가져올 행 수 (헤더 제외)")
    args = ap.parse_args()

    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src, sep="\t")
    df.iloc[: args.n].to_csv(dst, sep="\t", index=False)
    print(f"[done] {src} → {dst} (rows={min(args.n, len(df))})")


if __name__ == "__main__":
    main()