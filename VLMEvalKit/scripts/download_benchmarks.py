#!/usr/bin/env python3
"""Utility to fetch VLMEvalKit benchmark TSV files with one command.

Usage: python scripts/download_benchmarks.py Spatial457 MMBench_DEV_EN_V11

The script checks each dataset class for a HTTP(S) download link and stores the
TSV under ``$LMUData/<dataset>.tsv`` (default root: ``~/LMUData``). This mirrors
the layout that VLMEvalKit expects when running offline.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple
from urllib.parse import urlparse

# Reuse VLMEvalKit internals for dataset metadata and download helpers
from vlmeval.dataset import DATASET_CLASSES
from vlmeval.smp.file import LMUDataRoot, download_file, md5


def iter_dataset_metadata(dataset: str) -> Iterable[Tuple[str, Optional[str], Dict[str, str]]]:
    """Yield candidate (url, md5_map) pairs for the given dataset name."""

    for cls in DATASET_CLASSES:
        if dataset not in cls.supported_datasets():
            continue
        url_map = getattr(cls, "DATASET_URL", {})
        md5_map = getattr(cls, "DATASET_MD5", {})
        if isinstance(url_map, dict) and dataset in url_map:
            yield url_map[dataset], md5_map
        elif isinstance(url_map, str) and url_map:
            yield url_map, md5_map


def resolve_download_target(dataset: str, root: Path) -> Optional[Tuple[str, Path, Optional[str]]]:
    """Return (url, output_path, md5) if the dataset exposes a downloadable TSV."""

    for url, md5_map in iter_dataset_metadata(dataset):
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            continue
        target = root / f"{dataset}.tsv"
        checksum = md5_map.get(dataset) if isinstance(md5_map, dict) else None
        return url, target, checksum
    return None


def needs_download(target: Path, checksum: Optional[str], force: bool) -> bool:
    if force or not target.exists():
        return True
    if checksum is None:
        return False
    try:
        return md5(str(target)) != checksum
    except Exception:
        return True


def download_dataset(dataset: str, root: Path, force: bool) -> None:
    resolved = resolve_download_target(dataset, root)
    if resolved is None:
        print(f"[WARN] No HTTP download link registered for dataset '{dataset}'.", file=sys.stderr)
        return

    url, target, checksum = resolved
    os.makedirs(target.parent, exist_ok=True)

    if not needs_download(target, checksum, force):
        print(f"[SKIP] {dataset}: already exists at {target}.")
        return

    tmp_path = target.with_suffix(target.suffix + ".tmp")
    print(f"[INFO] Downloading {dataset} from {url} → {tmp_path}")
    download_file(url, str(tmp_path))
    if checksum is not None:
        file_md5 = md5(str(tmp_path))
        if file_md5 != checksum:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"MD5 mismatch for {dataset}: expected {checksum}, got {file_md5}."
            )

    tmp_path.replace(target)
    print(f"[DONE] Saved {dataset} to {target}")


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Download VLMEvalKit benchmark TSV files.")
    parser.add_argument("datasets", nargs="+", help="Dataset names (e.g., Spatial457 MMBench_DEV_EN_V11)")
    parser.add_argument("--root", type=Path, default=None, help="Override LMUData root directory")
    parser.add_argument("--force", action="store_true", help="Re-download even if file exists")
    args = parser.parse_args(list(argv) if argv is not None else None)

    root = Path(args.root) if args.root is not None else Path(LMUDataRoot())
    for dataset in args.datasets:
        try:
            download_dataset(dataset, root, args.force)
        except Exception as exc:  # pragma: no cover - surface error context to user
            print(f"[FAIL] {dataset}: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())

