#!/usr/bin/env python3
"""Assemble a runnable mflux model directory for the distilled Ideogram v4 variants.

Downloads:
  1. the distilled conditional transformer from VisualInference/ideogram-4-mlx-distilled
  2. the shared components (text encoder, VAE, tokenizer, scheduler, unconditional
     transformer, model_index.json) from Ideogram's own gated release
     ideogram-ai/ideogram-4-fp8 — you must have accepted its license on Hugging Face
     and be logged in (`huggingface-cli login`).

Then links everything into one directory that mflux can load directly.

Usage:
  python download_assemble.py --variant 6step          # or 12step
  python download_assemble.py --variant 6step --out ./ideogram4-6step
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download

DISTILL_REPO = "VisualInference/ideogram-4-mlx-distilled"
STOCK_REPO = "ideogram-ai/ideogram-4-fp8"
VARIANT_DIRS = {"12step": "cfg-fold-12step", "6step": "step-fold-6step"}
SHARED = ["text_encoder", "tokenizer", "vae", "unconditional_transformer", "scheduler"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANT_DIRS), required=True)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    sub = VARIANT_DIRS[a.variant]
    out = a.out or Path(f"./ideogram4-{a.variant}")

    print(f"[1/3] downloading distilled transformer ({sub}, ~8.7GB)…")
    distill = Path(snapshot_download(DISTILL_REPO, allow_patterns=[f"{sub}/*"]))
    print(f"[2/3] downloading stock components (gated; ~17GB)…")
    stock = Path(snapshot_download(
        STOCK_REPO,
        allow_patterns=["model_index.json"] + [f"{d}/*" for d in SHARED]))

    print(f"[3/3] assembling {out}")
    out.mkdir(parents=True, exist_ok=True)
    links = {"transformer": distill / sub / "transformer",
             "model_index.json": stock / "model_index.json",
             **{d: stock / d for d in SHARED}}
    for name, target in links.items():
        link = out / name
        if link.is_symlink() or link.is_file():
            link.unlink()
        elif link.is_dir():
            raise SystemExit(f"{link} is a real directory — remove it manually first")
        if not target.exists():
            raise SystemExit(f"missing {target} — incomplete download?")
        os.symlink(target, link)
        print(f"  {name} -> {target}")
    print(f"\ndone. Render with:\n  python generate.py --model {out} --variant {a.variant} "
          f"--prompt-file example_prompt.json --seed 4711")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
