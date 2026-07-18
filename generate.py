#!/usr/bin/env python3
"""Minimal runner for the distilled Ideogram v4 variants (mflux ≥ 0.18, Apple Silicon).

  python generate.py --model ./ideogram4-6step --variant 6step \
      --prompt-file example_prompt.json --seed 4711 --width 800 --height 992

Prompts are Ideogram's structured JSON (see PROMPT_NOTES.md). Distilled models run
single-branch: guidance 1.0, the unconditional transformer is never called.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

VARIANTS = {
    "6step": dict(steps=6, mu=0.5, std=1.75),
    "12step": dict(steps=12, mu=0.5, std=1.75),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True, help="dir from download_assemble.py")
    ap.add_argument("--variant", choices=list(VARIANTS), required=True)
    ap.add_argument("--prompt-file", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("out.png"))
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=992)
    ap.add_argument("--seed", type=int, default=4711)
    a = ap.parse_args()
    if a.width % 16 or a.height % 16:
        sys.exit(f"width/height must be multiples of 16 (got {a.width}x{a.height})")

    prompt = json.loads(a.prompt_file.read_text())

    from mflux.models.common.config import ModelConfig
    from mflux.models.ideogram4.model.ideogram4_scheduler.scheduler import (
        Ideogram4SamplerPreset, Ideogram4Scheduler)
    from ideogram_cadence import Ideogram4Cadence

    v = VARIANTS[a.variant]
    Ideogram4Scheduler.PRESETS["DISTILLED"] = Ideogram4SamplerPreset(
        num_steps=v["steps"], guidance_schedule=(1.0,) * v["steps"], mu=v["mu"], std=v["std"])

    print(f"[load] {a.model} ({a.variant}: {v['steps']} steps, single branch)")
    model = Ideogram4Cadence(model_config=ModelConfig.ideogram4_fp8(), model_path=str(a.model))
    t = time.perf_counter()
    img = model.generate_image(seed=a.seed, prompt=prompt, width=a.width, height=a.height,
                               preset="DISTILLED")
    img.save(str(a.out), overwrite=True)
    print(f"[done] seed {a.seed} in {time.perf_counter()-t:.0f}s -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
