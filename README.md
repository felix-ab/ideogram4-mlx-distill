# Ideogram v4 — Distilled for Apple Silicon (mflux/MLX)

Single-branch distillations of [Ideogram v4](https://huggingface.co/ideogram-ai/ideogram-4-fp8)
that make the full 9.3B model practical on Apple Silicon: **2× fewer forward passes**
(CFG folded into the conditional DiT) and optionally **6 denoising steps instead of 12** —
with the filmic/editorial character of the base model intact.

> This is an independent, non-commercial community distillation. It is **not** an official
> Ideogram, Inc. product and has not been endorsed, approved or validated by Ideogram, Inc.

## What's in the release

| variant | forwards @ n steps | warm e2e, 800×992, M4 Pro 48GB | character |
|---|---|---|---|
| stock dual-DiT (baseline) | 2×12 | ~250s | reference |
| **cfg-fold-12step** | 1×12 | ~135s | seed-compatible workhorse, hero-refine base |
| **cfg+step-fold-6step** | 1×6 | ~86s | fastest; style-faithful, not seed-exact vs 12-step |

Both are drop-in mflux model directories: our distilled **conditional transformer**
(fp8, 8.7GB) plus references to the stock release for the text encoder, VAE and scheduler
(you download those from Ideogram's own gated repo — this release redistributes only the
derivative transformer).

Measured on-device (M4 Pro, 48GB): the 6-step fold matches fal's official 8-step
`ideogram-v4-instant` on wall-clock while doing 25% less compute, and holds
filmic/editorial prompts better in our A/Bs (their release leans crisper/commercial).
Full benchmark method and numbers in [BENCHMARKS.md](BENCHMARKS.md).

## Requirements

- Apple Silicon Mac, **48GB unified memory recommended** (peak ~30GB at 800×992;
  1.5MP wide frames peak ~41GB). 32GB works at 512² only.
- macOS 15+, Python 3.12, [mflux](https://github.com/filipstrand/mflux) ≥ 0.18
  (`ideogram4` family support).
- Access to the gated [ideogram-ai/ideogram-4-fp8](https://huggingface.co/ideogram-ai/ideogram-4-fp8)
  repo (accept Ideogram's license) for the shared components.

## Install & run

```bash
pip install mflux>=0.18
huggingface-cli login                      # account with ideogram-4-fp8 access
python download_assemble.py --variant 6step    # pulls our transformer + stock components, links the model dir
python generate.py --model ./ideogram4-6step --variant 6step \
  --width 800 --height 992 --seed 4711 --prompt-file example_prompt.json
```

Key runtime facts (details in [INFERENCE.md](INFERENCE.md)):
- Run distilled models with `guidance 1.0` via the included cadence runner (skips the
  unconditional branch entirely — that's the speedup).
- 6-step preset: `mu 0.5, std 1.75` (the training grid). 12-step: stock preset, CFG ramp
  capped at 6.
- Prompts are Ideogram's structured JSON; dimensions must be multiples of 16. The
  included PROMPT_NOTES.md summarizes measured prompt-dial behavior.

## Limitations

- **Non-commercial use only** — inherited from the Ideogram 4 Non-Commercial Model
  Agreement (see NOTICE / LICENSE.md). All conditions of that agreement flow down.
- The 6-step variant recomposes vs the 12-step model at the same seed (style-faithful,
  not seed-exact). Use the 12-step fold when seed anchoring matters.
- Single-branch models don't support real CFG sweeps; guidance is baked.
- Incidental background text can garble (true of the base dual-DiT as well); declare
  wanted text explicitly as elements.
- Distilled from the fp8 release on self-generated teacher trajectories (56 captions,
  72 trajectories); no additional image data was used for the distillations.

## Attribution

Ideogram 4 was created by Ideogram, Inc. and is used under the Ideogram Non-Commercial
Model Agreement. This repository's modifications: CFG-fold distillation (dual→single
branch) and 12→6 step distillation, by Felix Brener, 2026. The `NOTICE` file carries the
license-required attribution; `MODIFICATIONS.md` documents exactly which tensors were
changed and how.
