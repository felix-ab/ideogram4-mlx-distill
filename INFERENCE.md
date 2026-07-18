# Inference notes

## The one rule

Distilled variants run **single-branch**: guidance 1.0 for every step. The included
`Ideogram4Cadence` subclass skips the unconditional transformer entirely when the step's
guidance weight is 1.0 — that skip *is* the speedup. Running these weights with a stock
guidance ramp (>1) will call the unconditional branch against fold-trained weights and
degrade output. `generate.py` sets this up correctly.

## Presets

| variant | steps | guidance | mu | std |
|---|---|---|---|---|
| 6step | 6 | 1.0 ×6 | 0.5 | 1.75 |
| 12step | 12 | 1.0 ×12 | 0.5 | 1.75 |

The 6-step schedule is the training grid (the even nodes of the 12-step logit-normal
grid) — don't substitute other step counts on the 6-step model; intermediate step counts
(8–10) DO work on the 12-step model and trade detail for time (~10s/step at 1.5MP).

## Memory & speed (M4 Pro 48GB, warm model)

| resolution | e2e 6step | e2e 12step | peak |
|---|---|---|---|
| 512×512 | ~33s | ~50s | ~22GB |
| 800×992 | ~86s | ~135s | ~30GB |
| 1472×816 | ~150s | ~258s | ~41GB |

Cold model load adds ~33s. 32GB machines: stay at 512². First generation after load
includes mx.compile warmup.

## Practical rules (measured, not vibes)

- Dimensions must be multiples of 16.
- Render people at ≥800×992; 512² is a draft resolution (anatomy and text both suffer).
- Wanted text: declare it explicitly as elements with y-first bboxes (×1000 coords).
  Incidental background text garbles in every lane, including the stock dual model —
  add "no text" to aesthetics when you don't want any.
- The refusal placards of the stock dual model don't occur in single-branch operation;
  if an output ever comes back near-solid-color, re-roll the seed.
- A quality-recovery pattern worth knowing: render 12step at base resolution, lanczos
  ×1.28, then re-noise to σ≈0.35 and run 6 distilled tail steps at the higher resolution
  (a cheap "hero" upscale that sharpens without recomposing). Implementation sketch in
  BENCHMARKS.md sources.
