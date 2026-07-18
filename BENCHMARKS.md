# Benchmarks

All numbers: Mac mini M4 Pro (20-core GPU, 48GB), macOS 26, mflux 0.18, fp8 weights,
warm model, 800×992 unless noted. e2e = prompt-in to PNG-out including VAE decode.

## Speed ladder

| configuration | forwards | e2e | speedup |
|---|---|---|---|
| stock dual-DiT, 12 steps (TURBO preset) | 24 | ~250s | 1× |
| cfg-fold-12step (this release) | 12 | ~135s | 1.85× |
| step-fold-6step (this release) | 6 | ~86s | 2.9× |
| fal ideogram-v4-instant, 8 steps (converted to this runtime) | 8 | 84–89s | 2.9× |

Per-forward cost is ~3.7s at this resolution and is bandwidth-bound; fp8 beats a bf16
conversion of the same weights by 2.4× wall-clock on unified memory (measured — small
working set wins over cheaper math on Apple Silicon).

## Quality method

- Reference = 48-step dual-DiT "gold" renders at fixed prompt+seed+resolution.
- Fidelity gates during distillation: bit-deterministic golden regression (SSIM 1.0000
  required for stack changes), fused-model ≡ adapter-model equivalence, text-rendering
  checks, LAB/palette drift metrics, and blind human A/Bs on a fixed brief set.
- The 12-step fold holds gold-level color energy and guidance character (the pre-fold
  single-branch baseline is visibly hazy — that's what the distillation restores).
- The 6-step fold is style-faithful but recomposes at fixed seed vs 12-step; it slightly
  exceeds 12-step on "aliveness" (flyaway hair, motion) and slightly trails on
  contrast/eyelash-level detail. Step counts 8–10 on the 12-step model interpolate the
  trade (each step count composes differently — they are not the same image sharper).
- vs fal's official 8-step instant (converted into this same runtime for a
  fair single-variable A/B): wall-clock tie with 25% fewer forwards; their release reads
  crisper/commercial-graphic, this one holds filmic/atmospheric briefs (dragged shutter,
  halation, film-scan looks) closer to the base model's character.
- Refusal-placard matrix (harmless prompts, production res): stock dual 50–73% seed-
  dependent false positives; both folds and fal's release: 0%.

Renders in `examples/` are unretouched outputs of the released weights (prompts included).
