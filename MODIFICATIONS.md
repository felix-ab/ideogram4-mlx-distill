# Modifications to Ideogram 4 (license §3(iv) notice)

Base: `ideogram-ai/ideogram-4-fp8`, snapshot `ee79a7237b519f1402ceacf952f30c8a31ec5073`.
Both released variants modify **only the conditional transformer**
(`transformer/diffusion_pytorch_model.safetensors`). The text encoder, VAE, tokenizer,
scheduler configuration and unconditional transformer are NOT redistributed here — the
assembly script references Ideogram's own gated release for those, unmodified.

## cfg-fold-12step

Classifier-free guidance folded into the conditional branch by LoRA self-distillation:
a rank-32 adapter over all 34 layers' attention (`qkv`, `o`) and feed-forward
(`w1`, `w2`, `w3`) projections (170 target matrices) was trained to match the guided
dual-DiT output (guidance ramp capped at 6) on 56 self-generated caption briefs, then
fused into the fp8 weights. Fusion = dequantize per-row (`W = fp8(u8) · scale[:,None]`),
add `B·A` deltas, requantize per-row (`scale' = rowmax|W'|/448`). Every fp8 weight tensor
and its `weight_scale` in the conditional transformer therefore differs from stock;
biases, norms, embeddings and non-quantized tensors are unchanged except where listed in
the adapter target set above.

## step-fold-6step

The cfg-fold-12step transformer further distilled from 12 to 6 denoising steps: teacher
trajectories captured from the fused 12-step model through its compiled predictor, a
second rank-32 adapter (same 170-matrix target set) trained on trajectory states at the
6-step grid (the even nodes of the 12-step grid — exact subset by construction), then
fused with the same per-row requantization. The 6-step sampler preset (`mu 0.5, std 1.75`,
guidance 1.0) is part of the variant's intended configuration.

## Behavioral notes

- Both variants are single-branch models: the unconditional transformer is bypassed at
  inference (guidance 1.0). Guidance behavior is baked; CFG sweeps are not supported.
- The dual-DiT's seed-dependent refusal placards do not occur in single-branch operation
  (measured 0% on our harmless-prompt matrix; fal's official single-branch distillation
  measures the same). No content-related training was performed; users remain bound by
  Ideogram's Acceptable Use Policy.
- No image data beyond the base model's own outputs was used in either distillation.
