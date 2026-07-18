#!/usr/bin/env python3
"""CFG-cost reduction for Ideogram v4 (training-free analog of fal's CFG distillation).

Ideogram v4 does true dual-DiT CFG: every step runs a conditional AND an unconditional
transformer (2 forwards/step). The unconditional velocity — and more importantly the CFG
*delta* d = pos_v - neg_v — varies slowly across adjacent timesteps. So we compute the
unconditional DiT only every N steps and reuse the cached delta in between:

    v = pos_v + (guidance - 1) * d          # exact rewrite of  g*pos + (1-g)*neg

  cadence N=1  -> refresh every step        == stock behavior (validated: SSIM≈1.0)
  cadence N=2  -> uncond ~half the steps    -> forwards 12->18 on TURBO_12
  cadence N=3  -> uncond ~third             -> forwards 12->16
  guidance==1  -> uncond weight is 0        -> skip it entirely (draft tier)

Implemented as a subclass so the pinned mflux site-packages is never edited. The eager
`_denoise_step` seam (ideogram4.py:213-237) sits OUTSIDE mx.compile, so this branching is
free of any compile conflict. `cfg_cadence` is a plain attribute — set it per generation.
"""
from __future__ import annotations

from typing import Any

import mlx.core as mx

from mflux.models.ideogram4 import Ideogram4


class Ideogram4Cadence(Ideogram4):
    cfg_cadence: int = 1  # N; 1 == stock

    def generate_image(self, *args, **kwargs):
        # reset per-generation CFG-delta cache + counters
        self._cfg_step = 0
        self._cfg_delta = None
        self._cfg_uncond_calls = 0
        return super().generate_image(*args, **kwargs)

    def _denoise_step(
        self,
        *,
        z: mx.array,
        t_value: float,
        s_value: float,
        guidance_value: float,
        text_z_padding: mx.array,
        llm_features: mx.array,
        inputs: dict[str, Any],
        negative_inputs: dict[str, mx.array],
        predict_conditional,
        predict_unconditional,
    ) -> mx.array:
        t = mx.full((1,), t_value, dtype=mx.float32)
        pos_v = predict_conditional(  # conditional DiT always runs
            z=z, t=t, text_z_padding=text_z_padding, llm_features=llm_features, inputs=inputs
        )
        k = getattr(self, "_cfg_step", 0)
        n = max(1, int(getattr(self, "cfg_cadence", 1)))
        if abs(guidance_value - 1.0) < 1e-6:
            v = pos_v  # unconditional branch has zero weight -> skip the DiT entirely
        else:
            refresh = (self._cfg_delta is None) or (k % n == 0)  # always refresh first step
            if refresh:
                neg_v = predict_unconditional(z=z, t=t, negative_inputs=negative_inputs)
                self._cfg_delta = pos_v - neg_v
                self._cfg_uncond_calls += 1
            v = pos_v + (guidance_value - 1.0) * self._cfg_delta
        self._cfg_step = k + 1
        return z + v * (s_value - t_value)
