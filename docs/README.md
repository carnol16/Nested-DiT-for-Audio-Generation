# Nested DiT PoC — Implementation Plan Index

A two-stage hierarchical Diffusion Transformer for audio generation, fitting inside 10 GB VRAM on a single consumer GPU.

**Spec:** [nested_dit_poc_plan.md](../../../nested_dit_poc_plan.md)

---

## Phases

| Phase | File | Goal | Depends On |
|---|---|---|---|
| 1 | [phase-1-foundation.md](phase-1-foundation.md) | Environment, DAC validation, latent pipeline, coarse resolution decision | Nothing |
| 2 | [phase-2-structural-dit.md](phase-2-structural-dit.md) | StructuralDiT (Stage 1) — implement and train to convergence | Phase 1 |
| 3 | [phase-3-detail-dit.md](phase-3-detail-dit.md) | DetailDiT (Stage 2) — implement with cross-attention, train with teacher forcing | Phase 2 |
| 4 | [phase-4-integration-evaluation.md](phase-4-integration-evaluation.md) | End-to-end pipeline, distribution alignment, 50-sample evaluation | Phase 3 |

---

## Key Architectural Decisions to Make in Phase 1

These must be pinned before any model code is written:

- **Dataset**: NSynth (recommended) vs. AudioCaps
- **VAE**: DAC (recommended) vs. AudioLDM VAE
- **Temporal downsampling factor**: 4x, 6x, or 8x between Stage 1 and Stage 2

---

## File Structure (Full Project)

```
Text_to_Audio/
├── nested_dit_poc_plan.md          ← original spec
├── requirements.txt                ← pinned deps (Phase 1)
├── config.yaml                     ← shared constants (downsampling factor, etc.)
├── data/
│   ├── latents/                    ← encoded .pt files (Phase 1)
│   └── stats.json                  ← latent mean/std/min/max (Phase 1)
│   └── dataset.py                  ← PyTorch Dataset (Phase 2)
├── models/
│   ├── structural_dit.py           ← Stage 1 model (Phase 2)
│   ├── detail_dit.py               ← Stage 2 model (Phase 3)
│   └── components/
│       ├── adaln.py                ← adaLN-Zero block (Phase 2)
│       ├── timestep_embedding.py   ← sinusoidal timestep embedding (Phase 2)
│       ├── transformer_block.py    ← base DiT block (Phase 2)
│       ├── cross_attention.py      ← Stage 1 → Stage 2 conditioning (Phase 3)
│       └── coarse_projector.py     ← coarse latent linear projection (Phase 3)
├── training/
│   ├── noise_schedule.py           ← linear DDPM schedule (Phase 2)
│   ├── sampler.py                  ← DDIM sampler (Phase 2)
│   ├── train_stage1.py             ← Stage 1 training loop (Phase 2)
│   ├── train_stage2.py             ← Stage 2 training loop (Phase 3)
│   └── finetune_stage2.py          ← distribution alignment fine-tune (Phase 4)
├── inference/
│   ├── pipeline.py                 ← end-to-end Stage 1 → Stage 2 → DAC (Phase 4)
│   └── vram_manager.py             ← sequential stage loading for 10 GB budget (Phase 4)
├── evaluation/
│   └── spectrogram_grid.py         ← spectrogram visualization (Phase 4)
├── scripts/
│   ├── preprocess.py               ← raw audio → DAC latent → .pt (Phase 1)
│   ├── compute_stats.py            ← compute latent distribution stats (Phase 1)
│   └── validate_coarse_resolution.py  ← listening test for downsampling factor (Phase 1)
├── checkpoints/
│   ├── stage1/                     ← Stage 1 checkpoints (Phase 2)
│   └── stage2/                     ← Stage 2 checkpoints (Phase 3)
├── results/
│   ├── samples/                    ← .wav files (Phase 4)
│   └── spectrograms/               ← spectrogram plots (Phase 4)
├── notebooks/
│   ├── 01_environment_check.ipynb  ← GPU + DAC smoke test (Phase 1)
│   ├── 02_structural_dit.ipynb     ← Stage 1 training + monitoring (Phase 2)
│   ├── 03_detail_dit.ipynb         ← Stage 2 training + conditioning validation (Phase 3)
│   ├── 04_integration.ipynb        ← end-to-end pipeline validation (Phase 4)
│   └── 05_evaluation.ipynb         ← 50-sample eval + written observations (Phase 4)
└── docs/
    └── plans/
        └── nested-dit-poc/
            ├── README.md           ← this file
            ├── phase-1-foundation.md
            ├── phase-2-structural-dit.md
            ├── phase-3-detail-dit.md
            └── phase-4-integration-evaluation.md
```

---

## Critical Failure Points (Read Before Starting)

These are the most common failure modes, ordered by phase:

| Phase | Failure | Cause | Fix |
|---|---|---|---|
| 1 | DAC decode produces silence | Sample rate mismatch — resample before encoding, not after | Resample audio to 44.1 kHz before calling DAC encoder |
| 1 | Latent std is 50+ | Float16 truncation or missing normalization | Store latents as float32 |
| 2 | Loss is NaN from step 1 | adaLN-Zero final projection not initialized to zero | Initialize last linear layer of adaLN MLP to zero |
| 2 | Samples are noise after 50K steps | Unnormalized latents | Verify normalization uses `data/stats.json` values |
| 3 | Cross-attention weights are uniform | Model is ignoring conditioning | Check K/V projections are receiving the coarse latent |
| 3 | Samples conditioned on A vs. B are indistinguishable | Conditioning dropout rate too high, or cross-attention not wired in | Reduce dropout rate; verify cross-attention residual connection |
| 4 | End-to-end samples are noise despite good Stage 1 and Stage 2 | Distribution mismatch | Fine-tune Stage 2 on Stage 1 outputs at 10% of original LR |
| 4 | Fine-tuning causes catastrophic forgetting | Learning rate too high | Reduce fine-tuning LR by another 5x |
