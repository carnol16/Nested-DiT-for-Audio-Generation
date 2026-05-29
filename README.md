# Nested DiT for Audio Generation (PoC)

A two-stage hierarchical Diffusion Transformer that separates global audio structure from fine-grained detail — running inside 10 GB VRAM on a single consumer GPU.

---

## What This Is Trying to Prove

This is a proof-of-concept with a specific claim to validate: a two-stage hierarchy is learnable, and the cross-attention conditioning relationship between Stage 1 and Stage 2 is real. Stage 1 must produce structurally coherent coarse latents; Stage 2 must produce full-resolution latents that are visibly conditioned on them; and the full pipeline must run within the VRAM budget on a single consumer GPU.

Everything else — text conditioning, scaling, formal evaluation metrics — comes after this claim is validated.

---

## The Problem

Flat diffusion models generate audio in a single pass. They learn local coherence well — individual notes, timbral texture, short-range patterns — but tend to lose global structure over longer time horizons. The result is audio that sounds plausible moment-to-moment but drifts structurally: the diffusion equivalent of writing grammatical sentences but incoherent paragraphs.

The root cause is architectural. A flat model must simultaneously learn the marginal distribution of the data *and* its long-range compositional structure, with no explicit mechanism to separate the two. For audio, this means rhythm, phrase boundaries, and macro dynamics compete with spectral detail for the same representational capacity.

---

## The Architecture

This project borrows a structural insight from **Recursive Neural Networks (RvNNs)**: meaning is hierarchical, and representations should reflect that hierarchy explicitly. In an RvNN, a root node represents the whole, child nodes represent parts, and leaves represent atomic units — each level composed from the one below. The hierarchy is *forced*, not hoped-for.

The two-stage DiT maps this directly to audio generation:

```
                      ┌──────────────────────────────┐
                      │        Stage 1               │
                      │      StructuralDiT            │
                      │  (coarse latent — rhythm,     │
                      │   phrase shape, macro form)   │
                      └──────────────┬───────────────┘
                                     │ cross-attention
                                     ▼
                      ┌──────────────────────────────┐
                      │        Stage 2               │
                      │       DetailDiT              │
                      │  (full-res latent — timbre,  │
                      │   harmonics, spectral detail) │
                      └──────────────┬───────────────┘
                                     │
                               ┌─────▼─────┐
                               │    DAC    │
                               │  Decoder  │
                               └─────┬─────┘
                                     │
                                 Audio Output
```

**Stage 1 (StructuralDiT)** operates on a temporally downsampled latent — a low-resolution view of the audio that retains rhythm and phrase structure while discarding spectral detail. It learns the global shape of the audio first.

**Stage 2 (DetailDiT)** operates on the full-resolution latent, conditioned on the Stage 1 output via cross-attention. It fills in timbral and spectral detail guided by the structural skeleton Stage 1 produced. Stage 2 is trained with teacher forcing (ground-truth coarse latents as conditioning), then optionally fine-tuned on Stage 1 model outputs to close the distribution gap at inference time.

**DAC (Descript Audio Codec)** is the frozen VAE that bridges both stages to waveform space. Audio is encoded to latents once offline; the raw waveform is never touched during training.

**VRAM management:** Both stages never occupy GPU memory simultaneously. Stage 1 runs, its output is saved to CPU, Stage 1 is offloaded, then Stage 2 loads. This keeps peak VRAM within the 10 GB budget across the full inference pipeline.

The cross-attention between stages is the architectural core — it is the direct analogue of RvNN's parent-to-child conditioning, implemented as the natural mechanism for transformer-based hierarchical conditioning.

---

## Dataset

Training uses **NSynth** (Google Magenta) — 305,979 four-second musical instrument note samples at 16 kHz. NSynth was chosen deliberately for its regularity: fixed duration, consistent structure, and a narrow distribution. A boring dataset is an advantage at PoC stage — if the model fails, the failure is architectural, not a data artifact.

The plan is to validate the pipeline on NSynth, then switch to **AudioCaps** (46K clips with natural language captions) when adding text conditioning in the next phase.

---

## Current Status

| Phase | Description | Status |
|---|---|---|
| 1 | Data pipeline — DAC validation, latent extraction, stats | Completed |
| 2 | StructuralDiT — implement and train Stage 1 | In Progress |
| 3 | DetailDiT — implement with cross-attention, teacher forcing | Not started |
| 4 | Integration — end-to-end pipeline, evaluation | Not started |

---

## What Comes Next

Once the PoC validates the two-stage hierarchy:

1. **Text conditioning** — Add CLAP or FLAN-T5 as a conditioning encoder for Stage 1. Switch dataset to AudioCaps. Stage 1 conditions on text to generate a structurally appropriate coarse latent; Stage 2 conditions on that latent as before.
2. **Scale** — Increase model size and training duration. This is where the VRAM ceiling starts to bind — distributed training or gradient offloading becomes necessary.
3. **Formal evaluation** — Replace informal listening with FAD (Fréchet Audio Distance) and CLAP similarity scores so architecture changes can be compared objectively.

---

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.11, CUDA 12.x, and an NVIDIA GPU with at least 10 GB VRAM.

**Preprocess audio to latents (one-time):**
```bash
python scripts/preprocess.py
```

**Compute latent distribution statistics:**
```bash
python scripts/compute_stats.py
```

---

## Project Structure

```
Text_to_Audio/
├── requirements.txt
├── config.yaml                  # shared constants (downsampling factor, etc.)
├── data/
│   ├── latents/                 # encoded .pt files
│   └── stats.json               # latent mean / std / min / max
├── models/
│   ├── structural_dit.py        # Stage 1
│   ├── detail_dit.py            # Stage 2
│   └── components/              # adaLN, cross-attention, timestep embedding
├── training/
│   ├── train_stage1.py
│   └── train_stage2.py
├── inference/
│   └── pipeline.py              # Stage 1 → Stage 2 → DAC decode
├── scripts/
│   ├── preprocess.py
│   ├── compute_stats.py
│   └── validate_coarse_resolution.py
├── notebooks/                   # per-phase experiment notebooks
└── docs/                        # architecture plan and phase breakdowns
```

---

## References

**Diffusion Fundamentals**
- Ho et al. (2020) — [Denoising Diffusion Probabilistic Models (DDPM)](https://arxiv.org/abs/2006.11239)
- Song et al. (2020) — [Denoising Diffusion Implicit Models (DDIM)](https://arxiv.org/abs/2010.02502)

**Architecture**
- Peebles & Xie (2023) — [Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748)
- Kumar et al. (2023) — [High-Fidelity Audio Compression with Improved RVQGAN (DAC)](https://arxiv.org/abs/2306.06546)

**Hierarchical Inspiration**
- Socher et al. (2011) — [Parsing Natural Scenes and Natural Language with Recursive Neural Networks](https://ai.stanford.edu/~ang/papers/icml11-RecursiveNeuralNetworks.pdf)
- Socher et al. (2013) — [Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank](https://aclanthology.org/D13-1170.pdf)
- Tai et al. (2015) — [Improved Semantic Representations From Tree-Structured LSTMs](https://arxiv.org/abs/1503.00075)

**Related Work**
- Liu et al. (2023) — [AudioLDM2](https://arxiv.org/abs/2308.05734)
- Shih et al. (2024) — [Whole-Song Hierarchical Generation with Cascaded Diffusion Models](https://arxiv.org/abs/2405.09901)
- ACE-Step Team (2025) — [ACE-Step 1.5: Pushing the Boundaries of Open-Source Music Generation](https://arxiv.org/abs/2602.00744)