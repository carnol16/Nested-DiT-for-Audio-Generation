# Phase 3: DetailDiT (Stage 2)

> **Goal:** Implement and train the full-resolution DiT that conditions on Stage 1's coarse latent via cross-attention. By the end of this phase, Stage 2 should produce samples that are visibly and audibly conditioned on the structure provided by Stage 1.

**Depends on:** Phase 2 complete — a trained Stage 1 checkpoint exists and Stage 1 samples have discernible structure.
**Unlocks:** Phase 4 (integration and evaluation).

---

## What Success Looks Like

- DetailDiT forward pass runs with a coarse latent as conditioning input, output shape matches full-resolution latent
- Cross-attention weights are non-uniform (the model is actually attending to the conditioning, not ignoring it)
- Stage 2 samples conditioned on different coarse latents are audibly distinguishable from each other
- Stage 2 samples conditioned on the same coarse latent are more similar to each other than to randomly conditioned samples
- At least one checkpoint saved

---

## Files to Create

| File | Purpose |
|---|---|
| `models/detail_dit.py` | DetailDiT model definition |
| `models/components/cross_attention.py` | Cross-attention block for Stage 1 → Stage 2 conditioning |
| `models/components/coarse_projector.py` | Linear projection upsampling Stage 1 latent to full resolution |
| `training/train_stage2.py` | Training loop for Stage 2 |
| `notebooks/03_detail_dit.ipynb` | Training launch, conditioning validation, sample evaluation |
| `checkpoints/stage2/` | Output directory for Stage 2 checkpoints |

---

## Task Breakdown

### Task 3.0 — Read AudioLDM2 Before Writing Anything

**What to do:**

Read sections 3 and 4 of the [AudioLDM2 paper](https://arxiv.org/abs/2308.05734) before writing any Stage 2 code. Focus on how they condition a latent diffusion model on structured representations. The specific mechanism will differ from what you build, but the conditioning pipeline design decisions are directly applicable.

Specifically understand:
- How cross-attention is used to inject conditioning signals into a diffusion model
- How the conditioning signal is projected to match the expected key/value dimensions
- What "conditioning dropout" means and why it helps robustness

**Verify:**
- You can describe in one paragraph how cross-attention injects a conditioning signal into a transformer block
- You have decided whether to use conditioning dropout during Stage 2 training (recommended: yes, 10–20% dropout rate)

---

### Task 3.1 — Implement the Coarse Latent Projector

**What to do:**

Implement `models/components/coarse_projector.py`. This module takes the coarse latent from Stage 1 (shape `[B, C, T_coarse]`) and projects it into a sequence of key/value vectors for cross-attention in Stage 2.

Do not use bilinear interpolation. Use a learned linear projection. The projection should map from the coarse latent's channel dimension to the Stage 2 hidden dimension.

The output should be a sequence of `T_coarse` vectors, each of dimension `D_stage2` (Stage 2's hidden dimension). These serve as the keys and values in the cross-attention layer.

**Verify:**
- Input: `[B, C_coarse, T_coarse]`
- Output: `[B, T_coarse, D_stage2]`
- The projection is trainable (it is not frozen)

---

### Task 3.2 — Implement Cross-Attention

**What to do:**

Implement `models/components/cross_attention.py`. This is a multi-head attention layer where:
- **Queries** come from the current Stage 2 hidden state: `[B, T_full, D_stage2]`
- **Keys and values** come from the projected coarse latent: `[B, T_coarse, D_stage2]`

The cross-attention attends across the time dimension: each position in the full-resolution sequence can attend to any position in the coarse sequence.

This is a standard multi-head attention module with separate Q, K, V projections — the only difference from self-attention is that K and V come from a different sequence (the coarse latent).

**Verify:**
- Forward pass: query `[B, T_full, D]` + context `[B, T_coarse, D]` → output `[B, T_full, D]`
- Attention weights shape: `[B, heads, T_full, T_coarse]`
- Attention weights sum to 1 along the last axis (they are proper distributions)
- With a random initialization, weights are not perfectly uniform (near-uniform is fine, perfectly uniform suggests a bug)

---

### Task 3.3 — Implement the Stage 2 Transformer Block

**What to do:**

Extend the DiT transformer block from Phase 2 to include a cross-attention sublayer. The block order should be:

1. Self-attention (with adaLN-Zero modulation, same as Stage 1)
2. Cross-attention on the coarse latent (no adaLN modulation on cross-attention — timestep modulation only on self-attention and MLP)
3. MLP (with adaLN-Zero modulation)

Each sublayer has a residual connection.

This design keeps the adaLN-Zero conditioning focused on the diffusion timestep while the cross-attention handles the structural conditioning. Mixing them adds complexity without a clear benefit at PoC scale.

**Verify:**
- Forward pass: `[B, T_full, D]` + conditioning `[B, T_coarse, D]` + timestep → `[B, T_full, D]`
- Removing the conditioning (zeroing it out) changes the output — the cross-attention is doing something

---

### Task 3.4 — Implement DetailDiT

**What to do:**

Assemble `models/detail_dit.py`. Architecture parameters:

- Layers: 8–12 transformer blocks (each containing self-attention + cross-attention + MLP)
- Hidden dimension: 384–512
- Attention heads: 4–8
- Input: noised full-resolution latent + Stage 1 coarse latent (as conditioning) + timestep
- Output: predicted noise at full resolution

Use the same adaLN-Zero conditioning on timestep as in StructuralDiT. Reuse the components from `models/components/` — do not duplicate logic.

**Verify:**
- Forward pass runs without error
- Output shape matches the full-resolution latent shape exactly
- Model parameter count is logged (target: under 100M parameters)
- The model fits in under 5 GB VRAM with a batch size of 4

---

### Task 3.5 — Validate Cross-Attention is Actually Conditioning

**What to do:**

Before beginning any training, run a quick test to confirm the cross-attention is actually routing information from the coarse latent into Stage 2's outputs.

Pass the same noised full-resolution latent through DetailDiT twice: once with coarse latent A, once with coarse latent B (two different examples from the dataset). The outputs should differ. If they are identical (or nearly identical), the cross-attention is not functioning — the model is ignoring its conditioning input.

If the outputs are identical, check:
- Is the coarse latent actually being passed into the cross-attention K and V projections?
- Is the cross-attention output being added back via the residual connection?
- Are the K and V projection weights initialized non-zero?

**Verify:**
- Outputs conditioned on A vs. B differ (even with random weights, they should differ)
- Attention weight visualization shows non-uniform patterns across the time axis

---

### Task 3.6 — Build the Stage 2 Dataset

**What to do:**

Extend `data/dataset.py` (or create a Stage 2 variant) to return pairs: `(full_resolution_latent, coarse_latent)`. The coarse latent is the ground-truth downsample of the same clip — not the Stage 1 model output.

This is teacher forcing: Stage 2 learns from perfect conditioning signals during initial training. Using Stage 1 model outputs during this phase introduces unnecessary noise and makes convergence slower and harder to diagnose.

**Verify:**
- Each batch item is a `(full_latent, coarse_latent)` pair from the same clip
- Both tensors are normalized using the same stats from `data/stats.json`
- Coarse latent has shape `[B, C, T_coarse]` and full latent has shape `[B, C, T_full]`

---

### Task 3.7 — Implement the Stage 2 Training Loop

**What to do:**

Implement `training/train_stage2.py` with:
- Optimizer: AdamW, `lr=1e-4`, `weight_decay=1e-2`
- Loss: MSE between predicted noise and actual noise (same as Stage 1)
- Mixed precision: fp16 (same as Stage 1)
- Gradient checkpointing: enabled
- Conditioning dropout: randomly replace the coarse latent with Gaussian noise for 10–20% of training steps. This makes Stage 2 more robust to imperfect Stage 1 outputs.
- Logging: loss every 100 steps
- Sampling: every 5K steps, sample from Stage 2 conditioned on a ground-truth coarse latent. Save `.wav` with step number in filename.
- Checkpointing: every 10K steps

Stage 1 is frozen during Stage 2 training. Do not update Stage 1 weights. Do not load Stage 1 into GPU during training — only load it when generating samples for monitoring (then move it back to CPU).

**Verify:**
- Stage 2 loss is finite from step 1
- Loss is trending downward after 100 steps
- Sample at step 5K sounds different when conditioned on different coarse latents (even if quality is poor — variance is what you are checking, not quality)

---

### Task 3.8 — Train Stage 2 to Near-Convergence

**What to do:**

Train for at least 50K steps on ground-truth coarse conditioning. Monitor:
- Loss curves
- Conditioning effectiveness every 5K steps: generate 4 samples from the same coarse latent and 4 samples from a different coarse latent. Are they distinguishable?
- VRAM usage

The conditioning is working when samples from the same coarse latent cluster together perceptually, even if absolute quality is low.

**Verify:**
- Loss curve has flattened
- Samples conditioned on the same coarse latent are more similar to each other than to randomly conditioned samples
- At least one checkpoint saved at 50K+ steps

---

### Task 3.9 — Log the Conditioning Effectiveness Evidence

**What to do:**

Before moving to Phase 4, produce explicit evidence that the conditioning is working. Generate spectrograms (using `librosa.display.specshow`) for:

- 5 samples conditioned on coarse latent A
- 5 samples conditioned on coarse latent B

Save the spectrograms side-by-side and write 2–3 sentences describing what structural features appear to be preserved across the A samples and how they differ from the B samples.

This is not a formal metric. It is a qualitative check you need to be able to articulate in writing before the conditioning claim is considered validated.

**Verify:**
- Spectrograms are saved
- Written observations exist in the notebook
- You can describe in plain language what the conditioning is and is not doing

---

## VRAM Constraints

- Stage 2 model target: under 5 GB during training
- Stage 1 must be on CPU during Stage 2 training
- Enable gradient checkpointing
- For sample generation during training: load Stage 1 to GPU → generate coarse latent → move Stage 1 to CPU → load Stage 2 to GPU → generate full latent → decode

---

## Decision Log Template

Minimum entries before moving to Phase 4:
- Number of transformer layers chosen for Stage 2 and why
- Conditioning dropout rate chosen and why
- Evidence that cross-attention is working (not ignored)
- Batch size that fit in VRAM

---

## Blockers That Should Stop You from Moving Forward

Do not begin Phase 4 if:

- Samples conditioned on different coarse latents are indistinguishable (conditioning is not working)
- Cross-attention weights are uniformly distributed across all positions (model is ignoring conditioning)
- Stage 2 loss is NaN (check adaLN-Zero initialization, same as Stage 1)
- No checkpoint has been saved
