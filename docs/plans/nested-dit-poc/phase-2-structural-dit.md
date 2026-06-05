# Phase 2: StructuralDiT (Stage 1)

> **Goal:** Implement and train the coarse-latent DiT. By the end of this phase, Stage 1 should produce samples that sound like plausible low-resolution audio with discernible rhythm and macro structure.

**Depends on:** Phase 1 complete — latents on disk, stats computed, downsampling factor chosen.
**Unlocks:** Phase 3 (DetailDiT) which conditions on Stage 1 outputs.

---

## What Success Looks Like

- StructuralDiT forward pass runs without error and output shape matches input shape
- Loss is finite on step 1 and trending downward after 100 steps
- After 50K steps, DDIM samples from Stage 1 decode to audio with discernible rhythm and rough harmonic shape (even without timbral detail)
- Loss curve has flattened (no longer aggressively decreasing)
- At least one checkpoint saved
- All training hyperparameters logged in a YAML block in the training notebook

---

## Files to Create

| File | Purpose |
|---|---|
| `models/structural_dit.py` | StructuralDiT model definition |
| `models/components/adaln.py` | Adaptive layer norm (adaLN-Zero) block |
| `models/components/timestep_embedding.py` | Sinusoidal timestep embedding |
| `models/components/transformer_block.py` | Single DiT transformer block |
| `data/dataset.py` | PyTorch Dataset wrapping the saved `.pt` latents |
| `training/train_stage1.py` | Training loop for Stage 1 |
| `training/noise_schedule.py` | Linear DDPM noise schedule |
| `training/sampler.py` | DDIM sampler for monitoring during training |
| `notebooks/02_structural_dit.ipynb` | Training launch, monitoring, and sample evaluation |
| `checkpoints/stage1/` | Output directory for Stage 1 checkpoints |

---

## Task Breakdown

### Task 2.0 — Study the DiT Paper Before Writing Anything

**What to do:**

Read sections 2 and 3 of [Peebles & Xie (2023)](https://arxiv.org/abs/2212.09748) before writing a single line of model code. Specifically understand:

- How patchification works and why patches are used instead of raw token positions
- How adaLN-Zero conditioning works and why it initializes the final projection to zero
- Why that zero-initialization matters (it is the source of NaN losses if you get it wrong)
- What epsilon parameterization means vs. x0 parameterization

This is not optional background. Misunderstanding adaLN-Zero is the single most common cause of failure at this stage.

**Verify:**
- You can explain in one sentence why the adaLN-Zero final projection is initialized to zero
- You have decided on epsilon vs. x0 parameterization and written it down

---

### Task 2.1 — Implement the Timestep Embedding

**What to do:**

Implement a sinusoidal timestep embedding in `models/components/timestep_embedding.py`. This maps a scalar timestep `t` to a vector that gets fed into adaLN conditioning.

The implementation should follow the standard transformer sinusoidal positional encoding formula, applied to timestep values rather than sequence positions.

**Verify:**
- Input: a batch of integer timesteps, shape `[B]`
- Output: a batch of embedding vectors, shape `[B, embedding_dim]`
- Embeddings for nearby timesteps are more similar than embeddings for distant timesteps (sanity check this manually)

---

### Task 2.2 — Implement adaLN-Zero COMPLETE

**What to do:**

Implement the adaptive layer norm block in `models/components/adaln.py`. This is the conditioning mechanism that lets the timestep control the scale and shift of each transformer block's layer norm.

The structure is:
- A small MLP maps the timestep embedding to 6 vectors: `shift_msa`, `scale_msa`, `gate_msa`, `shift_mlp`, `scale_mlp`, `gate_mlp`
- These 6 vectors modulate the attention and MLP sublocks of each transformer block
- The final linear projection of the MLP that produces these 6 vectors must be initialized to zero

**Verify:**
- At initialization (before any training), the gate values are zero, which means the residual connection passes through unchanged
- After a few gradient steps, the gate values are no longer zero (the model is learning to use conditioning)

**Watch out for:** If you do not initialize the final projection to zero, your loss will be NaN from step 1. This is the most common failure at this stage. Double-check your initialization.

---

### Task 2.3 — Implement the DiT Transformer Block COMPLETED

**What to do:**

Implement a single transformer block in `models/components/transformer_block.py` that combines:
- Multi-head self-attention with adaLN-Zero modulation
- MLP (feedforward) sublayer with adaLN-Zero modulation
- Residual connections on both

This block is the fundamental building unit. StructuralDiT stacks 6–8 of these.

**Verify:**
- Forward pass: input `[B, T, D]` → output `[B, T, D]` (shape preserved)
- The timestep conditioning is actually modifying the intermediate activations (check by passing different timesteps and confirming the outputs differ)

---

### Task 2.4 — Implement StructuralDiT COMPLETE

**What to do:**

Assemble `models/structural_dit.py` using the components above. Architecture parameters:

- Layers: 6–8 transformer blocks
- Hidden dimension: 256–384
- Attention heads: 4
- Input: noised coarse latent + timestep embedding
- Output: predicted noise (epsilon parameterization) at the same shape as input

The model takes a noised coarse latent (shape `[B, C, T_coarse]`) and a timestep, and outputs a tensor of the same shape predicting the noise added at that timestep.

A patchification layer may or may not be needed depending on how you treat the time axis — document your choice. If your latent is already compact (T_coarse is small), you may be able to treat each time frame as a token directly.

**Verify:**
- Forward pass runs without error
- Output shape matches input shape exactly
- Model parameter count is logged (target: under 50M parameters)
- The model fits in under 3 GB VRAM with a batch size of 4

---

### Task 2.5 — Build the Latent Dataset and DataLoader COMPLETE

**What to do:**

Implement `data/dataset.py` as a PyTorch `Dataset` that:
1. Lists all `.pt` files in `data/latents/`
2. Loads a latent on `__getitem__`
3. Applies the normalization from `data/stats.json` (subtract mean, divide by std)
4. Downsamples to the coarse resolution (using the factor chosen in Phase 1)

The DataLoader should use multiple workers and pin memory for GPU efficiency.

**Verify:**
- A batch loads correctly: shape `[B, C, T_coarse]`
- Normalized values are centered near zero with std ~1.0
- DataLoader iterates over the full dataset without error

---

### Task 2.6 — Implement the Noise Schedule COMPLETE

**What to do:**

Implement a linear DDPM noise schedule in `training/noise_schedule.py`. The schedule defines:
- `beta_t` for each timestep `t` (linearly spaced from `beta_start` to `beta_end`)
- `alpha_t = 1 - beta_t`
- `alpha_bar_t = cumulative product of alpha values` (used to compute the noised sample in one step)

Starting values: `beta_start = 1e-4`, `beta_end = 0.02`, `T = 1000` timesteps.

**Verify:**
- `alpha_bar_t` at `t=0` is close to 1.0 (almost no noise)
- `alpha_bar_t` at `t=999` is close to 0.0 (almost pure noise)
- The `q_sample` function (forward noising) produces visually noisy outputs at high timesteps

---

### Task 2.7 — Implement the DDIM Sampler

**What to do:**

Implement a DDIM sampler in `training/sampler.py` that can generate samples from StructuralDiT in 50 steps (for monitoring). The sampler takes a trained model, a noise schedule, and a number of steps, and returns a denoised latent.

Use the DDIM deterministic update rule (not stochastic DDPM). 50 steps is enough for qualitative monitoring during training.

**Verify:**
- Sampler runs on the untrained model (output will be noise — that is expected)
- Sampler output shape matches the coarse latent shape

---

### Task 2.8 — Implement the Training Loop

**What to do:**

Implement `training/train_stage1.py` with:
- Optimizer: AdamW, `lr=1e-4`, `weight_decay=1e-2`
- Loss: MSE between predicted noise and actual noise (epsilon parameterization)
- Mixed precision: `torch.cuda.amp.autocast()` with fp16 (not bfloat16)
- Gradient checkpointing: enabled on the model
- Logging: loss every 100 steps to stdout and a log file
- Sampling: run DDIM every 5K steps, decode through DAC, save the `.wav` with the step number in the filename
- Checkpointing: save model state dict every 10K steps to `checkpoints/stage1/`

At step 0 (before any training), run `torch.cuda.memory_summary()` and log the output. This is your baseline for VRAM tracking.

**Verify:**
- Loss is finite after step 1
- Loss is visibly decreasing after 100 steps
- A sample is generated and saved at step 5K
- No CUDA OOM on your target batch size

**Watch out for:** Do not train Stage 1 and Stage 2 simultaneously. Stage 1 uses the VRAM alone during training. Stage 2 will be trained separately in Phase 3.

---

### Task 2.9 — Train to Near-Convergence

**What to do:**

Run the training loop for at least 50K steps. Monitor:
- Loss curves (looking for flattening)
- Sample quality every 5K steps (listening)
- VRAM usage (should be stable throughout)

You do not need loss to reach a specific value — you need the curve to flatten. A flat curve means the model has converged on what it can learn at this scale. Samples from a converged Stage 1 should have discernible rhythm and rough harmonic shape, even without timbral detail.

**Verify:**
- Loss curve has flattened
- At least one checkpoint exists at 50K+ steps
- Stage 1 samples have perceptible macro structure (not pure noise)
- All hyperparameters are logged in the training notebook

---

## VRAM Constraints

- Stage 1 model target: under 3 GB
- Enable gradient checkpointing before the first training step
- Keep DAC on CPU during Stage 1 training; move to GPU only for decode during sample generation
- Batch size: start at 4, increase by 2 until OOM, then drop back 2. Document the stable batch size.

---

## Decision Log Template

Minimum entries before moving to Phase 3:
- Parameterization chosen (epsilon vs. x0) and why
- Hidden dimension and layer count chosen and why
- Batch size that fit in VRAM
- Whether patchification was used and why/why not

---

## Blockers That Should Stop You from Moving Forward

Do not begin Phase 3 if:

- Loss is NaN (diagnose adaLN-Zero initialization before continuing)
- Stage 1 samples after 50K steps sound like pure noise with no structure
- Loss curve is still steeply decreasing at 50K steps (train longer)
- No checkpoint has been saved
