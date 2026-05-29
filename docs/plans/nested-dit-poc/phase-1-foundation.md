# Phase 1: Foundation & Data Pipeline

> **Goal:** Get the environment working, build the offline data pipeline, and validate the VAE latent space before writing a single line of model code.

**Depends on:** Nothing — this is the starting point.
**Unlocks:** Phase 2 (StructuralDiT) and Phase 3 (DetailDiT) both consume the outputs produced here.

---

## What Success Looks Like

- Conda environment is pinned and reproducible from `requirements.txt`
- Raw audio → DAC latent → DAC decode round-trip sounds perceptually clean
- All training latents are saved to disk as float32 `.pt` files
- Latent distribution statistics (mean, std, min, max) are saved
- A downsampling factor is chosen, documented, and validated by ear
- At least 10 coarse-then-reconstructed clips confirm macro structure survives at the chosen resolution

---

## Files to Create

| File | Purpose |
|---|---|
| `requirements.txt` | Pinned dependencies for full environment |
| `scripts/preprocess.py` | Raw audio → resample → DAC encode → save `.pt` |
| `scripts/compute_stats.py` | Compute and save latent distribution statistics |
| `scripts/validate_coarse_resolution.py` | Downsample → upsample → DAC decode → save for listening |
| `data/latents/` | Output directory for encoded latents |
| `data/stats.json` | Latent mean, std, min, max |
| `notebooks/01_environment_check.ipynb` | Interactive verification of GPU, DAC, and first encode/decode |

---

## Task Breakdown

### Task 1.0 — Environment Setup :COMPLETED

**What to do:**

Install and pin all dependencies. The key packages are PyTorch (CUDA 12.x), DAC, `diffusers`, `einops`, `accelerate`, and `librosa`. Create a dedicated conda environment. Every package must have an exact version pinned in `requirements.txt` — no `>=` ranges.

**Verify:**
- `nvidia-smi` shows the 3080 with at least 9.5 GB free
- `python -c "import torch; print(torch.cuda.get_device_properties(0).total_memory)"` returns ~10 GB
- `import dac` in a Python REPL runs without error

**Watch out for:** DAC's pip installation sometimes pulls a mismatched PyTorch version. Install PyTorch first with the correct CUDA wheel, then install DAC.

---

### Task 1.1 — DAC Smoke Test : COMPLETED

**What to do:**

In `notebooks/01_environment_check.ipynb`, manually encode a 1-second audio clip and decode it. Listen to the output. This is a sanity check only — not part of the training pipeline.

**Verify:**
- Encoded latent has a shape you can reason about (write it down — you will reference it in every later phase)
- Decoded audio sounds like the original clip
- No CUDA OOM errors on a single clip

**Watch out for:** DAC expects 44.1 kHz by default. If your test clip is at 16 kHz or 22 kHz, resample it before encoding — not after decoding. Getting this wrong produces silence or clipping that is easy to misdiagnose.

---

### Task 1.2 — Dataset Download and Audit : COMPLETED - look at dataset_audit.md

**What to do:**

Download NSynth (recommended) or AudioCaps. Do not use both. Audit the dataset before preprocessing:

- How many files?
- What is the sample rate distribution?
- What is the duration distribution?
- Are there any corrupted or zero-length files?

Log these statistics. Write them down. Knowing your data before training begins will save you hours of confusion later.

**Verify:**
- You have a concrete file count
- You know the sample rate of the files (they should all be the same — if not, flag this now)
- No zero-length files exist in the dataset

---

### Task 1.3 — Write the Preprocessing Script: COMPLETED

**What to do:**

Write `scripts/preprocess.py` to do the following for each audio file:
1. Load the file
2. Resample to DAC's expected sample rate (44.1 kHz)
3. Encode with the frozen DAC encoder (eval mode, no gradients)
4. Save the resulting latent tensor as a `.pt` file in `data/latents/`

This is a one-time offline step. Once latents are on disk, training never touches the raw audio files again.

**Verify:**
- `data/latents/` contains one `.pt` file per audio file
- Each file loads correctly and has the expected shape
- Decoding a random sample of 5 saved latents sounds perceptually close to the source audio

**Watch out for:** Store latents in float32, not float16. Float16 truncation can silently degrade reconstruction quality in ways that are hard to debug during training.

---

### Task 1.4 — Compute and Save Latent Statistics

**What to do:**

Write `scripts/compute_stats.py` to iterate over all saved `.pt` latents, compute the global mean, standard deviation, minimum, and maximum, and save them to `data/stats.json`.

These statistics will be used to normalize the latent space before feeding it to the diffusion models. Skipping this step causes the model to spend its capacity learning the marginal distribution of the data rather than the noise schedule.

**Verify:**
- `data/stats.json` exists and contains `mean`, `std`, `min`, `max`
- The std is reasonable (should be in the range 0.5–5.0 for a well-conditioned latent space; if it's 50+, something went wrong in preprocessing)

---

### Task 1.5 — Choose and Validate the Coarse Resolution

**What to do:**

Decide on the temporal downsampling factor between Stage 1 and Stage 2. The reasonable range is 4x–8x along the time axis. This is a design decision — document it explicitly.

Write `scripts/validate_coarse_resolution.py` to:
1. Load a sample latent
2. Average-pool it down by your chosen factor
3. Upsample it back to original resolution (bilinear or repeat — method does not matter here, this is a listening test only)
4. Decode through DAC
5. Save the output as a `.wav` file

Listen to at least 10 of these. The coarse latent should preserve rhythm and rough harmonic shape while losing timbral detail. If it sounds like broadband noise, go with a smaller downsampling factor. If it sounds almost identical to the original, go larger.

**Verify:**
- A downsampling factor is chosen and written into a `config.yaml` or at the top of a shared constants file
- At least 10 coarse reconstructions have been listened to
- The choice is documented with a one-paragraph rationale in the notebook

**Watch out for:** Do not skip the listening step and just pick 8x because the plan says so. The right factor depends on your specific dataset's latent geometry.

---

## Decision Log Template

For each major choice made in this phase, fill this in:

```
Decision: [what you chose]
Alternatives considered: [what else you looked at]
Reason: [why you chose this]
Expected outcome: [what you expected to happen]
Actual outcome: [what actually happened — fill in after]
```

Minimum entries required before moving to Phase 2:
- DAC variant chosen (standard vs. 44khz vs. other)
- Dataset chosen (NSynth vs. AudioCaps)
- Downsampling factor chosen

---

## Blockers That Should Stop You from Moving Forward

Do not begin Phase 2 if any of these are true:

- Decoded latents sound noticeably different from source audio
- Latent std is outside the range 0.5–10.0 (investigate before normalizing blindly)
- Coarse reconstructions at your chosen factor sound like noise with no structure
- You have not listened to at least 10 coarse reconstructions
