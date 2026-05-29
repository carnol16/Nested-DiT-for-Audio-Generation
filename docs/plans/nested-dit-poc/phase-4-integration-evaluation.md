# Phase 4: Integration & Evaluation

> **Goal:** Connect Stage 1 and Stage 2 into an end-to-end sampling pipeline, diagnose and fix any distribution mismatch between training and inference, and produce the documentation deliverable that constitutes the PoC's conclusion.

**Depends on:** Phase 3 complete — a trained Stage 2 checkpoint exists and conditioning is demonstrably working.
**Unlocks:** The "What Comes Next" work described in the original spec (text conditioning, scaling, formal evaluation).

---

## What Success Looks Like

All five PoC success criteria from the original spec are met:

1. End-to-end pipeline (Stage 1 → Stage 2 → DAC decode) runs without crashing or producing silence
2. Stage 1 conditioning is doing visible work: samples from the same Stage 1 output cluster perceptually
3. Full inference runs within 10 GB VRAM without manual intervention
4. At least 10 samples are perceptually coherent (discernible audio structure, not noise)
5. At least 500 words of written qualitative notes exist in the evaluation notebook

---

## Files to Create

| File | Purpose |
|---|---|
| `inference/pipeline.py` | End-to-end sampling: Stage 1 → Stage 2 → DAC decode |
| `inference/vram_manager.py` | Stage loading/offloading to stay within 10 GB budget |
| `training/finetune_stage2.py` | Optional fine-tuning loop for distribution alignment |
| `evaluation/spectrogram_grid.py` | Generate spectrogram grids for visual comparison |
| `notebooks/04_integration.ipynb` | End-to-end pipeline validation and sample generation |
| `notebooks/05_evaluation.ipynb` | Formal evaluation: 50 samples, spectrograms, written observations |
| `results/samples/` | Output directory for evaluation samples |
| `results/spectrograms/` | Output directory for spectrogram plots |

---

## Task Breakdown

### Task 4.0 — Re-Read the Distribution Mismatch Warning Before Starting

**What to do:**

Before writing any code, re-read Step 7 and Step 8 in the original spec (`nested_dit_poc_plan.md`). Specifically understand what distribution mismatch is and why it is the most likely failure mode at this stage.

In short: Stage 2 was trained on ground-truth coarse latents (teacher forcing). Stage 1 model outputs are similar but not identical to ground-truth coarse latents. Stage 2 may produce noise when it receives an imperfect conditioning signal it was not trained on.

**Verify:**
- You can describe distribution mismatch in one sentence to a colleague
- You have a plan for detecting it (Step 4.4) and fixing it (Step 4.5) before you need them

---

### Task 4.1 — Implement the VRAM Manager

**What to do:**

Implement `inference/vram_manager.py` to handle stage loading and offloading. The constraint: during inference, both models must fit within 10 GB VRAM, but they do not need to be in GPU memory simultaneously.

The loading strategy is:
1. Load StructuralDiT to GPU
2. Run Stage 1 DDIM sampling → save coarse latent to CPU tensor
3. Move StructuralDiT to CPU (explicit `.cpu()` and `torch.cuda.empty_cache()`)
4. Load DetailDiT to GPU
5. Run Stage 2 DDIM sampling conditioned on the saved coarse latent → save full latent to CPU
6. Move DetailDiT to CPU
7. Load DAC to GPU
8. Decode the full latent → waveform
9. Move DAC to CPU

Use `torch.inference_mode()` (not just `torch.no_grad()`) for all inference steps. Delete intermediate tensors explicitly with `del` after each stage to free memory promptly.

**Verify:**
- After Stage 1 completes, VRAM usage drops back toward zero before Stage 2 loads
- No CUDA OOM during a full pipeline run on a single sample
- `torch.cuda.max_memory_allocated()` stays under 10 GB throughout

---

### Task 4.2 — Implement the End-to-End Inference Pipeline

**What to do:**

Implement `inference/pipeline.py` as a clean interface that takes no required arguments at call time (uses loaded model checkpoints) and returns a waveform tensor.

The pipeline orchestrates the VRAM manager, the DDIM samplers for both stages, and the DAC decoder. It should be callable from both the notebook and the command line.

Use 50 DDIM steps for Stage 1 and 50 DDIM steps for Stage 2 for the PoC. More steps improve quality but increase runtime.

**Verify:**
- Running the pipeline on a single sample completes without error
- Output is a waveform tensor (not a latent — DAC decoding must happen inside the pipeline)
- The output waveform plays back as audio (even if quality is poor)

---

### Task 4.3 — Generate 20 End-to-End Samples

**What to do:**

Run the pipeline 20 times. Save each output as a `.wav` file in `results/samples/` with a timestamp in the filename.

Listen to all 20 samples before evaluating anything. This is a listening step, not a metrics step. What you are looking for:
- Are any samples perceptually coherent (discernible audio structure)?
- Is there variety across samples, or do they all sound the same?
- Do the samples sound qualitatively similar to Stage 1's coarse samples or worse?

Write down observations after listening. Do not change any hyperparameters yet.

**Verify:**
- 20 `.wav` files exist in `results/samples/`
- You have listened to all 20 and written at least 3 sentences of observations

---

### Task 4.4 — Diagnose Distribution Mismatch

**What to do:**

If the 20 samples from Task 4.3 sound like noise even though Stage 1 and Stage 2 individually produced reasonable samples in their respective phases, you have a distribution mismatch.

To measure it quantitatively: take 20 clips from your validation set. Compute ground-truth coarse latents (downsample). Run Stage 1 on the same clips' full latents and get model-generated coarse latents. Compute the L2 distance between ground-truth and model-generated coarse latents.

If the L2 distance is large relative to the variance of the coarse latents (e.g., the distance is comparable to the std of the latent distribution), the mismatch is significant and you should proceed with fine-tuning (Task 4.5).

If the mismatch is small, the issue is likely elsewhere (check DAC decode, check normalization, check DDIM sampler).

**Verify:**
- L2 distance between ground-truth and Stage 1 outputs is measured and logged
- You have a written diagnosis: is this distribution mismatch or something else?

---

### Task 4.5 — Fine-Tune Stage 2 for Distribution Alignment (If Needed)

**What to do:**

If Task 4.4 confirmed distribution mismatch, fine-tune DetailDiT using Stage 1 model outputs as conditioning rather than ground-truth coarse latents.

Fine-tuning parameters:
- Learning rate: 10% of the original Stage 2 training LR (e.g., `1e-5` if Stage 2 trained at `1e-4`)
- Steps: 10–20% of the original Stage 2 training step count
- Otherwise, identical setup to the Stage 2 training loop

Save the fine-tuned checkpoint as a separate file (do not overwrite the Phase 3 checkpoint).

If fine-tuning causes catastrophic forgetting (Stage 2 loses its conditioning ability), the learning rate is too high. Reduce by another 5x.

**Verify:**
- End-to-end sample quality after fine-tuning is comparable to Stage 2 conditioned on ground-truth latents
- L2 distance between Stage 1 outputs and ground-truth coarse latents is measured again and has not changed (fine-tuning Stage 2 does not fix Stage 1 — it fixes Stage 2's tolerance for imperfect inputs)
- The fine-tuned Stage 2 still shows conditioning effectiveness (samples from same Stage 1 output cluster together)

---

### Task 4.6 — Validate Conditioning is Doing Work End-to-End

**What to do:**

Run Stage 1 once to generate a single coarse latent. Use that same coarse latent to condition Stage 2 five times. Save the 5 outputs.

Run Stage 1 again to generate a different coarse latent. Use that to condition Stage 2 five times. Save those 5 outputs.

Listen to both groups. The within-group samples should be more similar to each other than to the across-group samples. Generate spectrograms in `evaluation/spectrogram_grid.py` for visual confirmation.

**Verify:**
- 10 samples exist: 5 from coarse latent A, 5 from coarse latent B
- You can describe in writing what structural features differ between the two groups
- Spectrograms are saved to `results/spectrograms/`

---

### Task 4.7 — Generate 50 Evaluation Samples

**What to do:**

Generate 50 end-to-end samples. Save all of them. Listen to all of them.

Score each sample on three informal dimensions (no formal metric required):
- **Structural coherence**: Does it have discernible temporal structure? (Not just noise)
- **Timbral quality**: Does it sound like audio, not artifacts?
- **Diversity**: Is it different from the other samples?

A simple 1–3 scale per dimension is enough. Record the scores in the notebook.

**Verify:**
- 50 `.wav` files in `results/samples/`
- Scores recorded for each sample
- At least 10 samples score 2 or higher on structural coherence ("perceptually coherent" threshold)

---

### Task 4.8 — Write the Evaluation Notebook

**What to do:**

In `notebooks/05_evaluation.ipynb`, write the following sections in prose (not just code cells):

**Section 1: What Was Built**
One paragraph summarizing the architecture. Assume the reader has not read any other notebook.

**Section 2: What Worked**
Honest description of what the two-stage hierarchy achieved. Be specific. "Stage 1 conditioning is doing visible work" needs to be supported by what you observed in Task 4.6.

**Section 3: What Didn't Work**
Honest description of failure modes. If samples are noisy, say so. If conditioning dropout helped, say by how much. If distribution mismatch was severe, describe it.

**Section 4: Spectrogram Analysis**
Embed 10 spectrogram pairs: original audio vs. Stage 2 reconstruction. Annotate each pair with what structural features survived Stage 1 (rhythm, phrase boundaries) vs. what was added by Stage 2 (timbre, harmonics detail).

**Section 5: What I Would Do Differently**
Minimum 3 concrete items. Not wishes ("I'd use a bigger model") — specific architectural or training decisions you would change and why.

**Verify:**
- The notebook is at least 500 words of prose (not counting code cells)
- All 5 sections are present
- Spectrogram pairs are embedded and annotated

---

### Task 4.9 — VRAM Audit and Final Documentation

**What to do:**

Run the full inference pipeline one more time while logging `torch.cuda.max_memory_allocated()` at the end of each stage. Confirm the 10 GB budget is never exceeded.

Write a one-paragraph VRAM summary in the evaluation notebook: what each stage costs in peak VRAM, what the total sequential peak is, and whether the budget constraint was met.

Save all notebooks with date-stamped filenames. Do not overwrite. Archive the final run's samples, spectrograms, and checkpoints in a clearly named directory.

**Verify:**
- VRAM peak never exceeds 10 GB (9.5 GB or less is the practical target to leave headroom)
- VRAM breakdown is documented
- Notebooks are saved with date stamps
- Checkpoints, samples, and spectrograms are archived

---

## PoC Completion Checklist

The PoC is complete when all five original success criteria are confirmed:

- [ ] End-to-end pipeline runs without error and produces listenable audio
- [ ] Stage 1 conditioning is doing visible work (documented with spectrogram evidence)
- [ ] Full inference runs within 10 GB VRAM
- [ ] At least 10 of 50 samples are perceptually coherent
- [ ] At least 500 words of qualitative written observations exist

---

## What Comes After This PoC

The architecture is now validated. The next phase (not part of this PoC) involves:

1. **Text conditioning**: Add CLAP or FLAN-T5 as a conditioning encoder for Stage 1. Switch dataset to AudioCaps.
2. **Scale**: Increase model size and training duration. This is where the VRAM ceiling starts to bind — distributed training or gradient offloading becomes necessary.
3. **Formal evaluation**: Implement FAD and CLAP similarity scoring to replace informal listening.

Keep the Stage 1 and Stage 2 interfaces (conditioning inputs, output shapes) clean and documented. Everything built in the next phase plugs into the cross-attention mechanism and conditioning interface established here.
