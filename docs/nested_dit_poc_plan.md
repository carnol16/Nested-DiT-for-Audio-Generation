# Nested Diffusion Transformer (DiT) for Audio Generation — PoC Implementation Plan

---

## What This PoC Is Trying to Prove

A two-stage hierarchical DiT can generate structurally coherent audio by separating the problem of *global structure* from *fine-grained detail* — and that this separation can run inside 10GB of VRAM on a single consumer GPU.

---

## Background: RvNN Inspiration and the Two-Stage Architecture

### The RvNN Analogy

Recursive Neural Networks (RvNNs) process data as trees: a root node represents the whole, child nodes represent parts, and leaves represent atomic units. Each node is computed from its children, forcing the network to build hierarchical representations explicitly rather than hoping they emerge from a flat sequence model.

The key insight RvNNs encoded — and that this architecture borrows — is that **meaning is compositional and hierarchical**. For language, a sentence is built from clauses, clauses from phrases, phrases from words. For audio, a musical passage is built from sections, sections from phrases, phrases from motifs, motifs from individual events.

Standard flat diffusion models generate all of this in one pass. They can learn local coherence well but tend to lose global structure, especially over longer time horizons. This is the audio equivalent of a language model that writes grammatical sentences but incoherent paragraphs.

### How the Hierarchy Maps

| RvNN Concept | Audio DiT Equivalent |
|---|---|
| Root node | Stage 1 (StructuralDiT) — coarse latent, global rhythm and form |
| Leaf nodes | Stage 2 (DetailDiT) — fine-grained latent, timbral and spectral detail |
| Parent-to-child conditioning | Cross-attention from Stage 1 output into Stage 2 |
| Tree structure | Nested generation: structure first, detail second |

Stage 1 generates a low-resolution audio latent that captures tempo, phrase boundaries, and macro dynamics. Stage 2 receives that latent via cross-attention and generates the high-resolution detail conditioned on it. The VAE bridges both stages to and from waveform space.

### Foundational RvNN Literature

- **Socher et al. (2011) — Parsing Natural Scenes and Natural Language with Recursive Neural Networks**: The paper that established RvNNs for hierarchical representation learning. The core idea of composing representations up a tree is the direct philosophical ancestor of this architecture. [https://ai.stanford.edu/~ang/papers/icml11-RecursiveNeuralNetworks.pdf](https://ai.stanford.edu/~ang/papers/icml11-RecursiveNeuralNetworks.pdf)

- **Socher et al. (2013) — Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank**: Demonstrates that hierarchical composition generalizes across tasks and domains. Relevant for thinking about how structural conditioning can generalize across audio types. [https://aclanthology.org/D13-1170.pdf](https://aclanthology.org/D13-1170.pdf)

- **Tai et al. (2015) — Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks**: Tree-LSTM is the direct bridge between RvNN philosophy and modern sequence models. Understanding this paper clarifies why cross-attention is the natural mechanism for parent-to-child conditioning in a transformer context. [https://arxiv.org/abs/1503.00075](https://arxiv.org/abs/1503.00075)

---

## Core References for the Architecture

- **DiT — Scalable Diffusion Models with Transformers** (Peebles & Xie, 2023): The backbone of both stages. Understanding the patchification strategy and adaptive layer norm (adaLN) conditioning is essential before you start. [https://arxiv.org/abs/2212.09748](https://arxiv.org/abs/2212.09748)

- **DAC — Descript Audio Codec**: Recommended VAE for this PoC. Better reconstruction quality than EnCodec at equivalent bitrates, and the latent space is more structured for diffusion. [https://github.com/descriptinc/descript-audio-codec](https://github.com/descriptinc/descript-audio-codec) | [Paper](https://arxiv.org/abs/2306.06546)

- **AudioLDM** (Liu et al., 2023): If you need a drop-in pretrained VAE with a known-good latent space, AudioLDM's VAE is a solid alternative to DAC. The codebase is also a useful reference for how a diffusion pipeline wraps a frozen audio VAE. [https://github.com/haoheliu/AudioLDM](https://github.com/haoheliu/AudioLDM) | [Paper](https://arxiv.org/abs/2301.12503)

- **AudioLDM2** (Liu et al., 2023): More relevant than the original for conditioning strategies — introduces a latent diffusion model conditioned on general audio representations. Study the conditioning pipeline. [https://arxiv.org/abs/2308.05734](https://arxiv.org/abs/2308.05734)

---

## Datasets

Use one of the following. Do not use both at the same time in the PoC — pick one and stay consistent until you validate the pipeline.

- **AudioCaps**: 46K audio clips from AudioSet with natural language captions. Best choice if you want to eventually plug into text conditioning. Clips are 10 seconds. [https://audiocaps.github.io](https://audiocaps.github.io) | [HuggingFace](https://huggingface.co/datasets/audiocaps)

- **NSynth**: 300K annotated musical note samples from Google. Simpler distribution than AudioCaps — good for initial pipeline validation where you want the audio itself to be boring so you can focus on architecture behavior. [https://magenta.tensorflow.org/datasets/nsynth](https://magenta.tensorflow.org/datasets/nsynth) | [HuggingFace](https://huggingface.co/datasets/projectlosangeles/NSynth-WTF-Dataset)

**Recommendation for this PoC:** Start with NSynth. Its regularity (fixed pitch, instrument family, duration) makes it easier to verify that your model is learning something meaningful rather than memorizing dataset artifacts.

---

## Step-by-Step Notebook Workflow

### Step 0 — Environment Setup

**What to do:**
Install and pin your dependencies: PyTorch (CUDA 12.x), DAC, `diffusers`, `einops`, `accelerate`, and `librosa`. Create a dedicated conda environment. Pin everything to specific versions in a `requirements.txt` before writing a single line of model code.

Check that your GPU is visible, that `torch.cuda.get_device_properties(0).total_memory` returns approximately 10GB, and that you can load DAC and encode a short audio clip without error.

**Verify before moving on:**
- `nvidia-smi` shows your 3080 with at least 9.5GB free
- DAC encodes and decodes a 1-second clip and the output sounds correct
- A quick `torch.zeros(1, 512, 512, device='cuda')` allocation succeeds

**What failure looks like:**
DAC decode produces silence or clipping artifacts immediately. This almost always means a sample rate mismatch (DAC expects 44.1kHz by default — resample your audio before encoding, not after).

---

### Step 1 — Audio Preprocessing and Latent Extraction

**What to do:**
Write a preprocessing pipeline that takes raw audio files, resamples to the VAE's expected sample rate, encodes them with the frozen DAC encoder, and saves the resulting latents to disk as `.npy` or `.pt` files. Do this as a one-time offline step — do not encode on-the-fly during training.

Compute statistics over your latent dataset: mean, standard deviation, min, max. You will need these to normalize the latent space before feeding it to the diffusion model.

**Verify before moving on:**
- Latent files exist on disk and load correctly
- Decoded latents sound perceptually close to the original audio (ABX test by ear)
- Latent distribution statistics are logged and saved

**What failure looks like:**
The decoded audio from stored latents sounds noticeably worse than encoding fresh each time. This usually means you are storing unnormalized or float16-truncated latents. Store in float32.

---

### Step 2 — Define the Coarse Latent Resolution for Stage 1

**What to do:**
Decide on the spatial downsampling factor between Stage 1 and Stage 2. A factor of 4–8x along the time axis is a reasonable starting point. If your full DAC latent is shape `[C, T]`, your Stage 1 target is `[C, T//8]`.

Do not train anything yet. Manually downsample some latents (e.g., with average pooling), upsample them back, and run them through the DAC decoder. Listen to the result. This tells you what information survives at the resolution Stage 1 will be working with. If you can hear rhythm and rough harmonic content but not timbre, you have the right resolution. If it sounds like noise, you have gone too coarse.

**Verify before moving on:**
- You have a downsampling factor chosen and documented
- You have listened to at least 10 reconstructions from the coarse resolution and confirmed they preserve perceptible macro structure

**What failure looks like:**
At your chosen resolution, the coarse latent sounds like broadband noise with no discernible structure. Go coarser more slowly — try 2x, then 4x, then 8x, listening at each step.

---

### Step 3 — Implement StructuralDiT (Stage 1)

**What to do:**
Implement a standard DiT operating on the coarse latent. Use adaLN-Zero conditioning with a sinusoidal timestep embedding. Keep the model small: 6–8 transformer layers, hidden dimension 256–384, 4 attention heads. Your goal is a model that fits in under 3GB VRAM during training.

The input to StructuralDiT is the noised coarse latent. The output is the predicted noise (or predicted x0, your choice — epsilon parameterization is more stable for an initial PoC). No text conditioning at this stage.

**Verify before moving on:**
- Forward pass runs without error
- Output shape matches input shape
- Loss is finite on the first step and decreasing after 100 steps on a small batch
- Model parameter count is logged

**What failure looks like:**
Loss is NaN from step 1. Almost always caused by the adaLN scale initialization — adaLN-Zero initializes the final projection to zero, which is intentional. If you initialize it randomly, you get NaN losses immediately. Check your adaLN implementation against the original DiT paper.

---

### Step 4 — Train StructuralDiT to Convergence (or Near-Convergence)

**What to do:**
Train on your coarse latent dataset with a standard DDPM or DDIM noise schedule. Use a linear schedule for the PoC — cosine is better in the long run but adds a debugging variable. Batch size should be as large as VRAM allows (see VRAM section below). Train for at least 50K steps before evaluating quality.

Log loss curves. Log sample quality every 5K steps by running DDIM sampling (50 steps is enough for monitoring) and decoding the output through DAC. Listen to the samples.

**Verify before moving on:**
- Samples from Stage 1 sound like plausible coarse-grain audio — you can hear rhythm and rough structure even though detail is missing
- Loss curve has flattened (not still aggressively decreasing)
- You have saved at least one checkpoint

**What failure looks like:**
Stage 1 samples sound like static with no discernible structure after 50K steps. Check that your latent normalization is correct — unnormalized latents will cause the model to spend all its capacity learning the marginal distribution of the data rather than the noise schedule.

---

### Step 5 — Implement DetailDiT (Stage 2)

**What to do:**
Implement a second DiT that operates on the full-resolution latent. The critical addition is a cross-attention layer in each transformer block that attends to the Stage 1 output (the coarse latent, upsampled to the full resolution via a learned linear projection — not bilinear interpolation).

The cross-attention key and value projections take the Stage 1 latent as input. The query projections take the current Stage 2 hidden state. Use the same adaLN-Zero conditioning on timestep as in Stage 1.

DetailDiT should be somewhat larger than StructuralDiT but not dramatically so: 8–12 layers, hidden dimension 384–512.

**Verify before moving on:**
- Forward pass runs without error, passing a ground-truth coarse latent as the conditioning signal
- Cross-attention is actually attending to the conditioning signal (check attention weights — they should not be uniform across the sequence)
- Output shape matches the full-resolution latent shape

**What failure looks like:**
Cross-attention weights are uniform across all positions throughout training. This means DetailDiT is ignoring the Stage 1 conditioning — either the cross-attention projection is not initialized correctly, or the conditioning signal is not being passed through the forward call correctly.

---

### Step 6 — Train DetailDiT Conditioned on Ground-Truth Stage 1 Latents

**What to do:**
Train DetailDiT using the ground-truth coarse latent (the downsampled version of the target, not Stage 1 model outputs) as conditioning. This is called **teacher forcing** and it allows Stage 2 to learn the conditioning relationship without the confounding noise of Stage 1 being poorly trained.

Do not use Stage 1 model outputs for conditioning during this training phase. Only switch to model-generated coarse latents after Stage 2 has learned to use the conditioning signal effectively.

**Verify before moving on:**
- DetailDiT samples conditioned on ground-truth coarse latents sound significantly better than unconditioned noise
- The conditioning is doing visible work: samples conditioned on coarse latents from the same clip should be more similar to each other than to samples conditioned on coarse latents from different clips

**What failure looks like:**
DetailDiT samples conditioned on different ground-truth coarse latents are indistinguishable from each other. The model is not using the conditioning. Increase the cross-attention block's learning rate or add a conditioning dropout schedule to force the model to pay attention.

---

### Step 7 — Connect Stage 1 and Stage 2 for End-to-End Sampling

**What to do:**
Implement the inference pipeline:
1. Sample a coarse latent from StructuralDiT using DDIM
2. Pass the coarse latent as the cross-attention conditioning to DetailDiT
3. Sample a full-resolution latent from DetailDiT using DDIM
4. Decode the full-resolution latent through the frozen DAC decoder
5. Listen to the result

Run at least 20 samples this way before evaluating quality. Variance across samples is as informative as mean quality at this stage.

**Verify before moving on:**
- The pipeline runs end-to-end without error
- Some samples sound like plausible audio (they don't all have to be good)
- The Stage 1 and Stage 2 outputs are visually inspectable as spectrograms

**What failure looks like:**
All end-to-end samples sound like noise even though Stage 1 and Stage 2 samples individually sounded reasonable. This is a distribution mismatch: DetailDiT was trained on ground-truth coarse latents but is now receiving model-generated ones. See the next step.

---

### Step 8 — Fine-Tune for Distribution Alignment

**What to do:**
If Step 7 reveals a distribution mismatch (common), fine-tune DetailDiT using Stage 1 model outputs as conditioning rather than ground-truth coarse latents. Use a low learning rate (10% of the original training LR) and train for 10–20% of the original training steps.

Alternatively, use **conditioning dropout** during Stage 2 training from the start (randomly replace the conditioning with noise 10–20% of the time). This makes Stage 2 more robust to imperfect conditioning signals and reduces the fine-tuning burden.

**Verify before moving on:**
- End-to-end sample quality is comparable to Stage 2 conditioned on ground-truth latents
- The mismatch between Stage 1 outputs and ground-truth coarse latents has narrowed (measure via simple L2 distance on a validation set)

**What failure looks like:**
Fine-tuning causes DetailDiT to catastrophically forget how to use the conditioning signal. This means your fine-tuning learning rate is too high. Drop it by another 5x.

---

### Step 9 — Listen, Evaluate, and Document First Results

**What to do:**
Generate 50 samples. Listen to all of them. Score them on three dimensions (informally, no formal metric required for the PoC): structural coherence, timbral quality, and diversity across samples. Write down your observations in the notebook in prose — not just numbers.

Generate a spectrogram plot for at least 10 samples using `librosa.display.specshow`. The spectrogram is more informative than a waveform for diagnosing what is and is not working.

**Verify before moving on:**
- At least some samples are recognizably audio with discernible structure
- You have written observations, not just run metrics

**What failure looks like:**
You generate samples and immediately start changing hyperparameters before writing anything down. Do not do this. Write first.

---

## VRAM Management for the 3080 (10GB)

The two-stage architecture is VRAM-intensive if you are not careful. These constraints are specific to the 10GB 3080.

**Frozen VAE:** Load DAC in eval mode with `torch.no_grad()` and keep it on CPU during training. Only move it to GPU for encoding/decoding batches, then move it back. This saves approximately 500MB–1GB depending on the VAE variant.

**Gradient checkpointing:** Enable gradient checkpointing on both DiT models. This trades compute for memory — roughly 30–40% more compute per step but reduces activation memory by 50–70%. For a model this size, the tradeoff is strongly in favor of checkpointing.

**Training one stage at a time:** Do not attempt to train both stages simultaneously. Train Stage 1 to convergence, save the checkpoint, then train Stage 2. If you try to hold both models in VRAM simultaneously during training you will run out of memory.

**Batch size:** Start with a batch size of 4 and increase by 2 until you hit an OOM error, then drop back by 2. Log the batch size that works and document it.

**Mixed precision:** Use `torch.cuda.amp.autocast()` with fp16. Do not use bfloat16 — the 3080's bfloat16 support is soft (it falls back to fp32 ops in some places and the VRAM savings are inconsistent).

**Inference VRAM:** During end-to-end sampling, you need both models in memory simultaneously. Use `torch.inference_mode()` (not just `no_grad`) and explicitly delete intermediate tensors after each stage. Load Stage 1, generate the coarse latent, move Stage 1 to CPU, then load Stage 2.

**Monitoring:** Use `torch.cuda.memory_summary()` at the end of your first training step to get a breakdown of where your VRAM is going. Do this before you optimize anything.

---

## What to Capture and Document as You Go

This PoC feeds into a larger text-to-audio engine, and the documentation produced here is a deliverable in its own right — not an afterthought. Future-you (and any collaborator) needs to be able to reconstruct why decisions were made, not just what they were.

**For each major architectural decision, record:**
- What you chose and the specific alternatives you considered
- Why you chose what you chose (even if "it fit in VRAM" is the whole reason)
- What you expected to happen vs. what actually happened

**For each training run, record:**
- Hyperparameters (a YAML block in the notebook is fine)
- Loss curves (screenshot or inline plot)
- Sample audio files with the run ID in the filename
- At least 2–3 sentences of qualitative observations per checkpoint

**For the spectrogram analysis:**
- Side-by-side: original audio spectrogram vs. reconstructed spectrogram
- Annotate what structural features survived Stage 1 (rhythm, phrase boundaries) vs. what was added by Stage 2 (timbre, harmonics)

**Portfolio-specific notes:**
Since this PoC is building toward a text-to-audio portfolio piece, include a brief section at the end of the notebook titled "What I Would Do Differently." This is for future-you and for any write-up you publish — honest retrospectives are more compelling than polished post-hoc narratives.

Save every notebook iteration with a date-stamped filename. Do not overwrite notebooks.

---

## What Success Looks Like

The PoC is done when all of the following are true:

1. **End-to-end pipeline runs**: Stage 1 → Stage 2 → DAC decode produces listenable audio without crashing or producing silence.
2. **Stage 1 conditioning is doing visible work**: Samples conditioned on the same Stage 1 output are more similar to each other than to samples conditioned on different Stage 1 outputs. You can verify this by ear or by comparing spectrograms.
3. **VRAM budget is respected**: The full inference pipeline (both stages + VAE decode) runs within 10GB without manual intervention.
4. **At least 10 samples are perceptually coherent**: "Perceptually coherent" means a human listener (you) can identify that it is audio with discernible structure, not noise — even if quality is low.
5. **Observations are written down**: At least 500 words of qualitative notes exist in the notebook covering what worked, what did not, and why.

The bar is deliberately low. This is a proof-of-concept. You are not trying to match AudioLDM2 quality; you are trying to prove that the two-stage hierarchy is learnable and that the conditioning relationship between Stage 1 and Stage 2 is real.

---

## What Comes Next

This PoC validates the core architectural claim in isolation — no text conditioning, no complex data, no production concerns. The path from here to a full text-to-audio engine has three main steps:

**Text conditioning**: Add CLAP or a FLAN-T5 encoder as a conditioning signal to Stage 1. Stage 1 conditions on text to generate a structurally appropriate coarse latent; Stage 2 conditions on that latent as before. AudioCaps becomes the primary dataset at this point.

**Scale and quality**: Both stages are deliberately small for the PoC. Once the architecture is validated, the natural next move is increasing model size and training duration. This is where the VRAM ceiling starts to bind — moving to a distributed setup or gradient offloading becomes necessary.

**Evaluation framework**: Replace informal listening with FAD (Fréchet Audio Distance) and CLAP similarity scores so that architecture changes can be compared objectively. This is not necessary for the PoC but is required before any public claims about quality.

The two-stage hierarchy established in this PoC is the structural core of the larger engine. Everything built on top of it — text conditioning, speaker conditioning, style transfer — plugs into the cross-attention mechanism in Stage 2 and the conditioning interface in Stage 1. Keep those interfaces clean and documented, because you will be extending them.
