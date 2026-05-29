# Nested DiT PoC — Learning Guide

This document grows as you do. Fill in each section at the moment it calls for — comprehension questions before you write code, decision entries when you make a choice, observations after each milestone. The italicized line under each question is there to remind you why the question matters, so the answers stay useful months from now.

Use the `>` blockquotes as your answer space.

---

## How to Use This

- **Before You Start** — answer these before writing any code for the phase. If you can't answer them in your own words, go back to the reference material.
- **Decision Log** — fill these in at the moment you make the choice, with your actual numbers and your actual reasoning. "It fit in VRAM" is a valid reason.
- **After the Phase** — answer these once the phase's success criteria are met. Honest answers only. Bad results are more useful than polished ones.

---

---

## Phase 1 — Foundation & Data Pipeline

### Before You Start

**In your own words, what is a VAE latent space, and why does the diffusion model operate there instead of directly on the waveform?**
*Why this matters: if you can't answer this, you won't understand why latent normalization is critical or what it means when the decoder sounds wrong.*

> Your answer: A variational autoencoder(VAE) latent space is a compressed, continous mathmatical represtation of our data. 
Broken down:
Encoder taks high-dimensional input and reduces the nuimber of dimensions
Latent space is the bottleneck  where the compressed data lives
Decoder takes the compressed data from the latent space and reconstructs the original high dimensional waveform

The diffusion model uses this compressed variation due to computational efficently since the lower dimension leads to less datapoints.

---

**What does DAC do differently from a standard audio codec like MP3, and why does that matter for a generative model?**
*Why this matters: DAC's structured latent space is a load-bearing assumption of this entire architecture. Knowing why it was chosen informs every downstream decision.*

> Your answer:DAC used neural networks to compress audio into sound tokens, which is more efficent for training. This allows generative models to treat complex audio exactly like text, leveraging powerful text-porediction architectures, such as transformers, to easilt generate high fidelity music and speech. This differs from MP3 which creates a compressed bitstream by deleting impercepinble sounds, lowering its high fildelity.

---

**What would "too coarse" look like for Stage 1's resolution? What would "too fine" look like? What information lives at each extreme?**
*Why this matters: you are about to listen to coarse reconstructions and make a judgment call. You need to know what you are listening for before you listen.*

> Your answer:Too coarse would lead to the audio sounding muffled and slurred, leading to the audio being stripped away crisp transients, such as drum hits, until only broad volume and pitch contours remain. If it is too fine, the audio will subjectively sound pleasent but will be bloated with high frequency details and noisem, which leads to the training for the model to learn structral pattens, giving unneeded information. The ideal balance should sound like and accurate low quality sketch that still has the rhythm, melody, and sematic meaning intact.

---

### Decision Log

**Dataset chosen: NSynth or AudioCaps? Why?**
*Why this matters: the dataset's structure shapes what "learning something meaningful" looks like. Your answer here determines what success sounds like in Phase 2.*

> Dataset: NSynth
> Reason: The goal of the PoC is to verify that the architecture is viable. The job of stage one should be to learn pitch and envelope (ADSR). Stage 2 adds timbre. NSynth gives you a controlled, auditable signal
> Limitation: NSynth files are one-shots — they have no rhythmic or phrase-level structure. "Structure" in this context means ADSR envelope only. The PoC validates that hierarchical conditioning works (Stage 2 output changes when Stage 1 output changes), not that the model learns long-range musical structure. That claim requires a dataset with actual temporal structure (AudioCaps, music clips) and is deferred to the text-conditioning phase.


---

**Downsampling factor chosen: what is it, and what did the coarse reconstructions sound like at that factor?**
*Why this matters: this is the central architectural parameter of the whole PoC. Everything downstream — Stage 1's input size, Stage 2's cross-attention sequence length, VRAM budget — follows from this number.*

> Factor: 6x
> What 10+ coarse reconstructions sounded like: 4x: nearly identical to original with slight smearing — too conservative, Stage 1 would have little to learn. 6x: transient and pitch contour both preserved, tail audible — structure survives, detail is lost. 8x: transient often present but tail frequently gone — envelope information lost, too aggressive.
> Why you stopped here rather than going coarser or finer: 4x is not coarse enough to force a meaningful separation between stages. 8x drops the sustain and release of notes, which are part of the macro structure Stage 1 needs to model. 6x is the highest factor at which both attack and envelope consistently survive.

---

**DAC variant chosen and any surprises during the smoke test:**
*Why this matters: DAC version affects expected sample rate and latent shape. If something was surprising here, it will surface again during preprocessing.*

> DAC variant: 44100 Hz
> Anything unexpected during encode/decode: No

---

**Any corrupted, zero-length, or unexpected files found during dataset audit:**
*Why this matters: silent bugs from bad data will show up as mysteriously poor training runs with no obvious cause.*

> File count: 305,979 files
> Sample rate of files: 16 Hhz
> Any anomalies found:

---

### After the Phase

**What did the latent distribution statistics actually look like? (mean, std, min, max — and whether they surprised you)**
*Why this matters: a latent std outside the range 0.5–10 is a red flag. Knowing what "normal" looks like here calibrates your intuition for later.*

  "mean": -0.015405998565256596,
  "std": 3.7252843812262575,
  "min": -21.267919540405273,
  "max": 24.141265869140625
> Did anything surprise you?

---

**After listening to 10+ coarse reconstructions at your chosen factor, describe in plain language what survived and what was lost:**
*Why this matters: this is the clearest early evidence of what Stage 1 will and will not be able to learn. Writing it down now gives you a baseline to compare against Stage 1's actual samples in Phase 2.*

> What survived (rhythm, rough pitch, phrase shape, etc.): Transient and rough pitch
> What was lost (timbre, fine harmonics, etc.): Higher harmonics
> Anything unexpected?

---

**What would you do differently in the preprocessing pipeline if you started over?**
*Why this matters: honest retrospectives compound. A sentence here becomes a useful note before you build the next version.*

> Your answer: Nope

---

---

## Phase 2 — StructuralDiT (Stage 1)

### Before You Start

**Explain adaLN-Zero in your own words. Specifically: why is the final projection initialized to zero, and what goes wrong if you don't?**
*Why this matters: this is the single most common source of NaN losses in this project. You need to be able to diagnose it before you encounter it.*

> Your answer: adaLN-Zero makes the transformer's normalization respond to the noise level, and starts all the gating signals at zero so training doesn't explode

---

**What is the difference between epsilon (ε) parameterization and x0 parameterization in a diffusion model? Which did you choose and why?**
*Why this matters: the parameterization affects loss behavior, especially early in training. Understanding the tradeoff helps you interpret your loss curve.*

> Epsilon vs. x0 — the difference:
> Noise magnitude is consistent across timestamps while X0 parameterization can produce bllurry predictions at high noise levels due to having to guess the clean signal through heavy nopise. This is more diffcult and creates less stable gradients.
What was chosen: Epsilon
> Why:

---

**What does "convergence" mean for a diffusion model training run, and how will you recognize it from the loss curve?**
*Why this matters: "train until convergence" is meaningless if you don't know what a converged curve looks like. This is your definition before you see the data.*

> Your answer: COnvergance is when the MSE is long longer dropping, just flucating around the floor. Youll notice this if the loss has not improved after approx 5000 steps

---

**What is DDIM sampling, and why is it used for monitoring rather than DDPM during training?**
*Why this matters: you will be generating monitoring samples every 5K steps using DDIM. Knowing why it was chosen (and what it trades off) helps you interpret sample quality correctly.*

> Your answer: DDIM is faster than standard DDPM due to it being a deterministic version of DDPM sampling 

---

### Decision Log

**StructuralDiT architecture: layers, hidden dimension, attention heads, and the reasoning behind each:**
*Why this matters: these numbers determine VRAM usage and model capacity. Write down your reasoning now — "it fit" is fine, but be specific about what you tried and what didn't.*

> Layers:
> Hidden dimension:
> Attention heads:
> Reasoning:
> Parameter count:

---

**Stable batch size and how you found it:**
*Why this matters: future you will want to know the VRAM ceiling for this model, especially when loading both stages simultaneously in Phase 4.*

> Batch size:
> What you tried:
> Peak VRAM at this batch size:

---

**Did you use patchification? Why or why not?**
*Why this matters: patchification changes the effective sequence length and the nature of what each token represents. Your answer here informs how you think about Stage 2's cross-attention sequence lengths.*

> Used patchification: yes / no
> Reasoning:

---

**Optimizer settings: learning rate, weight decay, scheduler (if any):**

> LR:
> Weight decay:
> Scheduler:
> Any adjustments made mid-training and why:

---

### After the Phase

**Describe the Stage 1 loss curve: when did it start flattening, and what did "near-convergence" actually look like for this run?**
*Why this matters: your intuition about diffusion model convergence is forming right now. Write down what you observed so you have a reference for Phase 3.*

> Loss at step 1:
> Loss at step 10K:
> Loss at step 50K:
> When did flattening begin:
> Did it fully plateau or level off gradually:

---

**What do Stage 1 samples sound like after 50K steps? Be specific — what is perceptible, and what is still noise?**
*Why this matters: this is the ground truth for "what a coarse latent DiT can learn." It sets your expectations for what Stage 2 conditioning will add.*

> Perceptible structure (rhythm, pitch shape, etc.):
> What is still noise or missing:
> Did sample quality improve visibly between monitoring steps (5K vs 50K)?

---

**Did you hit NaN losses? If yes, what caused it and how did you fix it?**
*Why this matters: NaN diagnosis is a skill. Writing down root cause + fix is the most useful debugging log you can keep.*

> Occurred: yes / no
> If yes — root cause:
> Fix:

---

**What would you change about the Stage 1 architecture or training setup if you trained it again?**

> Your answer:

---

---

## Phase 3 — DetailDiT (Stage 2)

### Before You Start

**Explain cross-attention in your own words: what are queries, keys, and values, and where does each come from in this specific architecture?**
*Why this matters: the cross-attention mechanism is the core claim of this architecture. If you cannot describe it precisely, you will not be able to diagnose it when it fails to condition.*

> Your answer:

---

**What is teacher forcing? Why is it used for Stage 2's initial training instead of using Stage 1 model outputs?**
*Why this matters: teacher forcing is the reason distribution mismatch happens later. Understanding why it was chosen now makes Phase 4's diagnosis less mysterious.*

> Your answer:

---

**What is conditioning dropout, and what specific failure mode does it address?**
*Why this matters: conditioning dropout is the most practical lever for controlling how robust Stage 2 is to imperfect Stage 1 outputs. Your answer here should connect to distribution mismatch.*

> Your answer:

---

### Decision Log

**DetailDiT architecture compared to StructuralDiT — how and why it differs:**
*Why this matters: the size difference between Stage 1 and Stage 2 is an architectural decision with VRAM and quality implications.*

> Layers:
> Hidden dimension:
> Attention heads:
> How this differs from Stage 1 and why:
> Parameter count:

---

**Conditioning dropout rate chosen and reasoning:**
*Why this matters: too high and Stage 2 learns to ignore conditioning; too low and it becomes fragile to Stage 1 imperfections.*

> Rate:
> Reasoning:

---

**Evidence that cross-attention is actually attending (not uniform weights):**
*Why this matters: it is possible to wire cross-attention in and have it do nothing. You should confirm it is working before spending 50K steps training.*

> How you verified it:
> What the attention weights looked like:

---

### After the Phase

**What was the first concrete sign that the conditioning was doing something?**
*Why this matters: recognizing when a signal starts working is a skill. The specific observation here — what changed and at what step — is the most reusable thing you will learn in this phase.*

> Your answer:

---

**Describe the difference between samples conditioned on coarse latent A vs. coarse latent B. Be specific and perceptual — what did you actually hear?**
*Why this matters: "they sounded different" is not enough. The specific structural features that varied are the evidence that your conditioning claim is real.*

> What varied between A and B samples:
> What stayed similar:
> Did spectrograms confirm what you heard?

---

**Did you need to debug the cross-attention not conditioning? What did you find?**

> Occurred: yes / no
> If yes — what the problem was:
> Fix:

---

**What would you change about Stage 2's architecture or training if you built it again?**

> Your answer:

---

---

## Phase 4 — Integration & Evaluation

### Before You Start

**Explain distribution mismatch in your own words: why does it happen even when Stage 1 and Stage 2 both work individually?**
*Why this matters: this is the most likely failure mode in Phase 4. If you hit it and don't know what it is, you will start changing the wrong things.*

> Your answer:

---

**What does `torch.inference_mode()` do differently from `torch.no_grad()`, and why does it matter here?**
*Why this matters: you are managing a tight VRAM budget across three sequential loads. Understanding what each context manager actually frees (or doesn't) affects whether you stay under 10 GB.*

> Your answer:

---

### Decision Log

**Did distribution mismatch occur? What was the L2 evidence?**
*Why this matters: diagnosing from evidence rather than vibes is the habit. Write the actual number.*

> Occurred: yes / no
> L2 distance between Stage 1 outputs and ground-truth coarse latents:
> L2 as a fraction of coarse latent std (was it significant?):

---

**If you fine-tuned Stage 2: learning rate, step count, and outcome:**
*Why this matters: fine-tuning LR is the most sensitive hyperparameter in this phase. Your specific numbers are the useful thing to record.*

> Fine-tuned: yes / no
> If yes — LR:
> Steps:
> Did conditioning survive? How could you tell:
> Did sample quality improve? By how much:

---

**Peak VRAM at each stage during end-to-end inference:**
*Why this matters: staying under 10 GB is a PoC success criterion. These numbers are the evidence.*

> Stage 1 peak:
> Stage 2 peak:
> DAC decode peak:
> Total sequential peak (max across stages):

---

### After the Phase

**Of the 50 evaluation samples, how many would you call perceptually coherent? What distinguished the coherent ones from the rest?**
*Why this matters: 10 of 50 is the PoC bar. Your specific observation about what separates coherent from incoherent samples tells you where the model is spending and wasting its capacity.*

> Coherent count (out of 50):
> What the coherent ones had in common:
> What the incoherent ones had in common:

---

**Look at your spectrogram pairs (original vs. Stage 2 reconstruction). Describe what Stage 1 preserved and what Stage 2 added:**
*Why this matters: the spectrogram is the clearest visual evidence for whether the hierarchy is doing what it claimed. Describing it in words forces you to see it rather than just display it.*

> What survived Stage 1 (visible in coarse latent):
> What Stage 2 added (visible in full reconstruction vs. coarse):
> Anything that neither stage captured:

---

**Was the end-to-end quality better, worse, or about the same as Stage 2 conditioned on ground-truth coarse latents?**
*Why this matters: the gap here is the direct measure of distribution mismatch cost. Even if quality is low in absolute terms, the relative comparison tells you whether the two-stage pipeline is working as designed.*

> Your answer:

---

---

## Full Retrospective

*Answer these after the PoC success criteria are all checked off.*

---

**The core claim of this architecture is that separating global structure from fine-grained detail produces more coherent audio than a flat model. Do you believe the PoC supported this claim? What specific evidence would you point to?**

> Your answer:

---

**What was the most surprising failure mode across the whole project — something you didn't predict from reading the plan?**

> Your answer:

---

**What was the single most important thing you learned about diffusion models from building this? Not from reading — from building.**

> Your answer:

---

**The plan listed three "What Comes Next" directions: text conditioning, scaling, and formal evaluation. Having now built this, which do you think is the highest-leverage next step and why?**

> Your answer:

---

**Write 3 concrete things you would do differently if you started this PoC over today. Not wishes — specific decisions with specific alternatives.**

> 1.
> 2.
> 3.

---

**If you were to publish a write-up of this project, what would the headline claim be? One sentence.**

> Your answer:
