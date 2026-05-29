# Reading List — Nested DiT Text-to-Audio Project

---

## Phase 1 — Before You Write Code

### 1. Denoising Diffusion Probabilistic Models (DDPM)
**Authors:** Ho et al., 2020
**Why:** Defines the noise schedule, loss function, and epsilon parameterization used throughout this project. The DiT paper assumes you already know this — read sections 2 and 3 before anything else.
**Link:** [https://arxiv.org/abs/2006.11239]

---

### 2. Denoising Diffusion Implicit Models (DDIM)
**Authors:** Song et al., 2020
**Why:** You will implement a DDIM sampler in Phase 2. Read the abstract and section 4 before writing the sampler.
**Link:** [https://arxiv.org/abs/2010.02502]

---

### 3. Classifier-Free Diffusion Guidance
**Authors:** Ho & Salimans, 2022
**Why:** The paper that explains why conditioning dropout works. Read before Phase 3 — the plan tells you what rate to use, this explains why it helps.
**Link:** [https://arxiv.org/abs/2207.12598]

---

### 4. The Annotated Diffusion Model
**Authors:** Hugging Face (von Platen et al.)
**Why:** Line-by-line PyTorch walkthrough of a working DDPM implementation. Keep it open while implementing the noise schedule and training loop in Phase 2. Not a paper — a blog post.
**Link:** [https://huggingface.co/blog/annotated-diffusion]

---

### 5. Scalable Diffusion Models with Transformers (DiT)
**Authors:** Peebles & Xie, 2022
**Why:** Your core architecture. Read fully and carefully before anything else.
**Link:** [https://arxiv.org/abs/2212.09748]

---

### 6. Descript Audio Codec (DAC)
**Authors:** Kumar et al., 2023
**Why:** Your VAE backbone. Skim the paper, spend more time with the GitHub.
**Paper:** [https://arxiv.org/abs/2306.06546]
**GitHub:** [https://github.com/descriptinc/descript-audio-codec]

---

### 7. Whole-Song Hierarchical Generation of Symbolic Music Using Cascaded Diffusion Models
**Year:** 2024
**Why:** Closest existing analog to your project. Read fully — it will save you from mistakes they already made.
**Link:** [https://arxiv.org/abs/2405.09901]

---

## Phase 2 — While Building

### 8. AudioLDM 2: Learning Holistic Audio Generation with Self-Supervised Pretraining
**Authors:** Liu et al., 2023
**Why:** Study the conditioning pipeline section specifically. Read when you hit the cross-attention implementation.
**Link:** [https://arxiv.org/abs/2308.05734]

---

### 9. Hierarchical Text-Conditional Image Generation with CLIP Latents (DALL-E 2)
**Authors:** Ramesh et al., 2022
**Why:** The clearest articulation of why explicit two-stage conditioning works. Read intro and two-stage section only — read when wiring your two stages together.
**Link:** [https://arxiv.org/abs/2204.06125]

---

## Phase 3 — During Writeup

### 10. Parsing Natural Scenes and Natural Language with Recursive Neural Networks
**Authors:** Socher et al., 2011
**Why:** The paper that established RvNNs for hierarchical representation learning. Read abstract and intro only — enough to cite the lineage.
**Link:** [https://ai.stanford.edu/~ang/papers/icml11-RecursiveNeuralNetworks.pdf]

---

### 11. Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank
**Authors:** Socher et al., 2013
**Why:** Demonstrates hierarchical composition generalizes across tasks. Skim level — read during writeup to support your RvNN framing.
**Link:** [https://aclanthology.org/D13-1170.pdf]

---

### 12. Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks (Tree-LSTM)
**Authors:** Tai et al., 2015
**Why:** Bridges classical hierarchical modeling and modern attention mechanisms. Optional — only if you want to deepen the RvNN-to-transformer argument in your writeup.
**Link:** [https://arxiv.org/abs/1503.00075]

---

## What This List Teaches You Broadly

| Paper | Core Concept Learned |
|---|---|
| DDPM | Noise schedules, forward/reverse diffusion process, epsilon parameterization |
| DDIM | Deterministic sampling, fewer steps without retraining |
| Classifier-Free Guidance | Conditioning dropout, trading diversity for conditioning strength |
| Annotated Diffusion Model | Practical PyTorch implementation of the full DDPM pipeline |
| DiT | Transformers + diffusion unified, patchification, adaLN conditioning |
| DAC | Audio compression, quantization, learned latent spaces |
| Whole-Song Cascaded Diffusion | Cascade architectures, coarse-to-fine generation |
| AudioLDM 2 | Cross-modal conditioning, guiding generation across representation spaces |
| DALL-E 2 | Semantic representations, two-stage hierarchical generation |
| Socher 2011 | Origins of hierarchical representation learning |
| Socher 2013 | Generalization of hierarchical composition across domains |
| Tree-LSTM | Bridge between classical sequence modeling and modern attention |

---

*Reading order follows the arc: diffusion fundamentals → transformer-based diffusion → audio codec → hierarchical generation → conditioning strategies → RvNN lineage.*
