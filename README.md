# 🎵 Audio Genre Classification using Deep Learning

**Classifying noisy, multi-stem "mashup" audio clips into 10 music genres using CNNs, Transfer Learning, and a fine-tuned Audio Spectrogram Transformer (AST).**

[![Kaggle Score](https://img.shields.io/badge/Kaggle%20Macro%20F1-0.92064-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/jan-2026-dl-gen-ai-project/)
[![W&B Report](https://img.shields.io/badge/Weights%20%26%20Biases-Report-FFBE00?logo=weightsandbiases&logoColor=black)](https://api.wandb.ai/links/24f2008471-indian-institute-of-technology-madras/fz3d3f7b)
[![Hugging Face Demo](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/amanvish07/audio-genre-classifier)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)

---

## 🔗 Live Links

| Resource | Link |
|---|---|
| 🚀 **Live Demo (Hugging Face Spaces)** | [audio-genre-classifier](https://huggingface.co/spaces/amanvish07/audio-genre-classifier) |
| 📊 **W&B Experiment Report** | [View Full Report](https://api.wandb.ai/links/24f2008471-indian-institute-of-technology-madras/fz3d3f7b) |
| 📈 **W&B Project (all runs)** | [Dl-genai-26-t1](https://wandb.ai/24f2008471-indian-institute-of-technology-madras/Dl-genai-26-t1) |
| 🏆 **Kaggle Competition** | [Jan-2026 DL & GenAI Project](https://www.kaggle.com/competitions/jan-2026-dl-gen-ai-project/) |

---

## 📌 Overview

This project tackles automatic **music genre classification** on the *"Messy Mashup"* dataset from the IIT Madras BS Degree **DL & GenAI Kaggle competition (Jan-2026 term)**. Unlike clean genre-classification benchmarks, the task requires classifying audio clips built by mixing individual instrument **stems** (bass, drums, vocals, other) drawn from different songs — producing noisy, ambiguous audio across **10 genres**: `blues`, `classical`, `country`, `disco`, `hiphop`, `jazz`, `metal`, `pop`, `reggae`, `rock`.

Three progressively stronger architectures were built, trained, and benchmarked, with every experiment tracked end-to-end on **Weights & Biases**.

> **🏅 Final Result: 0.92064 Macro F1 on the Kaggle leaderboard** — 12 points above the 0.80 competition cutoff.

---

## 🧠 Key Finding

Pre-trained audio transformers substantially outperform convolutional baselines for this task. The **AST's** self-attention mechanism, combined with AudioSet pre-training, captures global spectro-temporal structure that local convolutional filters cannot — a **47-point Macro F1 improvement** over the from-scratch CNN baseline (0.45263 → 0.92064).

---

## 🏗️ Models & Results

| Model | Parameters | Train Macro F1 | Train Accuracy | Training Time | Kaggle Macro F1 | Best Use |
|---|---|---|---|---|---|---|
| Custom CNN (from scratch) | ~33K | 0.60 | 60.95% | ~5 hrs (20 ep) | **0.45263** | Learning / Baseline |
| ResNet-18 (transfer learning) | ~11.2M | 0.77 | 77.33% | ~72 min (12 ep) | **0.82880** | Fast production |
| **AST (fine-tuned)** | ~86M | 0.9351 | 93.50% | ~160 min (10 ep) | **🥇 0.92064** | Best accuracy |

**Architecture summaries:**
- **Custom CNN** — 3-block `Conv2d → ReLU → MaxPool2d` network built from scratch as a lower-bound baseline (32,906 params).
- **ResNet-18** — ImageNet-pretrained backbone (layers 1–2 frozen, 3–4 fine-tuned) with a custom `Dropout → Linear(512→10)` head and differential learning rates.
- **AST (Audio Spectrogram Transformer)** — `MIT/ast-finetuned-audioset-10-10-0.4593` fine-tuned end-to-end, with a sliding-window (5s window, 1.5s step) inference strategy that averages logits across overlapping segments for the final prediction.

---

## 🛠️ Tools & Frameworks

- **PyTorch & torchaudio** — model training, audio processing, Mel Spectrogram computation
- **Hugging Face Transformers** — AST architecture and feature extractor
- **librosa** — audio loading, resampling, tempo-stretch augmentation
- **Weights & Biases** — full experiment tracking (loss, accuracy, Macro F1, learning rate)
- **Kaggle** — competition platform, NVIDIA T4 GPU

---

## 📂 Dataset

- **1,000 songs** across 10 genres (100 songs/genre), each split into 4 stems: `bass.wav`, `drums.wav`, `vocals.wav`, `other.wav`
- **3,020 test files** — pre-mixed mashup clips requiring genre prediction
- Perfectly balanced classes → no resampling/class-weighting needed
- **On-the-fly data synthesis:** stems cross-mixed from different songs of the same genre every epoch, maximizing diversity and avoiding memorization

**Augmentations used:** random 5s cropping, cross-song stem mixing (volume jitter), tempo stretching (AST), ESC-50 background noise injection, and SpecAugment (frequency/time masking).

---

## 📊 Experiment Tracking

All training runs (loss curves, Macro F1, accuracy, learning-rate schedules) are logged and publicly viewable on Weights & Biases:

- 📈 **Full Report:** https://api.wandb.ai/links/24f2008471-indian-institute-of-technology-madras/fz3d3f7b
- 📁 **Project (raw runs):** https://wandb.ai/24f2008471-indian-institute-of-technology-madras/Dl-genai-26-t1

---

## 🚀 Live Demo

Try the fine-tuned AST model directly in your browser:

👉 **[audio-genre-classifier on Hugging Face Spaces](https://huggingface.co/spaces/amanvish07/audio-genre-classifier)**

Upload an audio clip and get real-time genre predictions across all 10 classes.

---

## 🔍 Key Insights

- **Pre-training > architectural depth:** a 47-point Kaggle F1 gap between the CNN (0.45) and AST (0.92) shows AudioSet pre-training matters far more than raw model capacity for audio tasks.
- **Rapid transfer learning:** ResNet-18 jumped from 0.34 → 0.72 Macro F1 within just 4 epochs thanks to ImageNet initialization.
- **Sliding-window inference is "free" robustness:** averaging logits over overlapping 5s windows improved test performance with zero extra training cost.
- **Cosine LR + warmup stabilized AST fine-tuning**, preventing early instability that could otherwise damage pretrained weights.

---

## 📈 Future Work

- Ensemble AST + ResNet-18 logits for an additional 1–3 F1 points
- Explore waveform-level pretraining (HuBERT, wav2vec 2.0)
- Per-genre threshold tuning instead of global argmax
- Systematic W&B hyperparameter sweeps
- ONNX export + Gradio deployment (already live — see demo above)

---

## 📚 References

1. Gong, Y., Chung, Y. A., & Glass, J. (2021). *AST: Audio Spectrogram Transformer.* Interspeech 2021. [arXiv:2104.01778](https://arxiv.org/abs/2104.01778)
2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition.* CVPR 2016. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
3. Park, D. S., et al. (2019). *SpecAugment.* Interspeech 2019. [arXiv:1904.08779](https://arxiv.org/abs/1904.08779)
4. Loshchilov, I., & Hutter, F. (2019). *Decoupled Weight Decay Regularization (AdamW).* ICLR 2019. [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)
5. Dosovitskiy, A., et al. (2021). *An Image is Worth 16x16 Words: ViT.* ICLR 2021. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
6. Wolf, T. et al. (2020). *Transformers: State-of-the-Art NLP.* EMNLP 2020.
7. Piczak, K. J. (2015). *ESC: Dataset for Environmental Sound Classification.* ACM Multimedia 2015.

---

## 👤 Author

**Roll Number:** 24f2008471 · **Program:** IIT Madras BS Degree
**Course:** BSDA2002P — Deep Learning & Generative AI

