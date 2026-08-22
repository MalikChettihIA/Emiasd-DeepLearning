# 🧠 Emiasd - Deep Learning

![EMIASD Dauphine](https://img.shields.io/badge/-EMIASD%20Dauphine-000000?style=flat) ![Python](https://img.shields.io/badge/Python-3.9-3776AB?style=flat&logo=python&logoColor=white) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-FF6F00?style=flat&logo=tensorflow&logoColor=white) ![Keras](https://img.shields.io/badge/Keras-3.9-D00000?style=flat&logo=keras&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat&logo=pytorch&logoColor=white) ![Jupyter](https://img.shields.io/badge/-Jupyter-F37626?style=flat&logo=jupyter&logoColor=white) ![WandB](https://img.shields.io/badge/Weights%20%26%20Biases-tracking-FFBE00?style=flat&logo=weightsandbiases&logoColor=black)

---

This repository gathers my work for the **Deep Learning** course of the **[EMIASD Executive Master](https://executive-education.dauphine.psl.eu/formations/executive-master-diplome-universite/ia-science-donnees) (Artificial Intelligence & Data Science) at Université Paris-Dauphine \| PSL**, taught by:

- **Alexandre Verine** — AI Research Fellow, Centre des Données, ENS PSL. Specialist in generative AI for images.
- **Tristan Cazenave** — Professor, LAMSADE, Paris-Dauphine \| PSL. PRAIRIE Chair, Editor-in-Chief of the ICGA, expert in strategy games.

**Learning objectives:** understand the fundamentals of deep learning and train neural networks on varied data with Keras and PyTorch — network architecture and training (MLP → CNN → ResNet → MobileNet), generative models (Autoencoders, VAE, GAN), computer vision and strategy games, and systematic hyperparameter ablation.

## Content

| Folder | Content |
|---|---|
| [`01.Cours/`](01.Cours/README.md) | Course slides and syllabus |
| [`02.TP/`](02.TP/README.md) | 11 labs (Keras + PyTorch) |
| [`03.Models/`](03.Models/README.md) | Reference implementations (AlphaZero, MobileNet, MixNet, ResNet) |
| [`04.Others/`](04.Others/README.md) | Research papers, notes and resources |
| [`05.ProjetGo/`](05.ProjetGo/README.md) | ⭐ **Main project** — Deep Learning applied to the game of Go |
| [`06.Livres/`](06.Livres/README.md) | Foundational books and papers |

## ⭐ Featured project — Deep Neural Networks for the Game of Go

*"Neural Networks for the Game of Go: Implementation and Convergence Diagnostics" — with Henri Balamou.*

Go is one of the hardest combinatorial problems tackled by AI: on a 19×19 board the number of legal positions is estimated at ~10⁶⁰⁰ (vs. ~10¹²⁰ for chess), placing it among NP-hard, arguably PSPACE-hard problems. The project retraces — and re-implements — the three innovations that made the game tractable for AI (Monte Carlo Tree Search → convolutional networks → AlphaGo's fusion of deep learning, MCTS and reinforcement learning), then builds and diagnoses several deep networks that jointly learn:

- **Policy** π(a\|P): the probability distribution over the 361 intersections of a 19×19 board (362 outputs incl. "pass") that an expert would play from position *P* — challenging because only ~20-50 of the 361 moves are ever actually viable (sparsity), and a move's value is highly context-dependent.
- **Value** V(P) ∈ [0,1]: the win probability from position *P*, replacing costly random-playout evaluation in MCTS — challenging because of long time horizons, territory/influence tradeoffs, and positional nuances that shift across opening/middlegame/endgame.

Training jointly optimizes both heads on a shared trunk, which introduces its own difficulties (conflicting gradients between the two objectives, loss-weighting, and a strong sensitivity to vanishing/exploding gradients).

**Data & augmentation** — training data comes from the **Golois** engine (a C++ module interfaced in Python, `getBatch()`/`getValidation()` pipelines) built on the KataGo dataset. Three augmentation strategies exploit Go's 8 geometric board symmetries (rotations/reflections): none, one random transform per batch, or all 8 transforms applied exhaustively (up to **60M effective training samples** vs. 7.5M without augmentation).

**Architectures** — five networks were implemented and compared: **BaseNet** (plain Conv+BN+ReLU stack), **ResNet** (identity skip connections, He et al. 2016), **MobileNet** (depthwise bottleneck + Squeeze-and-Excitation blocks), **MobileNetV2** (inverted residual blocks), and **MixNet** (multi-kernel depthwise convolutions) — all kept under a **100K-parameter budget**. Every model shares dual Policy/Value output heads.

**Training procedure** — **AdamW** (decoupled weight decay, correcting classic Adam's coupling between L2 regularization and the adaptive learning rate) combined with **Cosine Annealing with Warm Restarts** (Loshchilov & Hutter's SGDR) for the learning-rate schedule, **Swish activation** in the inverted residual blocks (per Cazenave et al.'s findings for Computer Go), gradient clipping, and a **1:4 Policy/Value loss weighting** favoring value-estimation quality. Runs span up to 300 epochs.

**Monitoring & diagnostics** — a custom `TrainingMonitor` class integrates with **Weights & Biases**: robust metric conversion across TensorFlow/NumPy types, automatic architecture logging (parameter counts, layer breakdown), **gradient health diagnostics** (vanishing threshold 1e-6, exploding threshold 10.0, calibrated empirically), and conditional checkpointing of the best-validation-loss model.

**Experimentation framework** — a factory-pattern test harness (`go_test_utils.py`) drives a systematic **7-axis ablation campaign**: architecture comparison, network depth, learning rate, batch size, MixNet-specific tuning, cosine-restart policy, and training-set size per epoch — each producing standardized comparative plots and validation reports. Best run: **~2.55 total validation loss** at epoch 240.

See [`05.ProjetGo/`](05.ProjetGo/README.md) for the full code breakdown, and the [full report](05.ProjetGo/Emiasd%20-%20Deep%20Learning%20-%20Rapport%20Projet.pdf) for the complete methodology and results.

## Tech stack

```
Frameworks    TensorFlow 2.16 / Keras 3.9  (Cazenave module)
              PyTorch 2.x                   (Verine module)
Tracking      Weights & Biases (wandb)
Hardware      Apple M4 Pro — Metal GPU
Environment   Python 3.9 / conda
```
