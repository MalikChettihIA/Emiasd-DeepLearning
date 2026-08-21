# 🧠 Emiasd - Deep Learning

![EMIASD Dauphine](https://img.shields.io/badge/-EMIASD%20Dauphine-000000?style=flat) ![Python](https://img.shields.io/badge/Python-3.9-3776AB?style=flat&logo=python&logoColor=white) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-FF6F00?style=flat&logo=tensorflow&logoColor=white) ![Keras](https://img.shields.io/badge/Keras-3.9-D00000?style=flat&logo=keras&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat&logo=pytorch&logoColor=white) ![Jupyter](https://img.shields.io/badge/-Jupyter-F37626?style=flat&logo=jupyter&logoColor=white) ![WandB](https://img.shields.io/badge/Weights%20%26%20Biases-tracking-FFBE00?style=flat&logo=weightsandbiases&logoColor=black)

---

This repository gathers my work for the **Deep Learning** course of the **EMIASD Executive Master (Artificial Intelligence & Data Science) at Université Paris-Dauphine**, taught by:

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

## ⭐ Featured project — Go & Deep Learning

The project trains deep neural networks to play **9×9 Go**, directly inspired by AlphaZero. The **Golois** evaluation engine (a C++ module interfaced in Python) provides the training data.

Several architectures were implemented and compared: **BaseNet, ResNet, MobileNet, MobileNetV2, MixNet**. Training is tracked with **Weights & Biases**, with a systematic ablation campaign across 7 axes (regularization, batch size, number of blocks, Policy/Value loss weighting...).

See [`05.ProjetGo/`](05.ProjetGo/README.md) for the full details.

## Tech stack

```
Frameworks    TensorFlow 2.16 / Keras 3.9  (Cazenave module)
              PyTorch 2.x                   (Verine module)
Tracking      Weights & Biases (wandb)
Hardware      Apple M4 Pro — Metal GPU
Environment   Python 3.9 / conda
```
