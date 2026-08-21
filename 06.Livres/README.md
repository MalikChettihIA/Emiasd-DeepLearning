# 📖 06 — Books & Papers

![Research](https://img.shields.io/badge/Papers-IEEE%20%7C%20Nature%20%7C%20AAAI-lightgrey?style=flat) ![Books](https://img.shields.io/badge/Books-MIT%20Press%20%7C%20Manning-4A90E2?style=flat)

---

Reference library for the course and the project.

## Books

| File | Authors | Publisher | Relevance |
|---|---|---|---|
| `Deep Learning with Python.pdf` | François Chollet | Manning, 2020 | **Main reference** for the Cazenave module — Keras, CNN, RNN, GAN, VAE |
| `Deep_Learning_and_the_Game_of_Go.pdf` | Pumperla & Ferguson | Manning, 2019 | Deep Learning applied to Go — AlphaGo, MCTS, policy networks |
| `Deep_Learning_and_the_Game_of_Go.epub` | Pumperla & Ferguson | Manning, 2019 | Ebook version |

## Papers — Computer Go & Self-Play

| File | Reference | Key contribution |
|---|---|---|
| `MobileNetworksForComputerGo.pdf` | Cazenave, IEEE ToG 2021 | Adapting MobileNet to Go — basis of the project |
| `ImprovingModelAndSearchForComputerGo.pdf` | Cazenave, IEEE CoG 2021 | Architecture + search optimization for Go |
| `CosineAnnealingMixnetAndSwishActivationForComputerGo.pdf` | Cazenave et al., ACG 2021 | Cosine Annealing + MixConv + Swish — project improvements |
| `2020 Accelerating Self-Play Learning in Go.pdf` | Wu, AAAI RLG 2020 | Speeding up self-play learning |
| `2020 Polygames Improved Zero Learning.pdf` | Cazenave et al., ICGA 2020 | Generalizing AlphaZero to multiple games |
| `2021 CosineAnnealingMixnetAndSwishActivationForComputerGo.pdf` | Cazenave et al., ACG 2021 | Conference version |

## Papers — Foundational architectures

| File | Reference | Key contribution |
|---|---|---|
| `resnet.pdf` | He et al., CVPR 2016 | **Residual Networks** — skip connections, training very deep networks |
| `sap.pdf` | Cazenave, IJCAI 2018 | Spatial Average Pooling for Computer Go |

## Recommended reading to understand the project

1. **Chollet** — *Deep Learning with Python*: theoretical and practical Keras foundations
2. **MobileNetworksForComputerGo**: how to adapt a lightweight architecture to Go
3. **ImprovingModelAndSearchForComputerGo**: optimizations implemented in the project
4. **CosineAnnealingMixnetAndSwish**: scheduler and activations used in the final experiments
5. **Pumperla & Ferguson** — *Deep Learning and the Game of Go*: AlphaGo/AlphaZero context
