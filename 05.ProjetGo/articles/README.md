<div align="center">

# Articles Scientifiques — Projet Go

**Base bibliographique du projet Deep Learning pour le jeu de Go**

![Research](https://img.shields.io/badge/Articles-7%20papers-lightgrey)
![Topics](https://img.shields.io/badge/Topics-ResNet%20%7C%20MobileNet%20%7C%20MCTS%20%7C%20Zero%20Learning-4A90E2)
![Authors](https://img.shields.io/badge/Auteur%20principal-Tristan%20Cazenave-003366)

</div>

Ces 7 articles constituent la **base scientifique du projet**. Ils forment une progression logique, des fondations (ResNet, PUCT) jusqu'aux architectures légères et techniques d'entraînement avancées qui ont directement guidé nos choix d'implémentation.

---

## Vue d'ensemble

| # | Article | PDF | Notes | Concepts clés |
|---|---|---|---|---|
| 1 | Residual Networks for Computer Go | [resnet.pdf](resnet.pdf) | [→ Notes](ResidualNetworksForComputerGo.md) | Skip connections, ResNet, Golois |
| 2 | Spatial Average Pooling for Computer Go | [sap.pdf](sap.pdf) | [→ Notes](SpatialAveragePoolingForComputerGo.md) | Value head, pooling spatial, territoire |
| 3 | Mobile Networks for Computer Go | [MobileNetworksForComputerGo.pdf](MobileNetworksForComputerGo.pdf) | [→ Notes](MobileNetworksForComputerGo.md) | MobileNetV2, depthwise conv, GAP |
| 4 | Improving Model and Search for Computer Go | [ImprovingModelAndSearchForComputerGo.pdf](ImprovingModelAndSearchForComputerGo.pdf) | [→ Notes](ImprovingModelAndSearchForComputerGo.md) | Squeeze & Excitation, GPUCT, KataGo dataset |
| 5 | Cosine Annealing, Mixnet and Swish for Computer Go | [CosineAnnealingMixnetAndSwishActivationForComputerGo.pdf](CosineAnnealingMixnetAndSwishActivationForComputerGo.pdf) | [→ Notes](CosineAnnealingMixnetAndSwishActivationForComputerGo.md) | Cosine annealing, MixConv, Swish |
| 6 | Accelerating Self-Play Learning in Go (KataGo) | [1902.10565v5.pdf](1902.10565v5.pdf) | [→ Notes](AcceleratingSelfPlayLearningInGo.md) | KataGo, global pooling, playout cap, 50× faster |
| 7 | Polygames: Improved Zero Learning | [2001.09832v1.pdf](2001.09832v1.pdf) | [→ Notes](PolygamesImprovedZeroLearning.md) | Fully conv, neuroplasticité, tournament mode |

---

## Progression logique

```
[1] ResNet (2016)          Fondation : skip connections → premier programme 3-dan
      ↓
[2] SAP (2018)             Value head améliorée : pooling spatial → représentation du territoire
      ↓
[6] KataGo (2019-2020)    Révolution efficacité : 50× moins de compute pour niveau superhuman
[7] Polygames (2020)       Généralisation : framework multi-jeux, fully convolutional
      ↓
[3] MobileNet Go (2021)    Architecture légère pour Go : MobileNetV2 > ResNet à paramètres égaux
      ↓
[4] Improving Model (2021) Optimisation : Squeeze & Excitation + GPUCT → meilleur Pareto
      ↓
[5] CosineAnnealing (2021) Fine-tuning : scheduler + MixConv + Swish → 3 améliorations cumulatives
```

---

## Correspondance avec notre code

| Article | Implémentation dans le projet |
|---|---|
| ResNet | `go_resnet.py`, `go_basenet.py` |
| SAP | Conception de la tête value (GAP dans `go_mobilenet.py`) |
| MobileNet Go | `go_mobilenet.py`, `go_mobilenetv2.py` |
| Improving Model | Bloc SE dans `go_mobilenet.py` |
| CosineAnnealing | Scheduler dans `go_train.py`, `go_mixnet.py` |
| KataGo | Source du dataset pour les benchmarks Cazenave ; concept de GAP value |
| Polygames | Architecture fully conv, global pooling value head |

---

## Auteurs principaux

- **Tristan Cazenave** (LAMSADE, Paris-Dauphine) — articles 1, 2, 3, 4, 5, 7 — notre encadrant de cours
- **David J. Wu** (Jane Street) — article 6 (KataGo)
- **Cazenave, Chen, Synnaeve et al.** (Facebook AI + LAMSADE) — article 7 (Polygames)
