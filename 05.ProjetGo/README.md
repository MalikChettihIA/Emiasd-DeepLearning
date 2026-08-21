# 🎯 05 — Go Project · Deep Learning

**Deep Learning applied to the game of Go (9×9)**

![Python](https://img.shields.io/badge/Python-3.9-3776AB?style=flat&logo=python&logoColor=white) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-FF6F00?style=flat&logo=tensorflow&logoColor=white) ![Keras](https://img.shields.io/badge/Keras-3.9-D00000?style=flat&logo=keras&logoColor=white) ![WandB](https://img.shields.io/badge/Weights%20%26%20Biases-tracked-FFBE00?style=flat&logo=weightsandbiases&logoColor=black) ![Apple Silicon](https://img.shields.io/badge/GPU-Metal%20M4%20Pro-000000?style=flat&logo=apple&logoColor=white) ![Go](https://img.shields.io/badge/Golois-C%2B%2B%20engine-555555?style=flat)

---

## Objective

Train deep neural networks to predict the moves of an expert player on a **9×9 Go board**, directly inspired by the AlphaZero architecture. The model learns both:
- **Policy**: probability of playing each intersection (362 outputs)
- **Value**: estimate of the win probability

Training data is provided by **Golois**, a C++ engine compiled as a Python extension.

## Code architecture

```
05.ProjetGo/
├── go_basenet.py          Base network (conv + BN + ReLU × N blocks)
├── go_mobilenet.py        MobileNetV3-Small adapted for Go (SE bottleneck blocks)
├── go_mobilenetv2.py      MobileNetV2 adapted for Go (inverted blocks)
├── go_resnet.py           ResNet adapted for Go (skip connections)
├── go_mixnet.py           MixNet adapted for Go (multi-kernel convolutions)
├── go_callbacks.py        Keras callbacks: checkpointing, monitoring, plots
├── go_train.py            Main training script
├── training_monitor.py    Real-time metrics tracking
├── go_test_utils.py       Experiment utilities
├── go_test_model_summary.py  Architecture summary and comparison
├── wandb.config           Weights & Biases configuration
└── golois.cpython-39-darwin.so  Go engine (compiled C++ module)
```

## Implemented models

| Model | File | Specificity |
|---|---|---|
| **BaseNet** | `go_basenet.py` | Stack of Conv+BN+ReLU blocks — simple baseline |
| **MobileNet** | `go_mobilenet.py` | Depthwise bottleneck blocks + Squeeze-and-Excitation |
| **MobileNetV2** | `go_mobilenetv2.py` | Inverted blocks (expand→depthwise→project) + skip |
| **ResNet** | `go_resnet.py` | Identity skip connections — inspired by He et al. 2016 |
| **MixNet** | `go_mixnet.py` | Multi-kernel depthwise convolutions (3×3, 5×5, 7×7) |

**Shared output heads:** Policy head (softmax, 362) + Value head (scalar sigmoid), trained multi-task.

## Experiment campaign — Ablation study

7 axes of systematic study, each with a dedicated script:

| Script | Axis studied | Variable |
|---|---|---|
| `go_test00000.py` | Base configuration | Dropout, L2, ClipNorm — 4 variants |
| `go_test00001-Models.py` | Architecture comparison | BaseNet vs MobileNet vs ResNet vs MixNet |
| `go_test00001-ModelsV2.py` | Comparison v2 | Refined variants |
| `go_test00002-BatchSize.py` | Batch size | Impact on convergence and generalization |
| `go_test00003-N.py` | Number of samples | Training data volume |
| `go_test00004-BlockNum.py` | Depth | Number of residual blocks |
| `go_test00005-PolicyWeight.py` | Policy loss weight | Policy/Value balance |
| `go_test00006-ValueWeight.py` | Value loss weight | Value/Policy balance |

Each experiment generates loss curves and is tracked on **Weights & Biases**.

## Test results

```
go_test00000/   Baseline results (curves, plots)
go_test00005/   Policy Weight test results
go_test00006/   Value Weight test results
```

The W&B runs (Backups/20250429/Test1 to Test10) document the evolution of the validation loss over 220+ training epochs.

## Training

```bash
# Run a training session
python go_train.py

# Run an ablation experiment
python go_test00001-Models.py
```

**Required data:**
- `games.data` — played games (not versioned, several GB)
- `validation.data` — validation data

## Experiment tracking (W&B)

Runs are organized by date in `Backups/20250429/wandb/`. Tracked metrics:
- `val_policy_loss` — move prediction loss
- `val_value_loss` — win estimation loss
- `val_loss` — total loss (best run: **~2.55** at epoch 240)

## Backups — Development history

| Folder | Content |
|---|---|
| `Backups/20250329/` | First exploration notebooks (MobileNet, importGolois) |
| `Backups/20250331/` | ProjetV0 and ProjetV1 (first trainable version) |
| `Backups/20250404/` | AlphaZero lite — simplified self-play |
| `Backups/20250405–20250407/` | GoMobileNetV2 — first iterations |
| `Backups/20250426/` | GoMobileNetV2/V3 — versions 0.1 to 0.2 |
| `Backups/20250429/` | Complete version with all models + W&B + Tests 1–10 |
| `Backups/20250701/` | Final version — importGolois v0/v1 |
| `Backups/project2022/` | Golois C++ source (Board.h, Game.h, golois.cpp, Makefile) |
| `Backups/project2025/` | Multi-platform Golois binaries |

## Scientific foundations — Reference papers

The project is built on a bibliography of 7 papers that guided every architectural and algorithmic decision, forming a progression from the foundations (ResNet, PUCT) to fine-grained optimizations (MixConv, Swish, Squeeze & Excitation).

| Paper | Contribution to the project |
|---|---|
| [Residual Networks for Computer Go](articles/ResidualNetworksForComputerGo.md) | Justifies `go_resnet.py` and `go_basenet.py` — skip connections as a baseline |
| [Spatial Average Pooling for Computer Go](articles/SpatialAveragePoolingForComputerGo.md) | Value head design — spatial pooling vs. GAP |
| [Mobile Networks for Computer Go](articles/MobileNetworksForComputerGo.md) | **Central paper** — justifies `go_mobilenet.py` and `go_mobilenetv2.py` |
| [Improving Model and Search for Computer Go](articles/ImprovingModelAndSearchForComputerGo.md) | Squeeze & Excitation block in `go_mobilenet.py`, depth/width study |
| [Cosine Annealing, Mixnet and Swish for Computer Go](articles/CosineAnnealingMixnetAndSwishActivationForComputerGo.md) | `go_mixnet.py`, cosine scheduler in `go_train.py`, Swish activation |
| [Accelerating Self-Play Learning in Go (KataGo)](articles/AcceleratingSelfPlayLearningInGo.md) | Source of the reference dataset, global pooling bias concept |
| [Polygames: Improved Zero Learning](articles/PolygamesImprovedZeroLearning.md) | Fully convolutional architecture, size-invariant value head |

→ [See the full articles folder](articles/README.md)

## Report

Final report documenting the methodology, architectures, ablation campaign and results: [PDF](Emiasd%20-%20Deep%20Learning%20-%20Rapport%20Projet.pdf) · [DOCX](Emiasd%20-%20Deep%20Learning%20-%20Rapport%20Projet.docx)
