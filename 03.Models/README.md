# 🏗️ 03 — Reference Models

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-FF6F00?style=flat&logo=tensorflow&logoColor=white) ![Keras](https://img.shields.io/badge/Keras-3.9-D00000?style=flat&logo=keras&logoColor=white) ![Python](https://img.shields.io/badge/Python-3.9-3776AB?style=flat&logo=python&logoColor=white)

---

From-scratch implementations of the reference architectures studied in class, used as the starting point for the Go project.

## AlphaZero/

Simplified implementation of the AlphaZero architecture (Silver et al., 2018).

| File | Role |
|---|---|
| `config.py` | Centralized hyperparameters (lr, batch size, number of blocks…) |
| `model.py` | Dual-head network: shared residual trunk → Policy head + Value head |
| `main.py` | Self-play training loop |

**Architecture:** residual convolutional trunk → two output heads (move probability / win estimate), as in AlphaGo Zero.

## MobileNetV2/

Full implementation of **MobileNetV2** (Sandler et al., 2018) — a lightweight architecture for embedded vision.

| File | Role |
|---|---|
| `mobilenet_v2.py` | Inverted bottleneck blocks (expand → depthwise → project) |
| `mobilenet_v2.ipynb` | Exploration and visualization notebook |
| `train.py` | Training script |
| `data/convert.py` | Data preprocessing |
| `model/hist.csv` | Training history |
| `1801.04381v4.pdf` | Original MobileNetV2 paper |

**Key concept:** depthwise separable convolutions — reduces the parameter count by ~8-9× vs. a classic CNN, with residual connections on stride-1 blocks.

## MobileNetV3/

Implementation of **MobileNetV3-Small** and **MobileNetV3-Large** (Howard et al., 2019).

| File | Role |
|---|---|
| `model/mobilenet_base.py` | Shared base class |
| `model/mobilenet_v3_small.py` | Small variant (tabulated architecture) |
| `model/mobilenet_v3_large.py` | Large variant |
| `model/LR_ASPP.py` | Lite Reduced ASPP for segmentation |
| `train_cls.py` | Classification training |
| `test.ipynb` | Tests and benchmarks |

**New vs. V2:** SE (Squeeze-and-Excitation) blocks, h-swish activation, NAS-optimized.

## MixNet/

Implementation of the **keras-mixnets** package — MixConv (Tan & Le, 2019).

| File | Role |
|---|---|
| `keras_mixnets/mixnets.py` | MixNet-S / MixNet-M / MixNet-L architecture |
| `keras_mixnets/config.py` | Variant configurations |
| `keras_mixnets/custom_objects.py` | Custom layers |
| `test.ipynb` | Demonstration |

**Key concept:** Mixed Depthwise Convolutions — depthwise convolutions with **several kernel sizes simultaneously** (3×3, 5×5, 7×7...) concatenated per channel. Better accuracy/efficiency than MobileNet.

## MixNet2/

Experimental MixNet variant implemented as a notebook (`mixnet.ipynb`), adapted to the constraints of the Go project.

## Link with the Go project

These implementations were the direct basis for [`05.ProjetGo/`](../05.ProjetGo/README.md):

| Reference model | Adapted into |
|---|---|
| AlphaZero | `go_basenet.py` |
| MobileNetV2 | `go_mobilenetv2.py` |
| MobileNetV3-Small | `go_mobilenet.py` |
| MixNet | `go_mixnet.py` |
| ResNet | `go_resnet.py` |
