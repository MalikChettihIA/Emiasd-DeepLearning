<div align="center">

# 03 — Modèles de référence

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-FF6F00?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-3.9-D00000?logo=keras&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python&logoColor=white)

</div>

Implémentations from-scratch des architectures de référence étudiées en cours, utilisées comme base de départ pour le projet Go.

---

## AlphaZero/

Implémentation simplifiée de l'architecture AlphaZero (Silver et al., 2018).

| Fichier | Rôle |
|---|---|
| `config.py` | Hyperparamètres centralisés (lr, batch size, nb blocs…) |
| `model.py` | Réseau dual-head : tronc résiduel partagé → tête Policy + tête Value |
| `main.py` | Boucle d'entraînement self-play |

**Architecture :** tronc convolutif résiduel → deux têtes de sortie (probabilité de coup / estimation de victoire), comme dans AlphaGo Zero.

---

## MobileNetV2/

Implémentation complète de **MobileNetV2** (Sandler et al., 2018) — architecture légère pour vision embarquée.

| Fichier | Rôle |
|---|---|
| `mobilenet_v2.py` | Blocs bottleneck inversés (expand → depthwise → project) |
| `mobilenet_v2.ipynb` | Notebook d'exploration et visualisation |
| `train.py` | Script d'entraînement |
| `data/convert.py` | Prétraitement des données |
| `model/hist.csv` | Historique d'entraînement |
| `1801.04381v4.pdf` | Paper original MobileNetV2 |

**Concept clé :** convolutions dépthwise séparables — réduction du nombre de paramètres par ~8-9× vs CNN classique, avec connexions résiduelles sur les blocs à stride 1.

---

## MobileNetV3/

Implémentation de **MobileNetV3-Small** et **MobileNetV3-Large** (Howard et al., 2019).

| Fichier | Rôle |
|---|---|
| `model/mobilenet_base.py` | Classe de base partagée |
| `model/mobilenet_v3_small.py` | Variante Small (architecture tabulée) |
| `model/mobilenet_v3_large.py` | Variante Large |
| `model/LR_ASPP.py` | Lite Reduced ASPP pour segmentation |
| `train_cls.py` | Entraînement classification |
| `test.ipynb` | Tests et benchmarks |

**Nouveautés vs V2 :** blocs SE (Squeeze-and-Excitation), activation h-swish, NAS-optimisé.

---

## MixNet/

Implémentation du package **keras-mixnets** — MixConv (Tan & Le, 2019).

| Fichier | Rôle |
|---|---|
| `keras_mixnets/mixnets.py` | Architecture MixNet-S / MixNet-M / MixNet-L |
| `keras_mixnets/config.py` | Configurations des variantes |
| `keras_mixnets/custom_objects.py` | Couches personnalisées |
| `test.ipynb` | Démonstration |

**Concept clé :** Mixed Depthwise Convolutions — convolutions dépthwise avec **plusieurs tailles de kernel simultanées** (3×3, 5×5, 7×7...) concaténées par canal. Meilleure précision/efficacité que MobileNet.

---

## MixNet2/

Variante expérimentale de MixNet implémentée en notebook (`mixnet.ipynb`), adaptée aux contraintes du projet Go.

---

## Lien avec le projet Go

Ces implémentations ont servi de base directe pour le [`05.ProjetGo/`](../05.ProjetGo/README.md) :

| Modèle référence | Adapté en |
|---|---|
| AlphaZero | `go_basenet.py` |
| MobileNetV2 | `go_mobilenetv2.py` |
| MobileNetV3-Small | `go_mobilenet.py` |
| MixNet | `go_mixnet.py` |
| ResNet | `go_resnet.py` |
