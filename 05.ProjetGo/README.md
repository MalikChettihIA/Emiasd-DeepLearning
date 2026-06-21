<div align="center">

# 05 — Projet Go · Deep Learning

**Application du Deep Learning au jeu de Go (9×9)**

![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-FF6F00?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-3.9-D00000?logo=keras&logoColor=white)
![WandB](https://img.shields.io/badge/Weights%20&%20Biases-tracked-FFBE00?logo=weightsandbiases&logoColor=black)
![Apple Silicon](https://img.shields.io/badge/GPU-Metal%20M4%20Pro-000000?logo=apple&logoColor=white)
![Go](https://img.shields.io/badge/Golois-C%2B%2B%20engine-555555)

</div>

---

## Objectif

Entraîner des réseaux de neurones profonds pour prédire les coups d'un joueur expert au **jeu de Go sur plateau 9×9**, en s'inspirant de l'architecture d'AlphaZero. Le modèle apprend à la fois :
- **Policy** : probabilité de jouer chaque intersection (362 sorties)
- **Value** : estimation de la probabilité de victoire

Les données d'entraînement sont fournies par **Golois**, un moteur C++ compilé comme extension Python.

---

## Architecture du code

```
05.ProjetGo/
├── go_basenet.py          Réseau de base (conv + BN + ReLU × N blocs)
├── go_mobilenet.py        MobileNetV3-Small adapté Go (blocs bottleneck SE)
├── go_mobilenetv2.py      MobileNetV2 adapté Go (blocs inversés)
├── go_resnet.py           ResNet adapté Go (skip connections)
├── go_mixnet.py           MixNet adapté Go (convolutions multi-kernel)
├── go_callbacks.py        Callbacks Keras : sauvegarde, monitoring, plots
├── go_train.py            Script d'entraînement principal
├── training_monitor.py    Suivi en temps réel des métriques
├── go_test_utils.py       Utilitaires pour les expériences
├── go_test_model_summary.py  Résumé et comparaison des architectures
├── wandb.config           Configuration Weights & Biases
└── golois.cpython-39-darwin.so  Moteur Go (module C++ compilé)
```

---

## Modèles implémentés

| Modèle | Fichier | Particularité |
|---|---|---|
| **BaseNet** | `go_basenet.py` | Empilement de blocs Conv+BN+ReLU — baseline simple |
| **MobileNet** | `go_mobilenet.py` | Blocs bottleneck dépthwise + Squeeze-and-Excitation |
| **MobileNetV2** | `go_mobilenetv2.py` | Blocs inversés (expand→depthwise→project) + skip |
| **ResNet** | `go_resnet.py` | Skip connections identité — inspiré He et al. 2016 |
| **MixNet** | `go_mixnet.py` | Convolutions dépthwise multi-kernel (3×3, 5×5, 7×7) |

**Têtes de sortie communes :** tête Policy (softmax 362) + tête Value (sigmoid scalaire), entraînées en multi-tâche.

---

## Campagne d'expériences — Ablation

7 axes d'étude systématique, chacun avec un script dédié :

| Script | Axe étudié | Variable |
|---|---|---|
| `go_test00000.py` | Configuration de base | Dropout, L2, ClipNorm — 4 variantes |
| `go_test00001-Models.py` | Comparaison architectures | BaseNet vs MobileNet vs ResNet vs MixNet |
| `go_test00001-ModelsV2.py` | Comparaison v2 | Variantes affinées |
| `go_test00002-BatchSize.py` | Taille de batch | Impact sur convergence et généralisation |
| `go_test00003-N.py` | Nombre de samples | Volume de données d'entraînement |
| `go_test00004-BlockNum.py` | Profondeur | Nombre de blocs résiduels |
| `go_test00005-PolicyWeight.py` | Poids loss Policy | Équilibre Policy/Value |
| `go_test00006-ValueWeight.py` | Poids loss Value | Équilibre Value/Policy |

Chaque expérience génère des courbes de loss et est trackée sur **Weights & Biases**.

---

## Résultats des tests

```
go_test00000/   Résultats baseline (courbes, plots)
go_test00005/   Résultats test Policy Weight
go_test00006/   Résultats test Value Weight
```

Les runs W&B (Backups/20250429/Test1 à Test10) documentent l'évolution de la loss de validation au fil des 220+ epochs d'entraînement.

---

## Entraînement

```bash
# Lancer un entraînement
python go_train.py

# Lancer une expérience d'ablation
python go_test00001-Models.py
```

**Données requises :**
- `games.data` — parties jouées (non versionné, ~plusieurs GB)
- `validation.data` — données de validation

---

## Suivi expérimental (W&B)

Les runs sont organisés par date dans `Backups/20250429/wandb/`. Métriques suivies :
- `val_policy_loss` — loss de prédiction des coups
- `val_value_loss` — loss d'estimation de victoire
- `val_loss` — loss totale (meilleur run : **~2.55** à l'epoch 240)

---

## Backups — Historique de développement

| Dossier | Contenu |
|---|---|
| `Backups/20250329/` | Premiers notebooks d'exploration (MobileNet, importGolois) |
| `Backups/20250331/` | ProjetV0 et ProjetV1 (première version entraînable) |
| `Backups/20250404/` | AlphaZero lite — self-play simplifié |
| `Backups/20250405–20250407/` | GoMobileNetV2 — premières itérations |
| `Backups/20250426/` | GoMobileNetV2/V3 — versions 0.1 à 0.2 |
| `Backups/20250429/` | Version complète avec tous les modèles + W&B + Tests 1–10 |
| `Backups/20250701/` | Version finale — importGolois v0/v1 |
| `Backups/project2022/` | Source C++ de Golois (Board.h, Game.h, golois.cpp, Makefile) |
| `Backups/project2025/` | Binaires Golois multi-plateforme |

---

## Rapport

`Emiasd - Deep Learning - Rapport Projet.docx` — rapport final documentant la méthodologie, les architectures, la campagne d'ablation et les résultats.
