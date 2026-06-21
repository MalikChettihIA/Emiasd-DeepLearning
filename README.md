# Emiasd — Deep Learning

Cours et projet de Deep Learning — Master IASD (IA & Science des Données), Paris-Dauphine / PSL.

## Structure du dépôt

```
.
├── 01.Cours/           Cours magistraux (slides PDF)
├── 02.TP/
│   ├── Module Cazenave/    TPs 1–6 (réseaux convolutifs, Go, AlphaZero)
│   └── Module Verrine/     TPs 1–5 (MNIST, segmentation, CNN)
├── 03.Models/          Implémentations de référence
│   ├── AlphaZero/      Architecture AlphaZero (config + modèle)
│   ├── MixNet/         MixConv — convolutions multi-kernel (Keras)
│   ├── MixNet2/
│   ├── MobileNetV2/    MobileNetV2 from scratch
│   └── MobileNetV3/    MobileNetV3 (Large + Small)
├── 04.Others/          Articles de recherche, notes, ressources complémentaires
├── 05.ProjetGo/        Projet principal
│   ├── go_*.py         Code source — modèles et entraînement
│   ├── go_test000XX*   Scripts d'expériences (ablation, hyperparamètres)
│   ├── Backups/        Snapshots de travail datés (mars → juillet 2025)
│   └── Rapport .docx   Rapport de projet final
└── 06.Livres/          Livres et articles (Deep Learning + Go)
```

## Projet Go — `05.ProjetGo/`

Application du Deep Learning au jeu de Go, en s'appuyant sur le moteur **Golois**.

### Modèles implémentés

| Fichier | Architecture |
|---|---|
| `go_basenet.py` | Réseau de base (conv + batchnorm) |
| `go_mobilenet.py` | MobileNet adapté Go |
| `go_mobilenetv2.py` | MobileNetV2 adapté Go |
| `go_resnet.py` | ResNet adapté Go |
| `go_mixnet.py` | MixNet adapté Go |

### Entraînement

```bash
python go_train.py
```

Les expériences d'ablation sont numérotées `go_test000XX-*.py` :

| Script | Variable testée |
|---|---|
| `go_test00000` | Configuration de base / régularisation |
| `go_test00001` | Comparaison d'architectures |
| `go_test00002` | Batch size |
| `go_test00003` | Nombre de samples (N) |
| `go_test00004` | Nombre de blocs |
| `go_test00005` | Poids de la loss Policy |
| `go_test00006` | Poids de la loss Value |

### Dépendances

- Python 3.9
- TensorFlow / Keras
- `golois.cpython-39-darwin.so` — module C++ d'interface avec le moteur Go (inclus)
- Weights & Biases (`wandb`) pour le suivi des expériences

## Références clés

- [MobileNetworks for Computer Go](04.Others/MobileNetworksForComputerGo.pdf)
- [Improving Model and Search for Computer Go](04.Others/ImprovingModelAndSearchForComputerGo.pdf)
- [Cosine Annealing + MixNet + Swish for Computer Go](04.Others/CosineAnnealingMixnetAndSwishActivationForComputerGo.pdf)
- [AlphaGo Zero — Nature](04.Others/nature24270.epdf)
