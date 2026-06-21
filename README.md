<div align="center">

# Deep Learning — Master IASD

**Paris-Dauphine · PSL · CNRS**

*Cours, travaux pratiques et projet de recherche en Deep Learning appliqué au jeu de Go*

---

![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-FF6F00?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-3.9-D00000?logo=keras&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?logo=jupyter&logoColor=white)
![WandB](https://img.shields.io/badge/Weights%20&%20Biases-tracking-FFBE00?logo=weightsandbiases&logoColor=black)
![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M4%20Pro-000000?logo=apple&logoColor=white)

</div>

---

## Contexte

Cours de Deep Learning du Master **IASD** (Intelligence Artificielle & Science des Données), assuré par :

- **Alexandre Verine** — IA Research Fellow, Centre des Données, ENS PSL. Spécialiste en IA Générative pour l'image.
- **Tristan Cazenave** — Professeur, LAMSADE, Paris-Dauphine–PSL. Chaire PRAIRIE, éditeur en chef de l'ICGA, expert en jeux de stratégie.

### Objectifs pédagogiques

> Comprendre les fondamentaux du Deep Learning et savoir entraîner des réseaux de neurones sur des données variées avec Keras et PyTorch.

**Compétences développées :**
- Architecture et entraînement de réseaux de neurones (MLP → CNN → ResNet → MobileNet)
- Modèles génératifs : Autoencoders, VAE, GAN
- Application à la vision par ordinateur et aux jeux de stratégie
- Expérimentation systématique et ablation d'hyperparamètres

---

## Structure du dépôt

| Dossier | Contenu |
|---|---|
| [`01.Cours/`](01.Cours/README.md) | Slides et syllabus du cours |
| [`02.TP/`](02.TP/README.md) | 11 travaux pratiques (Keras + PyTorch) |
| [`03.Models/`](03.Models/README.md) | Implémentations de référence (AlphaZero, MobileNet, MixNet, ResNet) |
| [`04.Others/`](04.Others/README.md) | Articles de recherche, notes et ressources |
| [`05.ProjetGo/`](05.ProjetGo/README.md) | **Projet principal** — Deep Learning appliqué au jeu de Go |
| [`06.Livres/`](06.Livres/README.md) | Livres et articles fondateurs |

---

## Projet phare — Go & Deep Learning

Le projet consiste à entraîner des réseaux de neurones profonds pour jouer au **jeu de Go (9×9)**, en s'inspirant directement d'AlphaZero. Le moteur d'évaluation **Golois** (module C++ interfacé en Python) fournit les données d'entraînement.

Plusieurs architectures ont été implémentées et comparées : **BaseNet, ResNet, MobileNet, MobileNetV2, MixNet**. L'entraînement est suivi avec **Weights & Biases** et une campagne d'ablation systématique a été conduite sur 7 axes (régularisation, batch size, nombre de blocs, poids des losses Policy/Value...).

➜ Voir [`05.ProjetGo/`](05.ProjetGo/README.md) pour le détail complet.

---

## Stack technique

```
Frameworks    TensorFlow 2.16 / Keras 3.9  (module Cazenave)
              PyTorch 2.x                   (module Verine)
Suivi         Weights & Biases (wandb)
Matériel      Apple M4 Pro — GPU Metal
Environnement Python 3.9 / conda
```
