<div align="center">

# 04 — Ressources complémentaires

![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?logo=jupyter&logoColor=white)
![Research](https://img.shields.io/badge/Articles-IEEE%20%7C%20Nature%20%7C%20AAAI-lightgrey)

</div>

Scripts d'exploration, articles de recherche fondateurs et notes de travail complémentaires au cours.

---

## Notebooks & scripts

### 01_Machine_Learnia/ — RNN from scratch

Implémentation d'un RNN (Réseau de Neurones Récurrent) depuis zéro, sans framework.

| Fichier | Contenu |
|---|---|
| `01_RNN_FromScratch.ipynb` | RNN forward/backward pass implémenté en NumPy pur |
| `_01_rnn_binary_logreg.py` | Régression logistique binaire avec RNN |
| `_02_dog_cats_application.py` | Application classification chiens/chats |
| `utilities.py` | Fonctions utilitaires (activation, loss, initialisation) |
| `datasets/trainset.hdf5` | Dataset d'entraînement |
| `datasets/testset.hdf5` | Dataset de test |

---

### 02_gradient_vanishing/ — Gradient Vanishing

Étude du phénomène de disparition du gradient et solutions.

| Fichier | Contenu |
|---|---|
| `Gradient_vanishing_1_visualization.ipynb` | Visualisation du gradient à travers les couches profondes |
| `gradient_vanishing_2_fixing.ipynb` | Solutions : BatchNorm, ReLU, initialisation He/Xavier, skip connections |

---

## Articles de recherche

Articles clés lus et référencés pendant le cours et le projet.

| Article | Auteurs | Conférence |
|---|---|---|
| `AlphagoMatapli.pdf` | Silver et al. | Nature 2016 — AlphaGo original |
| `nature24270.epdf` | Silver et al. | Nature 2017 — AlphaGo Zero (sans données humaines) |
| `science.aar6404` | Silver et al. | Science 2018 — AlphaZero (Go + Chess + Shogi) |
| `resnet.pdf` | He et al. | CVPR 2016 — Deep Residual Networks |
| `sap.pdf` | Cazenave | IJCAI 2018 — Spatial Average Pooling for Computer Go |
| `MobileNetworksForComputerGo.pdf` | Cazenave | IEEE ToG 2021 — MobileNet pour le Go |
| `ImprovingModelAndSearchForComputerGo.pdf` | Cazenave | IEEE CoG 2021 — Amélioration modèle + recherche |
| `CosineAnnealingMixnetAndSwishActivationForComputerGo.pdf` | Cazenave et al. | ACG 2021 — Cosine Annealing + MixNet + Swish |

---

## Notes de travail

| Fichier | Contenu |
|---|---|
| `Notes Rapports.rtf` | Notes personnelles de lecture et de cours |
| `Test à faire.rtf` | Liste d'expériences à conduire |
| `ResultatTestGo.docx` | Tableau de résultats des premières expériences Go |
