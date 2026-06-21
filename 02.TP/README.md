<div align="center">

# 02 — Travaux Pratiques

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-FF6F00?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-3.9-D00000?logo=keras&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?logo=jupyter&logoColor=white)
![Apple Silicon](https://img.shields.io/badge/GPU-Metal%20M4%20Pro-000000?logo=apple&logoColor=white)

</div>

11 travaux pratiques organisés en deux modules. Le module Cazenave utilise **Keras/TensorFlow**, le module Verine utilise **PyTorch**.

---

## Module Cazenave — Keras / TensorFlow

### [TP01 — Dense Networks](Module%20Cazenave/Cazenave_TP01.ipynb)
Premiers réseaux de neurones entièrement connectés.

| Exercice | Dataset | Tâche |
|---|---|---|
| Réseau XOR | Synthétique | Classification binaire (non-linéairement séparable) |
| MLP 512 → 10 | MNIST | Classification 10 classes — 98%+ accuracy |
| MLP 16 → 16 → 1 | IMDB (50k reviews) | Analyse de sentiment — régularisation L2 + Dropout |
| MLP 128 → 46 | Reuters | Classification multiclasse (46 topics) |
| MLP 64 → 64 → 1 | Boston Housing | Régression — validation croisée k-fold |

---

### [TP02 — Convolutional Networks](Module%20Cazenave/Cazenave_TP02.ipynb)
Introduction aux réseaux de convolution.

| Exercice | Dataset | Architecture |
|---|---|---|
| CNN 2 couches | MNIST | Conv2D(32) → Conv2D(64) → Dense — 99% accuracy |
| CNN profond + Dropout | CIFAR-10 | 5× Conv2D(64) → Dense(64) — ~59% accuracy |

---

### [TP03 — Residual Networks](Module%20Cazenave/Cazenave_TP03.ipynb)
Connexions résiduelles et Functional API de Keras.

| Exercice | Dataset | Architecture |
|---|---|---|
| CNN Functional API | MNIST | Input → Reshape → Conv2D → Flatten → Dense |
| ResNet simple | CIFAR-10 | Skip connections manuelles — ~43% accuracy |
| ResNet 5 blocs | CIFAR-10 | 5× (Conv → Conv → Add) — ~66% accuracy |

---

### [TP04 — Autoencoders & VAE](Module%20Cazenave/Cazenave_TP04.ipynb)
Modèles génératifs non-supervisés.

| Exercice | Dataset | Architecture |
|---|---|---|
| Autoencoder convolutif | Fashion MNIST | Encoder CNN → Latent → Decoder ConvTranspose |
| Autoencoder débruiteur | Fashion MNIST | Entraînement sur images bruitées |
| VAE | Fashion MNIST | Echantillonneur latent + divergence KL |

---

### [TP05 — VAE (suite)](Module%20Cazenave/Cazenave_TP05.ipynb)
Variational Autoencoder complet avec espace latent 2D, visualisation de la distribution latente et génération par interpolation.

---

### [TP06 — Mobile Networks](Module%20Cazenave/Cazenave_TP06.ipynb)
Implémentation des blocs bottleneck de MobileNet en Keras.

| Exercice | Dataset | Architecture |
|---|---|---|
| Bottleneck block | CIFAR-10 | PointwiseConv → DepthwiseConv → PointwiseConv + résiduel |
| MobileNet 5 blocs | CIFAR-10 | Stack de 5 bottleneck blocks + BatchNorm |

---

## Module Verine — PyTorch

### [TP1 — MLP from scratch (PyTorch)](Module%20Verrine/TP1.ipynb)
Implémentation d'un MLP en PyTorch pur, avec loop d'entraînement manuel.

- MNIST — réseau 784 → 64 → 32 → 10
- Backpropagation manuelle (`loss.backward()` + `optimizer.step()`)
- Étude de l'impact du learning rate (SGD) — de 5×10⁻⁴ à 0.9
- Visualisation des courbes de loss train/test
- **Résultat :** 97% accuracy avec lr=5×10⁻²

---

### [TP3 — Autoencoders (PyTorch)](Module%20Verrine/TP3.ipynb) · [Correction](Module%20Verrine/TP3_correction.ipynb)
Étude complète des autoencoders avec sélection d'hyperparamètres.

| Modèle | Architecture | Tâche |
|---|---|---|
| AElin | Linear encoder/decoder | Compression MNIST — impact de la dim latente (2 à 500) |
| SimpleAE | FC 784→256→h→256→784 | Débruitage + détection d'anomalies |
| ConvAE | Conv2D + ConvTranspose2D | Reconstruction haute qualité |

**Applications :** débruitage (σ de 0.01 à 0.5), détection d'anomalies (transposition, artefacts, mélange de patches)

---

### [TP4](Module%20Verrine/TP4.ipynb)

---

### [TP5 — GANs (PyTorch)](Module%20Verrine/TP5.ipynb)
Introduction aux réseaux antagonistes génératifs.

| Exercice | Données | Architecture |
|---|---|---|
| GAN linéaire | 2D Gaussian Mixture | Générateur/Discriminateur linéaires |
| GAN dense | 2D Gaussian Mixture | 4 couches cachées LeakyReLU (256→512→1024) |
| GAN sur MNIST | MNIST 14×14 | Dense GAN — 100 epochs — génération de chiffres |

**Concepts clés :** adversarial training, mode collapse, binary cross-entropy adversariale, équilibre Nash

---

## Données

| Dataset | Taille | Localisation |
|---|---|---|
| MNIST | 60k train / 10k test | `Module Verrine/MNIST/raw/` (téléchargé automatiquement) |
| Segmentation cellulaire | custom | `Module Verrine/nuclei_cells_segmentations.pck` |
| CIFAR-10, IMDB, Reuters, Fashion MNIST | standard | téléchargement automatique via Keras/PyTorch |
