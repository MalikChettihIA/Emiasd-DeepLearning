# 🧪 02 — Labs

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-FF6F00?style=flat&logo=tensorflow&logoColor=white) ![Keras](https://img.shields.io/badge/Keras-3.9-D00000?style=flat&logo=keras&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat&logo=pytorch&logoColor=white) ![Jupyter](https://img.shields.io/badge/-Jupyter-F37626?style=flat&logo=jupyter&logoColor=white) ![Apple Silicon](https://img.shields.io/badge/GPU-Metal%20M4%20Pro-000000?style=flat&logo=apple&logoColor=white)

---

11 labs organized into two modules. The Cazenave module uses **Keras/TensorFlow**, the Verine module uses **PyTorch**.

## Cazenave module — Keras / TensorFlow

### [Lab 01 — Dense Networks](Module%20Cazenave/Cazenave_TP01.ipynb)
First fully-connected neural networks.

| Exercise | Dataset | Task |
|---|---|---|
| XOR network | Synthetic | Binary classification (non-linearly separable) |
| MLP 512 → 10 | MNIST | 10-class classification — 98%+ accuracy |
| MLP 16 → 16 → 1 | IMDB (50k reviews) | Sentiment analysis — L2 regularization + Dropout |
| MLP 128 → 46 | Reuters | Multiclass classification (46 topics) |
| MLP 64 → 64 → 1 | Boston Housing | Regression — k-fold cross-validation |

### [Lab 02 — Convolutional Networks](Module%20Cazenave/Cazenave_TP02.ipynb)
Introduction to convolutional networks.

| Exercise | Dataset | Architecture |
|---|---|---|
| 2-layer CNN | MNIST | Conv2D(32) → Conv2D(64) → Dense — 99% accuracy |
| Deep CNN + Dropout | CIFAR-10 | 5× Conv2D(64) → Dense(64) — ~59% accuracy |

### [Lab 03 — Residual Networks](Module%20Cazenave/Cazenave_TP03.ipynb)
Residual connections and Keras' Functional API.

| Exercise | Dataset | Architecture |
|---|---|---|
| Functional API CNN | MNIST | Input → Reshape → Conv2D → Flatten → Dense |
| Simple ResNet | CIFAR-10 | Manual skip connections — ~43% accuracy |
| 5-block ResNet | CIFAR-10 | 5× (Conv → Conv → Add) — ~66% accuracy |

### [Lab 04 — Autoencoders & VAE](Module%20Cazenave/Cazenave_TP04.ipynb)
Unsupervised generative models.

| Exercise | Dataset | Architecture |
|---|---|---|
| Convolutional autoencoder | Fashion MNIST | CNN encoder → Latent → ConvTranspose decoder |
| Denoising autoencoder | Fashion MNIST | Trained on noisy images |
| VAE | Fashion MNIST | Latent sampler + KL divergence |

### [Lab 05 — VAE (continued)](Module%20Cazenave/Cazenave_TP05.ipynb)
Full Variational Autoencoder with a 2D latent space, latent distribution visualization, and generation by interpolation.

### [Lab 06 — Mobile Networks](Module%20Cazenave/Cazenave_TP06.ipynb)
Implementing MobileNet bottleneck blocks in Keras.

| Exercise | Dataset | Architecture |
|---|---|---|
| Bottleneck block | CIFAR-10 | PointwiseConv → DepthwiseConv → PointwiseConv + residual |
| 5-block MobileNet | CIFAR-10 | Stack of 5 bottleneck blocks + BatchNorm |

## Verine module — PyTorch

### [Lab 1 — MLP from scratch (PyTorch)](Module%20Verrine/TP1.ipynb)
Implementing an MLP in pure PyTorch, with a manual training loop.

- MNIST — 784 → 64 → 32 → 10 network
- Manual backpropagation (`loss.backward()` + `optimizer.step()`)
- Study of learning rate impact (SGD) — from 5×10⁻⁴ to 0.9
- Train/test loss curve visualization
- **Result: 97% accuracy with lr=5×10⁻²**

### [Lab 3 — Autoencoders (PyTorch)](Module%20Verrine/TP3.ipynb) · [Correction](Module%20Verrine/TP3_correction.ipynb)
Full study of autoencoders with hyperparameter selection.

| Model | Architecture | Task |
|---|---|---|
| AElin | Linear encoder/decoder | MNIST compression — impact of latent dimension (2 to 500) |
| SimpleAE | FC 784→256→h→256→784 | Denoising + anomaly detection |
| ConvAE | Conv2D + ConvTranspose2D | High-quality reconstruction |

**Applications:** denoising (σ from 0.01 to 0.5), anomaly detection (transposition, artifacts, patch mixing)

### [Lab 4](Module%20Verrine/TP4.ipynb)

### [Lab 5 — GANs (PyTorch)](Module%20Verrine/TP5.ipynb)
Introduction to generative adversarial networks.

| Exercise | Data | Architecture |
|---|---|---|
| Linear GAN | 2D Gaussian Mixture | Linear generator/discriminator |
| Dense GAN | 2D Gaussian Mixture | 4 hidden LeakyReLU layers (256→512→1024) |
| GAN on MNIST | MNIST 14×14 | Dense GAN — 100 epochs — digit generation |

**Key concepts:** adversarial training, mode collapse, adversarial binary cross-entropy, Nash equilibrium

## Data

| Dataset | Size | Location |
|---|---|---|
| MNIST | 60k train / 10k test | `Module Verrine/MNIST/raw/` (downloaded automatically) |
| Cell segmentation | custom | `Module Verrine/nuclei_cells_segmentations.pck` |
| CIFAR-10, IMDB, Reuters, Fashion MNIST | standard | downloaded automatically via Keras/PyTorch |
