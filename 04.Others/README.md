# 📎 04 — Additional Resources

![Python](https://img.shields.io/badge/Python-3.9-3776AB?style=flat&logo=python&logoColor=white) ![Jupyter](https://img.shields.io/badge/-Jupyter-F37626?style=flat&logo=jupyter&logoColor=white) ![Research](https://img.shields.io/badge/Papers-IEEE%20%7C%20Nature%20%7C%20AAAI-lightgrey?style=flat)

---

Exploration scripts, foundational research papers, and working notes complementing the course.

## Notebooks & scripts

### 01_Machine_Learnia/ — RNN from scratch

Implementation of a Recurrent Neural Network from scratch, without a framework.

| File | Content |
|---|---|
| `01_RNN_FromScratch.ipynb` | RNN forward/backward pass implemented in pure NumPy |
| `_01_rnn_binary_logreg.py` | Binary logistic regression with RNN |
| `_02_dog_cats_application.py` | Cat/dog classification application |
| `utilities.py` | Utility functions (activation, loss, initialization) |
| `datasets/trainset.hdf5` | Training dataset |
| `datasets/testset.hdf5` | Test dataset |

### 02_gradient_vanishing/ — Gradient Vanishing

Study of the vanishing gradient phenomenon and its fixes.

| File | Content |
|---|---|
| `Gradient_vanishing_1_visualization.ipynb` | Visualizing the gradient across deep layers |
| `gradient_vanishing_2_fixing.ipynb` | Fixes: BatchNorm, ReLU, He/Xavier initialization, skip connections |

## Research papers

Key papers read and referenced during the course and the project.

| Paper | Authors | Venue |
|---|---|---|
| `AlphagoMatapli.pdf` | Silver et al. | Nature 2016 — original AlphaGo |
| `nature24270.epdf` | Silver et al. | Nature 2017 — AlphaGo Zero (no human data) |
| `science.aar6404` | Silver et al. | Science 2018 — AlphaZero (Go + Chess + Shogi) |
| `resnet.pdf` | He et al. | CVPR 2016 — Deep Residual Networks |
| `sap.pdf` | Cazenave | IJCAI 2018 — Spatial Average Pooling for Computer Go |
| `MobileNetworksForComputerGo.pdf` | Cazenave | IEEE ToG 2021 — MobileNet for Go |
| `ImprovingModelAndSearchForComputerGo.pdf` | Cazenave | IEEE CoG 2021 — Model + search improvements |
| `CosineAnnealingMixnetAndSwishActivationForComputerGo.pdf` | Cazenave et al. | ACG 2021 — Cosine Annealing + MixNet + Swish |

## Working notes

| File | Content |
|---|---|
| `Notes Rapports.rtf` | Personal reading/course notes |
| `Test à faire.rtf` | List of experiments to run |
| `ResultatTestGo.docx` | Results table from the first Go experiments |
