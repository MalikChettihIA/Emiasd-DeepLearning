# Cosine Annealing, Mixnet and Swish Activation for Computer Go

**Auteurs :** Tristan Cazenave, Julien Sentuc, Mathurin Videau — LAMSADE, Université Paris-Dauphine, PSL, CNRS  
**Publication :** Advances in Computer Games (ACG), 2021

---

## Résumé

Cet article présente trois améliorations indépendantes et cumulatives appliquées aux MobileNets pour le jeu de Go en apprentissage supervisé : un meilleur **scheduler de learning rate** (cosine annealing), des **convolutions dépthwise multi-noyaux** (MixConv), et une **fonction d'activation améliorée** (Swish). Ces trois modifications améliorent conjointement la précision de la policy et la MSE de la value.

---

## Problématique

Les MobileNets surpassent les ResNet pour le Go. Mais leurs performances peuvent encore être améliorées en agissant sur trois leviers orthogonaux :

1. **Optimisation** : le scheduler de learning rate influence la convergence — peut-on faire mieux que diviser le LR à des epochs fixes ?
2. **Architecture convolutive** : les convolutions dépthwise utilisent un seul noyau 3×3 — les noyaux de tailles variées capturent-ils mieux les patterns du Go ?
3. **Non-linéarité** : ReLU est l'activation standard — une activation lisse apporte-t-elle un gain ?

---

## Concepts appliqués

### 2.1 Cosine Annealing (SGDR)

Formule du learning rate au temps *t* pour le cycle *i* :

```
η_t = η_min^i + (1/2)(η_max^i - η_min^i)(1 + cos(T_cur/T_i × π))
```

- Le LR descend en cosinus de η_max à η_min sur T_i steps, puis repart
- Comparé à la **division annealing** (LR divisé par 10 aux epochs 100, 150, 200)
- Un seul cycle sans restart dans les expériences finales

Avantage : courbe d'apprentissage **plus lisse** (pas de sauts brusques), meilleure exploration du paysage de loss.

### 2.2 MixConv (MixNet)

Proposition de Tan et al. (EfficientDet) : remplacer la convolution dépthwise 3×3 par un **mélange de noyaux de tailles différentes** :
- Mix de noyaux 3×3 et 5×5 (la moitié de chaque)
- Même nombre de paramètres, mais capture de patterns à plusieurs échelles
- Permet d'utiliser des noyaux plus grands avec le même coût computationnel qu'un seul 3×3

### 2.3 Swish Activation

Activation proposée par Ramachandran et al. (recherche automatique d'activations) :

```
f(x) = x · sigmoid(x)
```

- **Lisse** (dérivable partout) vs ReLU qui est discontinue en 0
- Légèrement négative pour x < 0 (contrairement à ReLU qui coupe tout)
- Drop-in replacement pour ReLU dans les blocs MobileNet

---

## Protocole expérimental

- **Dataset** : parties auto-jouées de KataGo (label value = Q du MCTS, non le résultat final)
- Mélange convolutif : moitié noyaux 3×3, moitié noyaux 5×5
- Évaluation jouée : 400 parties entre deux réseaux (200 en tant que Black, 200 en tant que White)
- Randomisation des 20 premiers coups selon les probabilités de la policy
- **PUCT constant** : best constant trouvé expérimentalement (0.40 pour 16 blocs, 100 playouts)
- Algorithme de recherche : Batch MCTS

---

## Résultats

### 3.1 Cosine Annealing vs Division Annealing
Sur le réseau MobileNet 16 blocs :
- Cosine annealing termine avec une meilleure accuracy **et** une meilleure MSE
- La courbe de cosine annealing est plus régulière (pas de bumps aux changements de LR)
- Différence mesurée : petite mais consistante — cosine annealing adopté pour toute la suite

### 3.2 Petits réseaux — MixNet + Swish (16 blocs)
| Configuration | Accuracy | MSE |
|---|---|---|
| Cosine 3×3 seulement | ~0.540 | ~0.033 |
| Cosine + MixNet | ~0.555 | ~0.031 |
| Cosine + MixNet + Swish | **~0.570** | **~0.029** |
| Cosine + 5×5 seulement | ~0.555 | ~0.031 |

MixNet ≈ 5×5 en accuracy, mais avec moins de paramètres. Swish apporte un gain supplémentaire.

### 3.3 Grands réseaux — MixNet + Swish (48 blocs)
Même tendance : Mixnet+Swish > Mixnet > 3×3 seul.

### 3.4 Résultats de jeu
| Configuration (16 blocs) | Win rate vs réseau de base |
|---|---|
| MobileNet standard (constant 0.40) | baseline |
| MixNet (constant 0.20) | 0.585 |
| MixNet + Swish (constant 0.40) | **0.695** |

Le meilleur constant pour Mixnet+Swish est 0.40. Le réseau avec Swish gagne **69.5%** des parties contre le réseau sans Mixnet ni Swish.

### 3.5 Conclusion
> Cosine Annealing + MixConv + Swish sont trois améliorations **cumulatives et complémentaires** qui renforcent significativement les MobileNets pour le Go.

---

## Lien avec notre projet

Ces trois techniques ont été intégrées dans les expériences du projet Go :
- **Cosine annealing** : scheduler utilisé dans `go_train.py` (via `CosineAnnealingScheduler`)
- **MixConv** : implémenté dans `go_mixnet.py` avec des convolutions dépthwise multi-kernel (3×3, 5×5, 7×7)
- **Swish** : testé comme alternative à ReLU dans les blocs bottleneck de `go_mobilenet.py`

Le script `go_test00001-Models.py` compare directement MixNet vs MobileNet vs BaseNet, reproduisant la Section 3 de cet article.
