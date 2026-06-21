# Improving Model and Search for Computer Go

**Auteur :** Tristan Cazenave — LAMSADE, Université Paris-Dauphine, PSL, CNRS  
**Publication :** IEEE Conference on Games (CoG), 2021

---

## Résumé

Cet article prolonge les travaux sur les MobileNets pour le Go en y ajoutant le mécanisme **Squeeze and Excitation (SE)**, en explorant systématiquement le compromis profondeur/largeur des réseaux, et en proposant une généralisation de l'algorithme de recherche PUCT appelée **GPUCT**.

---

## Problématique

Les MobileNets avec têtes convolutives ont montré leur supériorité sur les ResNet AlphaZero. Deux questions restent ouvertes :

1. Peut-on encore améliorer les MobileNets en ajoutant un mécanisme d'**attention par canal** (Squeeze and Excitation) ?
2. L'algorithme **PUCT** (utilisé dans AlphaZero) est-il optimal, ou peut-on le généraliser pour mieux explorer ?

---

## Concepts appliqués

### Squeeze and Excitation (SE)
Bloc d'attention par canal inséré à la fin de chaque bloc MobileNet **avant l'addition résiduelle** :

```python
def SE_Block(t, filters, ratio=16):
    se = GlobalAveragePooling2D()(t)
    se = Reshape((1, 1, filters))(se)
    se = Dense(filters // ratio, activation='relu', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', use_bias=False)(se)
    x = multiply([t, se])
    return x
```

Le bloc apprend à **recalibrer l'importance de chaque canal** selon le contexte global du plateau.

### Dataset KataGo
- **1 000 000** parties auto-jouées les plus récentes de KataGo (le programme Go open-source le plus fort)
- Qualité bien supérieure aux datasets Leela et ELF précédemment utilisés
- Validation : 100 000 parties tirées aléatoirement, 1 état par partie

### Profondeur vs Largeur
L'article explore une grille de combinaisons (depth × width) pour les MobileNets SE :
- depth ∈ {16, 32, 48, 64, 80}
- width (trunk) ∈ {64, 96, 128, 160, 192, 224, 256}
- Résultats compilés dans des tableaux de Pareto (accuracy vs vitesse GPU)

### GPUCT — Generalized PUCT
Remplacement de la racine carrée dans la formule PUCT par une exponentielle :

```
V(s,a) = Q(s,a) + c × P(s,a) × e^(τ × log(N(s))) / (1 + N(s,a))
```

- τ = 0.5 → PUCT classique
- Paramètres optimaux trouvés : **τ = 0.737**, **c = 0.057**
- Algorithme d'optimisation sur les données de Table I : `argmin_{τ,c}(Σ_d |c×e^(τ×log(d)) - c_d×e^(0.5×log(d))|)`

---

## Protocole expérimental

- 250 epochs d'entraînement sur dataset KataGo
- Epoch = 1 000 000 états, batch 32, L2 = 0.0001
- Annealing LR : 0.0005 → 0.00005 → 0.000005 → 0.0000005 (paliers de 50 epochs)
- Évaluation GPUCT vs PUCT : 400 parties par configuration, budgets de 16 à 2048 descentes

---

## Résultats

### Précision — Mobile+SE vs réseau résiduel
| Architecture | Accuracy (%) | MSE |
|---|---|---|
| AlphaZero (résiduel) | ~45% | ~0.20 |
| Mobile (sans SE) | ~53% | ~0.18 |
| Mobile+SE | **~57%** | **~0.16** |
| Polygames | ~50% | ~0.19 |

Le réseau **se.64.1152.192** (64 blocs, 1152 filtres internes, trunk 192) atteint **65%+ accuracy** — bien au-dessus des réseaux résiduels à vitesse équivalente.

### Compromis Profondeur / Largeur
- Les réseaux très profonds et étroits **ou** très larges et peu profonds sont dominés (Pareto front)
- Ratio optimal width/depth ≈ **2.67 à 6.00**
- Augmenter conjointement profondeur et largeur est la meilleure stratégie

### GPUCT vs PUCT
- Pour de petits budgets (≤ 256 descentes) : GPUCT ≈ PUCT
- Pour grands budgets (1024-2048 descentes) : **GPUCT gagne 68-73% contre PUCT à 0.15**
- GPUCT corrige le drift du meilleur constant PUCT quand le budget augmente

### Front de Pareto (GPU speed vs accuracy)
Les réseaux dominés incluent les résiduels (20 et 40 blocs, 256 filtres). Les MobileNets SE dominent largement, tant pour la policy que pour la value.

---

## Lien avec notre projet

Cet article a motivé l'intégration du bloc **Squeeze and Excitation** dans `go_mobilenet.py`. Le script `go_test00004-BlockNum.py` explore l'axe profondeur des blocs, reproduisant partiellement l'étude depth/width de cet article. L'architecture MobileNet+SE est le modèle le plus sophistiqué testé dans notre campagne d'ablation.
