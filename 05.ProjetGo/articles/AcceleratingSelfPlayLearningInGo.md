# Accelerating Self-Play Learning in Go (KataGo)

**Auteur :** David J. Wu — Jane Street Group  
**Publication :** arXiv:1902.10565v5, novembre 2020

---

## Résumé

Cet article décrit **KataGo**, un programme de Go par auto-jeu (zero learning) qui atteint le niveau de ELF OpenGo après seulement 19 jours sur 28 GPUs V100, contre 2 semaines sur 2000 GPUs pour ELF et plusieurs jours sur 5000 TPUs pour AlphaZero original. Soit un facteur **50× d'accélération**. L'article présente les techniques algorithmiques et architecturales responsables de cette efficacité.

---

## Problématique

AlphaZero et ses réplications (ELF, Leela Zero) nécessitent des ressources de calcul colossales pour atteindre le niveau superhuman. Comment rendre l'apprentissage par auto-jeu beaucoup plus efficace, sans utiliser de connaissance humaine préexistante ?

Contributions principales :
1. Techniques **générales** (transférables à d'autres problèmes)
2. Techniques **domaine-spécifiques** au Go

---

## Concepts appliqués

### Architecture de base
Réseau convolutif résiduel avec architecture **préactivation** (batch norm + activation avant la conv). Deux têtes :
- **Policy head** : prédit les bons coups
- **Game outcome value head** : prédit gagné/perdu

Formule PUCT de KataGo :
```
PUCT(c) = V(c) + c_PUCT × P(c) × sqrt(Σ_{c'} N(c')) / (1 + N(c))
```
avec c_PUCT = 1.1 et bruit de Dirichlet α = 0.03 × 19²/N.

### 3.1 Playout Cap Randomization
**Problème :** tension entre policy training (besoin de nombreux playouts par coup pour bonne exploration) et value training (besoin de nombreuses parties pour des labels non-bruités).

**Solution :** sur une proportion *p* des coups, faire une recherche complète avec *N* playouts (pour la policy). Sur les autres coups, faire une recherche rapide avec *n << N* playouts (pour la value). Seuls les coups à recherche complète entrent dans les targets de policy.

KataGo : p = 0.25, (N, n) = (600, 100) → (1000, 200) après 2 jours.

### 3.2 Policy Target Pruning + Forced Playouts
**Problème :** avec le bruit Dirichlet, de mauvais coups reçoivent des playouts forcés → le réseau policy apprend à prédire ces bad moves.

**Solution :** 
- *Forced playouts* : garantir à chaque enfant un minimum de playouts proportionnel à sa prior policy
- *Policy target pruning* : soustraire les playouts forcés de la distribution target, ne garder que ce que PUCT aurait naturellement choisi

Résultat : découplage de la policy target des dynamiques d'exploration de MCTS.

### 3.3 Global Pooling Bias Structure
**Problème :** les convolutions ont un rayon de perception limité et ne peuvent pas raisonner sur le contexte global du plateau (ex. situations de ko).

**Solution :** couches de global pooling insérées dans le réseau, dont les sorties biaisent les couches convolutives :

```
X (spatial features b×b×c_X)
G (global features b×b×c_G)
→ BN+ReLU(G) → GlobalPool → 3c_G valeurs (mean, mean×width, max)
→ Dense → c_X valeurs
→ Addition avec X
```

Ces couches permettent aux convolutions de conditionner leurs activations sur la situation globale du plateau.

### 3.4 Auxiliary Policy Targets
Idée reprise de l'apprentissage supervisé : entraîner le réseau à prédire simultanément les coups des positions futures (pas seulement le coup courant). Améliore la policy notamment en début de partie.

### 3.5 Ownership and Score Targets (domaine-spécifique)
- **Ownership target** : probabilité que chaque intersection soit territoire Black à la fin
- **Score target** : score final estimé
Ces cibles auxiliaires aident le réseau à développer une compréhension du territoire, similaire à ce que SAP cherche à induire architecturalement.

---

## Protocole expérimental

- Run principal : 19 jours, maximum 28 GPUs V100 (~1.4 GPU-years)
- 241 millions d'exemples d'entraînement, 4.2 millions de parties
- Stochastic weight averaging : snapshots tous les 250 000 exemples, moving average avec decay 0.75
- Gating : un nouveau réseau remplace le courant s'il gagne ≥100/200 parties test

---

## Résultats

### Efficacité computationnelle
| Système | Ressources | Durée |
|---|---|---|
| AlphaZero (Go) | 5 000 TPUs | plusieurs jours (~41 TPU-years) |
| ELF OpenGo | 2 000 V100 GPUs | 13-14 jours (~74 GPU-years) |
| Leela Zero | distribué, années | multi-year projet |
| **KataGo** | **28 V100 GPUs** | **19 jours (~1.4 GPU-years)** |

Facteur d'accélération : **~50× vs ELF**.

### Ablation studies (Section 5.2)
- Playout cap randomization : +fort impact en début de run
- Global pooling : +fort impact en milieu et fin de run
- Policy target pruning : amélioration constante
- Chaque technique contribue indépendamment

### Conclusion
> KataGo démontre qu'une combinaison de techniques algorithmiques (playout cap randomization, policy pruning) et architecturales (global pooling bias) permet d'atteindre le niveau superhuman au Go avec 50× moins de calcul qu'AlphaZero.

---

## Lien avec notre projet

KataGo est la source de données utilisée dans les articles de Cazenave (ImprovingModel, CosineAnnealing). Le dataset KataGo sur lequel nos modèles comparables sont entraînés est constitué de parties auto-jouées avec ces techniques. Le concept de **global pooling** a inspiré la tête value avec Global Average Pooling dans notre architecture MobileNet. La technique d'**ownership target** est conceptuellement liée au SAP de Cazenave. Notre projet utilise une approche supervisée (Golois fournit les données), mais les architectures sont directement informées par KataGo.
