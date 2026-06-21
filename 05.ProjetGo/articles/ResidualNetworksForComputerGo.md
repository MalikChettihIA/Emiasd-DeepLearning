# Residual Networks for Computer Go

**Auteur :** Tristan Cazenave — Université Paris-Dauphine, PSL Research University, CNRS, LAMSADE  
**Publication :** IEEE Transactions on Computational Intelligence and AI in Games (TCIAIG)

---

## Résumé

Article fondateur proposant l'utilisation des **réseaux résiduels (ResNet)** pour le jeu de Go en apprentissage supervisé. Publié juste après la victoire d'AlphaGo contre Lee Sedol (mars 2016), il établit que les skip connections accélèrent l'apprentissage, permettent d'entraîner des réseaux plus profonds et améliorent significativement la précision de prédiction des coups.

---

## Problématique

Les réseaux convolutifs profonds (CNN) pour le Go souffrent du **problème du gradient qui disparaît** : plus un réseau est profond, plus il est difficile à entraîner. Les blocs résiduels (He et al., CVPR 2016) résolvent ce problème en ajoutant l'entrée de chaque bloc à sa sortie, créant un chemin direct pour le gradient.

Questions abordées :
- Les réseaux résiduels améliorent-ils la précision de la prédiction de coups pour le Go ?
- Peut-on utiliser une couche résiduelle spéciale à l'entrée du réseau (couche input résiduelle) ?
- Quelle profondeur de réseau est optimale ?

---

## Concepts appliqués

### Couche résiduelle standard
```
Input → Conv → ReLU → Conv → Add(Input) → ReLU → Output
```
La sortie d'un bloc = transformée(entrée) + entrée. Le gradient peut revenir directement par le skip.

### Couche résiduelle d'entrée (input residual layer)
Couche spéciale pour la première couche du réseau :
- Branche 1 : convolution 5×5
- Branche 2 : convolution 1×1
- Somme des deux branches → ReLU

Cette couche capture à la fois des patterns locaux larges (5×5) et ponctuels (1×1).

### Données d'entrée
45 plans 19×19 :
- 3 plans : couleurs des intersections
- Plans de liberties (ami/ennemi, jusqu'à ≥5)
- Plans de ladders (capture, fuite, menace)
- 5 derniers coups joués
- Plans ko, troisième ligne, etc.

### Dataset
- **KGS** : parties jouées entre 2000 et 2014 sur Kiseido Go Server par des joueurs 6+ dan  
  → ~160M positions (avec 8 symétries)
- **GoGoD** : parties professionnelles de 1900 à 2014

### Entraînement
- Framework : Torch (Lua)
- Batch : 50 états aléatoires, 1 symétrie aléatoire par état
- Epoch = 5 000 000 exemples d'entraînement
- LR : 0.2 initial, divisé par 2 si la loss ne décroît plus sur les 1000 derniers steps vs 2000 steps avant
- Output : plan 19×19, 1 pour le coup joué, 0 ailleurs → softmax + entropie croisée

---

## Protocole expérimental

Comparaison entre :
- **Réseau vanilla** : 13 couches CNN, 256 filtres 3×3 (architecture style AlphaGo/DarkForest)
- **ResNet 20 couches** : 20 couches résiduelles, 256 filtres 3×3

Évaluation sur KGS test set (parties de 2015, 500 000 positions) et GoGoD test set.

Bagging : le même réseau est appliqué aux 8 symétries du plateau, les sorties sont moyennées.

Tournament round-robin entre les 10 derniers checkpoints du réseau 28 couches (epochs 70-79) pour sélectionner le meilleur.

---

## Résultats

### Précision sur KGS
| Réseau | Accuracy test | Avec bagging |
|---|---|---|
| Vanilla 13 couches 256 filtres | ~56% | — |
| ResNet 20 couches 256 filtres | 58.2456% | **58.5450%** |

Comparaison : AlphaGo policy network avec bagging atteint 57.0%, DarkForest 57.3%.

### Précision sur GoGoD
- ResNet 20 couches : ~53.5%
- ResNet 28 couches : meilleur checkpoint à 55.0306% (epoch 71 raw), 54.9954% moyen

Passer de 20 à 28 couches : amélioration faible sur GoGoD.

### Golois — Niveau de jeu
- **Golois4** (20 couches + bagging) : niveau 3-dan sur KGS, joue 24h/24 contre de nombreux adversaires
- **Golois6** (28 couches) : niveau 4-dan sur KGS
- Le programme utilise PUCT avec parallélisme d'arbres (12 threads, 4 GPUs K40)

### Conclusion
> Les réseaux résiduels **accélèrent l'apprentissage** et permettent des réseaux plus profonds et plus précis. Un ResNet de 20 couches dépasse AlphaGo et DarkForest sur le même dataset, atteignant 3-dan en jeu réel.

---

## Lien avec notre projet

Cet article est la **fondation historique** de notre projet. Il justifie l'utilisation des skip connections dans `go_resnet.py`. C'est également l'article qui a établi le programme Golois — le moteur C++ (`golois.cpython-39-darwin.so`) que nous utilisons comme source de données. Notre architecture BaseNet (`go_basenet.py`) est l'équivalent simplifié du réseau vanilla de cet article, que nous comparons à nos MobileNets et MixNets.
