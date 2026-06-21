# Spatial Average Pooling for Computer Go

**Auteur :** Tristan Cazenave — Université Paris-Dauphine, PSL Research University, CNRS, LAMSADE  
**Publication :** IJCAI 2018 (Workshop)

---

## Résumé

Cet article propose une amélioration architecturale de la **tête value** (réseau d'évaluation de position) des programmes de Go basés sur AlphaZero. En remplaçant la tête value standard par des couches de **Spatial Average Pooling** progressif, le réseau peut représenter des probabilités de propriété territoriale et mieux évaluer les positions de fin de partie.

---

## Problématique

Dans AlphaZero, le réseau value prédit directement le résultat de la partie (gagné/perdu) à partir d'une position. Cette architecture utilise des couches entièrement connectées après le tronc, ce qui agrège l'information spatiale de manière brutale.

Or, au Go, la valeur d'une position est directement liée à **qui contrôle quel territoire** — une information intrinsèquement spatiale. Une architecture qui préserve cette hiérarchie spatiale devrait mieux généraliser.

Questions posées :
- Peut-on améliorer la tête value en utilisant du pooling spatial progressif plutôt que Global Average Pooling ?
- Le Spatial Average Pooling aide-t-il le réseau à apprendre des représentations de territoire ?

---

## Concepts appliqués

### Spatial Average Pooling (SAP)
Le SAP applique un pooling moyenné sur une fenêtre de cellules adjacentes :
- SAP 2×2 avec stride 2 sur un plan 19×19 (avec padding 1) → plan 10×10
- Appliqué une seconde fois : plan 10×10 → plan 6×6

Contrairement au **Global Average Pooling** (qui réduit tout à un scalaire par canal), le SAP **préserve la structure spatiale** en réduisant progressivement la résolution.

### Architecture SAP(6,3,3)
La tête value utilisant SAP est nommée SAP(6,3,3) :
```
6 blocs résiduels (128 filtres)
→ SAP 2×2 stride 2 → plans 10×10
→ 3 blocs résiduels (128 filtres)
→ SAP 2×2 stride 2 → plans 6×6
→ 3 blocs résiduels (128 filtres)
→ Flatten (6×6×128 = 4608)
→ Dense 50 (ReLU)
→ Dense 9 (Sigmoid)    ← 9 sorties pour gérer les jeux handicap
```

### Baseline α(9,256)
Réseau value de référence : 9 blocs résiduels, 256 filtres, architecture standard.

### Justification théorique
Si les neurones de chaque plan représentent la probabilité qu'une intersection appartienne à Black en fin de partie, alors moyenner ces probabilités par fenêtres spatiales donne une estimation de la valeur globale de la position. SAP **force** le réseau à développer des représentations de propriété territoriale.

---

## Protocole expérimental

### Entraînement du réseau policy
- Policy network de Golois : 9 blocs résiduels, 3 sorties (3 prochains coups)
- Entraîné sur parties professionnelles 1900-2015
- Niveau atteint : 4-dan sur KGS

### Génération des données d'auto-jeu
- La policy joue contre elle-même (randomisée : coups avec proba > best_proba - 0.2)
- 1 600 000 parties auto-jouées générées
- Label value : résultat final de chaque auto-partie

### Comparaison
- Époque = 5 000 000 exemples, batch 50
- Même set d'auto-parties pour les deux réseaux
- Évaluation : 200 parties PUCT vs PUCT (constant c = 0.3, 40-80 descentes/coup, 4 GPU, 12 threads)

---

## Résultats

### Évolution de la loss d'entraînement
| Epochs | α(9,256) loss | SAP(6,3,3) loss |
|---|---|---|
| 1 | 677 | **654** |
| 3 | 560 | **554** |
| 7 | 532 | **530** |
| 15 | 522 | 521 |
| 31 | 516 | 515 |
| 63 | 510 | **511** |

SAP démarre avec une loss plus faible mais les deux réseaux convergent vers des valeurs similaires après 63 epochs.

### Force de jeu (résultats)
Les expériences montrent que SAP améliore la qualité d'évaluation, notamment pour les positions de fin de partie où le territoire est décisif. La représentation spatiale apprise par SAP est plus cohérente avec la logique du Go.

### Conclusion
> SAP apporte une convergence plus rapide en début d'entraînement et aide le réseau value à développer des représentations spatiales cohérentes avec le concept de territoire au Go. C'est une alternative au Global Average Pooling qui respecte mieux la structure spatiale du problème.

---

## Lien avec notre projet

Cet article a influencé le design de la **tête value** dans notre projet. Notre tête value utilise un Global Average Pooling (basé sur les travaux Cazenave 2021 de MobileNets), qui est conceptuellement lié à SAP — les deux s'appuient sur l'idée que l'agrégation spatiale aide la tête value à représenter la probabilité de victoire de manière géographiquement cohérente avec le plateau de Go.
