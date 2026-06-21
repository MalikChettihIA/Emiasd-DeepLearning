# Polygames: Improved Zero Learning

**Auteurs :** Tristan Cazenave, Yen-Chi Chen, Guan-Wei Chen, Shi-Yu Chen, Xian-Dong Chiu, Julien Dehos, Maria Else, et al. — Facebook AI Research + LAMSADE  
**Publication :** arXiv:2001.09832v1, janvier 2020

---

## Résumé

Polygames est un framework open-source de **zero learning** (apprentissage de jeux depuis zéro, sans connaissance humaine) développé par Facebook AI Research et plusieurs partenaires académiques. Il généralise AlphaZero à de nombreux jeux grâce à des architectures entièrement convolutives et board-size invariantes, un mécanisme de neuroplasticité pour faire croître les réseaux pendant l'entraînement, et un mode tournoi pour combattre l'oubli catastrophique. Il a battu des humains experts au Hex 19×19 et remporté plusieurs compétitions TAAI 2019.

---

## Problématique

AlphaZero est puissant mais limité à un jeu et une taille de plateau fixe. Peut-on construire un framework zero learning **générique** capable de :
1. Jouer à de nombreux jeux différents sans modification d'architecture ?
2. Être efficace sur des jeux réputés difficiles pour les ordinateurs (Hex 19×19, Havannah) ?
3. Éviter l'oubli catastrophique et les oscillations de performance pendant l'auto-jeu ?

---

## Concepts appliqués

### 2.1 Modèles entièrement convolutifs (Fully Convolutional)

**Observation :** au Go, la policy (meilleur coup) est naturellement spatiale — c'est une carte de probabilités sur le plateau. Utiliser des couches entièrement connectées pour la policy head brise l'invariance de taille de plateau.

**Solution :** supprimer toutes les couches fully connected :
- **Policy head** : entièrement convolutif → sortie = plan board_size × board_size × nb_actions
- Applicable à n'importe quelle taille de plateau sans modification

### 2.2 Modèles invariants à la taille (Scale-Invariant)

**Problème :** la value head nécessite une représentation globale qui n'est pas naturellement convolutive.

**Solution :** utiliser le **Global Pooling** à la place des FC layers :
- Pour chaque canal c de taille board_size × board_size, calculer : max(c) et mean(c)
- Représentation de taille 2×nb_channels, indépendante de la taille du plateau

Résultat : un modèle entraîné sur Hex 13×13 est **immédiatement compétitif sur Hex 19×19** sans réentraînement.

### 2.3 Neuroplasticité

Polygames permet de **faire croître le réseau pendant l'entraînement** :
- Ajouter des blocs résiduels
- Ajouter des canaux
- Augmenter la taille des noyaux (3×3 → 5×5 → 7×7)

Ces modifications sont **neutres** à l'initialisation (nouveaux poids proches de 0) → le réseau étendu produit les mêmes sorties que le petit réseau, puis apprend progressivement à utiliser la capacité supplémentaire. L'outil `convert` de Polygames automatise cette transition.

### 2.4 Mode Tournoi

**Problème :** en zero learning, le réseau peut osciller (red queen effect) ou oublier des positions clés (catastrophic forgetting).

**Solution :** maintenir un **pool de 10 modèles** avec scores ELO calculés entre eux :
- À chaque save, le pire modèle du pool est retiré
- Chaque client joue contre le modèle courant ("dev") et contre un modèle du pool sélectionné avec probabilité proportionnelle à exp(-(ELO_dev - ELO) / 400)
- Assure que le réseau courant reste fort contre toutes les versions précédentes

### Architecture réseau
Réseau dual-head standard :
- Tronc : blocs résiduels 3D (conv sur channels + positions)
- Policy head : entièrement convolutive (1×1 conv → board_size × board_size)
- Value head : global pooling → Dense → scalaire

Inputs : tenseur de l'état du jeu (spécifique à chaque jeu, défini dans une classe Python)

---

## Protocole expérimental

### Framework
- Code client en C++ (MCTS, génération de parties)
- Réseau en Python/PyTorch (serveur d'entraînement)
- Architecture serveur-clients : le serveur reçoit des 3-tuples (state, policy, result) et entraîne le réseau ; les clients génèrent des parties

### Jeux testés
- Hex 11×11, 13×13, 19×19
- Havannah 8×8
- Breakthrough, Othello10, Connect6
- Einstein Würfelt Nicht, MiniShogi...

---

## Résultats

### 3.1 Hex 19×19 — Victoire contre des humains experts
- Premier programme à battre des joueurs humains experts au Hex 19×19
- Victoire décisive dans un match organisé

### 3.2 TAAI 2019
Polygames classé **1er** dans :
- Othello10
- Breakthrough
- Connect6

Testé également contre WZebra et Ltbel (Othello8, vainqueurs TAAI 2018) — Polygames gagne tous les matchs.

### 3.3 Havannah 8×8
Polygames résout le jeu de Havannah 8×8 — un jeu réputé très difficile pour les ordinateurs en raison de la diversité des conditions de victoire.

### Impact de la neuroplasticité
La possibilité de faire grandir le réseau pendant l'entraînement permet d'éviter de relancer l'entraînement depuis zéro quand on augmente la capacité du modèle.

### Impact du mode tournoi
Réduit significativement les oscillations de performance observées dans l'auto-jeu standard.

### Conclusion
> Polygames démontre qu'un framework zero learning **générique et efficace** est possible. L'architecture entièrement convolutive + global pooling + neuroplasticité + mode tournoi constituent un ensemble de techniques robustes, applicables bien au-delà du Go.

---

## Lien avec notre projet

Polygames est développé en partie par Cazenave, le responsable du module dans lequel s'inscrit notre projet. Les concepts de **politique entièrement convolutive** et de **global average pooling** pour la tête value sont directement repris dans nos architectures MobileNet (`go_mobilenet.py`). Le mode tournoi de Polygames est conceptuellement proche de la stratégie d'ablation que nous menons avec W&B — comparer des configurations via des métriques de validation sur des données superhuman (Golois).
