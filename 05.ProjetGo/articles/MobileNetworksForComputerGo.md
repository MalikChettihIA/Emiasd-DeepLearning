# Mobile Networks for Computer Go

**Auteur :** Tristan Cazenave — LAMSADE, Université Paris-Dauphine, PSL, CNRS  
**Publication :** IEEE Transactions on Games, 2021

---

## Résumé

Cet article évalue l'intérêt des architectures MobileNet pour l'apprentissage supervisé du jeu de Go. Il compare les MobileNets aux réseaux résiduels (style AlphaZero) en testant différentes combinaisons de têtes policy et value, en mesurant la précision, la MSE, l'efficacité en paramètres et la vitesse d'inférence.

---

## Problématique

Les réseaux résiduels (ResNet) sont le standard des programmes de jeu de Go depuis AlphaZero. Leur utilisation a apporté +600 ELO par rapport aux CNN classiques. Mais sont-ils optimaux ? Les **MobileNets**, conçus pour la vision par ordinateur avec peu de paramètres, pourraient être plus efficaces pour le Go, tant en vitesse qu'en précision.

Questions posées :
- Les MobileNets surpassent-ils les réseaux résiduels à nombre de paramètres équivalent ?
- Quelle tête policy est la meilleure (classique dense vs entièrement convolutive) ?
- Quelle tête value est la meilleure (AlphaZero vs Global Average Pooling) ?

---

## Concepts appliqués

### Architecture MobileNetV2
- Blocs **inverted residual** : 1×1 pointwise (expand) → 3×3 depthwise → 1×1 pointwise (project)
- Le trunk (épine dorsale) a peu de canaux ; les blocs internes en ont beaucoup
- Convolutions dépthwise : chaque canal convolutionné séparément → beaucoup moins de paramètres

### Têtes policy testées
- **AlphaZero policy head** : 1×1 conv → 2 canaux → flatten → Dense 362
- **Fully convolutional policy head** : 1×1 conv → 1 canal → flatten 361 sorties (sans couche Dense)

### Têtes value testées
- **AlphaZero value head** : 1×1 conv → flatten → Dense 256 → Dense 1
- **Global Average Pooling value head** : GAP sur chaque canal → Dense 50 → Dense 1 (sigmoid)

### Données d'entrée
21 plans 19×19 : couleur à jouer (1 plan), liberties (6 plans), statut ladder (4 plans), 4 états précédents (8 plans), couleur à jouer globale (1 plan), etc.

### Datasets
- **Leela dataset** : 2 000 000 dernières parties auto-jouées de Leela Zero (niveau superhuman)
- **ELF dataset** : 1 347 184 parties auto-jouées de ELF (niveau légèrement inférieur)

---

## Protocole expérimental

- **Époque** = 1 000 000 états tirés aléatoirement
- **Batch** : 32, avec 1 symétrie aléatoire parmi 8 pour chaque sample
- **Scheduler** : learning rate 0.005 (0-100 epochs) → 0.0005 (100-150) → 0.00005 (150-200)
- **Régularisation** : L2 poids 0.0001
- **Validation** : 100 000 parties réservées, 1 état par partie

Réseaux testés :
| Nom | Architecture | Paramètres |
|---|---|---|
| a0.small | 8 blocs résiduels 66 filtres + AlphaZero heads | ~988k |
| a0.small.avg | Même + Global Avg Pooling value | ~987k |
| mobile.small | 25 blocs MobileNet, trunk 64/200 filtres | ~997k |
| mobile.small.conv.avg.bin | Idem + fully-conv policy + GAP value + BCE | ~970k |
| a0.20.256 (unbounded) | 20 blocs résiduels 256 filtres | ~23M |
| mobile.avg.40.128.512 | 40 blocs MobileNet trunk 128/512 | <23M |

---

## Résultats

### Réseaux < 1M paramètres (dataset Leela)
- Le meilleur réseau utilise des **blocs MobileNet + fully convolutional policy + GAP value** : accuracy ~47.5% vs ~42.5% pour a0.small
- La tête value AlphaZero standard échoue souvent à converger sur le dataset Leela avec les petits réseaux
- Ajouter un poids ×4 sur la value loss des MobileNets améliore la MSE value sans dégrader la policy

### Réseaux unbounded
- Les MobileNets (40 blocs) ont une **meilleure accuracy** que les réseaux AlphaZero (20 blocs) avec moins de paramètres (Fig. 3)
- MobileNet est plus rapide en inférence : ~28 batches/sec vs ~22 batches/sec pour AlphaZero de taille équivalente

### Dataset ELF
- Les résultats se confirment : MobileNet entièrement convolutif surpasse AlphaZero avec 3× moins de paramètres

### Conclusion
> Les MobileNets avec tête policy entièrement convolutive et tête value Global Average Pooling **dominent les réseaux résiduels AlphaZero** en précision, MSE et efficacité paramétrique, pour les deux datasets testés.

---

## Lien avec notre projet

Cet article est la **base architecturale directe** du projet Go. Les fichiers `go_mobilenet.py` et `go_mobilenetv2.py` implémentent des variantes de ces architectures en Keras. La comparaison MobileNet vs ResNet menée dans `go_test00001-Models.py` reproduit l'expérience centrale de cet article sur plateau 9×9 avec le moteur Golois.
