---
source_type: pdf
source_id: 06_Transfert Learning.pdf
page_count: 10
---

<!-- loc page=1 -->

UTILISATION DE RÉSEAUX EXISTANTS

SOMMAIRE

1- Quelques réseaux profonds classiques 1

1.1- AlexNet
1.2- VGG
1.3- Inception
1.4- ResNet
1.5- SqueezeNet

2- Comment utiliser ces réseaux 6

2.1- Utilisation de réseaux pré-entraînés
2.2- Transfer learning et fine tuning

3- Que faire si j’ai peu de données ? 7

4- Partie pratique 9

Depuis la fin des années 90, de nombreux réseaux profonds ont vu le jour et se sont complexifié, diversifié,
pour répondre à des problèmes de plus en plus vastes. [1] propose une analyse comparative de ces réseaux
et décrit en particulier leurs performances en fonction du nombre d’opérations (figure 0-1).
Nous présentons dans la suite cinq réseaux profonds classiques. Nous montrons ensuite comment les utiliser
directement, ou comment les adapter pour répondre à une problématique précise, en lien avec leur utilisation
originale ou non. Nous introduisons enfin une manière d’apprendre un réseau à partir de peu de données.

1- QUELQUES RÉSEAUX PROFONDS CLASSIQUES

Les cinq réseaux présentés ici ont prouvé leur efficacité, notamment lors des compétitions organisées de
puis 2010 sur une base de données d’images nommée ImageNet. Initiée à l’Université de Stanford, cette
base de données comporte aujourd’hui plus de 14 millions d’images, classées en 21841 catégories (avions,
voitures, chats,...). Dans les compétitions ILSVRC (ImageNet Large Scale Visual Recognition Challenge),
les chercheurs se voient proposer une extraction de 1,2 millions d’images, catégorisées en 1000 classes, et
le gagnant est celui qui atteint la meilleure précision de reconnaissance sur les 5 premières classes (top-5).

<!-- loc page=2 -->

Inception-v3
ResNet-50
ResNet-101
ResNet-34
VGG-16
VGG-19
ResNet-18
GoogLeNet
ENet
BN-NIN
5M 35M 65M 95M 125M 155M
BN-AlexNet
AlexNet
Operations [G-Ops]

--- [FIGURE] ---
FIGURE 0-1 – Précision en fonction du nombre d'opérations nécessaire pour un calcul en passe avant. La taille des blobs est proportionnelle au nombre de paramètres du réseau (source : [1])
Cette figure est un diagramme de dispersion montrant la relation entre la précision Top-1 exprimée en pourcentage sur l'axe vertical et le nombre d'opérations nécessaires au calcul en passe avant sur l'axe horizontal, mesurées en G-Ops (Gigaoperations). Les points sont représentés par des cercles colorés dont la taille est proportionnelle au nombre de paramètres du réseau. L'axe vertical s'étend de 50 % à 80 % avec des graduations visibles, tandis que l'axe horizontal présente des marques discrètes étiquetées « 5M », « 35M », « 65M », etc., bien que le titre indique « Operations [G-Ops] ». Chaque cercle correspond à un modèle spécifique : Inception-v4 (bleu clair), ResNet-152 (rose), VGG-16 (vert foncé), VGG-19 (vert clair), ResNet-50 (rose), ResNet-34 (violet), ResNet-18 (pourpre), GoogLeNet (bleu foncé), ENet (noir), BN-NIN (rouge), BN-AlexNet (orange) et AlexNet (jaune). Les modèles sont disposés de manière à refléter leur complexité : ceux situés à gauche, comme AlexNet ou BN-AlexNet, présentent des valeurs basses en opérations et précision, tandis que les modèles à droite, tels que VGG-19 ou ResNet-152, nécessitent davantage d'opérations mais affichent une précision plus élevée. La taille des cercles permet de visualiser immédiatement le nombre de paramètres : par exemple, VGG-16 et VGG-19 ont des cercles larges, indiquant un grand nombre de paramètres, contrairement à AlexNet dont le cercle est petit. Sur le plan pédagogique, cette figure illustre clairement le compromis entre la complexité computationnelle (mesurée par les opérations) et la performance du modèle (précision Top-1), soulignant que l'augmentation de la taille des réseaux entraîne généralement une amélioration de la précision mais à un coût en ressources matérielles accru, ce qui est essentiel pour comprendre les choix architecturaux dans le développement d'algorithmes d'apprentissage automatique.
--- [/FIGURE] ---

1.1- AlexNet

En 2012, Krizhevsky et al [5] remportent ILSVRC avec un taux de reconnaissance de 84.6%, en utilisant AlexNet, un réseau convolutif composé de 5 couches de convolution et de pooling, suivies de 3 couches complètement connectées (figure 1-2).

--- [FIGURE] ---
FIGURE 1-2 – Architecture du réseau AlexNet. Les couches de convolution et d’activation sont en orange clair, les couches d’agrégation en orange foncé. Les couches complètement connectées sont en violet.
La figure présente un schéma architectural détaillé du réseau AlexNet, illustrant le flux de données à travers ses différentes couches. Il s'agit d'un diagramme en perspective avec des blocs rectangulaires colorés : les couches de convolution et d’activation sont représentées en orange clair, tandis que les couches d’agrégation (comme le pooling) apparaissent en orange foncé, et les couches complètement connectées sont en violet. À gauche, une couverture de livre intitulée « Apprentissage artificiel » est visible avec des annotations en français incluant le mot « Deep learning », ainsi que des dimensions numériques comme 55, 27, 13 et des étiquettes précises : « K 5×5 », « stride 4 », « 11×11 », « 96 », « 256 », « 384 », « K 3×3 », « 1 FCC tanh », « 4096 » et « Softmax ». Des flèches horizontales reliant chaque bloc indiquent le parcours des données, tandis que des lignes pointillées relient certaines parties à la couverture du livre. Les dimensions numériques (comme 55×55 ou 27×27) et les paramètres de convolution (« K 3×3 ») sont clairement affichés sur les blocs, avec des valeurs exactes comme « stride 4 » associée à une dimension « 11×11 ». La légende précise que les couches de convolution et d’activation sont en orange clair, les couches d’agrégation en orange foncé et les couches complètement connectées en violet. Sur le plan pédagogique, cette figure illustre la progression des données à travers les étapes clés du réseau : traitement par des couches de convolution avec des tailles de noyaux spécifiques (5×5 puis 3×3), application d’opérations d’agrégation (comme le pooling), suivi d’une transition vers des couches complètement connectées pour la classification finale, avec une sortie via la fonction Softmax. Les couleurs et les étiquettes permettent de distinguer visuellement chaque type de couche, facilitant ainsi la compréhension structurelle du modèle.
--- [/FIGURE] ---

Si la profondeur du réseau reste faible, le nombre de paramètres était déjà important. En regardant uniquement la première couche de convolution, on constate que :
— l'entrée est composée d’images 227×227×3
— les filtres de convolution sont de taille 11
— le pas de convolution (stride) est de 4
2

<!-- loc page=3 -->

Ainsi la sortie de la couche de convolution est de taille $55 \times 55 \times 96=290$ 400 neurones, chacun ayant $11 \times 11 \times 3=363$ poids et un biais. Cela implique, sur cette couche de convolution seulement, 105 705 600 paramètres à ajuster.
Ce réseau, amélioration d’un réseau existant (LeNet), apportait de nombreuses contributions, comme l’utilisation de couches ReLU, de dropout, ou du GPU (NVIDIA GTX 580) pendant la phase d’entraînement.

1.2- VGG

Les réseaux VGG (Visual Geometry Group, université d’Oxford) [8] ont été les premiers réseaux à utiliser de petits filtres de convolution ($3 \times 3$) et à les combiner pour décrire des séquences de convolution, l’idée étant d’émuler l’effet de larges champs réceptifs par cette séquence. Cette technique amène malheureusement à un nombre exponentiel de paramètres (le modèle entraîné qui peut être téléchargé a une taille de plus de 500 Mo). VGG a concouru à ILSVRC 2014, a obtenu un taux de bonne classification de 92.3% mais n’a pas remporté le challenge. Aujourd’hui VGG et une famille de réseaux profonds (de A à E) qui variant par leur architecture (figures 1-3 et 1-4). Le nombre de paramètres (en millions) pour les réseaux de A à E est 133, 133, 134, 138 et 144. Les réseaux VGG-D et VGG-E sont les plus précis et populaires.

| A | A-LRN | B | C | D | E |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 11 couches | 11 couches | 13 couches | 16 couches | 16 couches | 19 couches |

Entrée : image $224 \times 224$ RGB

conv3-64 conv3-64 LRN conv3-64 conv3-64 conv3-64 conv3-64 conv3-64

max pooling

conv3-128 conv3-128 conv3-128 conv3-128 conv3-128 conv3-128

max pooling

conv3-256 conv3-256 conv3-256 conv3-256 conv3-256 conv3-256 conv3-256 conv3-256

max pooling

conv3-512 conv3-512 conv3-512 conv3-512 conv3-512 conv3-512 conv3-512 conv3-512 conv3-512

max pooling

conv3-512 conv3-512 conv3-512 conv3-512 conv3-512 conv3-512 conv3-512 conv3-512

max pooling

Couche complètement connectée 4096 neurones

Couche complètement connectée 4096 neurones

Couche complètement connectée 1000 neurones

Classifieur softmax

FIGURE 1-3 – Architectures des réseaux VGG

<!-- loc page=4 -->

--- [FIGURE] ---
FIGURE 1-4 – Réseau VGG16
C'est un schéma architectural détaillé du réseau VGG16 présenté sous forme de diagramme en perspective avec des blocs rectangulaires colorés et des flèches indiquant le flux de données. L'image représente une séquence de couches successives : à gauche, l'entrée est un rectangle rouge marqué « 224 » (hauteur) et « 224 » (largeur), avec une dimension en profondeur de « 64×64 ». Ce bloc sert d'input pour la première couche convolutive « conv1 », dont les dimensions sont indiquées comme « 64×64 » sur le côté gauche et « 224 » en haut. Ensuite, chaque couche convolutive suivante (« conv2 », « conv3 », « conv4 », « conv5 ») est représentée par des blocs de couleur orange ou jaune avec des dimensions spécifiques : « conv2 » affiche « 128×128 » et « 128 » en profondeur, « conv3 » montre « 256×256 » et « 256 », « conv4 » indique « 512×512 » et « 512 », tandis que « conv5 » présente « 512×512 » et « 512 » avec une dimension spatiale réduite à « 14 ». Les flèches bleues reliant ces blocs illustrent le passage des données entre les couches. À droite, les couches complètement connectées (« fc6 », « fc7 », « fc8+softmax ») sont représentées par des barres violettes : « fc6 » et « fc7 » ont une dimension de « 4096 », tandis que « fc8+softmax » affiche « K ». Les étiquettes précises comme « conv1 », « conv2 », etc., ainsi que les dimensions spatiales (« 56 », « 28 », « 14 ») et les tailles de canal (« 64 », « 128 », « 256 », « 512 ») sont clairement visibles sur chaque bloc. Les couleurs diffèrent selon le type de couche : rouge pour l'entrée, orange/yellow pour les couches convolutives et violet pour les couches complètement connectées. Sur le plan pédagogique, cette figure illustre la structure hiérarchique du réseau VGG16 en montrant comment les dimensions spatiales diminuent progressivement (de 224×224 à 14×14) tout en augmentant le nombre de canaux (de 64 à 512), puis la transition vers des couches complètement connectées pour la classification finale via softmax. Elle permet aux étudiants de visualiser l'évolution des caractéristiques extraites par les convolutions et la réduction spatiale, ainsi que le rôle des couches FC dans la prédiction des classes (« K »).
--- [/FIGURE] ---

1.3- Inception

Inception, proposé par Google, est le premier réseau dont les performances ont été augmentées pas seulement en augmentant le nombre de couches, mais en pensant et optimisant le design et l’architecture. L’idée est ici d’utiliser plusieurs filtres, de tailles différentes, sur la même image et de concaténer les résultats pour générer une représentation plus robuste.

Inception n’est pas un réseau, c’est une famille de réseaux : Network in Network [6], Inception V1 [10], Inception V2 [11], Xception [2],...

L’idée du premier réseau (figure 1-5) est de connecter les couches de convolution par des perceptrons multicouches, introduisant des non linéarités dans les réseaux profonds. Mathématiquement, ces perceptrons sont équivalents à des convolutions par des filtres $1 \times 1$ et gardent donc la cohérence des réseaux. Cette nouvelle architecture rend moins indispensable les couches complètement connectées en fin de réseau. Les auteurs moyennent spatialement les cartes finales et donnent le résultat au classifier softmax. Le nombre de paramètres est alors réduit, diminuant de ce fait le risque de sur apprentissage.

--- [FIGURE] ---
FIGURE 1-5 – Réseau Network in Network
La figure présente un schéma monochrome en deux parties juxtaposées, illustrant une architecture de réseau neuronal. À gauche, trois rectangles verticaux alignés sur la gauche sont reliés par des lignes pointillées à un bloc central rectangulaire, qui est lui-même connecté via des traits pointillés à un rectangle plus petit situé au centre droit ; ce dernier est enfin lié à une petite boîte cubique située à l'extrême droite. À droite, cette structure de base est complétée par deux rectangles supplémentaires : le premier contient trois cercles disposés verticalement (avec des traits pointillés reliant chaque cercle au bloc central), et le second en contient deux (également reliés par des lignes pointillées). Des étiquettes numériques « 3 » et « 2 » sont visibles respectivement sur les côtés de ces deux rectangles. Les flèches pointillées représentent des connexions entre les éléments, sans indication d'échelle ou de couleurs spécifiques. Ce schéma illustre pédagogiquement la notion de « Network in Network », où l'architecture de base (gauche) est enrichie par l'intégration de sous-réseaux internes (droite), permettant d'extraire des caractéristiques plus fines via des structures comme des couches convolutionnelles imbriquées, sans modifier la structure globale du réseau. Les éléments visibles soulignent comment ces sous-réseaux (représentés par les cercles et leurs connexions) améliorent l'efficacité de traitement en intégrant des mécanismes d'apprentissage localisé au sein des couches principales.
--- [/FIGURE] ---

Inception V1, implémenté dans le réseau GoogLeNet vainqueur d’ILSVRC 2014, est une extension à des réseaux plus profonds de Network to Network. Le réseau est composé de 22 couches et atteint 93.3% de taux de reconnaissance. D’autres améliorations théoriques (fonctions de pertes associées aux couches intermédiaires dans la phase d’apprentissage, introduction de caractères épars dans le réseau) ont également permis d’améliorer les performances (de calcul et de classification).

Inception V2, puis V3 (figure 1-6) adoptent des techniques de factorisation (toute convolution par un filtre de taille plus grande que $3 \times 3$ peut être exprimée de manière plus efficace avec une série de filtre de taille réduite) et de normalisation pour améliorer encore les performances.

<!-- loc page=5 -->

Inception V4 [9] propose une version rationalisée, à l’architecture uniforme et aux performances accrues.

--- [FIGURE] ---
FIGURE 1-6 – Architecture d’inception V3
La figure présente un schéma horizontal détaillant l'architecture d'un réseau de neurones Inception V3. Elle est constituée d'une séquence linéaire de blocs interconnectés par des flèches rouges (représentées ici comme des petits cercles rouges), organisés en plusieurs étapes successives. Chaque bloc contient des rectangles colorés selon une légende explicite : les rectangles orange correspondent aux opérations de Convolution, les bleus à AvgPool, les verts à MaxPool, les roses à Concat, les violets à Dropout, les mauves à Fully connected et les rouges à Softmax. On observe notamment plusieurs modules en cascade où chaque étape intègre des couches de convolution (orange) avec parfois des opérations de pooling (bleu ou vert), tandis que certaines branches se détachent pour former des chemins alternatifs : une branche, par exemple, part vers le bas pour intégrer un Dropout (violet) et un Concat (rose) avant de rejoindre le flux principal via une connexion rouge. À la fin du schéma, on distingue clairement les couches Fully connected (mauve) suivies de Softmax (rouge), indiquant la phase finale de classification. Les formes géométriques sont principalement des ovales ou des rectangles contenant ces blocs colorés, avec une répartition variée du nombre de rectangles par module (par exemple, certains modules présentent huit rectangles orange tandis que d'autres en ont six). La figure illustre pédagogiquement la structure hiérarchique et modulaire de l'architecture Inception V3, mettant en évidence comment les chemins parallèles avec des opérations variées (convolution, pooling, concaténation) permettent d'extraction multi-échelle des caractéristiques visuelles, essentielle pour la classification d'images. Les couleurs et les connexions visibles soulignent le rôle central de la concaténation dans l'intégration des informations issues de différentes branches, tout en montrant la progression logique des données vers la sortie finale via les couches complètes et la fonction Softmax.
--- [/FIGURE] ---

1.4- ResNet

En 2015, Microsoft remporte la compétition ILSVRC avec ResNet [3], un réseau à 152 couches qui utilise un module ResNet. Le taux de bonne reconnaissance est de 96.4%. Un réseau résiduel (ou ResNet) résout le problème de vanishing gradient de la manière la plus simple possible, en permettant des raccourcis entre chaque couche du réseau. Dans un réseau classique, l’activation en sortie de couche est de la forme $y = \sigma(x)$, et lors de la rétropropagation, le gradient doit nécessairement repasser par $\sigma(x)$, ce qui peut causer des problèmes en raison de la (forte) non linéarité induite par $\sigma$. Dans un réseau résiduel, la sortie de chaque couche est calculée par $y = \sigma() + x$, où $+x$ est le raccourci entre chaque couche, qui permet au gradient de transiter directement sans passer par $\sigma$.

Cette représentation donne l’idée générale, mais la réalité est un peu plus complexe, et prend la forme d’un module ResNet (figure 1-7).

--- [FIGURE] ---
FIGURE 1-7 – Module ResNet (source : [3])
La figure présente un schéma détaillé d'un module ResNet, illustrant une architecture de réseau neuronal résiduel. On observe une entrée 'x' qui traverse successivement deux blocs rectangulaires étiquetés "weight layer", séparés par une activation ReLU (indiquée comme "relu" en minuscules). Parallèlement, une connexion identité (marquée "identity") relie directement l'entrée 'x' à un nœud d'addition situé après les deux blocs de poids. Les sorties des deux chemins sont additionnées via le symbole "+" pour former F(x) + x, qui est ensuite soumis à une activation ReLU finale (étiquetée "relu"). À gauche du schéma, l'étiquette "F(x)" indique la sortie intermédiaire issue de la transformation des couches de poids et de l'activation ReLU. Ce diagramme illustre pédagogiquement le principe fondamental des réseaux résiduels : en combinant la transformation du signal (F(x)) avec l'entrée originale via une connexion identité, il permet d'éviter les problèmes liés aux gradients disparus dans les architectures profondes, facilitant ainsi l'apprentissage par la construction de blocs résiduels qui préservent le flux des gradients.
--- [/FIGURE] ---

1.5- SqueezeNet

SqueezeNet [4] est un réseau produit en 2016, qui n’est pas tant remarquable par ses performances (il atteint les mêmes niveaux de reconnaissance qu’AlexNet), mais par sa légèreté (le modèle entraîné sur ImageNet a une taille de 4.9 Mo, et possède 50 fois moins de paramètres qu’AlexNet par exemple) et la rapidité avec laquelle il peut être entrainé.

SqueezeNet introduit des modules "Fire" (figure 1-8), composés d’une couche de convolution "squeeze",
5

<!-- loc page=6 -->

dotée de filtres de taille 1×1, et d’une couche d’expansion dotée de filtres de taille 1×1 et 3×3. L’utilisation de filtres 1×1 permet une réduction du nombre de paramètres.

--- [FIGURE] ---
FIGURE 1-8 – Module Fire (source : [4])
La figure est un schéma détaillant l'architecture du Module Fire sous forme de diagramme structuré avec deux sections principales. Une ellipse orange en haut porte l'étiquette « squeeze » et contient trois blocs rectangulaires marron représentant des filtres de convolution 1x1, reliés par une flèche vers le bas étiquetée « ReLU ». En dessous, une ellipse verte plus grande est marquée « expand », incluant quatre petits blocs gris (pour les filtres 1x1) suivis de quatre grands blocs gris avec des éléments circulaires multiples (indiquant les filtres 3x3), également reliés par une flèche vers le bas étiquetée « ReLU ». Les couleurs utilisées sont l'orange pour la section squeeze et le vert pour expand, avec des blocs marron et gris respectivement. Ce schéma illustre pédagogiquement la structure du Module Fire : la phase squeeze réduit les dimensions de caractéristiques via des filtres 1x1 et une activation ReLU, tandis que la phase expand combine des filtres 1x1 et 3x3 pour augmenter la résolution spatiale tout en maintenant l'efficacité computationnelle.
--- [/FIGURE] ---

Le réseau est composé d’une couche de convolution classique, d’une couche d’agrégation max, suivie de 9 modules Fires entrecoupées d’agrégation max, et d’une couche de convolution finale. Le nombre de filtres est progressivement augmenté entre chaque module (figure 1-9).

| layer name/type | output size | filter size / stride (if not a fire layer) | depth | $s_{1x1}$ (#1x1 squeeze) | $e_{1x1}$ (#1x1 expand) | $e_{3x3}$ (#3x3 expand) | $s_{1x1}$ sparsity | $e_{1x1}$ sparsity | $e_{3x3}$ sparsity | # bits | #parameter before pruning | #parameter after pruning |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| input image | 224x224x3 | | | | | | | | | | - | - |
| conv1 | 111x111x96 | 7x7/2 (x96) | 1 | | | | 100% (7x7) | | | 6bit | 14,208 | 14,208 |
| maxpool1 | 55x55x96 | 3x3/2 | 0 | | | | | | | | | | |
| fire2 | 55x55x128 | | 2 | 16 | 64 | 64 | 100% | 100% | **33%** | 6bit | 11,920 | 5,746 |
| fire3 | 55x55x128 | | 2 | 16 | 64 | 64 | 100% | 100% | **33%** | 6bit | 12,432 | 6,258 |
| fire4 | 55x55x256 | | 2 | 32 | 128 | 128 | 100% | 100% | **33%** | 6bit | 45,344 | 20,646 |
| maxpool4 | 27x27x256 | 3x3/2 | 0 | | | | | | | | | | |
| fire5 | 27x27x256 | | 2 | 32 | 128 | 128 | 100% | 100% | **33%** | 6bit | 49,440 | 24,742 |
| fire6 | 27x27x384 | | 2 | 48 | 192 | 192 | 100% | **50%** | **33%** | 6bit | 104,880 | 44,700 |
| fire7 | 27x27x384 | | 2 | 48 | 192 | 192 | **50%** | 100% | **33%** | 6bit | 111,024 | 46,236 |
| fire8 | 27x27x512 | | 2 | 64 | 256 | 256 | 100% | **50%** | **33%** | 6bit | 188,992 | 77,581 |
| maxpool8 | 13x12x512 | 3x3/2 | 0 | | | | | | | | | | |
| fire9 | 13x13x512 | | 2 | 64 | 256 | 256 | **50%** | 100% | **30%** | 6bit | 197,184 | 77,581 |
| conv10 | 13x13x1000 | 1x1/1 (x1000) | 1 | | | | 20% (3x3) | | 6bit | 513,000 | 103,400 |
| avgpool10 | 1x1x1000 | 13x13/1 | 0 | | | | | | | | | |

FIGURE 1-9 – Architecture de SqueezeNet (source : [4])

2- COMMENT UTILISER CES RÉSEAUX

Il est possible de redéclarer les réseaux classiques depuis TensorFlow ou Keras, en définissant une à une les couches et leur paramètres, qui sont décrits dans les articles correspondants. On imagine assez bien le travail que cela peut représenter sur le ResNet de Microsoft...
Fort heureusement, il existe d’autres manières d’utiliser ces réseaux.

<!-- loc page=7 -->

2.1- Utilisation de réseaux pré-entraînés

Il est possible avec TensorFlow et Keras de charger / sauvegarder des réseaux qui ont été entraînés. Il est également possible, pendant l'entraînement, de créer des sauvegardes (checkpoints) pour reprendre éventuellement l'entraînement en cours d’itérations. On peut sauvegarder tout le réseau (architecture + optimiseur + poids), ou seulement les poids.
Pour charger / sauver les poids d’un modèle model Keras, on utilise simplement les fonctions

```python
model = create\_model()\
model.load\_weights('./checkpoints/my\_checkpoint')
```

et

```python
model.save\_weights('./checkpoints/my\_checkpoint')
```

Pour charger / sauver le modèle complet, on utilise

```python
model = keras.models.load\_model('my\_model.h5')
```

et

```python
model = create\_model()
model.fit(xtrain, ytrain)
model.save('my\_model.h5')
```

La procédure sous Tensorflow est un peu plus complexe et est décrite ici.

2.2- Transfer learning et fine tuning

Il est possible d’utiliser les réseaux classiques pré-entraînés pour de nouvelles tâches. L’idée sous-jacente et que les premières couches capturent des caractéristiques bas niveau, et que la sémantique vient avec les couches profondes. Ainsi, dans un problème de classification, où les classes n’ont pas été apprises, on peut supposer qu’en conservant les premières couches on extraira des caractéristiques communes des images, et qu’en changeant les dernières couches (information sémantique et haut niveau et étage de classification), c’est à dire en réapprenant les connexions, on spécifiera le nouveau réseau pour la nouvelle tâche de classification.

Cette approche rentre dans le cadre des méthodes de Transfer Learning [7] et de fine tuning, cas particulier d’adaptation de domaine :

— les méthodes de transfer learning prennent un réseau déjà entraîné, enlèvent la dernière couche complètement connectée, et traitent le réseau restant comme un extracteur de caractéristiques. Un nouveau classifieur est alors entraîné sur les caractéristiques calculées sur le nouveau problème

— les méthodes de fine tuning ré-entrainent le classifier du réseau, et remettent à jour les poids du réseau pré-entraîné par rétropropagation.

Plusieurs facteurs influent sur le choix de la méthode à utiliser : la taille des données d’apprentissage du nouveau problème, et la ressemblance du nouveau jeu de données avec celui qui a servi à entraîner le réseau initial :

— pour un jeu de données similaire de petite taille, on utilise du transfer learning, avec un classifieur utilisé sur les caractéristiques calculées sur les dernières couches du réseau initial

— pour un jeu de données de petite taille et un problème différent, on utilise du transfer learning, avec un classifieur utilisé sur les caractéristiques calculées sur les premières couches du réseau initial

— pour un jeu de données, similaire ou non de grande taille, on utilise le fine tuning

A noter qu’il est toujours possible d’augmenter la taille du jeu de données par des techniques de "Data Augmentation" (changement de couleurs des pixels, rotations, cropping, homothéties, translations...)

3- QUE FAIRE SI J’AI PEU DE DONNÉES ?

Les méthodes supervisées nécessitent pour de bonnes performances un ensemble d’apprentissage $S$ de grand cardinal. Si seulement peu d’exemples $S_s = \{(x_i, y_i), i \in [1\cdots m]\}$ sont disponibles, avec $m$ petit, les

<!-- loc page=8 -->

techniques précédemment décrites ne sont pour la plupart plus applicables.
Les méthodes de Few-Shot Learning ont été introduites pour traiter ce manque de données. Les exemples
applicatifs sont nombreux, allant de la classification d’images à l’analyse de sentiments à partir de textes,
ou encore à la reconnaissance d’objets.
Vu sous l’angle de la minimisation du risque empirique, l’hypothèse $h$ construite sur la minimisation de

$$R(h) = \sum_{i=1}^{m} \ell(y_i, h(x_i))$$

conduit à un sur apprentissage et un risque $R(h)$ très loin du risque réel. Pour pallier ce problème, des
connaissances a priori doivent être utilisées. Le Few-shot learning propose trois alternatives. Nous détaillons
ici l’une d’entre elles, l’augmentation de données.

Les approches de cette catégorie utilisent des connaissances a priori sur les données pour enrichir $S_s$. On
les regroupe parfois sous le vocable de méthodes d’augmentation de données. Si elles sont faciles à mettre
en oeuvre et à comprendre, ces méthodes restent cependant dépendantes du domaine d’étude et ne peuvent
être facilement généralisées.
Les principales stratégies sont résumées dans le tableau .1 et un exemple d’illustration est donné figure
3-10.

| Transformation... | Entrée | Opérateur | Sortie |
| :--- | :--- | :--- | :--- |
| ... de données de $S_s$ | $(x_i, y_i) \in S_s$ | $t : X \rightarrow X$ | $(t(x_i), y_i)$ |
| ... d’un ensemble de données
non étiquetées | $(x, -)$ | $h : X \rightarrow Y$ entraîné sur $S_s$ | $(x, h(x))$ |
| ... d’un ensemble de données
similaires | $\{(\hat{x}_j, \hat{y}_j)\}$ | Opérateur de combinaison $c$ | $(c(\{\hat{x}_j\}), c(\{\hat{y}_j\}))$ |

TABLE .1 – Techniques d’augmentation de données

--- [FIGURE] ---
FIGURE 3-10 – Exemple d’augmentation de données. De gauche à droite : image originale, rotation de $20^\circ$, flip, ajout de bruit gaussien, déformation élastique, changement de contraste par canal RGB.
La figure présente une série horizontale de six couvertures de livres identiques en termes de titre et de sous-titre, chacune illustrant une technique d'augmentation de données appliquée à la même image de base. Les titres visibles sur chaque couverture sont « Apprentissage artificiel » en caractères noirs gras et « Deep learning, concepts et algorithmes » en petits caractères sous le titre principal. La première image est l'originale : fond blanc avec des éléments graphiques vert clair et un bandeau rouge en haut à droite indiquant « 2e édition ». La deuxième couverture montre une rotation de 20° vers la gauche, visible par l'inclinaison légère du texte et des formes vertes. La troisième représente un flip horizontal : les éléments sont miroirs par rapport à l'axe vertical, avec le titre « Apprentissage artificiel » inversé horizontalement mais encore lisible. La quatrième image présente un ajout de bruit gaussien, caractérisée par une texture granuleuse sur le fond blanc et des formes vertes floues. La cinquième illustre une déformation élastique avec des contours vert clair distordus et irréguliers, comme si l'image était étirée ou pliée. La sixième montre un changement de contraste par canal RGB : le fond est teinté d'un brun-vert plus sombre, tandis que les éléments graphiques conservent une forme approximative mais avec des nuances colorées modifiées. Aucune échelle numérique ou axe n'est visible, mais chaque couverture est présentée comme un exemple concret de transformation appliquée à la même base visuelle. Sur le plan pédagogique, cette figure illustre concrètement comment les techniques d'augmentation de données modifient l'apparence des images pour générer des variations synthétiques, permettant ainsi d'améliorer la robustesse et la généralisation des modèles d'apprentissage automatique sans nécessiter de nouvelles données réelles.
--- [/FIGURE] ---
8

<!-- loc page=9 -->

4- PARTIE PRATIQUE

Vous avez à disposition un notebook vous proposant d'apprendre, à partir d’un réseau pré-entrainé (MobileNetV2, assez léger et performant en classification), de fine-tuner ce dernier pour un problème où les données ne font pas partie de la base d'entraînement initiale (l’objectif ici est d’apprendre à classifier des fleurs en 5 catégories). Votre travail consiste à :
— Appliquer une technique d’augmentation de données pour augmenter le nombre d’images de fleurs. Pour ce faire, vous utiliserez l’objet ImageDataGenerator, que nous vous laissons prendre en main...
— Créer le modèle. vous chargerez tout d’abord le réseau MobileNetV2 (fonction MobileNetV2). Vous supprimerez ensuite la dernière couche de ce réseau (la couche de classification) à l’aide de la fonction pop, puis ajouterez votre propre couche de classification. Vous indiquerez ensuite quels poids ne doivent pas être réappris (en taggant les couches correspondantes avec le drapeau layer.trainable = False), et réentrainerez le nouveau réseau sur la nouvelle base augmentée. Vous pourrez tester l’influence du nombre de couches non réentrainées.
— Evaluer la précision de votre réseau.
9

<!-- loc page=10 -->

[1] Alfredo Canziani, Adam Paszke, and Eugenio Culurciello. An analysis of deep neural network models for practical applications. CoRR, abs/1605.07678, 2016.

[2] François Chollet. Xception: Deep learning with depthwise separable convolutions. CoRR, abs/1610.02357, 2016.

[3] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. CoRR, abs/1512.03385, 2015.

[4] Forrest N. Iandola, Matthew W. Moskewicz, Khalid Ashraf, Song Han, William J. Dally, and Kurt Keutzer. SqueezeNet: Alexnet-level accuracy with 50x fewer parameters and <1mb model size. CoRR, abs/1602.07360, 2016.

[5] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton. Imagenet classification with deep convolutional neural networks. In Advances in Neural Information Processing Systems, page 2012, 2012.

[6] Min Lin, Qiang Chen, and Shuicheng Yan. Network in network. CoRR, abs/1312.4400, 2013.

[7] Sinno Jialin Pan and Qiang Yang. A survey on transfer learning. IEEE Trans. on Knowl. and Data Eng., 22(10):1345–1359, October 2010.

[8] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. CoRR, abs/1409.1556, 2014.

[9] Christian Szegedy, Sergey Ioffe, and Vincent Vanhoucke. Inception-v4, inception-resnet and the impact of residual connections on learning. CoRR, abs/1602.07261, 2016.

[10] Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott E. Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, and Andrew Rabinovich. Going deeper with convolutions. CoRR, abs/1409.4842, 2014.

[11] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. CoRR, abs/1512.00567, 2015.