---
source_type: pdf
source_id: 05_RNN.pdf
page_count: 11
source: 05_RNN
---

<!-- loc page=1 -->

RÉSEAUX RÉCURRENTS

SOMMAIRE

1- Définition 1

2- Entraînement des réseaux récurrents 2

3- Quelques architectures 4

### 3.1- LSTM
### 3.2- GRU
### 3.3- Réseaux récurrents bidirectionnels
### 3.4- Machines de Turing neuronales

4- Quelques applications 7

5- Partie pratique 9

### 5.1- Classification de séquences
### 5.2- Prévision de séries temporelles

## 1- DÉFINITION

Les réseaux de neurones récurrents (RNN, Recurrent Neural Networks) sont des réseaux à propagation avant, permettant de prendre en compte le temps. Comme dans les réseaux classiques, il n'existe pas de cycle, mais les arcs ajoutés pour introduire la notion de temps (les arcs récurrents) peuvent en revanche former des cycles, y compris de longueur 1 (connexion d'un neurone avec lui-même). À l'instant $t$, les neurones possédant des arcs récurrents reçoivent en entrée la donnée courante $\mathbf{x}_t$ et les valeurs des neurones cachés $h_{t-1}$ informant sur l'état précédent du réseau. La sortie $\hat{y}_t$ est calculée étant donné l'état $\mathbf{x}_t$ des neurones cachés à l'instant $t$. La donnée $\mathbf{x}_{t-1}$ peut influencer $\hat{y}_t$ et la sortie aux instants suivants, à la aide des arcs récurrents.
1

<!-- loc page=2 -->

Deux équations permettent de calculer les quantités nécessaires à l’instant $t$ dans la phase de propagation avant d’un réseau récurrent simple (comme celui de la figure 1-1 gauche) :

$$h_t = \sigma \left( W_{hx}^{\top} x_t + W_{hh}^{\top} x_{t-1} + b_h \right)$$
$$\hat{y}_t = \text{softmax} \left( W_{yh} h_t + b_y \right)$$

où $W_{hx}$ est la matrice des poids reliant l’entrée à la couche cachée et $W_{hh}$ celle des poids des arcs récurrents. Les biais sont notés $b_h$ et $b_y$.

La dynamique du réseau peut être décrite en dépliant ce réseau dans le temps (figure 1-1 droite). Le réseau devient donc un réseau profond, avec une couche par instant $t$ et un partage de poids au cours du temps. Ce dernier peut donc être entraîné de manière classique par l’algorithme de rétropropagation du gradient, indicé par le temps (Backpropagation through time, BPTT algorithm).

--- [FIGURE] ---
FIGURE 1-1 – Réseau récurrent et sa version dépliée dans le temps.
La figure est un schéma montrant une architecture de réseau récurrent et sa version dépliée dans le temps. À gauche, on observe une chaîne verticale avec trois cercles : en haut, ŷ_t (prédiction), au milieu h_t (état caché), et en bas x_t (entrée). Un bloc vert marqué 'C' est relié à h_t par des flèches noires vers le haut et vers le bas, avec une boucle rouge autour de C indiquant un retour temporel. À droite, l'égalité entre les deux représentations montre plusieurs chaînes horizontales : pour chaque temps t (0, 1, 2, ..., t), on a ŷ_0 à ŷ_t en haut, h_0 à h_t au milieu, et x_0 à x_t en bas. Chaque bloc vert 'C' est connecté par des flèches rouges horizontales entre eux, formant une séquence linéaire de C reliant les étapes temporelles. Les couleurs sont claires : cercles gris pour ŷ et h, cercles verts pour C, flèches noires verticales et rouges horizontales. La légende précise que cette figure illustre comment un réseau récurrent, représenté par la boucle de retour sur la gauche, correspond à une décomposition temporelle où chaque étape (x_0, x_1, ..., x_t) est traitée séparément via des blocs C connectés en série, permettant ainsi d'expliquer le fonctionnement récursif du modèle sans boucle explicite.
--- [/FIGURE] ---

Avec ces réseaux, il est possible de traiter des séquences de longueur quelconque, la taille du modèle étant indépendante de cette longueur. Plusieurs architectures peuvent être déclinées sur ce principe et le tableau .1 donne un panorama de certaines d’entre elles, avec des exemples d’applications.

## 2- ENTRAÎNEMENT DES RÉSEAUX RÉCURRENTS

L’apprentissage de dépendances long terme peut être difficile. Les problèmes d’évanescence (vanishing) ou d’explosion du gradient peuvent rapidement survenir, lors de la rétropropagation sur plusieurs pas de temps. Prenons un exemple simple pour comprendre : considérons un réseau à un neurone d'entrée, un neurone récurrent caché et un neurone de sortie. On donne au réseau une entrée à l’instant $t_0$ et on calcule l’erreur à l’instant $t > t_0$, en supposant des entrées nulles entre $t_0$ et $t$. Le lien entre les poids au cours du temps fait que le poids sur l’arc récurrent ne change jamais. La contribution de l’entrée au temps $t_0$ à la sortie au temps $t$ deviendra de plus en plus importante, ou se rapprochera de zéro, de manière exponentielle à mesure que $t - t_0$ croît. Et la dérivée de l’erreur par rapport à l’entrée explosera ou disparaît, selon que le poids de l’arc récurrent a une valeur absolue plus grande ou plus petite que 1 et selon la fonction d’activation du neurone caché (le problème du gradient évanescent est très présent avec une sigmoïde et une activation ReLU force davantage l’explosion).

Plusieurs solutions ont été proposées (régularisation, retropropagation tronquée, conception d’architecture et heuristiques) pour résoudre ces problèmes.

<!-- loc page=3 -->

Architecture
Réseau
Applications

Un vers plusieurs
Génération de musique, légendage d’images

Plusieurs vers un
Classification de sentiments

Plusieurs vers plusieurs
Reconnaissance d'entité dans des textes, annotation de vidéos

Plusieurs vers plusieurs
Traduction

--- [FIGURE] ---
TABLE .1 – Différentes architectures de réseaux récurrents et leurs applications
La première sous-figure présente des cercles verts étiquetés « C », des cercles blancs étiquetés « h₀ », « h₁ », « h₂ » et « hₜ », ainsi que des cercles gris étiquetés « ŷ₀ », « ŷ₁ », « ŷ₂ » et « ŷₜ ». Une flèche noire solide relie un cercle blanc « x » au premier cercle vert « C », puis ce dernier est relié par une flèche noire solide à « h₀ », qui est lui-même connecté à « ŷ₀ » via une flèche noire solide. Le deuxième cercle vert « C » est relié par une flèche noire solide à « h₁ », avec une flèche diagonale noire solide provenant du premier cercle vert « C ». Le troisième cercle vert « C » est connecté par une flèche noire solide à « h₂ », tandis que des flèches noires pointillées relient le deuxième cercle vert « C » à « h₂ » et le troisième cercle vert « C » à « hₜ ». La structure montre un flux d'entrée vers les états cachés (h), qui génèrent des sorties (ŷ), avec des connexions entre temps successifs indiquées par des flèches solides ou pointillées. Pédagogiquement, cette figure illustre une architecture récurrente où les informations se propagent entre les étapes temporelles via des connexions directes (solides pour les étapes actuelles, pointillées pour les futures), soulignant la dynamique inter-temporelle dans un réseau récurrent.

La deuxième sous-figure affiche quatre cercles verts étiquetés « C » alignés horizontalement. Chaque cercle vert est relié par une flèche noire solide vers un cercle blanc étiqueté « x₀ », « x₁ », « x₂ » et « xₜ ». Des flèches rouges relient chaque cercle vert au suivant (premier à deuxième, deuxième à troisième, troisième à quatrième), indiquant une connexion récurrente entre les étapes temporelles. Aucun autre nœud ou étiquette n'est visible. Pédagogiquement, cette figure démontre la structure de base d'un réseau récurrent où les entrées à différentes étapes temporelles alimentent des états cachés connectés par des flèches rouges, mettant en évidence la dépendance temporelle dans le flux de données sans couche de sortie.

La troisième sous-figure montre trois cercles verts étiquetés « C » alignés horizontalement. Chaque cercle vert est relié par une flèche noire solide vers un cercle blanc étiqueté « x₀ », « x₁ » et « x₂ ». Des flèches rouges relient chaque cercle vert au suivant (premier à deuxième, deuxième à troisième), indiquant une connexion récurrente entre les étapes temporelles. Pédagogiquement, cette figure simplifie la structure récurrente en se concentrant sur le flux d'entrée vers les états cachés sans inclure de couches de sortie ou de connexions supplémentaires, soulignant le traitement séquentiel des entrées par les états cachés.

La quatrième sous-figure présente quatre cercles verts étiquetés « C » alignés horizontalement. Chaque cercle vert est relié par une flèche noire solide vers un cercle blanc étiqueté « x₀ », « x₁ », « x₂ » et « xₜ ». Des flèches rouges relient chaque cercle vert au suivant, tandis que des cercles blancs étiquetés « h₀ », « h₁ », « h₂ » et « hₜ » sont placés au-dessus de chaque cercle vert, chacun relié par une flèche noire solide à un cercle gris étiqueté « ŷ₀ », « ŷ₁ », « ŷ₂ » et « ŷₜ ». Pédagogiquement, cette figure illustre l'architecture standard d'un réseau récurrent où les entrées alimentent des états cachés (h), qui génèrent des sorties (ŷ), avec des flèches rouges indiquant la récurrence entre les temps.

La cinquième sous-figure affiche quatre cercles verts étiquetés « C » alignés horizontalement. Chaque cercle vert est relié par une flèche noire solide vers un cercle blanc étiqueté « x₀ », « x₁ », « x₂ » et « xₜₓ ». Des flèches rouges relient chaque cercle vert au suivant, tandis que des cercles blancs étiquetés « h₁ » et « hₜᵧ » sont placés au-dessus du deuxième et quatrième cercle vert respectivement, chacun relié par une flèche noire solide à un cercle gris étiqueté « ŷ₁ » et « ŷₜᵧ ». Pédagogiquement, cette figure montre une architecture étendue où plusieurs sorties sont générées à différentes étapes temporelles (par exemple, ŷ₁ et ŷₜᵧ), démontrant comment les réseaux peuvent gérer des séquences avec des structures de sortie variées ou des dépendances temporelles spécifiques.
--- [/FIGURE] ---
3

<!-- loc page=4 -->

## 3- QUELQUES ARCHITECTURES

### 3.1- LSTM

Les réseaux Long Short-Term Memory (LSTM) ont été introduits en 1997 [4] pour résoudre le problème de l'évanescence du gradient. Ce modèle ressemble à un réseau récurrent classique à une couche cachée, mais chaque neurone de la couche cachée est remplacé par une cellule de mémoire.

Dans la suite, on note $x_t$ l'entrée de la cellule à l'instant $t$, $h_{t-1}$ la sortie de la couche cachée calculée au temps $t - 1$. Au lieu de calculer une sortie du type $\sigma(W^\top x + b)$, la cellule contient plusieurs éléments distincts aux fonctions particulières. Les LSTM introduisent la notion de portes, qui sont des unités d'activation de type sigmoïde qui prennent comme arguments $x_t$ et $h_{t-1}$ et viennent pondérer des valeurs calculées dans la cellule. En particulier, si la valeur d'une porte est nulle, alors le flot est coupé dans le graphe, alors qu'il transite intégralement si la valeur de la porte est égale à 1.

On retrouve dans une cellule (figure 3-3) les éléments suivants:

— Neurone d'entrée : ce neurone prend en entrée $x_t$ et $h_{t-1}$ et calculue, à la manière d'un neurone classique, une sortie $g^t = \sigma(W_C^\top [x_t, h_{t-1}] + b_C)$.

— Porte d'entrée (ou de mise à jour) : la porte calcule $i^t = \sigma(W_i^\top [x_t, h_{t-1}] + b_i)$ et vient pondérer la valeur du neurone d'entrée pour décider de l'importance à lui donner au temps $t$.

— Porte d'oubli : cette porte calcule $f^t = \sigma(W_f^\top [x_t, h_{t-1}] + b_f)$ et permet au réseau d'oublier son état interne.

— État interne : le cœur de la cellule de mémoire est son état interne, noté $C^t$, composé d'un neurone récurrent à poids fixe unité, assurant que le gradient peut passer par cet arc de nombreuses fois sans disparaître ou exploser. La mise à jour de l'état interne est effectuée par une opération du type $C^t = g^t.i^t + C^{(t-1)}.f^t$.

— Porte de sortie : la valeur $h_t$ produite par la cellule de mémoire est calculée comme le produit de $\tanh(C^t)$ par la valeur de la porte de sortie $o^t$. Cette porte sélectionne la part de $C^t$ à fournir en sortie et est calculée par $o^t = \sigma(W_o^\top [x_t, h_{t-1}] + b_o)$.

--- [FIGURE] ---
FIGURE 3-2 – Cellule LSTM
La figure présente un schéma architectural d'une cellule LSTM (Long Short-Term Memory), représentée sous forme de diagramme fonctionnel avec des éléments graphiques clairs. On observe trois cercles en entrée : $c^{t-1}$ (état cellulaire précédent), $h_{t-1}$ (état caché précédent) et $x_t$ (entrée courante), ainsi que deux cercles en sortie : $c^t$ (état cellulaire courant) et $h_t$ (état caché courant). À l'intérieur d'un rectangle délimitant la cellule, quatre blocs rectangulaires étiquetés $\sigma$ (sigmoïde) sont connectés par des flèches aux labels $f^t$, $g^t$, $i^t$ et $o^t$. Une flèche pointe vers un symbole « + » qui combine deux flux, tandis qu'une autre flèche relie ce résultat à un bloc ovale étiqueté « tanh ». Des multiplications (« × ») sont visibles entre les sorties des sigmoïdes et d'autres composants. Les flèches indiquent le flux de données : $f^t$ et $g^t$ se connectent à une multiplication, $i^t$ est relié à un autre « × », et $o^t$ s'associe au résultat du bloc « tanh » avant d'être transmis vers $h_t$. Aucune échelle, couleur ou étiquette supplémentaire n'est visible. Sur le plan pédagogique, cette figure illustre la structure interne de l'LSTM en montrant comment les portes (gates) contrôlent le flux d'informations : $f^t$ gère l'oubli du passé, $i^t$ et $g^t$ influencent la mise à jour de l'état cellulaire via une fonction tanh, et $o^t$ module la sortie vers l'état caché, tout en intégrant les entrées $x_t$ et $h_{t-1}$ pour maintenir un état contextuel.
--- [/FIGURE] ---

<!-- loc page=5 -->

En résumé, un LSTM effectue donc les opérations suivantes à l’instant $t$ :

$$g^t = \sigma \left( W_C^\top [x_t, h_{t-1}] + b_C \right)$$
$$i^t = \sigma \left( W_i^\top [x_t, h_{t-1}] + b_i \right)$$
$$f^t = \sigma \left( W_f^\top [x_t, h_{t-1}] + b_f \right)$$
$$o^t = \sigma \left( W_o^\top [x_t, h_{t-1}] + b_o \right)$$
$$C^t = g^t.i^t + C^{t-1}.f^t$$
$$h_t = o^t \tanh(C^t)$$

### 3.2- GRU

En 2014 [1], une version simplifiée des réseaux LSTM a été introduite, qui nécessite moins de paramètres.
Les GRU (Gated Recurrent Units) sont en effet des réseaux sans mémoire interne $C^t$, ni porte de sortie $o^t$.
Ces réseaux sont composés de deux portes au lieu de trois :
— une porte reset $r^t$, qui détermine la manière de combiner la nouvelle entrée au temps $t$ avec la mémoire provenant du temps $t-1$.
— une porte de mise à jour $z^t$, qui détermine la quantité de mémoire précédente qui doit être conservée.
Cette porte est la combinaison des portes d'entrée et d’oubli des LSTM.

Formellement :
$$r^t = \sigma \left( W_r^\top [x_t, h_{t-1}] + b_r \right)$$
$$z^t = \sigma \left( W_z^\top [x_t, h_{t-1}] + b_z \right)$$
$$\tilde{h}^t = \tanh \left( W^\top [x_t, r^t h_{t-1}] + b_h \right)$$
$$h_t = (1-z^t)h_{t-1} + z^t \tilde{h}^t$$

Si, pour tout $t,r^t=1$ et $z^t=0$, alors on modélisse un réseau récurrent classique.

--- [FIGURE] ---
FIGURE 3-3 – Cellule GRU
La figure est un schéma d'architecture de cellule GRU (Gated Recurrent Unit) représentant le flux d'information temporel. Elle présente trois cercles : h_{t-1} en haut à gauche, x_t en bas au centre et h_t en haut à droite, correspondant respectivement à l'état caché précédent, l'entrée courante et l'état caché actuel. À l'intérieur d'un rectangle encadré, on observe trois blocs rectangulaires : deux avec la fonction σ (sigmoid) étiquetés r^t et z^t, ainsi qu'un bloc avec tanh étiqueté ħ^t. Des flèches indiquent des opérations : une multiplication (×), une addition (+), et une étiquette "1-" sur une flèche verticale. Les connexions montrent que h_{t-1} et x_t alimentent les blocs σ (r^t) et σ (z^t), tandis que z^t est relié à 1- avant l'addition avec ħ^t, et r^t influence la multiplication vers h_t. Ce schéma illustre pédagogiquement le rôle des portes de réinitialisation (r^t) et de mise à jour (z^t) dans les GRU pour réguler le flux d'information entre l'état caché précédent et l'entrée, en combinant des opérations élémentaires comme la multiplication et l'addition pour générer l'état courant h_t.
--- [/FIGURE] ---

### 3.3- Réseaux récurrents bidirectionnels

Les réseaux bidirectionnels ont été décrits pour la première fois en 1997 [5]. Dans ces réseaux, deux couches cachées sont présentes, chacune connectée à l'entrée et la sortie. La première couche cachée a

<!-- loc page=6 -->

--- [FIGURE] ---
FIGURE 3-4 – Réseau bidirectionnel.
La figure présente un schéma d'architecture de réseau neuronal structuré en séquence temporelle avec des nœuds circulaires colorés et des flèches directionnelles. On observe des cercles blancs étiquetés $x_0$, $x_1$, ..., $x_t$ (représentant les entrées), des cercles verts étiquetés $h_0$, $h_1$, ..., $h_t$ (étiquetés comme « hidden states »), des cercles orange étiquetés $z_0$, $z_1$, ..., $z_t$ (sans précision supplémentaire sur leur rôle), et des cercles blancs supérieurs étiquetés $\hat{y}_0$, $\hat{y}_1$, ..., $\hat{y}_t$ (représentant les sorties prédites). Les flèches noires relient chaque $x_t$ à son $h_t$ et chaque $z_t$ à son $\hat{y}_t$. Des flèches rouges, orientées vers la droite, connectent les $h_0$ à $h_1$, $h_1$ à $h_2$, ..., jusqu'à $h_{t-1}$ à $h_t$, indiquant un flux d'information unidirectionnel vers l'avant. Des flèches bleues, orientées vers la gauche, relient $z_1$ à $z_0$, $z_2$ à $z_1$, ..., jusqu'à $z_t$ à $z_{t-1}$, suggérant un flux d'information unidirectionnel vers l'arrière. Les couleurs (blanc pour les entrées/sorties, vert pour les états cachés, orange pour les nœuds $z$) et la disposition linéaire des éléments temporels ($x_0$, $h_0$, $z_1$, $\hat{y}_0$ en position gauche, puis $x_1$, $h_1$, $z_2$, $\hat{y}_1$, etc., jusqu'à $x_t$, $h_t$, $z_t$, $\hat{y}_t$ à droite) illustrent une architecture bidirectionnelle où les états cachés ($h$) traitent les données dans un sens (avant), tandis que les nœuds $z$ exploitent des informations provenant de l'arrière (vers le passé). Ce schéma, sans échelle ni axes numériques visibles, met en évidence la dualité de flux d'information dans un réseau neuronal conçu pour intégrer à la fois le contexte antérieur et futur, typique des modèles comme les réseaux bidirectionnels (BiLSTM) ou similaires.
--- [/FIGURE] ---

des connexions récurrentes depuis le passé vers le futur, tandis que l’autre transmet les activations depuis le futur vers le passé (figure 3-4).
Étant données une entrée et une sortie du réseau (des séquences), le réseau peut être entraîné par rétropropagation après avoir été déplié :

$$x_t = \sigma \left( W_h^\top [x_t, h_{t-1}] + b_h \right)$$
$$z_t = \sigma \left( W_z^\top [x_t, z_{t+1}] + b_z \right)$$
$$\hat{y}_t = \text{softmax} \left( W_y^\top [x_t, z_t] + b_y \right)$$

où $h_t$ (respectivement $z_t$) représente la valeur de la couche cachée dans le sens du temps (respectivement dans le sens inverse). Puisque le temps doit être fini dans les deux sens de parcours, les réseaux bidirectionnels ne peuvent traiter que des séquences finies.

### 3.4- Machines de Turing neuronales

Les réseaux récurrentes sont performants pour construire une représentation implicite de l’information, mais restent relativement peu adaptés à la conservation d’informations explicites (des dates précises par exemple). S’inspirant des mémoires de travail, théorisées par les neurosciences et qui sont responsables du raisonnement inductif et de la création de nouveaux concepts, l’idée est alors d’ajouter à ces modèles une « mémoire de travail » externe, ce qui permet de découpler la mémoire (assimilable à la RAM d’un ordinateur) des opérations liées à la tâche effectuée par le réseau (assimilable à la CPU). Puisque la mémoire des LSTM est distribuée dans chaque cellule, elle est donc liée au nombre de cellules et à la capacité de calcul et ce modèle ne répond pas directement au problème posé.

Graves et al. [3] proposent alors une architecture, appelée machine de Turing neuronale, constituée de deux éléments principaux : une mémoire et un contrôleur doté d’un mécanisme d’attention qui lit et écrit dans cette mémoire. Les accès mémoire sont ici des équivalents analogiques dérivables, pour permettre d’entraîner le contrôleur par descente de gradient. Typiquement, le contrôleur est un réseau de neurones ou un réseau récurrent type LSTM (figure 3-5).

Les têtes de lecture et d’écriture interagissent avec la mémoire. Chaque tête est contrôlée par un vecteur de poids, chaque composante définissant le degré d’interaction de la tête avec la zone mémoire correspondante.

Un mécanisme de mise à jour de ces poids, composé de quatre opérations, est mis en place pour permettre l’apprentissage du réseau :

1. Le réseau s’intéresse tout d’abord aux zones mémoire proches d’une clé $k_t$ donnée. Cela permet au modèle de retrouver une information spécifique, en recherchant si la zone mémoire $M_t(i)$ est proche

<!-- loc page=7 -->

de la clé, au sens d’une similarité $K$. Formellement, chaque poids correspondant à la zone mémoire $i$ est calculé par $w_t(i) = \text{softmax}(\beta_t K[k_t, M_t(i)])$.

2. Un mécanisme d’interpolation linéaire permet ensuite de mettre à jour les poids en fonction de leur valeur précédente (pour prendre plus ou moins en compte l’information issue de la clé, ou au contraire la valeur précédente du poids) : $w_t(i) = g_t.w_t(i) + (1 - g_t).w_{t-1}(i)$.

3. Un décalage par convolution translate ensuite les poids, à la manière du décalage classique de la tête dans une machine de Turing classique : $w_t(i) = \sum_j w_t(j)\mathbf{s}_t(i - j)$ où $\mathbf{s}_t$ est un vecteur qui définit un décalage des poids à l’instant $t$.

4. Enfin, le vecteur de poids est focalisé : $w_t(i) = w_t(i)^{\gamma_t}, \gamma_t > 1$.

Une fois que la tête a mis à jour les poids, elle interagit avec la mémoire :
— Dans le cas de la tête de lecture, elle calcule une combinaison linéaire des zones mémoire, pondérées par les poids $w_t(i)$, et produit le vecteur $\mathbf{r}_t$, fourni au contrôleur de l’instant suivant.
— Dans le cas de la tête d’écriture, le contenu de la mémoire est mis à jour selon la formule $M_t(i) = M_{t-1}(i)(1 - w_t(i)\mathbf{e}_t) + w_t(i)\mathbf{a}_t$, où $\mathbf{e}_t$ est un vecteur d’effacement, dont les composantes sont dans \{0,1\} et $\mathbf{a}_t$ est un vecteur d’ajout.

--- [FIGURE] ---
FIGURE 3-5 – Exemple de machine de Turing neuronale dépliée dans le temps, où le contrôleur est un LSTM. Les accès en écriture du LSTM dans la mémoire sont représentés par des flèche rouges, les accès en lecture en bleu.
La figure présente un schéma détaillé d'une machine de Turing neuronale dépliée dans le temps avec un contrôleur LSTM. On observe une séquence horizontale de blocs rectangulaires orange étiquetés LSTM₀ à LSTMₜ, reliés par des flèches oranges vers la droite pour représenter les transitions temporelles. Chaque LSTM est connecté en bas à un cercle vert portant l'étiquette r_i/x_i (par exemple, r₀/x₀, r₁/x₁, etc.), avec des flèches bleues pointant vers le haut depuis ces cercles jusqu'aux LSTMs (accès en lecture). Des flèches rouges descendantes partent de chaque LSTM vers des éléments étiquetés M₀ à Mₜ (non visibles directement mais suggérés par les flèches), indiquant les accès en écriture. En haut de chaque LSTM, un cercle blanc avec l'étiquette h_i (h₀, h₁, etc.) est relié à un autre cercle blanc supérieur portant ŷ_i (ŷ₀, ŷ₁, etc.). Des lignes pointillées bleues relient les M₀ à Mₜ en travers de la figure. Les couleurs sont codées : orange pour les LSTMs, vert pour les cellules de mémoire r_i/x_i, rouge pour les écritures et bleu pour les lectures.

Sur le plan pédagogique, cette illustration montre comment un contrôleur LSTM interagit avec une mémoire externe via des opérations d'écriture (flèches rouges) et de lecture (flèches bleues) à chaque étape temporelle. À chaque temps t, le LSTM traite l'entrée x_t, lit les données stockées dans la mémoire via r_t/x_t, modifie cette mémoire en écrivant des informations via M_t, puis génère une sortie ŷ_t à partir de son état caché h_t. La dépliage temporel permet de visualiser le flux d'information entre les composants et l'évolution dynamique de la mémoire au cours du traitement.
--- [/FIGURE] ---

## 4- QUELQUES APPLICATIONS

Comme les réseaux convolutifs, les réseaux récurrents ont depuis leur introduction trouvé de nombreuses applications.

Traitement automatique du langage
Les réseaux récurrents sont utilisés en traitement automatique du langage, notamment à des fins génératives. Ces réseaux permettent de modéliser un langage (prédire la probabilité d’un mot donné étant donnés les mots précédents) et de générer du texte à partir du modèle appris. De nombreuses applications découlent de cette modélisation : génération de texte « au style de » (génération d’un texte dans le style de Shakespeare, à partir d’un RNN appris sur le corpus des œuvres de l’auteur par exemple, génération de textes manuscrits (figure 3-6(a)), génération de pages Wikipedia, ou même génération d’articles scientifiques, à partir des sources LATEXd’un ouvrage et d’un LSTM multicouche.

Traduction automatique
7

<!-- loc page=8 -->

Apprentissage artifice, concepts et algorithmes
Apprentissage artifice, concepts et algorithmes
Apprentissage artifice, concepts et algorithmes.

(a) Génération de texte manuscrit (source [2])

(b) Génération de légendes (source [6])

--- [FIGURE] ---
FIGURE 3-6 – Quelques applications des réseaux récurrents.
L'image 1 est une photographie d'un marché extérieur où plusieurs personnes sont visibles : certaines assises ou debout, vêtues de couleurs variées (bleu, jaune), et des légumes comme des feuilles vertes et ce qui semble être des bananes sont disposés sur le sol. En arrière-plan, on observe d'autres étals avec des contenants et des personnes en mouvement. Aucun axe, échelle ou courbe n'est présent car il s'agit d'une image réelle sans éléments graphiques techniques. Sur le plan pédagogique, cette image sert de contexte visuel pour illustrer une application concrète de la reconnaissance d'images.

L'image 2 combine la photographie de l'image 1 à gauche avec un schéma technique au centre et du texte à droite. Au centre, on distingue clairement deux blocs : le premier, étiqueté « Vision Deep CNN », présente une structure triangulaire composée de cercles interconnectés ; le second, étiqueté « Language Generating RNN », est un rectangle contenant un symbole circulaire avec une flèche en boucle. À droite, le texte indique : « A group of people shopping at an outdoor market. There are many vegetables at the fruit stand. » Sur le plan pédagogique, cette image illustre la chaîne de traitement où un réseau convolutif profond (CNN) analyse une image et transmet les données à un réseau récurrent génératif (RNN) pour produire une description textuelle.

L'image 3 est exclusivement le schéma technique sans photographie ni texte. Elle montre uniquement le bloc « Vision Deep CNN » avec sa structure triangulaire de cercles interconnectés et le bloc « Language Generating RNN » avec son symbole circulaire en boucle. Aucune autre information visuelle n'est présente. Sur le plan pédagogique, cette image simplifiée met en évidence l'architecture fondamentale du modèle d'analyse visuelle vers texte, isolant les composants techniques sans contexte supplémentaire.
--- [/FIGURE] ---

La traduction automatique de texte procède de la même stratégie que la modélisation d’une langue. Deux réseaux récurrents sont entraînés, chacun dans une des langues, et le RNN traducteur calcule sa sortie en fonction de la couche cachée du premier réseau.

Analyse de sentiments
Détecter de manière automatique l’opinion du public sur un sujet donné intéresse de plus en plus le domaine commercial. Ce domaine, largement alimenté par les réseaux sociaux, les avis et recommandations déposées sur les sites Internet, est un champ de prédilection pour les réseaux profonds. Des réseaux récurrents (notamment LSTM structurés en arbres) sont utilisés à cet effet et servent de base à des systèmes de recommandation.

Résumé automatique
Les réseaux récurrents permettent de produire des résumés abstraits de textes (i.e. générer de nouvelles phrases, en opposition à extraire les mots les plus importants d’un texte). Les modèles utilisés sont des réseaux récurrents avec mécanisme d’attention. Un système d’encodage/décodage est mis en place dans le réseau, où l’encodeur est par exemple un GRU bidirectionnel et le décodeur un GRU dont l’état caché a la même taille que celui de l’encodeur. Les modèles sont appris et validés sur des corpus d’diéss (DUC, CNN/Daily Mail par exemple).

Reconnaissance de la parole
L’utilisation de réseaux LSTM bidirectionnels, qui permettent à la fois d’exploiter les contextes passé et futur, et de garder trace d’un contexte à longue échéance, a montré de bonnes performances dans la tache de reconnaissance de la parole.

Annotation d’images
Couplé à un réseau convolutif, un RNN permet de générer des descriptions (légendes) d’images non labelisées. Le réseau convolutif produit des descripteurs, qui servent d’entrée à un réseau récurrent type LSTM (figure 3-6(b)).
8

<!-- loc page=9 -->

## 5- PARTIE PRATIQUE

### 5.1- Classification de séquences

L’objectif de cette partie est d’appendre à l’aide de réseaux récurrents à classifier des séquences (ici des phrases) en plusieurs classes. Le cas d’étude est une base de données d’avis de films (IMDB) : pour chaque film, un avis est rédigé, et un label positif (1) ou négatif (0) est assigné à cet avis. Il s’agit alors d’appendre à reconnaître la qualité de l’avis, et de pouvoir assigner un label 0/1 à un nouvel avis, jamais vu par le réseau.

Vous avez à disposition un notebook donnant l’essentiel des codes permettant de :
— lire la base de données
— mettre en forme les données
— proposer un modèle de base auquel comparer vos résultats

Votre travail consiste alors à construire un réseau récurrent permettant d’effectuer la tâche et de le tester. Pour ceci, vous aurez à :
— utiliser la fonction keras.preprocessing.sequence.pad_sequences() pour prétraiter les données d’en-traînement. Cette fonction créera un tableau 2D, avec en ligne les 25000 avis et en colonnes les maxlen premiers mots de l’avis. Si l’avis est plus long, il sera coupé, si l’avis est plus court il sera complété par des 0.
— construire un réseau avec
  1. une couche d’Embedding, à la dimension d’entrée appropriée, et dont la dimension de sortie est égale à 10. Le modèle apprendra donc à représenter chaque mot par un vecteur de $\mathbb{R}^{10}$
  2. une (ou plusieurs) couche(s) LSTM à 32 neurones
  3. une couche dense de sortie, à activation sigmoïde (problème de classification binaire)

Le modèle sera compilé avec la fonction de perte adéquate pour un tel problème, et entraîné sur des batchs de taille 128.
— améliorer ce réseau en utilisant un modèle bidirectionnel. Pour ceci, il suffit d’encapsuler la ou les couche(s) LSTM dans une couche bidirectionnelle. Si le modèle sur apprend, vous pouvez ajouter une couche de Dropout.

### 5.2- Prévision de séries temporelles

En utilisant l’API Keras, on se propose d’implémenter un RNN permettant de faire de la prédiction de série temporelle. Le notebook RNN prevision propose de construire un réseau récurrent simple (sans module LSTM ou GRU) permettant de faire de la prévision méteo à partir de données de température, pression et humidité mesurées pendant 5 ans. Les données sont issues d’un challenge Kaggle. Elles sont mises en forme pour un RNN simple, implémenté sous Keras via la couche SimpleRNN. Votre travail consiste simplement à :
— définir (fonction myRNN) un réseau récurrent comportant une couche SimpleRNN et deux couches complètement connectées (figure 5-7).
— Proposer des graphiques de comparaison des courbes réelles et prédites.
— Comparer les prédictions avec celles calculées par des réseaux LSTM et GRU.

<!-- loc page=10 -->

5375555904

rnn: SimpleRNN
input: (None, 1, 8)
output: (None, 128)

Dense1: Dense
input: (None, 128)
output: (None, 32)

output: Dense
input: (None, 32)
output: (None, 1)

--- [FIGURE] ---
FIGURE 5-7 – Réseau récurrent à réaliser
La figure présente un schéma d'architecture de réseau neuronal structuré en blocs rectangulaires reliés par des flèches verticales orientées vers le bas, illustrant un flux de données séquentiel. En haut, un rectangle contient le texte « 5375555904 », probablement une entrée ou un identifiant numérique. Ce bloc est suivi d'un rectangle étiqueté « rnn: SimpleRNN » avec deux sous-étiquettes : « input: (None, 1, 8) » et « output: (None, 128) ». Ensuite, un autre rectangle nommé « Dense1: Dense » affiche « input: (None, 128) » et « output: (None, 32) », relié par une flèche à un dernier bloc intitulé « output: Dense » avec « input: (None, 32) » et « output: (None, 1) ». Les formes sont des rectangles simples sans couleurs spécifiques mentionnées, les étiquettes sont en texte noir sur fond blanc, et les flèches noires indiquent la direction du traitement. Sur le plan pédagogique, cette figure illustre une architecture de réseau récurrent (SimpleRNN) traitant des données temporelles ou séquentielles (indiqué par la forme d'entrée (None, 1, 8), suggérant un temps avec 8 caractéristiques), suivie de couches dense pour réduire les dimensions et produire une sortie scalaire finale (output: (None, 1)), typique d'un modèle classificateur ou prédicteur. La légende « FIGURE 5-7 – Réseau récurrent à réaliser » confirme que cette illustration vise à expliquer la construction d’un réseau récurrent simple intégrant des couches dense pour une tâche de classification ou de regression.
--- [/FIGURE] ---
10

<!-- loc page=11 -->

[1] J Chung, Ç Gülçehre, K Cho, and Y Bengio. Empirical evaluation of gated recurrent neural networks on sequence modeling. CoRR, abs/1412.3555, 2014.

[2] A. Graves. Generating sequences with recurrent neural networks. CoRR, abs/1308.0850, 2013.

[3] A Graves, G Wayne, and I Danihelka. Neural turing machines. CoRR, abs/1410.5401, 2014.

[4] S. Hochreiter and J. Schmidhuber. Long short-term memory. Neural Comput., 9(8) :1735–1780, November 1997.

[5] M. Schuster and K.K. Paliwal. Bidirectional recurrent neural networks. Trans. Sig. Proc., 45(11) :2673–2681, November 1997.

[6] O. Vinyals, A. Toshev, S. Bengio, and D. Erhan. Show and tell: A neural image caption generator. CoRR, abs/1411.4555, 2014.
11
BIBLIOGRAPHIE
