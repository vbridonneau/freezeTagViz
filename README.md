# Freeze Tag Problem — Visualiseur interactif

Outil de visualisation et d'exploration heuristique du **Freeze Tag Problem (FTP)** dans le disque unité euclidien.

## Le problème

Le Freeze Tag Problem consiste à réveiller $n$ robots endormis à partir d'un unique robot actif, en minimisant le **makespan** défini comme le temps qu'il faut pour que tous les robots soient réveillés. Les robots se déplacent à vitesse 1 et réveillent un robot endormi en le touchant. Dès qu'un robot est réveillé, il devient lui-même actif et peut en réveiller d'autres rendant le problème intrinsèquement **distribué**.

**Conjecture ouverte (Bonichon, Gavoille, Hanusse, Odak — CCCG 2024) :**
Pour tout $n \geq 1$ et toute configuration de $n$ robots dans le disque unité, il existe une stratégie de réveil de makespan au plus $1 + 2\sqrt{2} \approx 3.828$ atteinte dans le cas $n=4$.

L'objectif de ce projet est de proposé un outil pour explorer la conjecture et de testé différentes hyppothèses via une interface graphique. La fonctionnalité principale de cette interface est de permettre de visualiser une solution optimale affichée sous la forme d'un arbre de réveil possible (solution exacte).

---

## Fonctionnalités

Ce petit outil propose plusieurs fonctionnalités dans le but d'explorer graphiquement le problème et de tester des configurations dans le but de tester les limites de la conjecture précédemment montrée.

### Visualiseur principal
- Placement et déplacement interactif des robots dans le disque unité
- Calcul et affichage de l'**arbre de réveil optimal** (vert)
- Affichage du makespan courant et comparaison à la borne $1 + 2\sqrt{2}$

### Mode Simulation
- Échantillonnage aléatoire massivement parallèle de configurations
- Identification et affichage de la pire configuration trouvée
- Export CSV des résultats

### Mode Exploration
- Hill-climbing depuis la configuration courante
- Perturbation gaussienne à décroissance adaptative
- Recherche locale du pire makespan pour $n$ fixé

### Champ gravitationnel
- Visualisation 3D interactive du champ de potentiel induit par les robots
- Chaque robot exerce une "masse" gaussienne (rayon d'action $\sigma$, réglable)
- Calcul et affichage du **gradient en l'origine** $\nabla\varphi(0,0)$ — direction naturelle du premier réveil
- Droite de partage perpendiculaire au gradient, divisant le plan en deux sous-problèmes
- Vue de dessus avec courbes de niveau

---

## Stack technique

| Composant | Technologie |
|---|---|
| Langage | Python 3.10+ |
| Interface graphique | Matplotlib (widgets, 3D) |
| Calcul numérique | NumPy |
| Accélération JIT | Numba (`@njit`, LLVM) |
| Parallélisation | `multiprocessing.Pool` |
| Algorithme exact | DP bitmask bottom-up ($O(3^n \cdot n)$) |

---

## Installation

```bash
git clone <url-du-repo>
cd freeze-tag-viz
pip install -r requirements.txt
python main.py
```

Numba est optionnel mais fortement recommandé : il compile la DP bitmask en code machine via LLVM, apportant un gain d'environ ×15 sur le calcul du makespan exact (mesuré sur n=8 à 10).

Sans Numba, le script bascule automatiquement sur un fallback `lru_cache`.

---

## Utilisation

**Ajouter des robots** : pour ajouter un robot, il suffit de cliquer sur `+ Robot` puis cliquer dans le disque.
 
**Obtenir un configuration aléatoires** : il suffit d'utiliser le boutton `Aléatoire` pour que les robots présents soient bougés aléatoirements.

**Déplacer un robot** : cliquer-glisser directement dans le disque.

**Lancer une simulation** : sélectionner `Simulation` dans le menu, régler $N$ (nombre de configurations à tester), cliquer `Simuler`.

**Explorer** : sélectionner `Exploration`, régler le nombre d'itérations de hill-climbing, cliquer `Explorer`.

**Visualiser le champ** : cliquer dans coin haut droit du disque pour qu'une une fenêtre 3D s'ouvre, le slider $\sigma$ est interactif.

---

## Contexte scientifique

Travaux de référence :

- Bonichon, Casteigts, Gavoille, Hanusse — *DISC 2024* (arXiv:2402.03258) : norme L₁, makespan ≤ 5r, optimal.
- Bonichon, Gavoille, Hanusse, Odak — *CCCG 2024* : norme L₂ euclidienne, bornes serrées dans le plan.
- Gavoille, Hanusse, Le Bouder, Marcé — *PODC 2025* (arXiv:2503.22521) : version distribuée, makespan $O(\rho + \ell^2 \log(\rho/\ell))$.