# ── Utilitaires ────────────────────────────────────────────────────────────────
import math

import numpy as np


def dist(a : tuple[float, float], b : tuple[float, float]) -> float:
    """Calcule la distance euclidienne entre deux points du plan.

    Args:
        a: Premier point, sous la forme (x, y).
        b: Second point, sous la forme (x, y).

    Returns:
        Distance euclidienne ||a - b||_2.
    """
    return math.hypot(a[0]-b[0], a[1]-b[1])

def random_in_disk() -> np.ndarray:
    """Tire un point uniformément au hasard dans le disque unité ouvert.

    La méthode correcte pour obtenir une distribution uniforme sur le disque
    consiste à tirer r = sqrt(U) avec U ~ Uniform(0, 1) (et non U directement),
    afin de compenser la croissance de l'aire en r*dr.

    Returns:
        Tableau numpy de forme (2,) contenant les coordonnées (x, y)
        du point tiré, avec x^2 + y^2 < 1.
    """
    local_state = np.random.RandomState(None) # Au cas où le parallélisme est authorisé
    r           = math.sqrt(local_state.uniform(0, 1))
    theta       = local_state.uniform(0, 2 * math.pi)
    return np.array([r * np.cos(theta), r * np.sin(theta)])


def clamp_disk(x : float, y : float, R=1.0) -> tuple[float, float]:
    """Projette un point sur le bord du disque s'il en est sorti.

    Si le point (x, y) est à l'intérieur ou sur le cercle de rayon R,
    il est retourné inchangé. Sinon il est ramené sur le cercle par
    normalisation radiale (projection sur le bord le plus proche).

    Args:
        x: Abscisse du point.
        y: Ordonnée du point.
        R: Rayon du disque (défaut : 1.0).

    Returns:
        Paire (x', y') avec x'^2 + y'^2 <= R^2.
    """
    n2 = x*x + y*y
    if n2 > R*R:
        s = R / math.sqrt(n2)
        return x*s, y*s
    return x, y

# ── Numba JIT : DP bitmask bottom-up ─────────────────────────────────────────
try:
    import numpy as _np
    from numba import njit as _njit

    @_njit(cache=True)
    def dp_bitmask(n, dist_mat):
        """DP (Dynamic Programming) bitmask bottom-up compilée par Numba.

        one[mask, pos] : makespan minimal pour un robot en 'pos' qui doit
                         réveiller exactement les robots de 'mask'.
        two[mask, p]   : makespan minimal pour deux robots situés tous deux
                         en p (juste après un réveil) qui se partagent 'mask'.

        choix_i[mask, pos] : premier robot visité dans la solution optimale de one.
        choix_t[mask, p]   : sous-ensemble attribué au premier robot dans two.

        Complexité : O(3^n * n) en temps, O(2^n * n) en mémoire.
        """
        INF = 1e18
        total = 1 << n

        one = _np.full((total, n + 1), INF)
        two = _np.full((total, n + 1), INF)

        for p in range(n + 1):
            one[0, p] = 0.0
            two[0, p] = 0.0

        for mask in range(1, total):
            # one[mask, .] n'utilise que two[rest, .] avec rest < mask : déjà calculé.
            for pos in range(n + 1):
                best = INF
                for i in range(n):
                    if not ((mask >> i) & 1):
                        continue
                    c = dist_mat[pos, i] + two[mask ^ (1 << i), i]
                    best = min(best, c)
                one[mask, pos] = best

            # two[mask, .] utilise one[t, .] et one[mask^t, .], tous deux <= mask.
            # Seules les positions de robots (0..n-1) sont utiles ici.
            for p in range(n):
                best = INF
                t = mask
                # Pour chaque sous ensemble de robots de mask, on cherche quel partage des robots à se réveiller est optimal depuis la position de p
                while True:
                    c = max(one[t, p], one[mask ^ t, p])
                    best = min(best, c)
                    if t == 0:
                        break
                    t = (t - 1) & mask
                two[mask, p] = best

        return one[total - 1, n]

    @_njit(cache=True)
    def _dp_bitmask_and_tree(n, dist_mat):
        """DP bitmask bottom-up compilée par Numba.

        one[mask, pos] : makespan minimal pour un robot en 'pos' qui doit
                         réveiller exactement les robots de 'mask'.
        two[mask, p]   : makespan minimal pour deux robots situés tous deux
                         en p (juste après un réveil) qui se partagent 'mask'.

        choix_i[mask, pos] : premier robot visité dans la solution optimale de one.
        choix_t[mask, p]   : sous-ensemble attribué au premier robot dans two.

        Complexité : O(3^n * n) en temps, O(2^n * n) en mémoire.
        """
        INF = 1e18
        total = 1 << n

        one = _np.full((total, n + 1), INF)
        two = _np.full((total, n + 1), INF)
        choix_i = _np.full((total, n + 1), -1, dtype=_np.int32)
        choix_t = _np.zeros((total, n + 1), dtype=_np.int64)

        for p in range(n + 1):
            one[0, p] = 0.0
            two[0, p] = 0.0

        for mask in range(1, total):
            # one[mask, .] n'utilise que two[rest, .] avec rest < mask : déjà calculé.
            for pos in range(n + 1):
                best = INF
                arg = -1
                for i in range(n):
                    if not ((mask >> i) & 1):
                        continue
                    c = dist_mat[pos, i] + two[mask ^ (1 << i), i]
                    if c < best:
                        best = c
                        arg = i
                one[mask, pos] = best
                choix_i[mask, pos] = arg

            # two[mask, .] utilise one[t, .] et one[mask^t, .], tous deux <= mask.
            # Seules les positions de robots (0..n-1) sont utiles ici.
            for p in range(n):
                best = INF
                arg = 0
                t = mask
                while True:
                    c = max(one[t, p], one[mask ^ t, p])
                    if c < best:
                        best = c
                        arg = t
                    if t == 0:
                        break
                    t = (t - 1) & mask
                two[mask, p] = best
                choix_t[mask, p] = arg

        return one[total - 1, n], choix_i, choix_t

    import numpy as np

    def dp_bitmask_numpy(n, dist_mat):
        """Même DP que _dp_bitmask, vectorisée avec NumPy par couches de cardinal."""
        INF = 1e18
        total = 1 << n

        one = np.full((total, n + 1), INF)
        two = np.full((total, n + 1), INF)
        one[0, :] = 0.0
        two[0, :] = 0.0

        all_masks = np.arange(total, dtype=np.int64)
        bit_ids = np.arange(n, dtype=np.int64)
        # bits_all[mask, b] vaut 1 si le robot b appartient à mask
        bits_all = (all_masks[:, None] >> bit_ids) & 1
        popcount = bits_all.sum(axis=1)

        for k in range(1, n + 1):
            M = all_masks[popcount == k]   # tous les masques de cardinal k
            m = M.size

            # --- Calcul de one sur la couche ---
            # Pour chaque robot i, on traite d'un coup tous les masques qui le contiennent.
            for i in range(n):
                Mi = M[((M >> i) & 1) == 1]
                # cand[a, pos] = dist_mat[pos, i] + two[Mi[a] sans i, i]
                cand = dist_mat[:, i][None, :] + two[Mi ^ (1 << i), i][:, None]
                one[Mi] = np.minimum(one[Mi], cand)

            # --- Énumération vectorisée des sous-ensembles ---
            # Tous les masques de la couche ont exactement k bits, donc exactement
            # 2^k sous-ensembles : on peut les ranger dans un tableau rectangulaire.
            idx = np.nonzero(bits_all[M])[1].reshape(m, k)   # positions des bits à 1
            pw = np.int64(1) << idx                           # (m, k)
            j = np.arange(1 << k, dtype=np.int64)
            sel = (j[:, None] >> np.arange(k)) & 1            # (2^k, k)
            T = sel @ pw.T                                    # (2^k, m) : tous les t
            C = M[None, :] ^ T                                # (2^k, m) : les mask ^ t

            # --- Calcul de two sur la couche ---
            for p in range(n):
                two[M, p] = np.maximum(one[T, p], one[C, p]).min(axis=0)

        return one[total - 1, n]

    def dist_mat(robots : list[tuple[float, float]]):
        """
            Calcul la matrice des distances entre chaque robots présents dans le dique unité

            Args:
                robots : Liste des position donné comme une liste de tuple de floattant (coordonnées cartésiennes).

            Returns:
                Tableau 2D-numpy de n*n éléments où n=len(robots)
        """
        n   = len(robots)
        dm  = _np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dm[i,j] = math.hypot(robots[i][0]-robots[j][0], robots[i][1]-robots[j][1])
        return dm

    def makespan_exact(origin, robots):
        n   = len(robots)
        pts = list(robots) + [tuple(origin)]
        dm  = dist_mat(pts)
        return float(_dp_bitmask_and_tree(n, dm)[0])

    # Pré-compiler pour n=4 au démarrage (évite la latence au 1er vrai appel)
    def warmup_numba():
        pts = [(math.cos(2*math.pi*i/4), math.sin(2*math.pi*i/4)) for i in range(4)]
        makespan_exact((0.,0.), pts)

    _NUMBA_OK = True
    print("[freeze_tag] Numba détecté — DP JIT activée (×15 speedup)")

except ImportError:
    import sys

    sys.exit(1)

def compute_makespan(origin : tuple[float], robots : list[tuple[float]]) -> float:
    """Calcule le makespan optimal ou approche selon la taille de l'instance.

    Dispatch automatique : DP bitmask exacte dont la complexité est en O(3^n * n).

    Args:
        origin: Position du robot initialement eveille, sous la forme (x, y)
                ou np.ndarray de forme (2,).
        robots: Liste des positions (x, y) des robots endormis.

    Returns:
        Makespan en unite de distance (temps = distance car vitesse = 1).
        Vaut 0.0 si la liste de robots est vide.
    """
    n = len(robots)
    if n == 0: return 0.0
    return (makespan_exact(tuple(origin), tuple(map(tuple, robots))))


class FreezeTagInstance:
    def __init__(self, origin : tuple[float, float], robots : list[tuple[float, float]] | np.ndarray):
        self._wakeup_tree = []
        self._ms          = 0.0
        self.origin       = origin
        self.robots       = robots

    def _parents(self, choix_i, choix_t):
        """parent[i] = indice du noeud (robot, ou n pour l'origine) qui réveille i."""
        n = len(self.robots)
        parent = np.full(n, -1, dtype=np.int64)
        pile = [((1 << n) - 1, n)]          # état "un robot en pos doit réveiller mask"
        while pile:
            mask, pos = pile.pop()
            if mask == 0:
                continue
            i = int(choix_i[mask, pos])
            parent[i] = pos
            rest = mask ^ (1 << i)
            t = int(choix_t[rest, i])
            pile.append((t, i))             # premier robot partant de i
            pile.append((rest ^ t, i))      # second robot partant de i
        return parent

    def _build_tree(self, parents):
        self._wakeup_tree = []
        robots = self.robots + [self.origin]
        for dst, src in enumerate(parents):
            self._wakeup_tree.append((robots[src], robots[dst]))


    def build(self) -> None:
        """Construit les aretes de l'arbre de reveil optimal et calcul le makespan optimal

        La liste des aretes de l'arbre de reveil est construite sous forme de paires
        de points, utilisee pour l'affichage de la strategie optimale dans la
        fenetre principale.

        Args:
            origin: Position du robot initialement eveille, sous la forme (x, y)
                    ou np.ndarray de forme (2,).
            robots: Liste des positions (x, y) des robots endormis.

        Returns:
            Le makespan et la liste de paires ((x1, y1), (x2, y2)) representant chaque arete
            de l'arbre de reveil. Liste vide si robots est vide.
        """
        n = len(self.robots)
        if n == 0: return
        pts = list(map(tuple, self.robots + [self.origin]))

        self._ms, choix_i, choix_t   = _dp_bitmask_and_tree(n, dist_mat(pts))

        self._build_tree(self._parents(choix_i, choix_t))

        # self._ms, self._wakeup_tree = one((1<<n)-1, tuple(self.origin))

    def getMs(self):
        return self._ms

    def getWakeTree(self):
        return self._wakeup_tree

def solve(origin : tuple[float, float], robots : list[tuple[float, float]] | np.ndarray) -> tuple[float, list[list[tuple[float]]]]:
    instance = FreezeTagInstance(origin, robots)
    instance.build()
    wakeTree = instance.getWakeTree()
    return instance.getMs(), wakeTree

def wake_tree(origin : tuple[float, float], robots : list[tuple[float, float]]) -> list[list[tuple[float]]]:
    return solve(origin, robots)[1]
