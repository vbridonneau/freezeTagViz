# ── Utilitaires ────────────────────────────────────────────────────────────────
import math
from typing import List, Tuple

import numpy as np


def dist(a : Tuple[float], b : Tuple[float]) -> float:
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
    return np.array([r * math.cos(theta), r * math.sin(theta)])


def clamp_disk(x : float, y : float, R=1.0) -> Tuple[float]:
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
    from numba import njit as _njit
    import numpy as _np

    @_njit
    def _dp_bitmask(n, dist_mat):
        """DP bitmask bottom-up compilee par Numba en code machine (LLVM).
 
        Calcule simultanement deux tables :
          one[mask, pos]      : makespan minimal pour UN robot en 'pos'
                                qui doit reveiller exactement les robots de 'mask'.
          two[mask, p1, p2]   : makespan minimal pour DEUX robots en p1 et p2
                                qui se partagent le reveil de 'mask'.
 
        Convention des indices de position :
          0 .. n-1  : robots endormis
          n         : origine p0 (point de depart)
 
        Complexite : O(3^n * n) en temps, O(2^n * n^2) en memoire.
 
        Args:
            n        : nombre de robots endormis.
            dist_mat : matrice de distances (n+1) x (n+1), precalculee
                       par makespan_exact avant l'appel Numba.
 
        Returns:
            Makespan optimal (flottant) pour reveiller tous les n robots
            depuis l'origine.
        """
        INF   = 1e18
        total = 1 << n

        one = _np.full((total, n+1), INF)
        two = _np.full((total, n+1, n+1), INF)

        for p in range(n+1):
            one[0, p] = 0.
            for q in range(n+1):
                two[0, p, q] = 0.

        for mask in range(1, total):
            # fill one
            for pos in range(n+1):
                best = INF
                for i in range(n):
                    if not (mask >> i & 1): continue
                    rest = mask ^ (1 << i)
                    c = dist_mat[pos, i] + two[rest, i, i]
                    if c < best: best = c
                one[mask, pos] = best
            # fill two — itération sur sous-masques
            for p1 in range(n+1):
                for p2 in range(n+1):
                    best = INF
                    t = mask
                    while True:
                        v1 = one[t, p1]; v2 = one[mask ^ t, p2]
                        c  = v1 if v1 > v2 else v2
                        if c < best: best = c
                        if t == 0: break
                        t = (t - 1) & mask
                    two[mask, p1, p2] = best
        return one[total-1, n]

    def makespan_exact(origin, robots):
        n   = len(robots)
        pts = list(robots) + [tuple(origin)]
        dm  = _np.zeros((n+1, n+1))
        for i in range(n+1):
            for j in range(n+1):
                dm[i,j] = math.hypot(pts[i][0]-pts[j][0], pts[i][1]-pts[j][1])
        return float(_dp_bitmask(n, dm))

    # Pré-compiler pour n=4 au démarrage (évite la latence au 1er vrai appel)
    def warmup_numba():
        pts = [(math.cos(2*math.pi*i/4), math.sin(2*math.pi*i/4)) for i in range(4)]
        makespan_exact((0.,0.), pts)

    _NUMBA_OK = True
    print("[freeze_tag] Numba détecté — DP JIT activée (×15 speedup)")

except ImportError:
    def _dp_bitmask(n, dist_mat):
        pass

    # Fallback lru_cache si numba absent
    def makespan_exact(origin, robots):
        from functools import lru_cache
        n   = len(robots)
        pts = [tuple(r) for r in robots]
        @lru_cache(maxsize=None)
        def one(mask, pos):
            """Makespan minimal pour un seul robot en 'pos' devant reveiller les robots de 'mask'.
 
            Le robot choisit son premier robot cible i dans mask, se deplace
            en i (cout = dist(pos, i)), le reveille, et les deux robots sont
            alors en i. Le reste du travail est delegue a two(rest, i, i).
            On retient le meilleur premier choix (minimum sur tous les i).
 
            Cas de base : mask = 0 -> makespan = 0 (rien a faire).
 
            Note : les resultats sont mis en cache par lru_cache pour eviter
            de recalculer le meme sous-probleme plusieurs fois (memoisation).
 
            Args:
                mask : entier dont le bit i vaut 1 si le robot i est a reveiller.
                pos  : position de depart du robot eveille, sous forme (x, y).
 
            Returns:
                Makespan optimal (flottant) pour ce sous-probleme.
            """
            if mask == 0: return 0.0
            best = float('inf')
            for i in range(n):
                # Si le robot i n'est PAS à réveillé, on passe au suivant
                if not (mask >> i & 1): continue
                # On regarde la solution à partir de du robot r et en considérant maintenant qu'il y a deux robots réveillés en position r qui vont se partager l'exploration du graphe
                r = pts[i]; rest = mask ^ (1 << i)
                # Calcul du makespan optimal à partir de 
                c = dist(pos, r) + two(rest, r, r)
                if c < best: best = c
            return best
        @lru_cache(maxsize=None)
        def two(mask, p1, p2):
            """Makespan minimal pour deux robots en p1 et p2 reveillant 'mask'.
 
            Les deux robots se partagent mask en deux sous-ensembles (t, mask^t).
            Chacun traite sa moitie independamment via one(), et travaille en
            parallele : le makespan est le max des deux durees (on attend le
            plus lent). On minimise sur toutes les partitions possibles.
 
            Cas de base : mask = 0 -> makespan = 0 (rien a faire).
 
            L'enumeration naive parcourt range(mask+1) et filtre avec
            (t & mask) != t les entiers qui ne sont pas des sous-masques de mask.
            C'est moins efficace que la technique (t-1)&mask utilisee dans la
            version Numba, mais la memoisation de lru_cache compense en ne
            recalculant jamais two(mask, p1, p2) pour les memes arguments.
 
            Args:
                mask : entier dont le bit i vaut 1 si le robot i est a reveiller.
                p1   : position du premier robot eveille, sous forme (x, y).
                p2   : position du second robot eveille, sous forme (x, y).
 
            Returns:
                Makespan optimal (flottant) pour ce sous-probleme.
            """
            if mask == 0: return 0.0
            best = float('inf')
            # L'idée de la boucle suivante est de diviser l'ensemble des robots à réveiller en une partition de deux sous-ensembles de telle sorte que les deux robots se partagent les robots à réveiller.
            for t in range(mask + 1):
                # Si le test suivant renvoit False, alors les bits à 1 de 't' ne représente pas un sous-ensemble des bits à 1 de 'mask'
                if (t & mask) != t: continue
                # Calcul l'optimal sachant que deux robots sont à la position de 't' et doivent se partager les robots à réveiller
                c = max(one(t, p1), one(mask ^ t, p2))
                if c < best: best = c
            return best
        ms = one((1 << n) - 1, tuple(origin))
        one.cache_clear(); two.cache_clear()
        return ms

    def warmup_numba(): pass
    _NUMBA_OK = False
    print("[freeze_tag] Numba absent — fallback lru_cache")

def compute_makespan(origin : Tuple[float], robots : List[Tuple[float]]) -> float:
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
    def __init__(self, origin : Tuple[float], robots : List[Tuple[float]]):
        self._wakeup_tree = []
        self._ms          = 0.0
        self.origin       = origin
        self.robots       = robots

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
            Liste de paires ((x1, y1), (x2, y2)) representant chaque arete
            de l'arbre de reveil. Liste vide si robots est vide.
        """
        n = len(self.robots)
        if n == 0: return []
        from functools import lru_cache
        pts = list(map(tuple, self.robots))

        @lru_cache(maxsize=None)
        def one(mask, pos):
            if mask == 0: return 0.0, []
            best, be = float('inf'), []
            for i in range(n):
                if not (mask >> i & 1): continue
                r = pts[i]; rest = mask ^ (1<<i)
                cs, es = two(rest, r, r)
                c = dist(pos, r) + cs
                if c < best: best, be = c, [(pos, r)] + es
            return best, be

        @lru_cache(maxsize=None)
        def two(mask, p1, p2):
            if mask == 0: return 0.0, []
            best, be = float('inf'), []
            for t in range(mask+1):
                if (t & mask) != t: continue
                c1, e1 = one(t, p1); c2, e2 = one(mask^t, p2)
                c = max(c1, c2)
                if c < best: best, be = c, e1+e2
            return best, be

        self._ms, self._wakeup_tree = one((1<<n)-1, tuple(self.origin))
        one.cache_clear(); two.cache_clear()

    def getMs(self):
        return self._ms

    def getWakeTree(self):
        return self._wakeup_tree

def solve(origin : Tuple[float], robots : List[Tuple[float]]) -> tuple[float, List[List[Tuple[float]]]]:
    instance = FreezeTagInstance(origin, robots)
    instance.build()
    return instance.getMs(), instance.getWakeTree()

def wake_tree(origin : Tuple[float], robots : List[Tuple[float]]) -> List[List[Tuple[float]]]:
    return solve(origin, robots)[1]