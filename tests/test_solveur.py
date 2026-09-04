"""
Tests de la fonction get_wake_tree() et de la conjecture.

Couvre :
  - Cas de base et cas vides
  - Valeurs analytiques connues
  - Cohérence interne makespan / arbre de réveil
  - Conjecture 1+2√2 sur configurations aléatoires
"""

import math
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from freeze_tag import solve, makespan_exact, dist, random_in_disk

TARGET = 1 + 2 * math.sqrt(2)   # ≈ 3.8284
ORIGIN = (0.0, 0.0)
TOL    = 1e-9

# ── Helpers ───────────────────────────────────────────────────────────────────

def makespan_from_edges(origin, edges):
    """
    Recalcule le makespan depuis la liste d'arêtes retournée par get_wake_tree().

    Parcourt l'arbre en BFS depuis l'origine et calcule le temps d'arrivée
    cumulé à chaque nœud. Le makespan est le maximum de ces temps.
    Permet de vérifier la cohérence interne du résultat de get_wake_tree().
    """
    if not edges:
        return 0.0

    # Construction du graphe orienté parent -> enfants
    children = {}
    for p1, p2 in edges:
        children.setdefault(p1, []).append(p2)

    # BFS depuis l'origine
    from collections import deque
    queue    = deque([(origin, 0.0)])
    max_time = 0.0

    while queue:
        pos, t = queue.popleft()
        max_time = max(max_time, t)
        for child in children.get(pos, []):
            queue.append((child, t + dist(pos, child)))

    return max_time


def polygon_robots(n):
    """Retourne n robots placés aux sommets d'un polygone régulier inscrit."""
    return [
        (math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n))
        for i in range(n)
    ]


# ── Cas de base ───────────────────────────────────────────────────────────────

class TestWakeTreeBase:
    def test_liste_vide(self):
        """Aucun robot à réveiller : makespan = 0 et arbre vide."""
        _, edges = solve(ORIGIN, [])
        assert edges == []

    def test_retourne_un_tuple(self):
        """get_wake_tree() retourne bien un tuple de taille 2 : un 'float' et un 'list'."""
        result = solve(ORIGIN, [(0.5, 0.0)])
        assert isinstance(result, tuple) and len(result) == 2
        ms, edges = result
        assert isinstance(ms, float)
        assert isinstance(edges, list)



# ── Valeurs analytiques connues ───────────────────────────────────────────────

class TestMakespanExactAnalytique:
    def test_n1_sur_axe(self):
        """n=1, robot en (d, 0) : makespan = d."""
        for d in [0.3, 0.7, 1.0]:
            ms, _ = solve(ORIGIN, [(d, 0.0)])
            assert ms == pytest.approx(d, abs=TOL), f"échec pour d={d}"

    def test_n1_quelconque(self):
        """n=1, robot en (x, y) : makespan = sqrt(x²+y²)."""
        robot = (0.6, 0.8)
        ms, _ = solve(ORIGIN, [robot])
        assert ms == pytest.approx(dist(ORIGIN, robot), abs=TOL)

    def test_n2_symetrique(self):
        """
        n=2, robots en (1,0) et (-1,0).
        Optimal : p0 → (1,0) coût 1, puis ce robot va vers (-1,0) coût 2.
        Makespan = 3.
        """
        ms, _ = solve(ORIGIN, [(1.0, 0.0), (-1.0, 0.0)])
        assert ms == pytest.approx(3.0, abs=TOL)

    def test_n2_quasi_aligne(self):
        """
        n=2, robots en (1,0) et (0, ε).
        Optimal : p0 → (0,ε) coût ε, puis les deux robots repartent de (0,ε).
        L'un va vers (1,0) (coût dist((0,ε),(1,0)) = sqrt(1+ε²)), l'autre
        ne fait rien (partition vide). Makespan = ε + sqrt(1+ε²).
        """
        eps      = 0.01
        expected = eps + math.sqrt(1.0 + eps**2)
        ms, _    = solve(ORIGIN, [(1.0, 0.0), (0.0, eps)])
        assert ms == pytest.approx(expected, abs=TOL)

    def test_n3_optimal_connu(self):
        """
        n=3, robots en (1,0), (0,1), (-1,0).
        Optimal : p0 → (0,1) coût 1, puis un robot vers (1,0) et l'autre
        vers (-1,0), chacun à distance sqrt(2). Makespan = 1 + sqrt(2).
        """
        robots = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)]
        ms, _  = solve(ORIGIN, robots)
        assert ms == pytest.approx(1 + math.sqrt(2), abs=TOL)

    def test_n4_carre_inscrit(self):
        """
        n=4, robots aux quatre coins du carré inscrit dans le disque.
        Makespan optimal = 1 + 2√2 (la borne de la conjecture).
        """
        robots = polygon_robots(4)
        ms, _  = solve(ORIGIN, robots)
        assert ms == pytest.approx(TARGET, abs=TOL)

    def test_makespan_positif(self):
        """Le makespan est toujours >= 0."""
        for n in range(1, 8):
            robots = [random_in_disk() for _ in range(n)]
            ms, _  = solve(ORIGIN, robots)
            assert ms >= 0.0


# ── Cohérence interne makespan / arbre ────────────────────────────────────────

class TestMakespanExactCoherence:
    def test_arbre_couvre_tous_les_robots(self):
        """Chaque robot apparaît comme destination dans au moins une arête."""
        robots = polygon_robots(5)
        _, edges = solve(ORIGIN, robots)
        destinations = {p2 for _, p2 in edges}
        for r in robots:
            assert r in destinations, f"robot {r} absent de l'arbre"

    def test_nombre_aretes_egal_n(self):
        """Un arbre couvrant n noeuds depuis p0 a exactement n arêtes."""
        for n in range(1, 9):
            robots   = polygon_robots(n)
            _, edges = solve(ORIGIN, robots)
            assert len(edges) == n, f"n={n} : {len(edges)} arêtes au lieu de {n}"

    def test_coherence_makespan_arbre(self):
        """
        Le makespan retourné par get_wake_tree() correspond au plus long chemin
        dans l'arbre retourné (recalculé indépendamment par BFS).
        """
        for n in range(1, 9):
            robots      = polygon_robots(n)
            ms, edges   = solve(ORIGIN, robots)
            ms_recompute = makespan_from_edges(ORIGIN, edges)
            assert ms == pytest.approx(ms_recompute, abs=TOL), (
                f"n={n} : get_wake_tree()={ms:.6f}, BFS={ms_recompute:.6f}"
            )

    def test_coherence_sur_configurations_aleatoires(self):
        """Cohérence makespan/arbre vérifiée sur 20 configurations aléatoires."""
        rng = np.random.default_rng(seed=42)
        for _ in range(20):
            n      = rng.integers(1, 9)
            robots = [tuple(random_in_disk()) for _ in range(n)]
            ms, edges = solve(ORIGIN, robots)
            ms_bfs    = makespan_from_edges(ORIGIN, edges)
            assert ms == pytest.approx(ms_bfs, abs=TOL), (
                f"incohérence : get_wake_tree={ms:.6f}, BFS={ms_bfs:.6f}"
            )

    def test_aretes_partent_de_noeuds_eveilles(self):
        """
        Dans l'arbre, chaque arête (p1, p2) part d'un nœud déjà éveillé :
        soit l'origine, soit une destination d'une arête précédente.
        Propriété fondamentale du FTP.
        """
        robots        = polygon_robots(6)
        _, edges      = solve(ORIGIN, robots)
        eveilles      = {ORIGIN}
        # Les arêtes sont dans l'ordre BFS retourné par get_wake_tree
        for p1, p2 in edges:
            assert p1 in eveilles, f"{p1} n'est pas encore éveillé"
            eveilles.add(p2)


# ── Conjecture 1 + 2√2 ───────────────────────────────────────────────────────

class TestConjecture:
    N_SAMPLES = 200
    SEED      = 0

    @pytest.mark.parametrize("n", range(1, 9))
    def test_conjecture_polygone_regulier(self, n):
        """
        Pour le polygone régulier à n sommets inscrit dans le disque,
        le makespan doit être <= 1 + 2√2.
        """
        robots = polygon_robots(n)
        ms, _  = solve(ORIGIN, robots)
        assert ms <= TARGET + TOL, (
            f"n={n} (polygone) : makespan={ms:.6f} > TARGET={TARGET:.6f}"
        )

    @pytest.mark.parametrize("n", range(1, 9))
    def test_conjecture_aleatoire(self, n):
        """
        Sur N_SAMPLES configurations aléatoires de n robots dans le disque,
        le makespan doit toujours être <= 1 + 2√2.

        Note : ce test n'est pas une preuve formelle (la conjecture est
        ouverte pour 8 <= n <= 280), mais il valide empiriquement le
        comportement du get_wake_treer sur les petites instances.
        """
        rng = np.random.default_rng(seed=self.SEED + n)
        for i in range(self.N_SAMPLES):
            robots = [tuple(random_in_disk()) for _ in range(n)]
            ms, _  = solve(ORIGIN, robots)
            assert ms <= TARGET + TOL, (
                f"n={n}, tirage {i} : makespan={ms:.6f} > TARGET={TARGET:.6f}\n"
                f"robots={robots}"
            )

    def test_borne_atteinte_carre_inscrit(self):
        """
        Le carré inscrit dans le disque atteint exactement la borne 1+2√2.
        Vérifie que la borne est serrée (pas juste une majoration lâche).
        """
        robots = polygon_robots(4)
        ms, _  = solve(ORIGIN, robots)
        assert ms == pytest.approx(TARGET, abs=TOL)