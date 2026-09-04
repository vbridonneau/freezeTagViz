"""
Tests des fonctions géométriques utilitaires.

Couvre : dist, clamp_disk, random_in_disk.
"""

import math
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from freeze_tag import dist, clamp_disk, random_in_disk


# ── dist ──────────────────────────────────────────────────────────────────────

class TestDist:
    def test_point_identique(self):
        """Distance d'un point à lui-même vaut 0."""
        assert dist((0.0, 0.0), (0.0, 0.0)) == pytest.approx(0.0)

    def test_axe_horizontal(self):
        """Distance sur l'axe horizontal = différence des abscisses."""
        assert dist((0.0, 0.0), (3.0, 0.0)) == pytest.approx(3.0)

    def test_axe_vertical(self):
        """Distance sur l'axe vertical = différence des ordonnées."""
        assert dist((0.0, 0.0), (0.0, 4.0)) == pytest.approx(4.0)

    def test_triangle_rectangle_345(self):
        """Triangle rectangle 3-4-5 : distance = 5."""
        assert dist((0.0, 0.0), (3.0, 4.0)) == pytest.approx(5.0)

    def test_symmetrie(self):
        """dist(a, b) == dist(b, a)."""
        a, b = (0.3, 0.7), (-0.5, 0.2)
        assert dist(a, b) == pytest.approx(dist(b, a))

    def test_inegalite_triangulaire(self):
        """dist(a, c) <= dist(a, b) + dist(b, c) pour tout triplet."""
        a, b, c = (0.0, 0.0), (0.5, 0.3), (1.0, 0.0)
        assert dist(a, c) <= dist(a, b) + dist(b, c) + 1e-12

    def test_valeur_connue(self):
        """Distance entre (1,0) et (-1,0) vaut 2."""
        assert dist((1.0, 0.0), (-1.0, 0.0)) == pytest.approx(2.0)

    def test_points_diagonaux_disque(self):
        """Diagonale du carré inscrit dans le disque unité = 2."""
        p1 = (math.cos(math.pi / 4), math.sin(math.pi / 4))
        p2 = (math.cos(5 * math.pi / 4), math.sin(5 * math.pi / 4))
        assert dist(p1, p2) == pytest.approx(2.0)


# ── clamp_disk ────────────────────────────────────────────────────────────────

class TestClampDisk:
    def test_point_interieur_inchange(self):
        """Un point strictement intérieur est retourné tel quel."""
        x, y = 0.3, 0.4
        cx, cy = clamp_disk(x, y)
        assert cx == pytest.approx(x)
        assert cy == pytest.approx(y)

    def test_origine_inchangee(self):
        """L'origine (0, 0) est un point intérieur, retournée telle quelle."""
        cx, cy = clamp_disk(0.0, 0.0)
        assert cx == pytest.approx(0.0)
        assert cy == pytest.approx(0.0)

    def test_point_bord_inchange(self):
        """Un point exactement sur le cercle unité est retourné inchangé."""
        x, y = 1.0, 0.0
        cx, cy = clamp_disk(x, y)
        assert math.hypot(cx, cy) == pytest.approx(1.0)

    def test_point_exterieur_ramene_sur_bord(self):
        """Un point extérieur est ramené sur le cercle : norme = R."""
        cx, cy = clamp_disk(2.0, 0.0)
        assert math.hypot(cx, cy) == pytest.approx(1.0)

    def test_direction_preservee(self):
        """La projection conserve la direction du vecteur original."""
        x, y = 3.0, 4.0           # norme 5, direction (0.6, 0.8)
        cx, cy = clamp_disk(x, y)
        assert math.hypot(cx, cy) == pytest.approx(1.0)
        # angle préservé : cx/cy == x/y
        assert cx / cy == pytest.approx(x / y)

    def test_point_exterieur_diagonal(self):
        """Point hors disque sur la diagonale ramené à norme 1."""
        cx, cy = clamp_disk(1.0, 1.0)      # norme sqrt(2) > 1
        assert math.hypot(cx, cy) == pytest.approx(1.0)

    def test_rayon_personnalise(self):
        """Le paramètre R est respecté (rayon différent de 1)."""
        cx, cy = clamp_disk(3.0, 0.0, R=2.0)
        assert math.hypot(cx, cy) == pytest.approx(2.0)

    @pytest.mark.parametrize("x,y", [
        (0.5, 0.5), (0.0, 0.9), (-0.7, 0.0), (0.1, -0.1),
    ])
    def test_points_interieurs_parametres(self, x, y):
        """Plusieurs points intérieurs vérifiés : tous retournés inchangés."""
        cx, cy = clamp_disk(x, y)
        assert cx == pytest.approx(x)
        assert cy == pytest.approx(y)


# ── random_in_disk ────────────────────────────────────────────────────────────

class TestRandomInDisk:
    N_SAMPLES = 500

    def test_retourne_ndarray(self):
        """random_in_disk retourne un tableau numpy."""
        p = random_in_disk()
        assert isinstance(p, np.ndarray)

    def test_forme_2(self):
        """Le tableau retourné est de forme (2,)."""
        p = random_in_disk()
        assert p.shape == (2,)

    def test_tous_dans_le_disque(self):
        """Tous les points tirés sont dans le disque unité fermé."""
        for _ in range(self.N_SAMPLES):
            p = random_in_disk()
            assert p[0]**2 + p[1]**2 <= 1.0 + 1e-12, (
                f"Point hors disque : {p}"
            )

    def test_distribution_non_degeneree(self):
        """Les points ne sont pas tous concentrés à l'origine."""
        pts = np.array([random_in_disk() for _ in range(self.N_SAMPLES)])
        assert pts[:, 0].std() > 0.1
        assert pts[:, 1].std() > 0.1

    def test_distribution_couvre_les_quadrants(self):
        """Les quatre quadrants sont tous visités sur N tirages."""
        pts = np.array([random_in_disk() for _ in range(self.N_SAMPLES)])
        assert np.any((pts[:, 0] > 0) & (pts[:, 1] > 0)), "quadrant I absent"
        assert np.any((pts[:, 0] < 0) & (pts[:, 1] > 0)), "quadrant II absent"
        assert np.any((pts[:, 0] < 0) & (pts[:, 1] < 0)), "quadrant III absent"
        assert np.any((pts[:, 0] > 0) & (pts[:, 1] < 0)), "quadrant IV absent"

    def test_distribution_uniforme_rayon_moyen(self):
        """
        Pour une distribution uniforme sur le disque, E[r^2] = 1/2.
        On vérifie que la moyenne empirique est proche de 0.5.
        """
        pts = np.array([random_in_disk() for _ in range(2000)])
        r2  = pts[:, 0]**2 + pts[:, 1]**2
        assert r2.mean() == pytest.approx(0.5, abs=0.05)