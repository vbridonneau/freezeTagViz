import numpy as np

from freeze_tag import dp_bitmask, dist_mat

def regular_disk_sunflower(n : int):
    # Golden angle in radians
    golden_angle = np.pi * (3 - np.sqrt(5))

    indices = np.arange(n)

    # Calculate radius and angle for each index
    r = np.sqrt((indices + 0.5) / n)
    theta = indices * golden_angle

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    return np.column_stack((x, y))

def heat_map_from_robots(origin : tuple[float, float], robots : list[tuple[float, float]], Nsample : int) -> dict[tuple[float, float], float]:
    """
        Compute, for a set of robots, the makespan if a new point would be inserted in the disk

        Args:
            origin: coordinates of the origin
            robots: list of coordinates of robots to wakeup (the origin is not represented in the list
            Nsample: number of sample to test

        Return:
            a dictionnary with key : (x, y) coordinates of a robot r in the the unit 2D disk, value : the makespan for the set robots U {r} of robots.
    """
    res = {}

    # Sample the unit disk so that points are uniformly distributed
    new_robot_positions = regular_disk_sunflower(Nsample)
    for position in new_robot_positions:
        instance             = robots + [position, origin]
        res[tuple(position)] = dp_bitmask(len(instance) - 1, dist_mat(instance))
    return res

def feature_heat_map(origin : tuple[float, float], robots : list[tuple[float, float]]):
    return heat_map_from_robots(origin, robots, 1024*64)

# This is relevant to use the
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib import tri
from matplotlib.colorbar import Colorbar


class HeatMapCursor:
    def __init__(self, ax : plt.Axes, robots : list[tuple[float, float]], cbar : Colorbar):
        self.ax      = ax
        self.marker, = ax.plot([0],[0], marker="o", color="crimson", zorder=3)
        self.txt     = ax.text(0.0, 0.0, '', bbox={"facecolor": 'white', "alpha": 0.5})
        self.robots  = robots
        self.cbar    = cbar

        # --- Éléments dessinés sur la colorbar ---
        cax = cbar.ax
        vmin, _ = cax.get_ylim()

        # Ligne horizontale qui traverse toute la largeur de la barre
        self.cbar_line = cax.axhline(vmin, color="crimson", linewidth=2, visible=False)

        # Petit triangle à gauche de la barre : x en coordonnées d'axe (0 = bord gauche),
        # y en coordonnées de données (la valeur du makespan)
        trans = mtransforms.blended_transform_factory(cax.transAxes, cax.transData)
        self.cbar_arrow, = cax.plot([-0.4], [vmin], marker=">", markersize=10,
                                            color="crimson", transform=trans,
                                            clip_on=False, visible=False)

    def on_move(self, event):
        """
            Lorsque le curseur de la sourie bouge,
            calcul le makespan de la solution avec un robot en plus en position
            (event.xdata, event.ydata)
        """
        if not event.inaxes:return
        pos      = (event.xdata, event.ydata)
        instance = self.robots + [pos, (0., 0.)]
        n        = len(instance) - 1
        makespan = dp_bitmask(n, dist_mat(instance))

        # Afficher le texte
        self.txt.set_text(f"{makespan:.6f}")
        offset_pos = 0.05
        self.txt.set_position((pos[0] + offset_pos, pos[1] + offset_pos))

        # Position sur la colorbar, bornée à l'intervalle affiché
        vmin, vmax = self.cbar.ax.get_ylim()
        y = float(np.clip(makespan, vmin, vmax))
        self.cbar_line.set_ydata([y, y])
        self.cbar_arrow.set_ydata([y])
        self.cbar_line.set_visible(True)
        self.cbar_arrow.set_visible(True)

        self.ax.figure.canvas.draw_idle()

class HeatMap:
    def __init__(self):
        self.values = np.array([])
        self.x      = np.array([])
        self.y      = np.array([])
        self.tri    = None # tri.Triangulation([], [])

    def update(self, heat_map : dict):
        points      = list(heat_map.keys())
        self.values = np.array(list(heat_map.values()))
        self.x      = np.array([p[0] for p in points])
        self.y      = np.array([p[1] for p in points])

        self.tri    = tri.Triangulation(self.x, self.y)

# Ensure that when the mouse move, the makespan of the nearest position is displayed
class HeatMapUIFeature:
    CMAP_NAME = "jet"
    def __init__(self, ax : plt.Axes, cax : plt.Axes):
        self.cax = cax
        self.ax  = ax

        self.heatmap = HeatMap()
        # self.heatmap = self.ax.tricontourf(self.x, self.y, self.values, levels=100, cmap=HeatMapUIFeature.CMAP_NAME)
        self.cbar    = None # self.ax.figure.colorbar(heatmap, shrink=0.7, pad=0.05)


    def update(self, heat_map : dict):
        """
            Met à jour la heat map.
        """
        self.heatmap.update(heat_map)


    def draw(self):
        # self.ax.tripcolor(self.heatmap, self.values, shading="flat")
        assert self.heatmap.tri
        tpc = self.ax.tricontourf(self.heatmap.tri, self.heatmap.values, levels=100, cmap=HeatMapUIFeature.CMAP_NAME)
        if not self.cbar:
            self.cbar = self.cax.figure.colorbar(tpc, shrink=0.7, pad=0.05)
        else:
            self.cbar.update_normal(tpc)
        self.cbar.set_label('Intensité (Unités)', fontsize=11, weight='bold')
        self.cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')

    def create_cursor_event_handler(self, robots) -> HeatMapCursor:
        assert self.cbar
        return HeatMapCursor(self.cax, robots, self.cbar)



if __name__ == "__main__":
    from freeze_tag import random_in_disk

    def polygon_robots(n):
        """Retourne n robots placés aux sommets d'un polygone régulier inscrit."""
        return [
            (np.cos(2 * np.pi * i / n), np.sin(2 * np.pi * i / n))
            for i in range(n)
        ]
    N = 4
    # robots = polygon_robots(N)
    robots = [random_in_disk() for _ in range(N)]
    # robots = [(-.5, -.5), (-.5, .5), (.5, -.5), (.5, .5)]

    # Matplotlib setup
    fig = plt.figure(figsize=(10, 10), facecolor="#0a0a18") # 0a0a18
    assert isinstance(fig, plt.Figure)
    ax : plt.Axes = fig.add_axes([0, 0, 1, 1])  # ty: ignore[no-matching-overload]
    ax.set_facecolor("#0a0a18")
    ax.set_xlim(-1.35,1.35); ax.set_ylim(-1.35,1.35); ax.set_aspect('equal')
    ax.grid(True, color="#ffffff0d", linewidth=0.5, linestyle='--', zorder=0)
    ax.axhline(0, color="#ffffff25", linewidth=0.7, zorder=1)
    ax.axvline(0, color="#ffffff25", linewidth=0.7, zorder=1)
    ax.tick_params(colors='#555566', labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor('#1e1e33')

    # Calcul de la heat map
    Nsample  = 1024 * 64
    origin   = (0, 0)
    heat_map = heat_map_from_robots(origin, robots, Nsample)

    # Extraction des données pour l'affichage
    points, values = [], []
    for point, value in heat_map.items():
        points.append(point)
        values.append(value)
    # points = list(heat_map.keys())
    # values = np.array(list(heat_map.values()))
    x      = np.array([p[0] for p in points])
    y      = np.array([p[1] for p in points])

    # Dessin
    # Cercle unité
    ax.add_patch(plt.Circle((0,0), 1., fill=False, edgecolor="#3366bb",
                             linewidth=1.3, linestyle='--', zorder=2, alpha=0.65))
    ax.text(0.73, 0.73, "r=1", color="#3366bb", fontsize=7, alpha=0.6, zorder=3)

    for robot in robots:
        ax.scatter(robot[0], robot[1], s=75, color="#ee5577",
                   edgecolors="#ff99bb", linewidths=1.0, zorder=5)

    CMAP_NAME = "jet"
    # À mettre à la place de plt.scatter si vous voulez un effet de surface lisse :
    heatmap = plt.tricontourf(x, y, values, levels=100, cmap=CMAP_NAME)
    # heatmap = plt.scatter(x, y, c=values, cmap=CMAP_NAME, s=2, edgecolors="none")
    # 'shrink' ajuste la taille, 'pad' l'espace avec le graphique
    cbar = plt.colorbar(heatmap, shrink=0.7, pad=0.05)
    cbar.set_label('Intensité (Unités)', fontsize=11, weight='bold')
    cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')

    # Create a cursor to follow the point
    cursor = HeatMapCursor(ax, robots, cbar)
    plt.connect("motion_notify_event", cursor.on_move)

    plt.show()
