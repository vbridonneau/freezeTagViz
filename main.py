"""
Freeze Tag Problem — Visualiseur interactif
============================================
Modes (menu déroulant) :
  • Simulation  : tire N configurations aléatoires complètes, retient le pire
  • Exploration : part de la config courante, perturbe localement (hill-climbing)
CSV exportés dans ./output/
"""
import csv
import math
import os
from typing import *

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, TextBox

from features import HeatMapUIFeature, HeatMapCursor, heat_map_from_robots
from freeze_tag import (
    _NUMBA_OK,
    clamp_disk,
    compute_makespan,
    dist,
    makespan_exact,
    random_in_disk,
    wake_tree,
    warmup_numba,
)

# ── Constante cible ────────────────────────────────────────────────────────────
TARGET = 1 + 2 * math.sqrt(2)   # ≈ 3.8284

# ── Worker multiprocessing (doit être top-level pour être picklable) ──────────
import multiprocessing as _mp
N_WORKERS = max(1, _mp.cpu_count())

def _sim_worker_explorer(args):
    origin_xy, batch, sigma, cur_cfg, cur_ms = args

    best_ms = cur_ms; best_cfg = cur_cfg

    for i in range(batch):
        cand = []
        for r in cur_cfg:
            dx, dy = np.random.normal(0, sigma), np.random.normal(0, sigma)
            cx, cy = clamp_disk(r[0]+dx, r[1]+dy)
            cand.append(np.array([cx, cy]))
        ms = makespan_exact(origin_xy, cand)
        if ms > best_ms:
            best_ms  = ms
            best_cfg = cand
    return best_ms, best_cfg

def _sim_worker(args):
    """Évalue 'batch' configurations aléatoires et retourne la pire."""
    n, origin_xy, batch = args

    best_ms = -1.; best_cfg = None
    for _ in range(batch):
        cfg = tuple(random_in_disk() for _ in range(n))
        ms  = makespan_exact(origin_xy, cfg)
        if ms > best_ms: best_ms = ms; best_cfg = cfg
    return best_ms, best_cfg

# ── Export CSV ─────────────────────────────────────────────────────────────────
def export_csv(robots : List[Tuple[float]], ms: float, mode_tag : str, n : int) -> str:
    """Sauvegarde la pire configuration trouvee dans un fichier CSV.

    Cree le dossier ./output/ si necessaire, puis ecrit un fichier
    worst_{mode_tag}_n{n}.csv avec une ligne par robot. Le makespan
    est inscrit uniquement sur la premiere ligne (colonne "makespan").

    Format du CSV :
        X, Y, makespan
        x1, y1, <makespan>
        x2, y2,
        ...

    Args:
        robots:   Liste des positions np.ndarray ou (x, y) de la configuration.
        ms:       Makespan de la configuration (valeur a sauvegarder).
        mode_tag: Identifiant du mode ("simulation" ou "exploration"),
                  utilise dans le nom de fichier.
        n:        Nombre de robots, utilise dans le nom de fichier.

    Returns:
        Nom du fichier cree (sans le chemin), par exemple
        "worst_simulation_n8.csv".
    """
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(out_dir, exist_ok=True)
    fname = f"worst_{mode_tag}_n{n}.csv"
    with open(os.path.join(out_dir, fname), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["X", "Y", "makespan"])
        for k, r in enumerate(robots):
            w.writerow([f"{r[0]:.8f}", f"{r[1]:.8f}",
                        f"{ms:.8f}" if k == 0 else ""])
    return fname

# ══════════════════════════════════════════════════════════════════════════════
# Dropdown maison
# ════════════════════════════════════════════════════════════════════════════
class DropDown:
    MODES = ["Simulation", "Exploration", "Heatmap"]

    def __init__(self, fig, rect, on_change):
        """Initialise le menu deroulant et l'insere dans la figure.

        Cree un bouton principal affichant le mode courant, et des axes
        "option" caches qui s'affichent vers le haut lors du clic.

        Args:
            fig:       Figure matplotlib dans laquelle inserer le widget.
            rect:      Position et taille [x, y, largeur, hauteur] en
                       coordonnees normalisees (0..1).
            on_change: Callable appele avec le nom du mode selectionne
                       (str) a chaque changement de selection.
        """
        self._fig = fig; self._cb = on_change
        self._open = False; self._cur = 0
        x, y, w, h = rect

        # Bouton principal
        self._ax = fig.add_axes(rect)
        self._ax.set_facecolor("#1e1e3e")
        self._ax.set_xlim(0,1); self._ax.set_ylim(0,1); self._ax.axis('off')
        self._lbl = self._ax.text(0.08, 0.5, f"⊞  {self.MODES[0]}",
                                  color="#ccccff", fontsize=9, fontweight='bold',
                                  va='center', transform=self._ax.transAxes)
        self._ax.text(0.90, 0.5, "▴", color="#8888bb", fontsize=10,
                      va='center', transform=self._ax.transAxes)

        # Options déroulantes — s'ouvrent VERS LE HAUT
        self._opts = []
        self._otxt = []
        for i, name in enumerate(self.MODES):
            # i=0 → première option juste au-dessus du bouton
            ax = fig.add_axes([x, y + h*(i+1) + 0.005, w, h])
            ax.set_facecolor("#12122e"); ax.set_xlim(0,1); ax.set_ylim(0,1)
            ax.axis('off')
            t = ax.text(0.08, 0.5, name, color="#aaaadd", fontsize=9,
                        va='center', transform=ax.transAxes)
            ax.set_visible(False)
            self._opts.append(ax); self._otxt.append(t)

        fig.canvas.mpl_connect('button_press_event', self._click)

    def _click(self, ev):
        """Gere les clics souris sur le bouton principal et les options.

        Ouvre ou ferme le menu si le clic est sur le bouton principal.
        Selectionne une option si le clic est sur un axe d'option visible,
        puis ferme le menu et appelle le callback on_change.
        Ferme le menu si le clic est en dehors.

        Args:
            ev: Evenement matplotlib MouseEvent.
        """
        if ev.inaxes == self._ax:
            self._open = not self._open; self._refresh(); return
        for i, ax in enumerate(self._opts):
            if ev.inaxes == ax and ax.get_visible():
                self._cur = i; self._open = False; self._refresh()
                self._lbl.set_text(f"⊞  {self.MODES[i]}")
                self._fig.canvas.draw_idle()
                self._cb(self.MODES[i]); return
        if self._open:
            self._open = False; self._refresh()

    def _refresh(self):
        """Met a jour la visibilite et la couleur des options du menu.

        Affiche ou cache les axes d'option selon self._open,
        et met en evidence l'option actuellement selectionnee.
        Appelle draw_idle() pour rafraichir l'affichage.
        """
        for i, (ax, t) in enumerate(zip(self._opts, self._otxt)):
            ax.set_visible(self._open)
            ax.set_facecolor("#2a2a5a" if i == self._cur else "#1a1a3e")
            t.set_color("#ffffff" if i == self._cur else "#aaaadd")
        self._fig.canvas.draw_idle()

    @property
    def mode(self): return self.MODES[self._cur]


# ══════════════════════════════════════════════════════════════════════════════
# Application
# ══════════════════════════════════════════════════════════════════════════════
class FreezeTagViz:
    ORIGIN  = np.array([0., 0.])
    RADIUS  = 1.0
    DTHR    = 0.07

    def __init__(self):
        self.robots           = []
        self.add_mode         = False
        self._drag_idx        = None
        self._sim_best_robots = None; self._sim_best_ms = None
        self._exp_best_robots = None; self._exp_best_ms = None
        self._mode            = "Simulation"
        # widgets mode (recréés à chaque switch)
        self._ax_iter = None; self._txt_iter = None
        self._ax_run  = None; self._btn_run  = None

        # Handle the features
        self.heatmapui = None

        self._build()
        self._switch("Simulation")
        self._redraw()

    # ── Mise en page ───────────────────────────────────────────────────────────
    def _build(self):
        """Construit la mise en page de la fenetre principale.

        Cree et positionne tous les axes et widgets permanents :
        - ax       : zone de dessin du disque (gauche)
        - ax_info  : panneau de statistiques (droite)
        - Boutons  : Reset, +Robot, -Robot, Aleatoire, champ gravitationnel
        - Dropdown : selection du mode (Simulation / Exploration)
        - hint_txt : ligne de statut/feedback en bas de la zone de dessin

        Les widgets dependant du mode (TextBox N= et bouton Run) sont crees
        separement dans _switch() car ils sont recrées a chaque changement
        de mode.
        """
        self.fig = plt.figure(figsize=(11, 8), facecolor="#0a0a18")
        self.fig.canvas.manager.set_window_title("Freeze Tag — Visualiseur")

        self.ax      = self.fig.add_axes([0.07, 0.18, 0.54, 0.76])
        self.ax.set_facecolor("#0a0a18")
        self.ax_info = self.fig.add_axes([0.65, 0.18, 0.33, 0.76])
        self.ax_info.set_facecolor("#0a0a18"); self.ax_info.axis('off')

        # For colobars
        self.cax = self.fig.add_axes([0.56, 0.18, 0.05, 0.76])

        def mkbtn(attr, rect, lbl, col, hov, cb):
            ax  = self.fig.add_axes(rect)
            btn = Button(ax, lbl, color=col, hovercolor=hov)
            btn.label.set_color("white"); btn.label.set_fontsize(9)
            btn.label.set_fontweight('bold'); btn.on_clicked(cb)
            setattr(self, attr, btn)

        mkbtn("btn_reset", [0.07,0.04,0.09,0.08], "⟳ Reset",
              "#1e1e2e","#2e2e4e", self._on_reset)
        mkbtn("btn_add",   [0.18,0.04,0.09,0.08], "+ Robot",
              "#163a2f","#1f5c47", self._on_add_toggle)
        mkbtn("btn_rem",   [0.29,0.04,0.09,0.08], "− Robot",
              "#3a1616","#5c1f1f", self._on_remove_last)
        mkbtn("btn_rand",  [0.40,0.04,0.10,0.08], "⚄ Aléatoire",
              "#1a1a3a","#2a2a5a", self._on_randomize)

        # Dropdown — tout à droite, options vers le haut
        self.dropdown = DropDown(self.fig, [0.84, 0.04, 0.14, 0.08],
                                 self._switch)

        self.hint_txt = self.fig.text(
            0.33, 0.14, "", color="#aaaacc", fontsize=8.5,
            ha='center', va='center')
        self.fig.canvas.mpl_connect('button_press_event',   self._on_click)
        self.fig.canvas.mpl_connect('motion_notify_event',  self._on_drag)
        self.fig.canvas.mpl_connect('button_release_event', self._on_release)

    # ── Switch mode ────────────────────────────────────────────────────────────
    def _switch(self, mode):
        """Change le mode actif (Simulation ou Exploration).

        Detruit les widgets dependant du mode precedent (TextBox N= et
        bouton Run/Explorer) et les recrée avec les valeurs et callbacks
        appropries au nouveau mode. Appelle draw_idle() pour rafraichir.

        Args:
            mode: Nom du mode cible, "Simulation" ou "Exploration".
        """
        self._mode = mode
        # Supprimer widgets précédents
        for ax_attr in ("_ax_iter", "_ax_run"):
            ax = getattr(self, ax_attr)
            if ax is not None:
                ax.remove(); setattr(self, ax_attr, None)
        self._txt_iter = None; self._btn_run = None

        # Recréer selon le mode — N= et Run entre boutons fixes et dropdown
        ax_i = self.fig.add_axes([0.57, 0.04, 0.09, 0.08])
        txt  = TextBox(ax_i, "N= ", initial="200" if mode=="Simulation" else "500" if mode=="Heatmap" else "65536",
                       color="#12122a", hovercolor="#1e1e3a")
        txt.label.set_color("#aaaacc"); txt.label.set_fontsize(9)
        txt.text_disp.set_color("#eeeeff"); txt.text_disp.set_fontsize(10)
        self._ax_iter  = ax_i
        self._txt_iter = txt

        lbl = "▶ Simuler" if mode == "Simulation" else "▶ Explorer" if mode == "Explorer" else "Heatmap"
        col = "#2a1a4a"   if mode == "Simulation" else "#1a3a2a" if mode == "Explorer" else "#1a3a2b"
        hov = "#4a2a7a"   if mode == "Simulation" else "#2a6a4a" if mode == "Explorer" else "#2a6a4b"
        cb  = self._on_simulate if mode == "Simulation" else self._on_explore if mode == "Explorer" else self._on_heatmap

        ax_r = self.fig.add_axes([0.68, 0.04, 0.13, 0.08])
        btn  = Button(ax_r, lbl, color=col, hovercolor=hov)
        btn.label.set_color("white"); btn.label.set_fontsize(9)
        btn.label.set_fontweight('bold'); btn.on_clicked(cb)
        self._ax_run  = ax_r
        self._btn_run = btn

        # self.heatmapui = None
        self.cax.set_visible(mode != "Heatmap")

        self.fig.canvas.draw_idle()

    # ── Dessin ─────────────────────────────────────────────────────────────────
    def _redraw(self):
        """Redessine completement la zone de dessin et le panneau info.

        Efface les axes principaux puis trace dans l'ordre :
        - grille et axes cartesiens
        - cercle unite (reference geometrique)
        - arbre de reveil optimal (aretes vertes)
        - pire configuration connue du mode courant (cercles orange)
        - robots endormis (points rouges) et origine p0 (etoile dorée)
        - indicateur de mode "ajout" si actif
        Appelle _draw_info() pour mettre a jour le panneau de droite.
        """
        self.ax.cla(); self.ax_info.cla(); self.ax_info.axis('off')
        ax = self.ax
        ax.set_facecolor("#0a0a18")
        ax.set_xlim(-1.35,1.35); ax.set_ylim(-1.35,1.35); ax.set_aspect('equal')
        ax.grid(True, color="#ffffff0d", linewidth=0.5, linestyle='--', zorder=0)
        ax.axhline(0, color="#ffffff25", linewidth=0.7, zorder=1)
        ax.axvline(0, color="#ffffff25", linewidth=0.7, zorder=1)
        ax.tick_params(colors='#555566', labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor('#1e1e33')

        # Cercle unité
        ax.add_patch(plt.Circle((0,0), 1., fill=False, edgecolor="#3366bb",
                                 linewidth=1.3, linestyle='--', zorder=2, alpha=0.65))
        ax.text(0.73, 0.73, "r=1", color="#3366bb", fontsize=7, alpha=0.6, zorder=3)

        # Arbre de réveil
        if self.robots and not self._drag_idx:
            for (p1,p2) in wake_tree(self.ORIGIN, self.robots):
                ax.plot([p1[0],p2[0]], [p1[1],p2[1]],
                        color="#33dd99", linewidth=1.0, alpha=0.45, zorder=3)

        # Worst-case courant (cercles orange)
        bots = (self._exp_best_robots if self._mode=="Exploration"
                else self._sim_best_robots)
        bms  = (self._exp_best_ms     if self._mode=="Exploration"
                else self._sim_best_ms)
        if bots is not None:
            for r in bots:
                ax.scatter(r[0], r[1], s=130, color='none',
                           edgecolors="#ff8800", linewidths=1.7,
                           zorder=4, marker='o', alpha=0.75)
            ax.text(-1.30, -1.27, f"⚑ Worst ({self._mode}) : {bms:.4f}",
                    color="#ff8800", fontsize=7.5, zorder=7, alpha=0.88)

        # Robots
        for i, r in enumerate(self.robots):
            ax.scatter(r[0], r[1], s=75, color="#ee5577",
                       edgecolors="#ff99bb", linewidths=1.0, zorder=5)
            ax.text(r[0]+0.03, r[1]+0.04, f"r{i+1}",
                    color="#ff99bb", fontsize=6.5, zorder=6)
        # Origine
        ax.scatter(0,0, s=150, color="#ffcc00", edgecolors="#ffe066",
                   linewidths=1.3, zorder=6, marker='*')
        ax.text(0.04, 0.06, "p₀", color="#ffcc00",
                fontsize=8.5, fontweight='bold', zorder=7)

        mc = {"Simulation":"#9977ff","Exploration":"#44ddaa"}.get(self._mode,"#ccccff")
        ax.set_title(f"Freeze Tag  ·  Mode : {self._mode}",
                     color=mc, fontsize=10.5, pad=7)

        self._draw_info()

        if self._mode == "Heatmap" and self.heatmapui:
            self.heatmapui.draw()

        # Hint
        if self.add_mode:
            self.hint_txt.set_text("Mode ajout — cliquez dans le disque")
            self.hint_txt.set_color("#44ffaa")
            self.btn_add.label.set_text("✓ Ajout ON")
        elif self._drag_idx is not None:
            self.hint_txt.set_text(f"Déplacement r{self._drag_idx+1}…")
            self.hint_txt.set_color("#ffcc44")
            self.btn_add.label.set_text("+ Robot")
        else:
            self.hint_txt.set_text(
                "'+ Robot' pour ajouter  |  Glisser pour déplacer")
            self.hint_txt.set_color("#555577")
            self.btn_add.label.set_text("+ Robot")

        self.fig.canvas.draw_idle()

    def _draw_info(self):
        """Remplit le panneau de statistiques (axe de droite).

        Affiche dans ax_info :
        - nombre de robots, makespan courant (exact ou greedy), methode
          utilisee, cible 1+2*sqrt(2), statut OK/KO
        - liste des positions de tous les robots (tronquee si > ~8)
        - bloc de resultat du mode courant (pire makespan trouve)

        La couleur du makespan est verte si <= TARGET, rouge sinon.
        """
        ai = self.ax_info
        ai.set_xlim(0,1); ai.set_ylim(0,1)

        n  = len(self.robots)
        ms = compute_makespan(self.ORIGIN, self.robots) if n else 0.
        ok = ms <= TARGET + 1e-9
        mc = "#44ff88" if ok else "#ff4455"
        mth = "exact"
        ai.text(0.5,0.96,"Statistiques",color="#9999ee",fontsize=10.5,
                fontweight='bold',ha='center',va='top',transform=ai.transAxes)

        for lbl,val,vc,y in [
            ("Robots",   str(n),                   "#eeeeff", 0.87),
            ("Makespan", f"{ms:.4f}" if n else "—", mc,       0.79),
            ("Méthode",  mth,                       "#eeeeff", 0.71),
            ("Cible",    f"{TARGET:.4f}",            "#eeeeff", 0.63),
            ("Statut",   "✓ ≤ 1+2√2" if ok else "✗ > 1+2√2", mc, 0.55),
        ]:
            ai.text(0.04,y,lbl+":",color="#666688",fontsize=8.5,
                    transform=ai.transAxes)
            ai.text(0.96,y,val,color=vc,fontsize=8.5,ha='right',
                    fontweight='bold',transform=ai.transAxes)

        ai.plot([0,1],[0.49,0.49],color="#222244",linewidth=0.8,
                transform=ai.transAxes,clip_on=False)

        # Positions
        y = 0.44
        ai.text(0.5,y,"Positions",color="#9999ee",fontsize=8.5,
                ha='center',transform=ai.transAxes)
        y -= 0.055
        ai.text(0.04,y,"p₀",color="#ffcc00",fontsize=7.5,transform=ai.transAxes)
        ai.text(0.96,y,"(0.00, 0.00)",color="#ffcc00",fontsize=7.5,
                ha='right',transform=ai.transAxes)
        for i,r in enumerate(self.robots):
            y -= 0.05
            if y < 0.22:
                ai.text(0.5,y,"…",color="#444466",fontsize=7.5,
                        ha='center',transform=ai.transAxes); break
            ai.text(0.04,y,f"r{i+1}",color="#ff99bb",fontsize=7.5,
                    transform=ai.transAxes)
            ai.text(0.96,y,f"({r[0]:.2f},{r[1]:.2f})",color="#ff99bb",
                    fontsize=7.5,ha='right',transform=ai.transAxes)

        ai.plot([0,1],[0.20,0.20],color="#222244",linewidth=0.8,
                transform=ai.transAxes,clip_on=False)

        # Bloc résultat mode courant
        bms  = self._exp_best_ms     if self._mode=="Exploration" else self._sim_best_ms
        tcol = "#44ddaa"             if self._mode=="Exploration" else "#ff8800"
        tlbl = "⚑ Exploration"       if self._mode=="Exploration" else "⚑ Simulation"
        if bms is not None:
            y = 0.17
            ai.text(0.5,y,tlbl,color=tcol,fontsize=8.5,fontweight='bold',
                    ha='center',transform=ai.transAxes)
            y -= 0.06
            ai.text(0.04,y,"Worst ms:",color="#666688",fontsize=8,
                    transform=ai.transAxes)
            ai.text(0.96,y,f"{bms:.4f}",color=tcol,fontsize=8,ha='right',
                    fontweight='bold',transform=ai.transAxes)
            y -= 0.055
            sc = "#44ff88" if bms <= TARGET+1e-9 else "#ff4455"
            ai.text(0.04,y,"Statut:",color="#666688",fontsize=8,
                    transform=ai.transAxes)
            ai.text(0.96,y,"✓ ≤ 1+2√2" if bms<=TARGET+1e-9 else "✗ > 1+2√2",
                    color=sc,fontsize=8,ha='right',fontweight='bold',
                    transform=ai.transAxes)

    def _compute_heatmap(self):
        n = len(self.robots)
        if n == 0: self._warn("Ajoutez des robots avant de simuler"); return
        nb = self._read_n()
        if nb is None: return

        pts      = list(map(tuple, self.robots))
        heat_map = heat_map_from_robots(tuple(self.ORIGIN), pts, nb)

        if not self.heatmapui:
            self.heatmapui = HeatMapUIFeature(self.ax, self.cax)
        self.heatmapui.update(heat_map)

    def _on_heatmap(self, event):
        self._compute_heatmap()
        self._redraw()

    # ── Simulation (Monte-Carlo) ───────────────────────────────────────────────
    def _on_simulate(self, event):
        """Lance une simulation Monte-Carlo pour trouver la pire configuration.

        Tire N configurations aleatoires de n robots dans le disque unite,
        calcule le makespan de chacune, et retient la pire. Si N >= 500
        et plusieurs coeurs sont disponibles, la simulation est parallelisee
        via multiprocessing.Pool avec un worker par coeur.

        La pire configuration trouvee est affichee (cercles orange) et
        devient la configuration courante. Le resultat est exporte en CSV.

        Args:
            event: Evenement matplotlib (non utilise, requis par l'API Button).
        """
        n = len(self.robots)
        if n == 0: self._warn("Ajoutez des robots avant de simuler"); return
        nb = self._read_n()
        if nb is None: return

        origin_xy = (float(self.ORIGIN[0]), float(self.ORIGIN[1]))

        # Seuil : paralléliser seulement si nb assez grand (amortir l'overhead Pool)
        MIN_PAR = 300
        use_par = (N_WORKERS > 1) and (nb >= MIN_PAR)

        if use_par:
            n_workers  = N_WORKERS
            batch_size = max(100, nb // n_workers)
            batches    = [batch_size] * (n_workers - 1)
            batches.append(nb - batch_size * (n_workers - 1))
            args = [(n, origin_xy, b) for b in batches]

            self.hint_txt.set_text(
                f"Simulation… {nb} tirages sur {n_workers} cœurs")
            self.hint_txt.set_color("#ffcc44")
            self.fig.canvas.draw_idle(); self.fig.canvas.flush_events()

            with _mp.Pool(n_workers) as pool:
                results = pool.map(_sim_worker, args)

            best_ms, best_cfg_raw = max(results, key=lambda x: x[0])
            best_cfg = [np.array(list(r)) for r in best_cfg_raw]
            label = f"Simulation ({n_workers} cœurs)"

        else:
            # Séquentiel (nb petit ou 1 seul cœur)
            best_ms, best_cfg = -1., None
            for i in range(nb):
                cfg = [random_in_disk() for _ in range(n)]
                ms  = compute_makespan(self.ORIGIN, cfg)
                if ms > best_ms: best_ms, best_cfg = ms, [c.copy() for c in cfg]
                if (i+1) % 50 == 0 or i == nb-1:
                    self.hint_txt.set_text(
                        f"Simulation… {i+1}/{nb}  best={best_ms:.4f}")
                    self.hint_txt.set_color("#ffcc44")
                    self.fig.canvas.draw_idle(); self.fig.canvas.flush_events()
            label = "Simulation (séquentiel)"

        self._sim_best_ms     = best_ms
        self._sim_best_robots = best_cfg
        self.robots           = [c.copy() for c in best_cfg]
        fname = export_csv(best_cfg, best_ms, "simulation", n)
        self.hint_txt.set_text(
            f"{label} — worst={best_ms:.4f}  →  output/{fname}")
        self.hint_txt.set_color("#44ff88")
        self._redraw()

    # ── Exploration (hill-climbing local) ─────────────────────────────────────
    def _on_explore(self, event):
        """Lance un hill-climbing local pour maximiser le makespan.

        Demarre depuis la configuration courante et applique N iterations
        de perturbation gaussienne. A chaque iteration, tous les robots
        sont deplaces d'un vecteur gaussien de variance sigma, puis clampes
        dans le disque. La nouvelle configuration est acceptee si son
        makespan est >= au makespan courant (ascension de gradient bruitee).
        sigma decroit exponentiellement (x0.9998 par iteration) pour affiner
        la recherche locale au fil du temps.

        La meilleure configuration trouvee est affichee et exportee en CSV.

        Args:
            event: Evenement matplotlib (non utilise, requis par l'API Button).
        """
        n = len(self.robots)
        if n==0: self._warn("Placez des robots pour démarrer l'exploration"); return
        nb = self._read_n()
        if nb is None: return

        cur_cfg = [r.copy() for r in self.robots]
        cur_ms  = compute_makespan(self.ORIGIN, cur_cfg)
        best_ms, best_cfg = cur_ms, [r.copy() for r in cur_cfg]
        sigma   = 0.08      # amplitude perturbation initiale

        MIN_PAR = 1e18
        use_par = (N_WORKERS > 1) and (nb >= MIN_PAR)

        if use_par:
            n_workers  = N_WORKERS
            batch_size = max(100, nb // n_workers)
            batches    = [batch_size] * (n_workers - 1)
            batches.append(nb - batch_size * (n_workers - 1))
            origin_xy  = [0, 0]
            args = [(origin_xy, b, sigma, cur_cfg, cur_ms) for b in batches]

            self.hint_txt.set_text(
                f"Simulation… {nb} tirages sur {n_workers} cœurs")
            self.hint_txt.set_color("#ffcc44")
            self.fig.canvas.draw_idle(); self.fig.canvas.flush_events()

            with _mp.Pool(n_workers) as pool:
                results = pool.map(_sim_worker_explorer, args)

            best_ms, best_cfg_raw = max(results, key=lambda x: x[0])
            best_cfg = [np.array(list(r)) for r in best_cfg_raw]
            label = f"Simulation ({n_workers} cœurs)"

        else:
            for i in range(nb):
                cand = []
                for r in cur_cfg:
                    dx, dy = np.random.normal(0, sigma), np.random.normal(0, sigma)
                    cx, cy = clamp_disk(r[0]+dx, r[1]+dy)
                    cand.append(np.array([cx, cy]))
                ms = compute_makespan(self.ORIGIN, cand)
                # Hill-climbing vers le pire : on accepte si makespan ≥ actuel
                if ms >= cur_ms:
                    cur_cfg, cur_ms = cand, ms
                    if ms > best_ms:
                        best_ms, best_cfg = ms, [c.copy() for c in cand]
                # Recuit léger pour ne pas rester coincé
                sigma = max(0.008, sigma * 0.9998)
                if (i+1)%100==0 or i==nb-1:
                    self.hint_txt.set_text(
                        f"Exploration… {i+1}/{nb}  best={best_ms:.4f}  σ={sigma:.4f}")
                    self.hint_txt.set_color("#ffcc44")
                    self.fig.canvas.draw_idle(); self.fig.canvas.flush_events()

        self._exp_best_ms     = best_ms
        self._exp_best_robots = best_cfg
        self.robots           = [c.copy() for c in best_cfg]
        fname = export_csv(best_cfg, best_ms, "exploration", n)
        self.hint_txt.set_text(
            f"Exploration — worst={best_ms:.4f}  →  output/{fname}")
        self.hint_txt.set_color("#44ddaa")
        self._redraw()

    # ── Helpers ────────────────────────────────────────────────────────────────
    def _warn(self, msg):
        """Affiche un message d'avertissement dans la barre de statut.

        Args:
            msg: Texte a afficher (typiquement prefixe par "⚠").
        """
        self.hint_txt.set_text(msg); self.hint_txt.set_color("#ffaa44")
        self.fig.canvas.draw_idle()

    def _read_n(self):
        """Lit et valide la valeur entiere saisie dans le champ TextBox "N=".

        Returns:
            Entier strictement positif si la valeur est valide,
            None sinon (un avertissement est alors affiche).
        """
        if self._txt_iter is None: return None
        try:
            v = int(self._txt_iter.text.strip())
            assert v > 0; return v
        except Exception:
            self._warn("⚠ Entrez un entier positif dans N="); return None

    # ── Boutons permanents ────────────────────────────────────────────────────
    def _on_reset(self, event):
        """Remet l'application dans son etat initial.

        Vide la liste des robots, desactive le mode ajout, et efface
        les meilleurs resultats des modes Simulation et Exploration.

        Args:
            event: Evenement matplotlib (non utilise, requis par l'API Button).
        """
        self.robots = []; self.add_mode = False
        self._sim_best_robots = None; self._sim_best_ms = None
        self._exp_best_robots = None; self._exp_best_ms = None
        self._redraw()

    def _on_add_toggle(self, event):
        """Active ou desactive le mode ajout de robots.

        En mode ajout, le prochain clic dans le disque ajoute un robot
        a la position cliquee (si elle est dans le disque unite).

        Args:
            event: Evenement matplotlib (non utilise, requis par l'API Button).
        """
        self.add_mode = not self.add_mode; self._redraw()

    def _on_remove_last(self, event):
        """Supprime le dernier robot ajoute.

        Args:
            event: Evenement matplotlib (non utilise, requis par l'API Button).
        """
        if self.robots: self.robots.pop()
        self._redraw()

    def _on_randomize(self, event):
        """Repositionne aleatoirement tous les robots dans le disque.

        Conserve le nombre de robots existant mais tire de nouvelles
        positions independamment et uniformement dans le disque unite.
        Ne fait rien si la liste est vide.

        Args:
            event: Evenement matplotlib (non utilise, requis par l'API Button).
        """
        n = len(self.robots)
        if n==0: return
        self.robots = [random_in_disk() for _ in range(n)]
        self._redraw()

    # ── Souris ────────────────────────────────────────────────────────────────
    def _pick(self, x, y):
        """Trouve l'indice du robot le plus proche d'un point donne.

        Retourne l'indice uniquement si la distance est inferieure au seuil
        DTHR (0.07 unite), ce qui correspond au rayon de selection par clic.

        Args:
            x: Abscisse du point cible (coordonnees du disque).
            y: Ordonnee du point cible (coordonnees du disque).

        Returns:
            Indice dans self.robots du robot le plus proche si la distance
            est < DTHR, None sinon.
        """
        best_i, best_d = None, self.DTHR
        for i,r in enumerate(self.robots):
            d = dist((x,y), r)
            if d < best_d: best_d, best_i = d, i
        return best_i

    def _on_click(self, event):
        """Gere les clics souris dans la zone de dessin.

        Comportement selon le mode :
        - Mode ajout actif : place un nouveau robot au point clique
          (si dans le disque), ou commence un drag si un robot existant
          est proche du clic.
        - Mode normal : commence un drag si un robot est proche du clic.

        Args:
            event: Evenement matplotlib MouseEvent.
        """
        if event.inaxes != self.ax: return
        x, y = event.xdata, event.ydata
        if x is None: return
        if self.add_mode:
            idx = self._pick(x,y)
            if idx is not None:
                self._drag_idx=idx; self.add_mode=False; self._redraw(); return
            if x*x+y*y > 1.:
                self._warn("⚠ Hors du disque"); return
            self.robots.append(np.array([x,y])); self._redraw(); return
        idx = self._pick(x,y)
        if idx is not None:
            self._drag_idx = idx; self.fig.canvas.set_cursor(2)

    def _on_drag(self, event):
        """Deplace le robot selectionne en suivant le curseur.

        Appele a chaque mouvement souris. Si un robot est en cours de drag
        (self._drag_idx != None), met a jour sa position vers le curseur
        en la clampant dans le disque, puis redessine.

        Args:
            event: Evenement matplotlib MouseEvent.
        """
        if self._drag_idx is None or event.inaxes!=self.ax:
            if self._mode == "Heatmap" and self.heatmapui:pass
                # self.heatmapui.create_cursor_event_handler(self.robots).on_move(event)
                # self._redraw()
            return
        x,y = event.xdata, event.ydata
        if x is None: return
        x,y = clamp_disk(x,y)
        self.robots[self._drag_idx] = np.array([x,y])
        # self._redraw()

    def _on_release(self, event):
        """Termine le drag en cours et restaure le curseur normal.

        Args:
            event: Evenement matplotlib MouseEvent.
        """
        if self._drag_idx is not None:
            self._drag_idx = None
            self.fig.canvas.set_cursor(1)
            if self._mode == "Heatmap":
                self._compute_heatmap()
            self._redraw()

    # ── Fenêtre champ gravitationnel ──────────────────────────────────────────
    @staticmethod
    def _gravity_field(X, Y, robots, sigma=1.0):
        """Calcule le champ de potentiel gravitationnel sur une grille 2D.

        Le champ est une somme de gaussiennes centrees sur chaque robot.
        Avec sigma=1, la gaussienne vaut exp(-2) ~ 0.13 a distance 2,
        ce qui garantit une influence couvrant tout le disque unite
        (diametre = 2).

        Args:
            X:      Grille des abscisses, tableau numpy 2D (meshgrid).
            Y:      Grille des ordonnees, tableau numpy 2D (meshgrid).
            robots: Liste de paires (rx, ry) representant les positions
                    des robots (centres des gaussiennes).
            sigma:  Ecart-type des gaussiennes, controle le rayon d'action
                    (defaut : 1.0).

        Returns:
            Tableau numpy 2D de meme forme que X et Y contenant la valeur
            du champ phi(x, y) = sum_i exp(-||p - p_i||^2 / (2*sigma^2)).
        """
        Z = np.zeros_like(X, dtype=float)
        for rx, ry in robots:
            d2 = (X - rx)**2 + (Y - ry)**2
            Z += np.exp(-d2 / (2.0 * sigma**2))
        return Z

    def show(self): plt.show()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _mp.freeze_support()   # indispensable sur Windows avec multiprocessing
    if _NUMBA_OK:
        print("[freeze_tag] Compilation JIT Numba en cours...")
        warmup_numba()
        print("[freeze_tag] JIT prêt.")
    print(f"Freeze Tag Visualiseur  |  cible 1+2√2 ≈ {TARGET:.6f}")
    print(f"  {N_WORKERS} cœurs  |  Numba={'OUI (×15)' if _NUMBA_OK else 'NON (fallback lru_cache)'}")
    print("  exact pour n ≤ 10, greedy au-delà")
    FreezeTagViz().show()
