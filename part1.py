"""
Assignment 3, Part 1 (image warping + mosaicing).

The script first asks you to click any correspondences that are still missing (saved in points/, so you
only ever click once), then computes homographies, warps, blends, and writes everything to the output
folder.

    python3 mosaic.py --out                 # choose the output folder (otherwise you are asked)
    python3 mosaic.py --redo                # re-click everything (or:  --redo lake_a   --redo building_rect_1)
    python3 mosaic.py --no-click            # never open a window; use saved points only
    python3 mosaic.py --no-check            # build mosaics even if a homography looks insane (debugging bad clicks)

Two images appear side by side; click a feature in the green-bordered one,
then the SAME feature in the other, and repeat: 10-20 pairs on sharp static corners, spread over the WHOLE
overlap (high and low, left and right; not along one line). 

For a rectification the script shows the image and you click the 4 corners
(TL, TR, BR, BL) of something that is a rectangle in real life; no coordinates to type.
"""
import argparse, json, os, sys, time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.widgets import Button
from PIL import Image, ImageOps
from scipy.ndimage import gaussian_filter, distance_transform_edt

IMAGE_SETS = {
    "building":   ["Photoshoot/IMG_7647.jpeg", "Photoshoot/IMG_7648.jpeg", "Photoshoot/IMG_7649.jpeg"],
    "lake_a":     ["Photoshoot/IMG_7670.JPG", "Photoshoot/IMG_7671.JPG", "Photoshoot/IMG_7672.JPG", "Photoshoot/IMG_7673.JPG"],
}

# One entry per rectified image: a photo of something that is a RECTANGLE in the real world (a window,
# a door, a panel, a sign...). 
RECTIFICATIONS = [
    dict(name="building_rect_1", image="Photoshoot/IMG_7647.jpeg", aspect=1.5),
    dict(name="building_rect_2", image="Photoshoot/IMG_7648.jpeg", aspect=1.5),
]

POINTS_DIR, OUT_DIR = "points", "out"
MAX_DIM = 1400        # longest side of the working images (pixels)
CANVAS_MAX = 2400     # longest side of a mosaic (scaled down if it would be bigger)
CHECK = True          # --no-check turns the 'is this mosaic sane' test off
MAX_STRETCH = 6.0     # an image may grow / shrink by at most this area factor in a mosaic
_cache = {}

# image + point I/O

def autocrop_black(arr, thresh=10 / 255.0):
    """Remove pure-black borders (e.g. screen-recording letterbox bars) that would ruin feathering."""
    lit = arr.max(axis=2) > thresh
    rows, cols = np.where(lit.any(axis=1))[0], np.where(lit.any(axis=0))[0]
    if len(rows) == 0:
        return arr
    r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    return arr[r0:r1, c0:c1] if (r1 - r0) * (c1 - c0) < 0.98 * arr.shape[0] * arr.shape[1] else arr

def load_image(path):
    """Working copy (cached): rotated upright, black bars cropped, longest side <= MAX_DIM."""
    key = (os.path.abspath(path), MAX_DIM)
    if key not in _cache:
        im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        arr = autocrop_black(np.asarray(im, dtype=np.float32) / 255.0)
        s = MAX_DIM / max(arr.shape[:2])
        if s < 1:
            im = Image.fromarray((arr * 255 + 0.5).astype(np.uint8))
            im = im.resize((int(round(arr.shape[1] * s)), int(round(arr.shape[0] * s))), Image.LANCZOS)
            arr = np.asarray(im, dtype=np.float32) / 255.0
        _cache[key] = arr
    return _cache[key]

def save_image(path, im):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    Image.fromarray((np.clip(im, 0, 1) * 255 + 0.5).astype(np.uint8)).save(path, quality=93)

def size_of(im):
    return [int(im.shape[1]), int(im.shape[0])]          # [width, height]

def rescale_pts(pts, saved_size, cur_size):
    pts = np.asarray(pts, float)
    if saved_size is None or list(saved_size) == list(cur_size):
        return pts
    return pts * np.array([cur_size[0] / saved_size[0], cur_size[1] / saved_size[1]])

def get_sets(names=None):
    """{set_name: [paths]} from IMAGE_SETS; reports ALL problems at once."""
    if names:
        unknown = [n for n in names if n not in IMAGE_SETS]
        if unknown:
            raise SystemExit(f"unknown set(s) {unknown}; IMAGE_SETS has {list(IMAGE_SETS)}")
    sets, problems = {}, []
    for name, paths in IMAGE_SETS.items():
        if names and name not in names:
            continue
        paths = [os.path.expanduser(str(p)) for p in paths]
        if len(paths) < 2:
            problems.append(f"set '{name}' needs at least 2 images")
        problems += [f"set '{name}': file not found: {p}" for p in paths if not os.path.isfile(p)]
        sets[name] = paths
    if not sets:
        problems.append("IMAGE_SETS is empty")
    if problems:
        raise SystemExit("problem with the image lists:\n  " + "\n  ".join(problems))
    return sets

def pair_file(set_name, i):
    return os.path.join(POINTS_DIR, set_name, f"pair_{i}_{i+1}.json")

def rect_file(name):
    return os.path.join(POINTS_DIR, f"rect_{name}.json")

def save_pair_points(set_name, i, paths, imgs, left, right):
    f = pair_file(set_name, i)
    os.makedirs(os.path.dirname(f), exist_ok=True)
    json.dump({"file1": os.path.basename(paths[i]), "file2": os.path.basename(paths[i + 1]),
               "im1": np.asarray(left).round(1).tolist(), "im2": np.asarray(right).round(1).tolist(),
               "size1": size_of(imgs[i]), "size2": size_of(imgs[i + 1])}, open(f, "w"), indent=1)

def load_pair_points(set_name, i, paths, imgs):
    """Saved points for images i -> i+1, rescaled to the current working size. Rejects a re-ordered list."""
    f = pair_file(set_name, i)
    d = json.load(open(f))
    want = (os.path.basename(paths[i]), os.path.basename(paths[i + 1]))
    got = (d.get("file1"), d.get("file2"))
    if None not in got and got != want:
        raise ValueError(f"{f} was clicked on {got[0]} -> {got[1]} but positions {i},{i+1} of '{set_name}' are now "
                         f"{want[0]} -> {want[1]}. Re-click with: python3 mosaic.py --redo {set_name}")
    return (rescale_pts(d["im1"], d.get("size1"), size_of(imgs[i])),
            rescale_pts(d["im2"], d.get("size2"), size_of(imgs[i + 1])))


# point picker (the clicking window)

NON_INTERACTIVE = {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}

def interactive_backend():
    """Switch matplotlib to a GUI backend; True if a window can be opened."""
    if matplotlib.get_backend().lower() not in NON_INTERACTIVE:
        return True
    for b in ("TkAgg", "QtAgg", "Qt5Agg", "MacOSX", "GTK3Agg", "WXAgg"):
        try:
            plt.switch_backend(b)
            return True
        except Exception:
            continue
    return False

class PointPicker:
    """pairs mode: images=[left, right], order=[0, 1] (clicks alternate).  quad mode: 1 image, max_points=4."""
    STEP, BIG, ZOOM, MIN_VIEW = 1.0, 10.0, 0.75, 12.0

    def __init__(self, images, titles, title, order, sides, labels=None, max_points=None, min_points=4):
        for k in [k for k in plt.rcParams if k.startswith("keymap.")]:   # we use our own keys
            plt.rcParams[k] = []
        self.images, self.titles, self.title = images, titles, title
        self.order, self.sides, self.labels = order, sides, labels
        self.max_points, self.min_points = max_points, min_points
        n = len(images)
        self.fig, axs = plt.subplots(1, n, figsize=(8.2 * n, 7.8), squeeze=False)
        self.axes, self.full = list(axs[0]), []
        for ax, im in zip(self.axes, images):
            ax.imshow(im); ax.set_autoscale_on(False)
            h, w = im.shape[:2]
            self.full.append([-0.5, w - 0.5, -0.5, h - 0.5])
        self.view = [list(f) for f in self.full]
        self.pts, self.action, self.msg, self._warned = [], None, "", False          # pts: [axis index, x, y]
        self._artists, self._drag, self._last_ax = [], None, 0
        self.fig.subplots_adjust(top=0.86, bottom=0.14, left=0.04, right=0.99, wspace=0.08)
        self._sup = self.fig.suptitle("", fontsize=11)
        self.fig.text(0.5, 0.008, "wheel/+,- zoom   middle-drag/WASD pan   arrows nudge last point (Shift = 10 px)   "
                      "right-click/Backspace undo   R reset zoom   Enter save   Esc skip   Q quit",
                      ha="center", fontsize=8.5, color="dimgray")
        self._btns = []                                           # keep references or the buttons stop working
        for x, label, fn, col in ((0.30, "Save && next  (Enter)", lambda e: self._finish(), "#b7e4c7"),
                                  (0.46, "Undo  (Backspace)", lambda e: self._undo(), "#eeeeee"),
                                  (0.60, "Skip this pair  (Esc)", lambda e: self._close_with("skip"), "#eeeeee"),
                                  (0.76, "Quit  (Q)", lambda e: self._close_with("quit"), "#f4b6b6")):
            b = Button(self.fig.add_axes([x, 0.045, 0.14 if x < 0.7 else 0.09, 0.05]), label.replace("&&", "&"), color=col)
            b.on_clicked(fn); self._btns.append(b)
        for ev, fn in (("button_press_event", self._on_press), ("button_release_event", lambda e: setattr(self, "_drag", None)),
                       ("motion_notify_event", self._on_motion), ("scroll_event", self._on_scroll),
                       ("key_press_event", self._on_key), ("close_event", self._on_close)):
            self.fig.canvas.mpl_connect(ev, fn)
        self._apply_views(); self._redraw()

    # ---- state
    def _expected(self):
        if self.max_points is not None and len(self.pts) >= self.max_points:
            return None
        return self.order[len(self.pts) % len(self.order)]

    def _complete(self):
        n = len(self.pts)
        if self.max_points is not None:
            return n == self.max_points
        return n % len(self.order) == 0 and n // len(self.order) >= self.min_points

    def _status(self):
        need, n = self._expected(), len(self.pts)
        if self.max_points is not None:
            nxt = (f"click corner {self.labels[n]}  ({n}/{self.max_points})" if need is not None
                   else "all corners placed: press Enter (or Backspace to undo)")
        else:
            nxt = f"{n // len(self.order)} pair(s) placed (need {self.min_points}+); next: click in the {self.sides[need]} image"
        return f"{self.title}   |   {nxt}" + (f"\n{self.msg}" if self.msg else "")

    def _redraw(self):
        for a in self._artists:
            a.remove()
        self._artists, last, need = [], len(self.pts) - 1, self._expected()
        for k, (ai, x, y) in enumerate(self.pts):
            ax = self.axes[ai]
            if k == last:
                self._artists += ax.plot([x], [y], "+", color="cyan", ms=22, mew=2.2)
                self._artists += ax.plot([x], [y], "o", mfc="none", mec="cyan", ms=9, mew=1.5)
            else:
                self._artists += ax.plot([x], [y], "o", mfc="red", mec="white", ms=6, mew=0.8)
            t = ax.annotate(self.labels[k] if self.labels else str(k // len(self.order)), (x, y), xytext=(7, -7),
                            textcoords="offset points", color="yellow", fontsize=10, fontweight="bold")
            t.set_path_effects([pe.withStroke(linewidth=2.5, foreground="black")])
            self._artists.append(t)
        for i, ax in enumerate(self.axes):                       # green border = "click here next"
            on = i == need
            ax.set_title(self.titles[i], fontsize=10, fontweight="bold" if on else "normal", color="green" if on else "black")
            for s in ax.spines.values():
                s.set_edgecolor("limegreen" if on else "gray"); s.set_linewidth(3.5 if on else 1)
        self._sup.set_text(self._status())
        self.fig.canvas.draw_idle()

    # ---- views
    def _apply_views(self):
        for ax, (x0, x1, y0, y1) in zip(self.axes, self.view):
            ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)

    def _set_view(self, ai, x0, x1, y0, y1):
        fx0, fx1, fy0, fy1 = self.full[ai]
        w, h = x1 - x0, y1 - y0
        x0, y0 = min(max(x0, fx0), fx1 - w), min(max(y0, fy0), fy1 - h)
        self.view[ai] = [x0, x0 + w, y0, y0 + h]
        self._apply_views(); self.fig.canvas.draw_idle()

    def _zoom(self, ai, x, y, f):                                # f < 1 zooms in
        x0, x1, y0, y1 = self.view[ai]
        fx0, fx1, _, _ = self.full[ai]
        nx0, nx1, ny0, ny1 = x - (x - x0) * f, x + (x1 - x) * f, y - (y - y0) * f, y + (y1 - y) * f
        if nx1 - nx0 >= fx1 - fx0:
            self.view[ai] = list(self.full[ai]); self._apply_views(); self.fig.canvas.draw_idle()
        elif nx1 - nx0 >= self.MIN_VIEW:
            self._set_view(ai, nx0, nx1, ny0, ny1)

    def _target(self, e):
        if e.inaxes in self.axes and e.xdata is not None:
            self._last_ax = self.axes.index(e.inaxes)
            return self._last_ax, e.xdata, e.ydata
        ai = self._last_ax
        x0, x1, y0, y1 = self.view[ai]
        return ai, (x0 + x1) / 2, (y0 + y1) / 2

    def _toolbar_busy(self):
        mode = getattr(getattr(self.fig.canvas, "toolbar", None), "mode", "")
        return bool(getattr(mode, "value", mode))

    # ---- editing
    def _add(self, ai, x, y):
        need = self._expected()
        if need is None:
            self.msg = "all points placed: press Enter (Backspace undoes the last one)"
        elif ai != need:
            self.msg = f"click in the {self.sides[need]} image next (green border)"
        else:
            h, w = self.images[ai].shape[:2]
            self.pts.append([ai, float(np.clip(x, 0, w - 1)), float(np.clip(y, 0, h - 1))]); self.msg = ""; self._warned = False
        self._redraw()

    def _undo(self):
        if self.pts:
            self.pts.pop()
        self.msg = ""; self._redraw()

    def _nudge(self, dx, dy):
        if self.pts:
            ai, x, y = self.pts[-1]
            h, w = self.images[ai].shape[:2]
            self.pts[-1][1], self.pts[-1][2] = float(np.clip(x + dx, 0, w - 1)), float(np.clip(y + dy, 0, h - 1))
            self._redraw()

    def _finish(self):
        if not self._complete():
            self.msg = ("place all 4 corners first" if self.max_points else
                        f"need at least {self.min_points} complete pairs, and the last point must be in the RIGHT image")
            self._redraw(); return
        if self.max_points is None and not self._warned:        # pairs mode: refuse (once) a badly spread set
            bad = []
            for k in (0, 1):
                p = np.array([[x, y] for ai, x, y in self.pts if ai == k])
                h, w = self.images[k].shape[:2]
                sx, sy = np.ptp(p[:, 0]) / w, np.ptp(p[:, 1]) / h
                if min(sx, sy) < 0.25:
                    bad.append(f"{self.sides[k]} image: {sx:.0%} wide x {sy:.0%} tall")
            if bad:
                self._warned = True
                self.msg = ("WEAK POINTS (" + "; ".join(bad) + "). They must spread out in BOTH directions, not sit in a line or a corner: "
                            "add points higher/lower and further left/right. Press Enter again to save anyway.")
                self._redraw(); return
        self.action = "save"; plt.close(self.fig)

    # ---- events
    def _on_press(self, e):
        if e.inaxes not in self.axes or self._toolbar_busy():
            return
        ai, x, y = self._target(e)
        if e.button == 1:
            self._add(ai, x, y)
        elif e.button == 3:
            self._undo()
        elif e.button == 2:
            self._drag = (ai, e.inaxes.transData.inverted().transform((e.x, e.y)), list(self.view[ai]), e.inaxes.transData.inverted())

    def _on_motion(self, e):
        if e.inaxes in self.axes:
            self._last_ax = self.axes.index(e.inaxes)
        if self._drag is not None:
            ai, p0, v0, inv0 = self._drag
            p1 = inv0.transform((e.x, e.y))
            dx, dy = p0[0] - p1[0], p0[1] - p1[1]
            self._set_view(ai, v0[0] + dx, v0[1] + dx, v0[2] + dy, v0[3] + dy)

    def _on_scroll(self, e):
        if e.inaxes in self.axes and not self._toolbar_busy():
            ai, x, y = self._target(e)
            up = (e.button == "up") if e.button in ("up", "down") else (e.step > 0)
            self._zoom(ai, x, y, self.ZOOM if up else 1 / self.ZOOM)

    def _on_key(self, e):
        key = e.key or ""
        base, step = key.replace("shift+", ""), (self.BIG if key.startswith("shift+") else self.STEP)
        low = base.lower()
        if base in ("left", "right", "up", "down"):
            self._nudge({"left": -step, "right": step}.get(base, 0), {"up": -step, "down": step}.get(base, 0))
        elif base == "backspace" or low == "u":
            self._undo()
        elif base in ("enter", "return"):
            self._finish()
        elif base == "escape":
            self._close_with("skip")
        elif low == "q":
            self._close_with("quit")
        elif base in ("+", "=", "plus", "-", "_", "minus"):
            ai, x, y = self._target(e)
            self._zoom(ai, x, y, self.ZOOM if base in ("+", "=", "plus") else 1 / self.ZOOM)
        elif low == "r":
            self.view = [list(f) for f in self.full]; self._apply_views(); self.fig.canvas.draw_idle()
        elif low in "wasd" and len(low) == 1:
            ai, _, _ = self._target(e)
            x0, x1, y0, y1 = self.view[ai]
            self._set_view(ai, x0 + {"a": -.2, "d": .2}.get(low, 0) * (x1 - x0), x1 + {"a": -.2, "d": .2}.get(low, 0) * (x1 - x0),
                           y0 + {"w": -.2, "s": .2}.get(low, 0) * (y1 - y0), y1 + {"w": -.2, "s": .2}.get(low, 0) * (y1 - y0))

    def _close_with(self, action):
        self.action = action; plt.close(self.fig)

    def _on_close(self, e):
        if self.action is None:                                  # window closed with the X button
            self.action = "save" if self._complete() else "quit"

    def run(self):
        plt.show()
        return self.action

    def pairs(self):
        return tuple(np.array([[x, y] for ai, x, y in self.pts if ai == k]).reshape(-1, 2) for k in (0, 1))

    def corners(self):
        return np.array([[x, y] for _, x, y in self.pts]).reshape(-1, 2)


def collect_points(sets, rects, redo):
    """Phase 1: open the picker for every pair / rectification that has no saved points yet.
    Returns the list of things still missing (headless or skipped)."""
    redo_all, redo_names = redo == [], set(redo or [])
    pairs_todo, quads_todo = [], []
    for name, paths in sets.items():
        for i in range(len(paths) - 1):
            if redo is None or not (redo_all or name in redo_names):
                if os.path.exists(pair_file(name, i)):
                    continue
            pairs_todo.append(("pair", name, i))
    for job in rects:
        if "quad" in job:
            print(f"  rectification '{job['name']}': corners are typed into the config (delete `quad` there to click them instead)")
        elif os.path.exists(rect_file(job["name"])) and not (redo_all or job["name"] in redo_names):
            print(f"  rectification '{job['name']}': saved corners found ({rect_file(job['name'])}); "
                  f"to click them again run with  --redo {job['name']}")
        else:
            quads_todo.append(("quad", job["name"], job))
    todo = quads_todo + pairs_todo                                # the 4-click rectification windows come FIRST
    print("click plan: " + (f"{len(quads_todo)} rectification window(s) ({', '.join(t[1] for t in quads_todo)}), " if quads_todo else "")
          + f"{len(pairs_todo)} pair window(s)")
    if not todo:
        return []
    print(f"{len(todo)} click task(s) to do ...")
    if not interactive_backend():
        print("  cannot open a window here (no display / no GUI toolkit; on your own computer try: pip install PyQt5).\n"
              "  Run this on your own computer to click, or use --no-click to skip clicking.")
        return todo
    imgs_of = {n: [load_image(p) for p in ps] for n, ps in sets.items()}
    for kind, name, arg in todo:
        if kind == "pair":
            i, paths, imgs = arg, sets[name], imgs_of[name]
            pk = PointPicker([imgs[i], imgs[i + 1]],
                             [f"image {i}: {os.path.basename(paths[i])}", f"image {i+1}: {os.path.basename(paths[i+1])}"],
                             title=f"{name}: pair {i} -> {i+1}", order=[0, 1], sides=["LEFT", "RIGHT"])
            act = pk.run()
            if act == "quit":
                print(f"  window closed before pair {i}->{i+1} of '{name}' was complete (it needs 4+ pairs, then Enter / the green button), "
                      "so nothing was saved for it. Stopped clicking; earlier pairs are kept."); return todo
            if act == "skip":
                print(f"  skipped pair {i}->{i+1} of '{name}'")
            if act == "save":
                l, r = pk.pairs()
                save_pair_points(name, i, paths, imgs, l, r)
                print(f"  saved {len(l)} pairs -> {pair_file(name, i)}")
        else:
            im = load_image(arg["image"])
            pk = PointPicker([im], [os.path.basename(arg["image"])], title=f"{name}: click the 4 corners of a real-world rectangle",
                             order=[0], sides=["IMAGE"], labels=["TL", "TR", "BR", "BL"], max_points=4)
            act = pk.run()
            if act == "quit":
                print(f"  window closed before the 4 corners of '{name}' were placed; stopped clicking."); return todo
            if act == "save":
                os.makedirs(POINTS_DIR, exist_ok=True)
                json.dump({"quad": pk.corners().round(1).tolist(), "size": size_of(im)}, open(rect_file(name), "w"))
                print(f"  saved corners -> {rect_file(name)}")
    return [t for t in todo if not (os.path.exists(pair_file(t[1], t[2])) if t[0] == "pair" else os.path.exists(rect_file(t[1])))]



# A.2  RECOVER HOMOGRAPHIES

def build_system(p1, p2):
    """
    n point pairs -> the linear system A h = b for the 8 unknowns of H = [[a b c],[d e f],[g h 1]].

        x' = (a x + b y + c) / (g x + h y + 1),   y' = (d x + e y + f) / (g x + h y + 1)

    Multiply by the denominator and move the unknowns left; the equations become LINEAR:

        a x + b y + c           - g x x' - h y x' = x'
                  d x + e y + f - g x y' - h y y' = y'

    Each correspondence gives 2 rows, so we need n >= 4 points (H has 8 degrees of freedom).
    """
    p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
    n = len(p1)
    A, b = np.zeros((2 * n, 8)), np.zeros(2 * n)
    x, y, xp, yp = p1[:, 0], p1[:, 1], p2[:, 0], p2[:, 1]
    A[0::2, 0], A[0::2, 1], A[0::2, 2], A[0::2, 6], A[0::2, 7] = x, y, 1, -x * xp, -y * xp
    A[1::2, 3], A[1::2, 4], A[1::2, 5], A[1::2, 6], A[1::2, 7] = x, y, 1, -x * yp, -y * yp
    b[0::2], b[1::2] = xp, yp
    return A, b

def _normalize(pts):
    """Hartley normalisation (mean 0, mean distance sqrt(2)): raw pixel coordinates make A badly
    conditioned (x*x' ~ 1e6 next to 1s) and least squares then amplifies click noise."""
    c = pts.mean(axis=0)
    s = np.sqrt(2) / np.mean(np.linalg.norm(pts - c, axis=1))
    return (pts - c) * s, np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1.0]])

def computeH(im1_pts, im2_pts):
    """H = computeH(im1_pts, im2_pts): n-by-2 (x, y) arrays, returns 3x3 H with p2 ~ H p1.
    n = 4 solves exactly; n > 4 is overdetermined and solved by least squares (averages click noise)."""
    p1, p2 = np.asarray(im1_pts, float), np.asarray(im2_pts, float)
    if p1.shape != p2.shape or p1.ndim != 2 or p1.shape[1] != 2 or len(p1) < 4:
        raise ValueError("computeH needs two n-by-2 arrays with n >= 4")
    q1, T1 = _normalize(p1)
    q2, T2 = _normalize(p2)
    A, b = build_system(q1, q2)
    h, *_ = np.linalg.lstsq(A, b, rcond=None)
    H = np.linalg.inv(T2) @ np.append(h, 1.0).reshape(3, 3) @ T1
    return H / H[2, 2]

def apply_H(H, pts):
    ph = np.c_[np.asarray(pts, float), np.ones(len(pts))] @ H.T
    return ph[:, :2] / ph[:, 2:3]

def reprojection_errors(H, p1, p2):
    return np.linalg.norm(apply_H(H, p1) - np.asarray(p2, float), axis=1)


# A.3  WARP (inverse warping, both interpolations by hand)

def output_bbox(shape, H):
    """Bounding box (xmin, ymin, xmax, ymax) of the warped image: push the 4 corners through H."""
    h, w = shape[:2]
    c = apply_H(H, np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], float))
    return (int(np.floor(c[:, 0].min())), int(np.floor(c[:, 1].min())),
            int(np.ceil(c[:, 0].max())), int(np.ceil(c[:, 1].max())))

def _inverse_map(shape, H, bbox):
    """For every OUTPUT pixel find where it came from in the SOURCE (apply H^-1). Looping over output
    pixels instead of pushing source pixels forward leaves no holes."""
    xmin, ymin, xmax, ymax = bbox
    xs, ys = np.meshgrid(np.arange(xmin, xmax + 1), np.arange(ymin, ymax + 1))
    S = np.linalg.inv(H) @ np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)])
    d = S[2]
    h, w = shape[:2]
    cw = (H @ np.array([w / 2, h / 2, 1.0]))[2]
    ok = (d * cw) > 1e-9 * abs(cw)                    # drop the "behind the camera" mirror image
    d = np.where(ok, d, 1.0)
    return S[0] / d, S[1] / d, ok, (ymax - ymin + 1, xmax - xmin + 1)

def warpImageNearestNeighbor(im, H, bbox=None):
    """Round to the closest source pixel. Returns (warped, mask, (xmin, ymin)); mask = 1 where real data."""
    bbox = bbox or output_bbox(im.shape, H)
    sx, sy, ok, (oh, ow) = _inverse_map(im.shape, H, bbox)
    h, w = im.shape[:2]
    xi, yi = np.rint(sx).astype(np.int64), np.rint(sy).astype(np.int64)
    valid = ok & (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    out = np.zeros((oh * ow,) + im.shape[2:], im.dtype)
    out[valid] = im[yi[valid], xi[valid]]
    return out.reshape((oh, ow) + im.shape[2:]), valid.reshape(oh, ow).astype(np.float32), bbox[:2]

def warpImageBilinear(im, H, bbox=None):
    """Weighted average of the 4 surrounding pixels:
    I(x,y) = (1-dx)(1-dy) I00 + dx(1-dy) I10 + (1-dx)dy I01 + dx dy I11."""
    bbox = bbox or output_bbox(im.shape, H)
    sx, sy, ok, (oh, ow) = _inverse_map(im.shape, H, bbox)
    h, w = im.shape[:2]
    valid = ok & (sx >= 0) & (sx <= w - 1) & (sy >= 0) & (sy <= h - 1)
    x, y = np.clip(sx[valid], 0, w - 1), np.clip(sy[valid], 0, h - 1)
    x0, y0 = np.minimum(np.floor(x).astype(np.int64), w - 2), np.minimum(np.floor(y).astype(np.int64), h - 2)
    dx, dy = x - x0, y - y0
    if im.ndim == 3:
        dx, dy = dx[:, None], dy[:, None]
    vals = ((1 - dx) * (1 - dy) * im[y0, x0] + dx * (1 - dy) * im[y0, x0 + 1] +
            (1 - dx) * dy * im[y0 + 1, x0] + dx * dy * im[y0 + 1, x0 + 1])
    out = np.zeros((oh * ow,) + im.shape[2:], im.dtype)
    out[valid] = vals
    return out.reshape((oh, ow) + im.shape[2:]), valid.reshape(oh, ow).astype(np.float32), bbox[:2]

def best_time(fn, repeat=3):
    ts = []
    for _ in range(repeat):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return min(ts)



# A.4  MOSAIC

def homographies_to_reference(Hpair, idxs, ref):
    """Chain neighbouring homographies so every image maps into the reference image's frame
    (matrix products going one way, inverses the other). The middle image keeps stretching small."""
    out = {ref: np.eye(3)}
    for i in range(ref - 1, idxs[0] - 1, -1):
        out[i] = out[i + 1] @ Hpair[i]
    for i in range(ref + 1, idxs[-1] + 1):
        out[i] = out[i - 1] @ np.linalg.inv(Hpair[i - 1])
    return {k: v / v[2, 2] for k, v in out.items()}

def _poly_area(c):
    return 0.5 * abs(np.dot(c[:, 0], np.roll(c[:, 1], -1)) - np.dot(c[:, 1], np.roll(c[:, 0], -1)))

def warp_is_sane(shape, H, max_stretch=MAX_STRETCH):
    """Image must stay in front of the virtual camera and not be stretched absurdly."""
    if not CHECK:
        return True, 1.0
    h, w = shape[:2]
    pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1], [w / 2, h / 2]], float)
    ph = np.c_[pts, np.ones(5)] @ H.T
    if not (np.all(ph[:, 2] > 0) or np.all(ph[:, 2] < 0)):
        return False, "part of the image would land 'behind' the camera (view too wide for a plane)"
    ratio = _poly_area((ph[:, :2] / ph[:, 2:3])[:4]) / ((w - 1) * (h - 1))
    if not (1 / max_stretch <= ratio <= max_stretch):
        return False, f"image would be stretched by an area factor of {ratio:.1f}"
    return True, ratio

def plan_pieces(imgs, Hpair):
    """Cover the set with the fewest flat mosaics: greedily take the longest run of consecutive images
    that stays valid (middle image = reference), start the next piece at that run's last image."""
    n, pieces, notes, s = len(imgs), [], [], 0
    while s < n - 1:
        for L in range(n - s, 1, -1):
            idxs = list(range(s, s + L))
            ref = idxs[len(idxs) // 2]
            Hs = homographies_to_reference(Hpair, idxs, ref)
            checks = [(i, *warp_is_sane(imgs[i].shape, Hs[i])) for i in idxs]
            if all(c[1] for c in checks):
                break
            if s == 0 and L == n:
                bad = [c for c in checks if not c[1]][0]
                notes.append(f"all {n} images do not fit in one flat mosaic (image {bad[0]}: {bad[2]})")
        else:                                    # even the 2-image mosaic failed; `checks` is from that last attempt
            bad = [c for c in checks if not c[1]][0]
            raise RuntimeError(f"images {s} and {s+1} do not give a usable homography: image {bad[0]} {bad[2]}. "
                               "This almost always means the clicked points are weak: too few, bunched in one corner, "
                               "or a mismatched feature (reflections and repeated windows are easy to mix up). "
                               "Click 10-20 pairs spread over the whole overlap, then re-run with --redo <set>. "
                               "(--no-check builds the mosaic anyway so you can see what is wrong.)")
        pieces.append((idxs, ref, Hs))
        s += L - 1
    if len(pieces) > 1:
        notes.append("split into overlapping pieces " + ", ".join(f"{p[0][0]}..{p[0][-1]}" for p in pieces)
                     + " (a cylindrical/spherical projection would be needed for one panorama)")
    return pieces, notes

def feather_weight(shape):
    """Distance to the nearest image border, normalised: 0 at the edge, 1 at the centre."""
    h, w = shape[:2]
    d = distance_transform_edt(np.pad(np.ones((h, w)), 1))[1:-1, 1:-1]
    return (d / d.max()).astype(np.float32)

def build_mosaic(imgs, idxs, Hs, max_dim=None, sigma=None):
    """Warp the images into one canvas and blend two ways in one pass:
    feather (weighted average by distance-to-border), and
    twoband (low frequencies feathered with a blurred weight; high frequencies from the single image
    with the largest weight, so edges cannot ghost)."""
    max_dim = max_dim or CANVAS_MAX
    boxes = np.array([output_bbox(imgs[i].shape, Hs[i]) for i in idxs])
    span = max(boxes[:, 2].max() - boxes[:, 0].min(), boxes[:, 3].max() - boxes[:, 1].min())
    s = min(1.0, max_dim / span)
    Hs = {i: np.diag([s, s, 1.0]) @ Hs[i] for i in idxs}
    boxes = np.array([output_bbox(imgs[i].shape, Hs[i]) for i in idxs])
    bbox = (boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].max())
    oh, ow = bbox[3] - bbox[1] + 1, bbox[2] - bbox[0] + 1
    sigma, eps = sigma or max(3.0, ow / 300.0), 1e-8

    f_num, f_den = np.zeros((oh, ow, 3), np.float32), np.zeros((oh, ow), np.float32)
    l_num, l_den = np.zeros((oh, ow, 3), np.float32), np.zeros((oh, ow), np.float32)
    hi_sel, best, cover = np.zeros((oh, ow, 3), np.float32), np.zeros((oh, ow), np.float32), np.zeros((oh, ow), np.float32)
    for i in idxs:
        warped, mask, _ = warpImageBilinear(np.dstack([imgs[i], feather_weight(imgs[i].shape)]), Hs[i], bbox)
        rgb, w = warped[..., :3], warped[..., 3] * mask
        f_num += w[..., None] * rgb; f_den += w
        low = gaussian_filter(rgb * mask[..., None], (sigma, sigma, 0)) / (gaussian_filter(mask, sigma)[..., None] + eps)
        wb = gaussian_filter(w, sigma)
        l_num += wb[..., None] * low; l_den += wb
        sel = w > best
        hi_sel[sel] = (rgb - low)[sel]; best[sel] = w[sel]
        cover = np.maximum(cover, mask)
    feather = np.where(f_den[..., None] > 1e-6, f_num / np.maximum(f_den[..., None], eps), 0)
    twoband = np.where(l_den[..., None] > 1e-6, l_num / np.maximum(l_den[..., None], eps) + hi_sel, 0)
    out = {"feather": feather, "twoband": twoband}
    return {k: np.where(cover[..., None] > 0, np.clip(v, 0, 1), 0).astype(np.float32) for k, v in out.items()}


# figures

def fig_sources(name, imgs, path):
    n = len(imgs); cols = min(n, 5); rows = int(np.ceil(n / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.0 * rows), squeeze=False)
    for k, a in enumerate(ax.ravel()):
        a.axis("off")
        if k < n:
            a.imshow(imgs[k]); a.set_title(f"{name} / image {k}", fontsize=10)
    plt.tight_layout(); plt.savefig(path, dpi=90, pil_kwargs={"quality": 88}); plt.close()

def fig_correspondences(imgs, i, p1, p2, path):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    for a, im, p, t in ((ax[0], imgs[i], p1, f"image {i}"), (ax[1], imgs[i + 1], p2, f"image {i+1}")):
        a.imshow(im); a.set_title(f"{t}  ({len(p)} correspondences)"); a.axis("off")
        a.scatter(p[:, 0], p[:, 1], c="r", s=24, edgecolors="w", linewidths=0.5)
        for k, (x, y) in enumerate(p):
            a.text(x + 5, y - 5, str(k), color="yellow", fontsize=8)
    plt.tight_layout(); plt.savefig(path, dpi=100, pil_kwargs={"quality": 88}); plt.close()

def fig_mosaic(name, imgs, idxs, res, path):
    k = len(idxs)
    ch, cw = res["feather"].shape[:2]
    mh = 14 * ch / cw
    fig = plt.figure(figsize=(14, 2.6 + 2 * (mh + 0.4)))
    gs = fig.add_gridspec(3, k, height_ratios=[2.6, mh + 0.4, mh + 0.4])
    for c, i in enumerate(idxs):
        a = fig.add_subplot(gs[0, c]); a.imshow(imgs[i]); a.set_title(f"source {i}", fontsize=10); a.axis("off")
    for r, (key, title) in enumerate([("feather", "feathered weighted average"),
                                      ("twoband", "2-band blend (feather low freq., single-image high freq.)")], start=1):
        a = fig.add_subplot(gs[r, :]); a.imshow(res[key]); a.set_title(title, fontsize=11); a.axis("off")
    plt.tight_layout(); plt.savefig(path, dpi=80, pil_kwargs={"quality": 88}); plt.close()


# rectification (A.3)

def get_quad(job, im):
    """Corners from the config, else from points/rect_<name>.json, else None."""
    if "quad" in job:
        return rescale_pts(job["quad"], job.get("size"), size_of(im))
    f = rect_file(job["name"])
    if os.path.exists(f):
        d = json.load(open(f))
        return rescale_pts(d["quad"], d.get("size"), size_of(im))
    return None

def _stats(d):
    """d = array of signed errors in 0..255 units -> (rmse, mean abs, psnr dB)."""
    rmse = float(np.sqrt(np.mean(d ** 2))) if d.size else float("nan")
    return rmse, float(np.mean(np.abs(d))) if d.size else float("nan"), (20 * np.log10(255 / rmse) if rmse > 0 else float("inf"))

def interpolation_errors(nn, m_nn, bl, m_bl):
    """
    Numbers behind the nearest-neighbour vs bilinear comparison (0..255 pixel units), over the pixels that both
    methods filled: RMSE, mean |diff|, PSNR, max |diff|, and the % of pixels differing by more than 2.
    """
    both = (m_nn > 0) & (m_bl > 0)
    d = (nn - bl)[both] * 255
    return {"diff": _stats(d), "diff_max": float(np.abs(d).max()) if d.size else float("nan"),
            "diff_frac": float((np.abs(d).max(axis=1) > 2).mean()) if d.size else float("nan")}

def rectify(job, quad, outdir):
    """Homography that sends 4 clicked corners of a real-world rectangle to a true rectangle; everything
    on that plane then looks as if the camera pointed straight at it. quad order: TL, TR, BR, BL."""
    im = load_image(job["image"])
    aspect, rect_h, margin = job.get("aspect", 1.0), job.get("rect_h", 500), job.get("margin", 0.35)
    rect_w = rect_h * aspect
    dst = np.array([[0, 0], [rect_w, 0], [rect_w, rect_h], [0, rect_h]], float)
    H = computeH(quad, dst)
    mx, my = margin * rect_w, margin * rect_h
    bbox = (int(-mx), int(-my), int(rect_w + mx), int(rect_h + my))
    nn, m_nn, _ = warpImageNearestNeighbor(im, H, bbox)
    bl, m_bl, _ = warpImageBilinear(im, H, bbox)
    E = interpolation_errors(nn, m_nn, bl, m_bl)
    t_nn = best_time(lambda: warpImageNearestNeighbor(im, H, bbox))
    t_bl = best_time(lambda: warpImageBilinear(im, H, bbox))
    name = job["name"]
    save_image(f"{outdir}/{name}_nearest.png", nn)
    save_image(f"{outdir}/{name}_bilinear.png", bl)
    wr = [im.shape[1] / im.shape[0], nn.shape[1] / nn.shape[0], nn.shape[1] / nn.shape[0]]
    fig, ax = plt.subplots(1, 3, figsize=(17, 17 / sum(wr) * 1.15), gridspec_kw={"width_ratios": wr})
    ax[0].imshow(im); ax[0].set_title("original + clicked corners")
    q = np.vstack([quad, quad[:1]]); ax[0].plot(q[:, 0], q[:, 1], "r-o", ms=4)
    ax[1].imshow(nn); ax[1].set_title(f"rectified - nearest neighbour ({t_nn*1e3:.0f} ms)")
    ax[2].imshow(bl); ax[2].set_title(f"rectified - bilinear ({t_bl*1e3:.0f} ms)")
    for a in ax: a.axis("off")
    plt.tight_layout(); plt.savefig(f"{outdir}/{name}_compare.jpg", dpi=100, pil_kwargs={"quality": 88}); plt.close()
    cy, cx = nn.shape[0] // 2, nn.shape[1] // 2
    sl = (slice(cy - 35, cy + 35), slice(cx - 50, cx + 50))
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    ax[0].imshow(nn[sl], interpolation="nearest"); ax[0].set_title("nearest (zoom)")
    ax[1].imshow(bl[sl], interpolation="nearest"); ax[1].set_title("bilinear (zoom)")
    ax[2].imshow(np.abs(nn - bl)[sl].mean(2) * 4, cmap="magma", vmin=0, vmax=1); ax[2].set_title("|difference| x4")
    for a in ax: a.axis("off")
    plt.tight_layout(); plt.savefig(f"{outdir}/{name}_zoom.jpg", dpi=100, pil_kwargs={"quality": 88}); plt.close()
    return dict(name=name, image=job["image"], t_nn=t_nn, t_bl=t_bl, err=E)


# one image set (A.1 + A.2 + A.4)

def process_set(name, paths):
    outdir = os.path.join(OUT_DIR, name); os.makedirs(outdir, exist_ok=True)
    imgs = [load_image(p) for p in paths]
    n = len(imgs)
    res = dict(name=name, n=n, notes=[])
    fig_sources(name, imgs, f"{outdir}/sources.jpg")
    missing = [i for i in range(n - 1) if not os.path.exists(pair_file(name, i))]
    if missing:
        res["error"] = f"missing clicked points for pairs {missing}; run python3 mosaic.py on a computer with a display"
        print(f"[{name}] {res['error']}"); return res

    Hpair, res["pairs"] = {}, []
    for i in range(n - 1):
        try:
            p1, p2 = load_pair_points(name, i, paths, imgs)
            if len(p1) < 4:
                raise ValueError(f"pair {i}->{i+1} has only {len(p1)} points (need >= 4)")
        except ValueError as ex:
            res["error"] = str(ex); print(f"[{name}] {ex}"); return res
        for k, (p, im) in enumerate(((p1, imgs[i]), (p2, imgs[i + 1]))):
            sx, sy = np.ptp(p[:, 0]) / im.shape[1], np.ptp(p[:, 1]) / im.shape[0]
            if min(sx, sy) < 0.3 or len(p1) < 8:
                print(f"[{name}] pair {i}->{i+1}, image {i+k}: {len(p1)} points covering "f"{sx:.0%} x {sy:.0%} of the image")
        H = computeH(p1, p2); Hpair[i] = H
        err = reprojection_errors(H, p1, p2)
        fig_correspondences(imgs, i, p1, p2, f"{outdir}/corr_{i}_{i+1}.jpg")
        res["pairs"].append(dict(i=i, npts=len(p1), H=H, mean=float(err.mean()), max=float(err.max()), corr=f"{name}/corr_{i}_{i+1}.jpg"))
        print(f"[{name}] pair {i}->{i+1}: {len(p1)} pts, mean reprojection error {err.mean():.2f} px")

    try:
        pieces, res["notes"] = plan_pieces(imgs, Hpair)
    except RuntimeError as ex:
        res["error"] = str(ex); print(f"[{name}] {ex}"); return res
    res["pieces"] = []
    for k, (idxs, ref, Hs) in enumerate(pieces):
        suf = "" if len(pieces) == 1 else f"_p{k}"
        t = time.perf_counter()
        out = build_mosaic(imgs, idxs, Hs)
        for key, v in out.items():
            save_image(f"{outdir}/mosaic{suf}_{key}.jpg", v)
        fig_mosaic(name, imgs, idxs, out, f"{outdir}/mosaic{suf}_comparison.jpg")
        size = out["feather"].shape[1::-1]
        res["pieces"].append(dict(idxs=idxs, ref=ref, suf=suf, size=size))
        print(f"[{name}] mosaic{suf} of images {idxs} (reference {ref}): {size[0]}x{size[1]} in {time.perf_counter()-t:.1f}s")
    for x in res["notes"]:
        print(f"[{name}] NOTE: {x}")
    return res

# ======================================================================================
def main():
    global MAX_DIM, CANVAS_MAX, POINTS_DIR, OUT_DIR, CHECK
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--redo", nargs="*", metavar="NAME", help="re-click points (no names = everything; or a set / rectification name)")
    ap.add_argument("--no-click", action="store_true", help="never open a window; use saved points only")
    ap.add_argument("--no-check", action="store_true", help="build mosaics even if a homography looks insane")
    ap.add_argument("--max-dim", type=int, default=MAX_DIM)
    ap.add_argument("--canvas-max", type=int, default=CANVAS_MAX)
    ap.add_argument("--points", default=POINTS_DIR)
    ap.add_argument("--out", default=None, help="output folder (if omitted you are asked in the terminal)")
    a = ap.parse_args()
    CHECK = not a.no_check
    out = a.out
    if out is None:
        if sys.stdin.isatty():
            out = input(f"Output folder [{OUT_DIR}]: ").strip().strip("'\"") or OUT_DIR
        else:
            out = OUT_DIR
    MAX_DIM, CANVAS_MAX, POINTS_DIR, OUT_DIR = a.max_dim, a.canvas_max, a.points, os.path.expanduser(out)
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Results will be written to: {os.path.abspath(OUT_DIR)}")

    sets = get_sets()
    rects = [j for j in RECTIFICATIONS if os.path.isfile(j["image"]) or print(f"[rectify] skipping {j['name']}: its `image` is '{j['image']}', which does not exist. `image` must be the path of the "
                                                                                    f"PHOTO you want to rectify (an input, e.g. data/building/00.jpg). Results are written automatically to "
                                                                                    f"{OUT_DIR}/rectify/{j['name']}_*.png, so you never type an output name.")]

    if not a.no_click:                                           # phase 1: clicking (needs a display)
        collect_points(sets, rects, a.redo)
    plt.switch_backend("Agg")                                    # phase 2: just write files

    results = [process_set(nm, paths) for nm, paths in sets.items()]
    rect_results = []
    if rects:
        os.makedirs(f"{OUT_DIR}/rectify", exist_ok=True)
    for job in rects:
        quad = get_quad(job, load_image(job["image"]))
        if quad is None:
            print(f"[rectify] {job['name']}: no corners yet (run without --no-click on a computer with a display)"); continue
        r = rectify(job, quad, f"{OUT_DIR}/rectify")
        rect_results.append(r)
        E = r["err"]
        print(f"[rectify] {r['name']}: nearest {r['t_nn']*1e3:.0f} ms, bilinear {r['t_bl']*1e3:.0f} ms | "
              f"NN vs bilinear RMSE {E['diff'][0]:.2f} ({E['diff'][2]:.1f} dB)")
    nm = sum(len(s["pieces"]) for s in results if "error" not in s)
    print(f"\nDone: {nm} mosaic(s), {len(rect_results)} rectification(s). Output images are in: {os.path.abspath(OUT_DIR)}"
          + ("" if len(rect_results) >= 2 else "   (assignment wants >= 2 rectifications)"))


if __name__ == "__main__":
    main()