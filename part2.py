"""
Assignment 3, Part 2, built on part1.py.

Pipeline (following Brown, Szeliski & Winder, "Multi-Image Matching using Multi-Scale Oriented Patches",
with the simplifications from the handout):

    B.1  Harris corners  +  Adaptive Non-Maximal Suppression (ANMS)
    B.2  axis-aligned 8x8 descriptors sampled from a blurred 40x40 window, bias/gain normalised
    B.3  matching by Lowe's ratio test (1st NN / 2nd NN) on the 64-d descriptors
    B.4  4-point RANSAC (written from scratch) -> robust homography

    python3 part2.py                     # all sets, results in out2/
    python3 part2.py --sets building     # only one set
    python3 part2.py --ratio 0.7 --eps 3 --anms 500
    python3 part2.py --points points     # folder with part-1 clicks -> also writes manual-vs-auto comparisons
"""

import argparse, os, time
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.spatial import cKDTree

import part1 as P
plt.switch_backend("Agg")

IMAGE_SETS = {
    "building": ["Photoshoot/IMG_7647.jpeg", "Photoshoot/IMG_7648.jpeg", "Photoshoot/IMG_7649.jpeg"],
    "lake_a":   ["Photoshoot/IMG_7670.JPG", "Photoshoot/IMG_7671.JPG", "Photoshoot/IMG_7672.JPG", "Photoshoot/IMG_7673.JPG"],
    "lake_b":   ["Photoshoot/IMG_7671.JPG", "Photoshoot/IMG_7672.JPG", "Photoshoot/IMG_7673.JPG"],
}
OUT_DIR = "out2"


# ---------------------------------------------------------------- B.1 Harris + ANMS

def to_gray(im):
    return (0.299 * im[..., 0] + 0.587 * im[..., 1] + 0.114 * im[..., 2]).astype(np.float64)

def harris(gray, sigma_d=1.0, sigma_i=1.5, margin=20, rel_thresh=0.01, max_pts=6000, nms_size=7):
    """Single-scale Harris detector. Response is the harmonic mean det(M)/tr(M) used in Brown et al.
    Returns (pts as [x, y], strengths) of local maxima, sorted strongest first, `margin` px away from the border
    (the 40x40 descriptor window must fit)."""
    Ix = gaussian_filter(gray, sigma_d, order=(0, 1))
    Iy = gaussian_filter(gray, sigma_d, order=(1, 0))
    Sxx, Syy, Sxy = (gaussian_filter(a, sigma_i) for a in (Ix * Ix, Iy * Iy, Ix * Iy))
    R = (Sxx * Syy - Sxy ** 2) / (Sxx + Syy + 1e-12)
    peaks = (R == maximum_filter(R, size=nms_size)) & (R > rel_thresh * R.max())
    h, w = gray.shape
    peaks[:margin], peaks[-margin:], peaks[:, :margin], peaks[:, -margin:] = False, False, False, False
    ys, xs = np.nonzero(peaks)
    f = R[ys, xs]
    order = np.argsort(-f)[:max_pts]
    return np.c_[xs[order], ys[order]].astype(np.float64), f[order]

def anms(pts, strength, n_keep=500, c_robust=0.9, chunk=400):
    """Adaptive Non-Maximal Suppression (Brown et al. sec. 3). Each corner i gets a suppression radius
        r_i = min_j |x_i - x_j|   over corners j with  f(x_i) < c_robust * f(x_j)
    (the distance to the nearest clearly-stronger corner). Keeping the n corners with the LARGEST radii gives a
    set that is strong AND spatially spread out. pts/strength must be sorted strongest first."""
    n = len(pts)
    radius = np.full(n, np.inf)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        d2 = ((pts[s:e, None, :] - pts[None, :, :]) ** 2).sum(-1)
        stronger = strength[s:e, None] < c_robust * strength[None, :]
        d2 = np.where(stronger, d2, np.inf)
        radius[s:e] = d2.min(axis=1)
    keep = np.argsort(-radius, kind="stable")[:n_keep]        # strongest corner has radius inf -> always kept
    return pts[keep], radius[keep]


# ---------------------------------------------------------------- B.2 descriptors

def extract_descriptors(gray, pts, win=40, size=8, blur_sigma=2.0):
    """Axis-aligned descriptors: blur the image, then sample an 8x8 grid with spacing 5 px from the 40x40 window
    around each corner (so each descriptor value is a smoothed average over a big neighbourhood).
    Bias/gain normalise (zero mean, unit std) for robustness to brightness/contrast changes.
    Returns (descs [n, 64], pts that were kept)."""
    blurred = gaussian_filter(gray, blur_sigma)
    step = win // size                                          # 5
    offs = (np.arange(size) - size // 2) * step + step // 2 + 0  # -17,-12,...,18 (centred on the corner)
    h, w = gray.shape
    xi, yi = np.rint(pts[:, 0]).astype(int), np.rint(pts[:, 1]).astype(int)
    ok = (xi + offs.min() >= 0) & (xi + offs.max() < w) & (yi + offs.min() >= 0) & (yi + offs.max() < h)
    xi, yi, pts = xi[ok], yi[ok], pts[ok]
    patches = blurred[yi[:, None, None] + offs[None, :, None], xi[:, None, None] + offs[None, None, :]]   # n x 8 x 8
    flat = patches.reshape(len(pts), -1)
    flat = (flat - flat.mean(1, keepdims=True)) / (flat.std(1, keepdims=True) + 1e-8)
    return flat, pts


# ---------------------------------------------------------------- B.3 matching

def nn_ratios(d1, d2):
    """For every descriptor in image 1: index of its nearest neighbour in image 2 and the ratio
    d(1st NN) / d(2nd NN)  (small ratio = the best match is much better than the runner-up = distinctive)."""
    dists, idx = cKDTree(d2).query(d1, k=2)
    return idx[:, 0], dists[:, 0] / (dists[:, 1] + 1e-12)

def match_features(d1, d2, ratio=0.7):
    """Lowe ratio test: accept i -> its nearest neighbour j in image 2 if  d(1st NN) / d(2nd NN) < ratio.
    Returns index arrays (i, j) and the ratios."""
    j, r = nn_ratios(d1, d2)
    good = r < ratio
    return np.nonzero(good)[0], j[good], r[good]


# ---------------------------------------------------------------- B.4 RANSAC

def ransac_homography(p1, p2, n_iter=5000, eps=3.0, seed=0, min_inliers=8):
    """4-point RANSAC. Repeat: pick 4 random matches, solve H exactly (computeH from part 1), count matches whose
    reprojection error |H p1 - p2| is below eps pixels (the inliers). Keep the hypothesis with the most inliers
    (ties: lower total error), then re-fit H by least squares on all its inliers (twice, re-selecting inliers).
    Returns H, boolean inlier mask."""
    n = len(p1)
    if n < 4:
        raise ValueError(f"RANSAC needs >= 4 matches, got {n}")
    rng = np.random.default_rng(seed)
    best_inl, best_cost = None, np.inf
    best_n = 0
    with np.errstate(all="ignore"):
        for _ in range(n_iter):
            s = rng.choice(n, 4, replace=False)
            try:
                H = P.computeH(p1[s], p2[s])
            except Exception:
                continue
            if not np.all(np.isfinite(H)):
                continue
            err = P.reprojection_errors(H, p1, p2)
            inl = err < eps
            k = int(inl.sum())
            cost = float(err[inl].sum())
            if k > best_n or (k == best_n and cost < best_cost):
                best_n, best_cost, best_inl = k, cost, inl
    if best_inl is None or best_n < max(4, min_inliers):
        raise RuntimeError(f"RANSAC found only {best_n} inliers out of {n} matches (need >= {min_inliers}); "
                           "try a larger --ratio, more --anms corners, or a larger --eps")
    inl = best_inl
    for _ in range(2):                                           # least-squares refit on the inliers
        H = P.computeH(p1[inl], p2[inl])
        inl = P.reprojection_errors(H, p1, p2) < eps
    H = P.computeH(p1[inl], p2[inl])
    return H, P.reprojection_errors(H, p1, p2) < eps


# ---------------------------------------------------------------- figures

def fig_harris(im, all_pts, kept, path, title):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.6))
    ax[0].imshow(im); ax[0].scatter(*all_pts.T, s=5, c="r"); ax[0].set_title(f"Harris corners, no ANMS ({len(all_pts)})")
    ax[1].imshow(im); ax[1].scatter(*kept.T, s=14, c="cyan", edgecolors="k", linewidths=0.4)
    ax[1].set_title(f"after ANMS ({len(kept)} corners)")
    for a in ax: a.axis("off")
    fig.suptitle(title); plt.tight_layout(); plt.savefig(path, dpi=90, pil_kwargs={"quality": 88}); plt.close()

def fig_descriptors(gray, pts, descs, path, n=12, seed=1):
    """Show where some descriptors come from, the 40x40 window, and the normalised 8x8 descriptor."""
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(len(pts), min(n, len(pts)), replace=False))
    fig, ax = plt.subplots(2, len(pick), figsize=(1.6 * len(pick), 3.6), squeeze=False)
    for c, k in enumerate(pick):
        x, y = np.rint(pts[k]).astype(int)
        win = gray[max(0, y - 20):y + 20, max(0, x - 20):x + 20]
        ax[0, c].imshow(win, cmap="gray", vmin=0, vmax=1); ax[0, c].set_title(f"#{k}", fontsize=8)
        ax[1, c].imshow(descs[k].reshape(8, 8), cmap="gray", interpolation="nearest")
        for r in range(2): ax[r, c].axis("off")
    ax[0, 0].text(-0.15, 0.5, "40x40\nwindow", transform=ax[0, 0].transAxes, ha="right", va="center", fontsize=8)
    ax[1, 0].text(-0.15, 0.5, "8x8\nnormalised", transform=ax[1, 0].transAxes, ha="right", va="center", fontsize=8)
    plt.tight_layout(); plt.savefig(path, dpi=100, pil_kwargs={"quality": 88}); plt.close()

def fig_matches(im1, im2, p1, p2, inl, path, title):
    h = max(im1.shape[0], im2.shape[0])
    canvas = np.zeros((h, im1.shape[1] + im2.shape[1], 3), np.float32)
    canvas[:im1.shape[0], :im1.shape[1]] = im1; canvas[:im2.shape[0], im1.shape[1]:] = im2
    off = im1.shape[1]
    fig, ax = plt.subplots(figsize=(16, 16 * h / canvas.shape[1]))
    ax.imshow(canvas); ax.axis("off")
    for (a, b), ok in zip(zip(p1, p2), inl):
        ax.plot([a[0], b[0] + off], [a[1], b[1] + off * 0], lw=0)       # placeholder keeps autoscale tidy
        ax.plot([a[0], b[0] + off], [a[1], b[1]], "-", color="lime" if ok else "red", lw=0.8 if ok else 0.6, alpha=0.8)
    ax.scatter(p1[:, 0], p1[:, 1], s=8, c="yellow"); ax.scatter(p2[:, 0] + off, p2[:, 1], s=8, c="yellow")
    ax.set_title(title + f"   (green = RANSAC inliers {int(inl.sum())}, red = outliers {int((~inl).sum())})")
    ax.set_xlim(0, canvas.shape[1]); ax.set_ylim(h, 0)
    plt.tight_layout(); plt.savefig(path, dpi=90, pil_kwargs={"quality": 88}); plt.close()


# ---- B.3 figures

def fig_b3_matches(im1, im2, p1, p2, ratios, path, title, ratio_thresh):
    """All ratio-test matches (no RANSAC), line colour = ratio (dark = very distinctive)."""
    h = max(im1.shape[0], im2.shape[0]); off = im1.shape[1]
    canvas = np.zeros((h, off + im2.shape[1], 3), np.float32)
    canvas[:im1.shape[0], :off] = im1; canvas[:im2.shape[0], off:] = im2
    fig, ax = plt.subplots(figsize=(16, 16 * h / canvas.shape[1]))
    ax.imshow(canvas); ax.axis("off")
    cmap = plt.cm.viridis
    for k, (a, b, r) in enumerate(zip(p1, p2, ratios)):
        c = cmap(min(r / ratio_thresh, 1.0))
        ax.plot([a[0], b[0] + off], [a[1], b[1]], "-", color=c, lw=1.0, alpha=0.9)
        ax.text(a[0] + 4, a[1] - 4, str(k), color="yellow", fontsize=6)
        ax.text(b[0] + off + 4, b[1] - 4, str(k), color="yellow", fontsize=6)
    ax.scatter(p1[:, 0], p1[:, 1], s=10, c="red", edgecolors="w", linewidths=0.4)
    ax.scatter(p2[:, 0] + off, p2[:, 1], s=10, c="red", edgecolors="w", linewidths=0.4)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, ratio_thresh)); sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.01, label="ratio 1-NN / 2-NN (lower = more distinctive)")
    ax.set_title(f"{title}: {len(p1)} matches passing the ratio test (< {ratio_thresh})")
    ax.set_xlim(0, canvas.shape[1]); ax.set_ylim(h, 0)
    plt.tight_layout(); plt.savefig(path, dpi=90, pil_kwargs={"quality": 88}); plt.close()

def fig_patch_pairs(g1, g2, p1, p2, d1, d2, ratios, path, title, n=10):
    """The n most distinctive matches, side by side: 40x40 window and normalised 8x8 descriptor in each image."""
    order = np.argsort(ratios)[:n]
    fig, ax = plt.subplots(4, len(order), figsize=(1.6 * len(order), 7.2), squeeze=False)
    def window(g, p):
        x, y = np.rint(p).astype(int)
        return g[max(0, y - 20):y + 20, max(0, x - 20):x + 20]
    for c, k in enumerate(order):
        ax[0, c].imshow(window(g1, p1[k]), cmap="gray", vmin=0, vmax=1)
        ax[1, c].imshow(window(g2, p2[k]), cmap="gray", vmin=0, vmax=1)
        ax[2, c].imshow(d1[k].reshape(8, 8), cmap="gray", interpolation="nearest")
        ax[3, c].imshow(d2[k].reshape(8, 8), cmap="gray", interpolation="nearest")
        ax[0, c].set_title(f"match {k}\nratio {ratios[k]:.2f}", fontsize=8)
        for r in range(4): ax[r, c].axis("off")
    for r, lab in enumerate(["image A\n40x40", "image B\n40x40", "image A\n8x8 desc.", "image B\n8x8 desc."]):
        ax[r, 0].text(-0.15, 0.5, lab, transform=ax[r, 0].transAxes, ha="right", va="center", fontsize=8)
    fig.suptitle(title + ": matched feature pairs (most distinctive first)")
    plt.tight_layout(); plt.savefig(path, dpi=100, pil_kwargs={"quality": 88}); plt.close()

def fig_ratio_threshold(pa, da, pb, db, a, path, title):
    """Like Fig. 6b of Brown et al.: distribution of the 1-NN/2-NN ratio for correct vs incorrect matches, and how
    the number of matches / precision change with the threshold. There is no ground truth, so 'correct' =
    RANSAC inlier of the matches that pass a loose ratio test (0.9)."""
    j, r = nn_ratios(da, db)
    loose = np.nonzero(r < 0.9)[0]
    try:
        _, inl = ransac_homography(pa[loose], pb[j[loose]], n_iter=a.iters, eps=a.eps)
    except (RuntimeError, ValueError):
        return
    correct = np.zeros(len(da), bool); correct[loose[inl]] = True
    cand = r < 0.9
    ts = np.linspace(0.3, 0.9, 61)
    n_all = np.array([(r < t).sum() for t in ts]); n_ok = np.array([((r < t) & correct).sum() for t in ts])
    prec = np.where(n_all > 0, n_ok / np.maximum(n_all, 1), np.nan)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    bins = np.linspace(0, 0.9, 31)
    ax[0].hist(r[cand & correct], bins, alpha=0.7, color="green", label=f"correct (RANSAC inliers, {int((cand & correct).sum())})")
    ax[0].hist(r[cand & ~correct], bins, alpha=0.7, color="red", label=f"incorrect ({int((cand & ~correct).sum())})")
    ax[0].axvline(a.ratio, color="k", ls="--", label=f"chosen threshold {a.ratio}")
    ax[0].set_xlabel("ratio  d(1-NN) / d(2-NN)"); ax[0].set_ylabel("number of features"); ax[0].legend(fontsize=8)
    ax[0].set_title("ratio distribution (cf. Fig. 6b)")
    ax[1].plot(ts, n_ok, "g-", label="correct matches kept"); ax[1].plot(ts, n_all - n_ok, "r-", label="incorrect matches kept")
    ax[1].set_xlabel("ratio threshold"); ax[1].set_ylabel("matches")
    a2 = ax[1].twinx(); a2.plot(ts, prec, "b--", label="precision"); a2.set_ylabel("precision (correct / kept)"); a2.set_ylim(0, 1.05)
    ax[1].axvline(a.ratio, color="k", ls="--")
    h1, l1 = ax[1].get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    ax[1].legend(h1 + h2, l1 + l2, fontsize=8, loc="upper center"); ax[1].set_title("effect of the threshold")
    fig.suptitle(title); plt.tight_layout(); plt.savefig(path, dpi=100, pil_kwargs={"quality": 88}); plt.close()

def fig_manual_vs_auto(name, k, man, auto, path):
    fig, ax = plt.subplots(2, 1, figsize=(14, 2 * 14 * max(man.shape[0] / man.shape[1], auto.shape[0] / auto.shape[1]) + 1))
    ax[0].imshow(man); ax[0].set_title(f"{name}: manual correspondences (part 1)")
    ax[1].imshow(auto); ax[1].set_title(f"{name}: automatic (Harris + ANMS + descriptors + RANSAC)")
    for a in ax: a.axis("off")
    plt.tight_layout(); plt.savefig(path, dpi=80, pil_kwargs={"quality": 88}); plt.close()


# ---------------------------------------------------------------- one image set

def manual_pair_H(name, paths, imgs):
    """Part-1 (hand-clicked) homographies if all pair files exist, else None."""
    n = len(imgs)
    if not all(os.path.exists(P.pair_file(name, i)) for i in range(n - 1)):
        return None
    Hs = {}
    for i in range(n - 1):
        a, b = P.load_pair_points(name, i, paths, imgs)
        Hs[i] = P.computeH(a, b)
    return Hs

def process_set(name, paths, a):
    outdir = os.path.join(OUT_DIR, name); os.makedirs(outdir, exist_ok=True)
    imgs = [P.load_image(p) for p in paths]
    grays = [to_gray(im) for im in imgs]
    n = len(imgs)
    feats = []
    for k in range(n):
        pts, f = harris(grays[k])
        kept, _ = anms(pts, f, n_keep=a.anms)
        desc, kept = extract_descriptors(grays[k], kept)
        feats.append((kept, desc))
        fig_harris(imgs[k], pts, kept, f"{outdir}/harris_{k}.jpg", f"{name} / image {k}")
        fig_descriptors(grays[k], kept, desc, f"{outdir}/descriptors_{k}.jpg")
        print(f"[{name}] image {k}: {len(pts)} Harris corners -> {len(kept)} after ANMS")

    Hpair, stats = {}, []
    for i in range(n - 1):
        (pa, da), (pb, db) = feats[i], feats[i + 1]
        ia, ib, ratios = match_features(da, db, a.ratio)
        m1, m2 = pa[ia], pb[ib]
        fig_b3_matches(imgs[i], imgs[i + 1], m1, m2, ratios, f"{outdir}/b3_matches_{i}_{i+1}.jpg", f"{name}: image {i} -> {i+1}", a.ratio)
        fig_patch_pairs(grays[i], grays[i + 1], m1, m2, da[ia], db[ib], ratios, f"{outdir}/b3_patch_pairs_{i}_{i+1}.jpg", f"{name} {i}->{i+1}")
        fig_ratio_threshold(pa, da, pb, db, a, f"{outdir}/b3_ratio_threshold_{i}_{i+1}.jpg", f"{name}: image {i} -> {i+1}")
        try:
            H, inl = ransac_homography(m1, m2, n_iter=a.iters, eps=a.eps)
        except (RuntimeError, ValueError) as ex:
            print(f"[{name}] pair {i}->{i+1}: {ex}")
            fig_matches(imgs[i], imgs[i + 1], m1, m2, np.zeros(len(m1), bool), f"{outdir}/matches_{i}_{i+1}.jpg",
                        f"{name}: matches {i} -> {i+1} (RANSAC FAILED)")
            return dict(name=name, error=str(ex))
        Hpair[i] = H
        err = P.reprojection_errors(H, m1[inl], m2[inl])
        fig_matches(imgs[i], imgs[i + 1], m1, m2, inl, f"{outdir}/matches_{i}_{i+1}.jpg", f"{name}: matches {i} -> {i+1}")
        stats.append(dict(i=i, matches=len(ia), inliers=int(inl.sum()), mean=float(err.mean())))
        print(f"[{name}] pair {i}->{i+1}: {len(ia)} ratio-test matches, {int(inl.sum())} RANSAC inliers "
              f"({100 * inl.mean():.0f}%), mean inlier error {err.mean():.2f}px")
        np.savetxt(f"{outdir}/H_{i}_{i+1}.txt", H, fmt="%.6f",
                   header=f"homography mapping image {i} -> image {i+1} (RANSAC, p2 ~ H p1)")

    pieces, notes = P.plan_pieces(imgs, Hpair)
    man_H = manual_pair_H(name, paths, imgs) if a.points else None
    man_pieces = P.plan_pieces(imgs, man_H)[0] if man_H else None
    for k, (idxs, ref, Hs) in enumerate(pieces):
        suf = "" if len(pieces) == 1 else f"_p{k}"
        t = time.perf_counter()
        out = P.build_mosaic(imgs, idxs, Hs)
        for key, v in out.items():
            P.save_image(f"{outdir}/auto_mosaic{suf}_{key}.jpg", v)
        P.fig_mosaic(name, imgs, idxs, out, f"{outdir}/auto_mosaic{suf}_comparison.jpg")
        print(f"[{name}] auto mosaic{suf} of images {idxs}: {out['feather'].shape[1]}x{out['feather'].shape[0]} "
              f"in {time.perf_counter() - t:.1f}s")
        if man_pieces:
            for midx, mref, mHs in man_pieces:
                if midx == idxs:
                    mout = P.build_mosaic(imgs, midx, mHs)
                    fig_manual_vs_auto(name, k, mout["twoband"], out["twoband"], f"{outdir}/manual_vs_auto{suf}.jpg")
    for x in notes:
        print(f"[{name}] NOTE: {x}")
    return dict(name=name, stats=stats, pieces=len(pieces))


def main():
    global OUT_DIR
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="*", help=f"which sets to run (default all: {list(IMAGE_SETS)})")
    ap.add_argument("--anms", type=int, default=500, help="corners kept by ANMS")
    ap.add_argument("--ratio", type=float, default=0.7, help="Lowe ratio-test threshold (1st NN / 2nd NN)")
    ap.add_argument("--eps", type=float, default=3.0, help="RANSAC inlier threshold in (working-size) pixels")
    ap.add_argument("--iters", type=int, default=5000, help="RANSAC iterations")
    ap.add_argument("--points", default=None, help="folder with part-1 clicked points (enables manual-vs-auto figures)")
    ap.add_argument("--max-dim", type=int, default=P.MAX_DIM)
    ap.add_argument("--out", default=OUT_DIR)
    a = ap.parse_args()
    P.MAX_DIM, OUT_DIR = a.max_dim, a.out
    if a.points:
        P.POINTS_DIR = a.points
    names = a.sets or list(IMAGE_SETS)
    bad = [p for n in names for p in IMAGE_SETS[n] if not os.path.isfile(p)]
    if bad:
        raise SystemExit("missing image files:\n  " + "\n  ".join(bad))
    res = [process_set(n, IMAGE_SETS[n], a) for n in names]
    ok = [r for r in res if "error" not in r]
    print(f"\nDone: {len(ok)}/{len(res)} set(s) stitched automatically. Output in {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()
