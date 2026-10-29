"""
Assignment 3, Part 1: Image warping and mosaicing  (A.2, A.3, A.4)

Run:  python3 part1.py
Needs: numpy, scipy (only for gaussian_filter / distance transform), matplotlib, pillow.

NOT used anywhere (assignment forbids them): cv2.findHomography, cv2.warpPerspective,
cv2.getPerspectiveTransform, skimage ProjectiveTransform / warp, scipy.interpolate.

Conventions used throughout
---------------------------
* A point is (x, y) = (column, row).  Pixel centres sit at integer coordinates, so
  pixel (r, c) has its centre at x=c, y=r.  This makes bilinear interpolation clean.
* computeH(im1_pts, im2_pts) returns H with   p2 ~ H @ p1   (image 1  ->  image 2).
* Images are float arrays in [0, 1] with shape (rows, cols, channels).
"""
import json, os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter, distance_transform_edt

# ----------------------------------------------------------------------------- config
IMG_FILES = [f"data/img{i}.jpg" for i in range(5)]        # shot left -> right
POINTS_DIR = "points"                                      # pair_{i}_{i+1}.json
OUT_DIR = "out"
os.makedirs(OUT_DIR, exist_ok=True)


def load_image(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def save_image(path, im):
    Image.fromarray((np.clip(im, 0, 1) * 255 + 0.5).astype(np.uint8)).save(path)


# ======================================================================================
# A.2  RECOVER HOMOGRAPHIES
# ======================================================================================
def build_system(p1, p2):
    """
    Turn n point pairs into the linear system  A h = b  for the 8 unknowns of H.

    WHY: a homography is  [x', y', w']^T = H [x, y, 1]^T  and the pixel we land on is
    x'/w', y'/w'.  Write H = [[a b c],[d e f],[g h 1]] (we fix the bottom-right entry
    to 1 because H is only defined up to scale -> 8 unknowns, not 9). Then

        x' = (a x + b y + c) / (g x + h y + 1)
        y' = (d x + e y + f) / (g x + h y + 1)

    Multiply both sides by the denominator and move everything with an unknown to the
    left; the equations become LINEAR in (a..h):

        a x + b y + c           - g x x' - h y x' = x'
                  d x + e y + f - g x y' - h y y' = y'

    Each correspondence therefore gives 2 rows of A, so we need n >= 4 points.
    """
    p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
    n = len(p1)
    A = np.zeros((2 * n, 8))
    b = np.zeros(2 * n)
    x, y = p1[:, 0], p1[:, 1]
    xp, yp = p2[:, 0], p2[:, 1]
    A[0::2, 0], A[0::2, 1], A[0::2, 2] = x, y, 1
    A[0::2, 6], A[0::2, 7] = -x * xp, -y * xp
    A[1::2, 3], A[1::2, 4], A[1::2, 5] = x, y, 1
    A[1::2, 6], A[1::2, 7] = -x * yp, -y * yp
    b[0::2], b[1::2] = xp, yp
    return A, b


def _normalize(pts):
    """Hartley normalisation: shift points to mean 0 and scale so mean distance = sqrt(2).
    WHY: raw pixel coordinates are ~1000 while the '1' entries are 1, and products like
    x*x' are ~1e6. That makes A badly conditioned, so least squares amplifies click
    noise. Working with well-scaled numbers fixes that; we undo it afterwards."""
    c = pts.mean(axis=0)
    s = np.sqrt(2) / np.mean(np.linalg.norm(pts - c, axis=1))
    T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1.0]])
    return (pts - c) * s, T


def computeH(im1_pts, im2_pts):
    """
    H = computeH(im1_pts, im2_pts): n-by-2 arrays of (x, y), returns 3x3 H with p2 ~ H p1.

    With exactly 4 points the system is square and solved exactly, but any click error
    goes straight into H. With more points the system is OVERDETERMINED, so we solve it
    in the least-squares sense (minimise ||A h - b||^2), which averages out the noise.
    """
    p1, p2 = np.asarray(im1_pts, float), np.asarray(im2_pts, float)
    assert p1.shape == p2.shape and len(p1) >= 4, "need >= 4 matching points"
    q1, T1 = _normalize(p1)
    q2, T2 = _normalize(p2)
    A, b = build_system(q1, q2)
    h, *_ = np.linalg.lstsq(A, b, rcond=None)
    Hn = np.append(h, 1.0).reshape(3, 3)
    H = np.linalg.inv(T2) @ Hn @ T1          # undo the normalisation
    return H / H[2, 2]


def apply_H(H, pts):
    """Apply H to n-by-2 points, including the divide-by-w ('dehomogenise') step."""
    pts = np.asarray(pts, float)
    ph = np.c_[pts, np.ones(len(pts))] @ H.T
    return ph[:, :2] / ph[:, 2:3]


def reprojection_error(H, p1, p2):
    """Mean pixel distance between H*p1 and p2 (a quick sanity check on the clicks)."""
    return float(np.mean(np.linalg.norm(apply_H(H, p1) - np.asarray(p2), axis=1)))


# ======================================================================================
# A.3  WARP THE IMAGES  (inverse warping, two interpolation methods written by hand)
# ======================================================================================
def output_bbox(shape, H):
    """Bounding box (xmin, ymin, xmax, ymax) of the warped image: push the 4 corners
    through H. A homography maps straight lines to straight lines, so the warped image
    is the quadrilateral through the 4 warped corners and the box of those corners
    contains all of it."""
    h, w = shape[:2]
    corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], float)
    c = apply_H(H, corners)
    return (int(np.floor(c[:, 0].min())), int(np.floor(c[:, 1].min())),
            int(np.ceil(c[:, 0].max())), int(np.ceil(c[:, 1].max())))


def _inverse_map(shape, H, bbox):
    """
    For every pixel of the OUTPUT canvas, find where it came from in the SOURCE image.

    WHY INVERSE: forward warping (push each source pixel to H*p) leaves holes and
    overlaps in the output because pixels land on non-integer positions. Instead we loop
    over output pixels, apply H^-1 to find the source location, and READ the colour
    there (interpolating, since it is generally between pixels). No holes.
    """
    xmin, ymin, xmax, ymax = bbox
    xs, ys = np.meshgrid(np.arange(xmin, xmax + 1), np.arange(ymin, ymax + 1))
    P = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)])      # 3 x N
    Hinv = np.linalg.inv(H)
    S = Hinv @ P
    d = S[2]
    # Guard against the "behind the camera" mirror: for big rotations the denominator
    # can change sign, and those pixels would wrongly map back into the image.
    h, w = shape[:2]
    cw = (H @ np.array([w / 2, h / 2, 1.0]))[2]                  # w' at the image centre
    ok_sign = (d * cw) > 1e-9 * np.abs(cw)
    d_safe = np.where(ok_sign, d, 1.0)
    sx, sy = S[0] / d_safe, S[1] / d_safe
    out_shape = (ymax - ymin + 1, xmax - xmin + 1)
    return sx, sy, ok_sign, out_shape


def warpImageNearestNeighbor(im, H, bbox=None):
    """
    Warp `im` by H using NEAREST-NEIGHBOUR sampling (round to the closest source pixel).
    Returns (warped, mask, (xmin, ymin)); mask=1 where the output pixel has real data
    (an alpha channel, so empty pixels aren't confused with black), and (xmin, ymin) is
    where the output's top-left pixel sits in the destination coordinate frame.
    Fast but blocky: it can't represent anything between pixels -> jagged edges.
    """
    bbox = bbox or output_bbox(im.shape, H)
    sx, sy, ok, (oh, ow) = _inverse_map(im.shape, H, bbox)
    h, w = im.shape[:2]
    xi, yi = np.rint(sx).astype(np.int64), np.rint(sy).astype(np.int64)
    valid = ok & (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    out = np.zeros((oh * ow,) + im.shape[2:], im.dtype)
    out[valid] = im[yi[valid], xi[valid]]
    return out.reshape((oh, ow) + im.shape[2:]), valid.reshape(oh, ow).astype(np.float32), bbox[:2]


def warpImageBilinear(im, H, bbox=None):
    """
    Same interface, but BILINEAR sampling: the colour at a fractional location (x, y) is
    a weighted average of the 4 surrounding pixels, weighted by how close we are to each:

        I(x,y) = (1-dx)(1-dy) I00 + dx(1-dy) I10 + (1-dx)dy I01 + dx dy I11

    where (x0,y0) = floor(x,y) and (dx,dy) = fractional parts. Smoother, slightly
    blurrier, ~3-4x more arithmetic than nearest neighbour.
    """
    bbox = bbox or output_bbox(im.shape, H)
    sx, sy, ok, (oh, ow) = _inverse_map(im.shape, H, bbox)
    h, w = im.shape[:2]
    valid = ok & (sx >= 0) & (sx <= w - 1) & (sy >= 0) & (sy <= h - 1)
    x = np.clip(sx[valid], 0, w - 1)
    y = np.clip(sy[valid], 0, h - 1)
    x0 = np.minimum(np.floor(x).astype(np.int64), w - 2)
    y0 = np.minimum(np.floor(y).astype(np.int64), h - 2)
    dx, dy = x - x0, y - y0
    if im.ndim == 3:
        dx, dy = dx[:, None], dy[:, None]
    vals = ((1 - dx) * (1 - dy) * im[y0, x0] + dx * (1 - dy) * im[y0, x0 + 1] +
            (1 - dx) * dy * im[y0 + 1, x0] + dx * dy * im[y0 + 1, x0 + 1])
    out = np.zeros((oh * ow,) + im.shape[2:], im.dtype)
    out[valid] = vals
    return out.reshape((oh, ow) + im.shape[2:]), valid.reshape(oh, ow).astype(np.float32), bbox[:2]


# --------------------------------------------------------------------- rectification
def rectify(name, im, quad, aspect, rect_h=500, margin=0.35, log=None):
    """
    Rectification = a sanity test for computeH + warping using a single image.
    We know the 4 clicked points are the corners of a RECTANGLE in the real world
    (window frames, a wall panel...), so we ask for the homography that sends them to a
    true rectangle. Everything on that same plane then looks as if the camera were
    pointing straight at it.  quad order: TL, TR, BR, BL.  aspect = width / height of the
    real rectangle (a guess when we don't know it).
    """
    rect_w = rect_h * aspect
    dst = np.array([[0, 0], [rect_w, 0], [rect_w, rect_h], [0, rect_h]], float)
    H = computeH(quad, dst)
    mx, my = margin * rect_w, margin * rect_h            # show some surroundings too
    bbox = (int(-mx), int(-my), int(rect_w + mx), int(rect_h + my))

    t = time.perf_counter(); nn, _, _ = warpImageNearestNeighbor(im, H, bbox); t_nn = time.perf_counter() - t
    t = time.perf_counter(); bl, _, _ = warpImageBilinear(im, H, bbox); t_bl = time.perf_counter() - t

    A, b = build_system(quad, dst)
    if log is not None:
        log.write(f"=== {name} ===\nsource corners (TL,TR,BR,BL):\n{np.round(quad,1)}\n"
                  f"target rectangle ({rect_w:.0f} x {rect_h:.0f}):\n{dst}\n"
                  f"A (8x8) =\n{np.array2string(A, precision=1, max_line_width=200, suppress_small=True)}\n"
                  f"b = {b}\nH =\n{np.array2string(H, precision=6, suppress_small=True)}\n"
                  f"reprojection error (px in rectified frame) = {reprojection_error(H, quad, dst):.2e}\n"
                  f"warp time: nearest {t_nn*1e3:.0f} ms, bilinear {t_bl*1e3:.0f} ms\n\n")

    save_image(f"{OUT_DIR}/rect_{name}_nearest.png", nn)
    save_image(f"{OUT_DIR}/rect_{name}_bilinear.png", bl)

    fig, ax = plt.subplots(1, 3, figsize=(16, 5.2))
    ax[0].imshow(im); ax[0].set_title("original + clicked corners")
    q = np.vstack([quad, quad[:1]]); ax[0].plot(q[:, 0], q[:, 1], "r-o", ms=5)
    for k, lab in enumerate(["TL", "TR", "BR", "BL"]):
        ax[0].text(quad[k, 0] + 5, quad[k, 1] - 5, lab, color="yellow", fontsize=9)
    ax[1].imshow(nn); ax[1].set_title(f"rectified, nearest neighbour ({t_nn*1e3:.0f} ms)")
    ax[2].imshow(bl); ax[2].set_title(f"rectified, bilinear ({t_bl*1e3:.0f} ms)")
    for a in ax: a.axis("off")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/rect_{name}_compare.png", dpi=110); plt.close()

    # zoomed crop so the NN-vs-bilinear difference is visible
    cy, cx = nn.shape[0] // 2, nn.shape[1] // 2
    sl = (slice(cy - 40, cy + 40), slice(cx - 60, cx + 60))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].imshow(nn[sl], interpolation="nearest"); ax[0].set_title("nearest (zoom)")
    ax[1].imshow(bl[sl], interpolation="nearest"); ax[1].set_title("bilinear (zoom)")
    for a in ax: a.axis("off")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/rect_{name}_zoom.png", dpi=110); plt.close()
    return H


# ======================================================================================
# A.4  BLEND THE IMAGES INTO A MOSAIC
# ======================================================================================
def load_pair_H(n_imgs):
    """H_{i -> i+1} from the clicked correspondences of each neighbouring pair."""
    Hs, errs = {}, {}
    for i in range(n_imgs - 1):
        d = json.load(open(f"{POINTS_DIR}/pair_{i}_{i+1}.json"))
        p1, p2 = np.array(d["im1"]), np.array(d["im2"])
        Hs[i] = computeH(p1, p2)
        errs[i] = reprojection_error(Hs[i], p1, p2)
    return Hs, errs


def homographies_to_reference(Hpair, idxs, ref):
    """
    Chain neighbouring homographies so every image maps into ONE reference image's frame.
    WHY: homographies compose by matrix multiplication: if H01 maps 0->1 and H12 maps
    1->2, then H12 @ H01 maps 0->2. Going the other way we use the matrix inverse.
    The reference is left (almost) unwarped, which keeps the middle of the mosaic
    sharp and limits stretching at the ends.
    """
    out = {ref: np.eye(3)}
    for i in range(ref - 1, idxs[0] - 1, -1):           # images left of the reference
        out[i] = out[i + 1] @ Hpair[i]
    for i in range(ref + 1, idxs[-1] + 1):              # images right of the reference
        out[i] = out[i - 1] @ np.linalg.inv(Hpair[i - 1])
    return {k: v / v[2, 2] for k, v in out.items()}


def feather_weight(shape):
    """Per-pixel weight = distance to the nearest image border (normalised to 0..1).
    It is 0 at the edge and largest at the centre, so when two images overlap, each one
    dominates where it is 'most trustworthy' (away from its own borders/vignetting)."""
    h, w = shape[:2]
    padded = np.pad(np.ones((h, w)), 1)                  # zeros outside so the border -> distance 1
    d = distance_transform_edt(padded)[1:-1, 1:-1]
    return (d / d.max()).astype(np.float32)


def build_mosaic(imgs, idxs, ref, Hpair, max_dim=3000, method="feather", sigma=8):
    """
    1. map every image into the reference frame (chained H),
    2. size the canvas by warping all image corners,
    3. warp each image (bilinear) into that shared canvas, together with its feather weight
       (the weight is warped as a 4th channel so it follows the image exactly),
    4. blend with a weighted average (feather) or a 2-band Laplacian-style blend.
    """
    Hs = homographies_to_reference(Hpair, idxs, ref)
    # canvas size
    boxes = np.array([output_bbox(imgs[i].shape, Hs[i]) for i in idxs])
    xmin, ymin, xmax, ymax = boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].max()
    s = min(1.0, max_dim / max(xmax - xmin, ymax - ymin))   # shrink if the mosaic is huge
    S = np.diag([s, s, 1.0])
    Hs = {i: S @ Hs[i] for i in idxs}
    boxes = np.array([output_bbox(imgs[i].shape, Hs[i]) for i in idxs])
    bbox = (boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].max())

    layers, weights, masks = [], [], []
    for i in idxs:
        stacked = np.dstack([imgs[i], feather_weight(imgs[i].shape)])      # RGB + weight
        warped, mask, _ = warpImageBilinear(stacked, Hs[i], bbox)
        layers.append(warped[..., :3]); weights.append(warped[..., 3] * mask); masks.append(mask)

    W = np.stack(weights)                                   # (k, h, w)
    L = np.stack(layers)                                    # (k, h, w, 3)
    M = np.stack(masks)
    eps = 1e-8
    if method == "feather":
        out = (W[..., None] * L).sum(0) / (W.sum(0)[..., None] + eps)
    elif method == "twoband":
        # Low frequencies (colour / exposure) are blended smoothly with the feather weights;
        # high frequencies (edges, texture) are taken from ONE image per pixel (the one with
        # the biggest weight) so they cannot ghost by being averaged out of alignment.
        low = np.stack([gaussian_filter(L[k] * M[k][..., None], (sigma, sigma, 0)) /
                        (gaussian_filter(M[k], sigma)[..., None] + eps) for k in range(len(L))])
        Wb = np.stack([gaussian_filter(w, sigma) for w in W])
        low_blend = (Wb[..., None] * low).sum(0) / (Wb.sum(0)[..., None] + eps)
        high = L - low
        best = np.argmax(W, axis=0)
        high_sel = np.take_along_axis(high, best[None, ..., None], axis=0)[0]
        out = low_blend + high_sel
    else:
        raise ValueError(method)
    out = np.where(M.max(0)[..., None] > 0, out, 0)
    return np.clip(out, 0, 1), Hs


def naive_overwrite(imgs, idxs, Hs, bbox_from_blend=None):
    """For comparison only: paste images over each other (shows the edge artifacts)."""
    boxes = np.array([output_bbox(imgs[i].shape, Hs[i]) for i in idxs])
    bbox = (boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].max())
    canvas = None
    for i in idxs:
        w, m, _ = warpImageBilinear(imgs[i], Hs[i], bbox)
        canvas = np.zeros_like(w) if canvas is None else canvas
        canvas = np.where(m[..., None] > 0, w, canvas)
    return canvas


def show_correspondences(imgs, i, p1, p2):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    for a, im, p, t in ((ax[0], imgs[i], p1, f"image {i}"), (ax[1], imgs[i + 1], p2, f"image {i+1}")):
        a.imshow(im); a.set_title(t); a.axis("off")
        a.scatter(p[:, 0], p[:, 1], c="r", s=22)
        for k, (x, y) in enumerate(p):
            a.text(x + 6, y - 6, str(k), color="yellow", fontsize=8)
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/corr_{i}_{i+1}.png", dpi=100); plt.close()


# ======================================================================================
if __name__ == "__main__":
    imgs = [load_image(p) for p in IMG_FILES]
    log = open(f"{OUT_DIR}/homographies.txt", "w")

    # ---- A.2: correspondences + homographies for each neighbouring pair
    Hpair, errs = load_pair_H(len(imgs))
    for i in range(len(imgs) - 1):
        d = json.load(open(f"{POINTS_DIR}/pair_{i}_{i+1}.json"))
        p1, p2 = np.array(d["im1"]), np.array(d["im2"])
        show_correspondences(imgs, i, p1, p2)
        A, b = build_system(p1, p2)
        log.write(f"=== pair {i}->{i+1}: {len(p1)} points, A is {A.shape} ===\n"
                  f"first 4 rows of A | b:\n{np.array2string(np.c_[A[:4], b[:4]], precision=1, max_line_width=220, suppress_small=True)}\n"
                  f"H =\n{np.array2string(Hpair[i], precision=6, suppress_small=True)}\n"
                  f"mean reprojection error = {errs[i]:.2f} px\n\n")
        print(f"pair {i}->{i+1}: mean reprojection error {errs[i]:.2f} px")

    # ---- A.3: rectification examples  (corners TL, TR, BR, BL clicked on the real rectangles)
    # Two window bays on the building facade; windows in a row share top/bottom lines.
    # aspect = width/height of that rectangle in real life: a GUESS (~1.5); change it to taste.
    rectify("facade_img0", imgs[0], np.array([[623, 430], [710, 436], [711, 524], [624, 529]], float),
            aspect=1.5, log=log)
    rectify("facade_img1", imgs[1], np.array([[228, 434], [320, 444], [320, 533], [223, 538]], float),
            aspect=1.5, log=log)
    log.close()

    # ---- A.4: three mosaics, feather vs two-band blend
    jobs = {"mosaic1_imgs012": ([0, 1, 2], 1), "mosaic2_imgs123": ([1, 2, 3], 2),
            "mosaic3_imgs234": ([2, 3, 4], 3)}
    # NB: all 5 images in ONE projective mosaic blows up (total pan too wide; a plane can't
    # hold a >~120 degree view). That is what cylindrical/spherical projection is for.
    for name, (idxs, ref) in jobs.items():
        t = time.perf_counter()
        feath, Hs = build_mosaic(imgs, idxs, ref, Hpair, method="feather")
        two, _ = build_mosaic(imgs, idxs, ref, Hpair, method="twoband")
        naive = naive_overwrite(imgs, idxs, Hs)
        save_image(f"{OUT_DIR}/{name}_feather.png", feath)
        save_image(f"{OUT_DIR}/{name}_twoband.png", two)
        save_image(f"{OUT_DIR}/{name}_naive.png", naive)
        fig, ax = plt.subplots(1, len(idxs), figsize=(4 * len(idxs), 3.2))
        for a, i in zip(np.atleast_1d(ax), idxs):
            a.imshow(imgs[i]); a.set_title(f"source {i}"); a.axis("off")
        plt.tight_layout(); plt.savefig(f"{OUT_DIR}/{name}_sources.png", dpi=90); plt.close()
        print(f"{name}: {feath.shape[1]}x{feath.shape[0]}  ({time.perf_counter()-t:.1f}s)")
