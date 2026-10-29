"""
Click your OWN correspondences (the assignment wants hand-picked points).
Run on your own computer (needs a window):   python3 click_points.py pairs     or     python3 click_points.py quad 0
  pairs : for each neighbouring pair, click a feature in the LEFT image, then the SAME
          feature in the RIGHT image, alternating. Aim for 10-20 well-spread points on
          distant/static things (corners, window edges). Press Enter when done.
          Saves points/pair_i_i+1.json
  quad N: click the 4 corners (TL, TR, BR, BL) of a real rectangle in image N and print them.
"""
import json, os, sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from part1 import IMG_FILES, POINTS_DIR

mode = sys.argv[1]
if mode == "pairs":
    os.makedirs(POINTS_DIR, exist_ok=True)
    for i in range(len(IMG_FILES) - 1):
        fig, ax = plt.subplots(1, 2, figsize=(16, 6))
        ax[0].imshow(Image.open(IMG_FILES[i])); ax[1].imshow(Image.open(IMG_FILES[i + 1]))
        fig.suptitle(f"pair {i}->{i+1}: click left, right, left, right ... Enter to finish")
        pts = np.array(plt.ginput(-1, timeout=0))
        pts = pts[: len(pts) // 2 * 2].reshape(-1, 2, 2)    # (n, [left,right], xy)
        json.dump({"im1": pts[:, 0].round(1).tolist(), "im2": pts[:, 1].round(1).tolist()},
                  open(f"{POINTS_DIR}/pair_{i}_{i+1}.json", "w"), indent=1)
        plt.close(fig)
elif mode == "quad":
    plt.imshow(Image.open(IMG_FILES[int(sys.argv[2])])); plt.title("click TL, TR, BR, BL")
    print(np.array(plt.ginput(4, timeout=0)).round(1).tolist())
