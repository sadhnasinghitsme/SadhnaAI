import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from config import CATEGORIES, OUT_PLOT_DIR


def plot_2d_category_map(records: list[dict]):
    if not records:
        print("[!] No data to plot.")
        return

    cat_index = {c: i for i, c in enumerate(CATEGORIES)}
    files     = list({r["file"] for r in records})
    cmap      = plt.cm.get_cmap("tab10", len(files))
    colors    = {f: cmap(i) for i, f in enumerate(files)}

    fig, ax = plt.subplots(figsize=(13, 7))

    for r in records:
        y = cat_index.get(r["category"], len(CATEGORIES) - 1)
        y += np.random.uniform(-0.3, 0.3)
        ax.scatter(r["confidence"], y, color=colors[r["file"]], alpha=0.65, s=40, edgecolors="none")

    ax.axvline(x=0.6, color="red", linestyle="--", linewidth=1, label="60% threshold")
    ax.set_yticks(range(len(CATEGORIES)))
    ax.set_yticklabels(CATEGORIES, fontsize=12)
    ax.set_xlabel("Confidence Score", fontsize=12)
    ax.set_title("BOQ 2D Category Map", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x", linestyle=":", alpha=0.5)

    patches = [mpatches.Patch(color=colors[f], label=f) for f in files]
    patches.append(plt.Line2D([0], [0], color="red", linestyle="--", label="60% threshold"))
    ax.legend(handles=patches, loc="lower right", fontsize=8)

    plt.tight_layout()
    os.makedirs(OUT_PLOT_DIR, exist_ok=True)
    out = os.path.join(OUT_PLOT_DIR, "category_map.png")
    plt.savefig(out, dpi=150)
    print(f"[OK] Plot saved -> {out}")
    plt.close()
