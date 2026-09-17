"""
Publication-styled figure for the inverse-design screen: predicted ZT for
the top dopant/site combinations (at each one's own best ratio+temperature),
colored by whether the element has already been reported as an AgSbTe2
dopant in our corpus vs. a novel/unexplored candidate.

Usage: python3 src/make_inverse_design_figure.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
import matplotlib.pyplot as plt

from style import apply_publication_style

ML_RUN_DIR = Path(__file__).resolve().parent.parent
PRED_DIR = ML_RUN_DIR / "outputs" / "predictions"
FIG_DIR = ML_RUN_DIR / "outputs" / "figures"

apply_publication_style()

KNOWN_COLOR = "#2166ac"
NOVEL_COLOR = "#b2182b"


def main():
    top20 = pd.read_csv(PRED_DIR / "top20_single_dopant_site_combos.csv")
    top15 = top20.sort_values("predicted_ZT", ascending=False).head(15).iloc[::-1]

    labels = [f"{r.dopant_element} @ {r.doped_site} ({r.dopant_ratio*100:.0f}%, {int(r.temperature_K)} K)"
              for r in top15.itertuples()]
    colors = [KNOWN_COLOR if seen else NOVEL_COLOR for seen in top15["seen_in_training_data"]]

    fig, ax = plt.subplots(figsize=(11, 10))
    bars = ax.barh(labels, top15["predicted_ZT"], color=colors, edgecolor="black", linewidth=1.2)
    for b, v in zip(bars, top15["predicted_ZT"]):
        ax.text(v + 0.02, b.get_y() + b.get_height() / 2, f"{v:.2f}", va="center", fontsize=13, fontweight="bold")

    ax.set_xlabel("Model-predicted ZT")
    ax.set_title("Top 15 model-screened dopant/site/ratio/temperature\ncombinations for AgSbTe2 (best ZT model)",
                  fontsize=19, fontweight="bold")
    ax.set_xlim(0, top15["predicted_ZT"].max() * 1.15)

    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor=KNOWN_COLOR, edgecolor="black", label="Element already reported as an AgSbTe2 dopant"),
        Patch(facecolor=NOVEL_COLOR, edgecolor="black", label="Novel/unexplored dopant candidate"),
    ]
    ax.legend(handles=legend_elems, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=12, frameon=True)

    plt.tight_layout()
    out = FIG_DIR / "inverse_design_top_candidates.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
