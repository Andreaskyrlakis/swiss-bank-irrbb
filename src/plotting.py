"""Shared figure style; figures are saved to figures/ and copied to report/figures/."""

import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .data import ROOT  # noqa: E402

COLORS = ["#1f4e79", "#c0504d", "#9bbb59", "#8064a2", "#f79646", "#4bacc6"]
FIGURES = ROOT / "figures"
REPORT_FIGURES = ROOT / "report" / "figures"


def style():
    plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})


def save(fig, name):
    FIGURES.mkdir(exist_ok=True)
    REPORT_FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    path = FIGURES / f"{name}.png"
    fig.savefig(path, dpi=160)
    shutil.copy(path, REPORT_FIGURES / path.name)
    plt.close(fig)
    print(f"figure {path.name}")
