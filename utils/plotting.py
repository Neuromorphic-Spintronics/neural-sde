from typing import Final

import matplotlib.axes
import matplotlib.pyplot as plt

COLOURS: Final[list[str]] = [
    "#ffbe0b",
    "#fb5607",
    "#ff006e",
    "#8338ec",
    "#3a86ff",
    "#49DCB1",
]


def set_default_plotting_style(use_tex: bool = True) -> None:
    """Set default matplotlib style parameters."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
            "font.size": 11,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "figure.titlesize": 14,
            "text.usetex": use_tex,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times",
            "mathtext.it": "Times:italic",
            "mathtext.bf": "Times:bold",
            "mathtext.default": "it",
        }
    )


def style_axis_clean(ax: matplotlib.axes.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
