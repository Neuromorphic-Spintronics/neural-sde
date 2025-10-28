import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.text import Text

from neural_dynamics.core.utils import COLOURS, set_default_plotting_style


def setup_matplotlib_style(use_tex: bool = True) -> None:
    """Configure matplotlib using the shared neural_dynamics style helper."""
    try:
        set_default_plotting_style(use_tex=use_tex)
    except Exception:
        if use_tex:
            print("LaTeX not found, using default plotting style.")
            set_default_plotting_style(use_tex=False)
        else:
            raise
    plt.rcParams["axes.linewidth"] = 1.0


def finalise_plot(fig, filename, output_dir, os):
    """Common plot finalization: tight layout, save, and show."""
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, filename), bbox_inches="tight", dpi=300)
    plt.show()


def _measure_pair_kerned_advances(fig, sample_text, fontprops):
    """Return per‑character advances (in pixels) that include pair kerning.

    We measure the width of the prefix s[:i+1] and subtract the width of s[:i].
    This captures the font's kerning pairs (including TeX rendering when enabled).
    """
    # Ensure a renderer exists
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    cum_widths = []
    for i in range(1, len(sample_text) + 1):
        t = Text(0, 0, sample_text[:i], fontproperties=fontprops)
        t.set_figure(fig)  # required so the text resolves fonts correctly
        bbox = t.get_window_extent(renderer=renderer)
        cum_widths.append(bbox.width)

    advances = []
    prev = 0.0
    for w in cum_widths:
        adv = max(w - prev, 0.01)  # keep strictly positive to avoid collapse
        advances.append(adv)
        prev = w
    return np.array(advances)

def _apply_pair_adjustments(text, advances):
    """Fine-tune kerning for tricky pairs like 'MR' or 'R ' by shrinking advances."""
    adjustments = {
        ("M", "R"): -0.15,   # tighten MR
        ("R", " "): -0.05,   # tighten R + space
        (" ", "["): -0.05,   # tighten space + [
    }
    new_adv = advances.copy()
    for i in range(len(text) - 1):
        pair = (text[i], text[i + 1])
        if pair in adjustments:
            new_adv[i] += adjustments[pair]
            new_adv[i + 1] -= adjustments[pair]
    new_adv = np.clip(new_adv, 0.01, None)  # avoid collapsing to zero
    return new_adv



def plot_nanoring_results(
    *,
    time_axis: np.ndarray,
    true_sequence: torch.Tensor,
    predicted_sequence_ode: torch.Tensor,
    predicted_sequence_sde: torch.Tensor,
    h_field: torch.Tensor,
    all_train_sequences: np.ndarray,
    output_dir: str,
):
    """
    Generates and saves a comprehensive plot for the nanoring experiment results.
    """
    try:
        set_default_plotting_style(use_tex=True)
    except Exception:
        print("LaTeX not found, using default plotting style.")
        set_default_plotting_style(use_tex=False)

    fig, ax2 = plt.subplots(figsize=(12, 6), dpi=120)

    # --- Trajectories ---
    ax2_h = ax2.twinx()
    
    # Plot training sequences (already denormalized in the caller)
    for train_seq in all_train_sequences:
        ax2.plot(time_axis, train_seq, color='black', alpha=0.01, label='_nolegend_', zorder=2)
    
    # Plot predictions and ground truth
    true_seq_np = true_sequence.cpu().numpy() if hasattr(true_sequence, 'cpu') else true_sequence
    pred_ode_np = predicted_sequence_ode.cpu().numpy() if hasattr(predicted_sequence_ode, 'cpu') else predicted_sequence_ode
    pred_sde_np = predicted_sequence_sde.cpu().numpy() if hasattr(predicted_sequence_sde, 'cpu') else predicted_sequence_sde
    
    ax2.plot(time_axis, true_seq_np, label="Ground Truth", color=COLOURS[3], linewidth=2, zorder=3)
    ax2.plot(time_axis, pred_ode_np, label="Neural ODE", color=COLOURS[4], linestyle='-.', linewidth=2, zorder=3)
    ax2.plot(time_axis, pred_sde_np, label="Neural SDE", color=COLOURS[2], linestyle='--', linewidth=2, zorder=3)

    h_colour = COLOURS[1]
    ax2_h.plot(time_axis, h_field.cpu().numpy(), label=r'$H(t)$', color=h_colour, linestyle=':', linewidth=2, zorder=3)

    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("AM Rresponse  [V]")
    ax2.legend(loc='upper left')
    ax2.grid(False)
    ax2_h.set_ylabel(r'$H(t)$ [Oe]')

    # Set x-axis limits to start at 0 and end at the last time point
    ax2.set_xlim(0, float(time_axis[-1]))
    ax2_h.set_xlim(0, float(time_axis[-1]))

    # Ensure renderer exists for measurements below
    fig.canvas.draw()

    # --- Style right axis ---
    ax2_h.spines['right'].set_visible(False)
    ax2_h.yaxis.label.set_color(h_colour)
    ax2_h.tick_params(axis='y', colors=h_colour)
    ax2_h.plot([1, 1], [0, 1], transform=ax2_h.transAxes, linestyle=':', color=h_colour, linewidth=2, alpha=0.95, clip_on=False, zorder=4)

    # --- Gradient styling for left axis ---
    amr_cmap = LinearSegmentedColormap.from_list("amr_grad", [COLOURS[2], COLOURS[3]])
    ymin, ymax = ax2.get_ylim()
    for tick in ax2.yaxis.get_major_ticks():
        y_pos = tick.get_loc()
        norm_pos = (y_pos - ymin) / (ymax - ymin)
        color = amr_cmap(norm_pos)
        tick.label1.set_color(color)
        tick.tick1line.set_color(color)
        tick.tick2line.set_color(color)

    # --- Replace y-label with kerning-aware gradient glyphs ---
    # Keep the original temporarily to read its geometry and font props
    temp_label = ax2.set_ylabel(ax2.get_ylabel())
    fig.canvas.draw()

    bbox_disp = temp_label.get_window_extent()  # display coords
    bbox_ax = ax2.transAxes.inverted().transform(bbox_disp)  # -> axes coords
    label_text = temp_label.get_text()
    fontsize = temp_label.get_fontsize()
    fontprops = temp_label.get_fontproperties()

    # Center x in axes coords; y-span where characters will be placed
    x_center_ax = bbox_ax[0, 0] + 0.5 * (bbox_ax[1, 0] - bbox_ax[0, 0])
    y_lo_ax, y_hi_ax = bbox_ax[0, 1], bbox_ax[1, 1]
    pad = (y_hi_ax - y_lo_ax) * 0.05
    y_lo_ax += pad
    y_hi_ax -= pad

    # Pair-kerning aware advances from the actual renderer/font
    advances_px = _measure_pair_kerned_advances(fig, label_text, fontprops)
    advances_px = _apply_pair_adjustments(label_text,   advances_px)

    # Normalized cumulative positions for baseline of each glyph (0..1)
    norm_adv = advances_px / advances_px.sum()
    # Positions (bottom to top) along the label span
    char_positions_ax = y_lo_ax + np.cumsum(np.concatenate([[0.0], norm_adv[:-1]])) * (y_hi_ax - y_lo_ax)

    # Hide the original y-label
    temp_label.set_visible(False)

    # Draw each character with color sampled from the data-to-axes mapping
    ymin_data, ymax_data = ax2.get_ylim()
    for y_ax, ch in zip(char_positions_ax, label_text):
        # Map the axes y to data y for color sampling
        _, y_data = ax2.transData.inverted().transform(ax2.transAxes.transform((0, y_ax)))
        norm_pos = (y_data - ymin_data) / (ymax_data - ymin_data)
        color = amr_cmap(norm_pos)
        ax2.text(
            x_center_ax, y_ax, ch,
            transform=ax2.transAxes,
            rotation=90,
            ha='center', va='center',
            fontsize=fontsize,
            fontproperties=fontprops,
            color=color,
        )

    # --- Gradient spine as inset ---
    ax_bbox = ax2.get_window_extent()
    one_px_frac = 20.0 / max(ax_bbox.width, 3.0)
    ax2.spines[['top','left','right']].set_visible(False)
    ax2_h.spines[['top','left']].set_visible(False)

    grad_ax = inset_axes(ax2, width=one_px_frac, height="100%", loc="lower left",
                         bbox_to_anchor=(0, 0, 1, 1), bbox_transform=ax2.transAxes, borderpad=0)
    grad_ax.set_axis_off()
    gradient = np.linspace(0, 1, 256).reshape(-1, 1)
    grad_ax.imshow(gradient, aspect='auto', origin='lower', cmap=amr_cmap,
                   extent=(0, 1, 0, 1), zorder=1, interpolation='nearest')
    grad_ax.patch.set_alpha(0)

    fig.subplots_adjust(hspace=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "trajectories.pdf"), bbox_inches="tight")
    plt.show()
    plt.close(fig)
