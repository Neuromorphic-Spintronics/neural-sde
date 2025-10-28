"""
Standalone plotting script for Duffing horizon comparison results.

Reads CSV files saved by duffing_horizon_comparison.py and generates plots.

Generates a combined 3-panel PDF (MSE, CPU, Memory vs Horizon), separate PDFs for each metric,
and converts existing PNG rollout figures (e.g., horizon_96_rollout.png) to PDF format.

Usage:
    uv run python -m examples.studies.plot_duffing_results --output-dir <path>
"""

import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def apply_study_plot_style() -> None:
    """Apply consistent plot styling."""
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "TeX Gyre Termes"],
        }
    )


def plot_mse_cpu_mem_vs_horizon(
    mse_csv_path: Path,
    metrics_csv_path: Path,
    output_path: Path,
    show_errorbars: bool = True,
) -> None:
    """Plot MSE (top), CPU time (middle), and Memory (bottom) vs horizon on a single 3x1 figure.

    Args:
        mse_csv_path: Path to mse_vs_horizon.csv.
        metrics_csv_path: Path to metrics_vs_horizon.csv.
        output_path: Path to save the combined figure.
        show_errorbars: If True, show MSE errorbars; otherwise scatter only.
    """
    apply_study_plot_style()

    # Read CSVs
    mse_data = np.loadtxt(mse_csv_path, delimiter=",", skiprows=1)
    metrics_data = np.loadtxt(metrics_csv_path, delimiter=",", skiprows=1)

    horizons_mse = mse_data[:, 0].astype(int)
    mse_means = mse_data[:, 2]
    mse_stds = mse_data[:, 3]

    horizons_metrics = metrics_data[:, 0].astype(int)
    cpu_seconds = metrics_data[:, 1]
    memory_mib = metrics_data[:, 3]

    # Build union of horizons so x positions align if any file is missing rows
    unique_horizons = sorted(set(horizons_mse) | set(horizons_metrics))
    x_pos = np.arange(len(unique_horizons))
    horizon_to_x = {h: x for x, h in zip(x_pos, unique_horizons)}

    # Map rows to x positions
    x_vals_mse = np.array([horizon_to_x[int(h)] for h in horizons_mse])
    x_vals_metrics = np.array([horizon_to_x[int(h)] for h in horizons_metrics])

    # Sanitize arrays
    mse_means = np.asarray(mse_means, dtype=float)
    mse_stds = np.asarray(mse_stds, dtype=float)

    cpu_arr = np.asarray(cpu_seconds, dtype=float)
    cpu_arr[cpu_arr <= 0] = (
        np.min(cpu_arr[cpu_arr > 0]) if np.any(cpu_arr > 0) else 1e-9
    )

    mem_arr = np.asarray(memory_mib, dtype=float)
    mem_arr[mem_arr <= 0] = (
        np.min(mem_arr[mem_arr > 0]) if np.any(mem_arr > 0) else 1e-9
    )

    # Prepare MSE errorbars (clamp lower error to avoid negative)
    lower_err = mse_stds.copy()
    eps_frac = 1e-6
    min_allowed = np.maximum(mse_means * eps_frac, 1e-16)
    lower_err = np.minimum(lower_err, mse_means - min_allowed)
    lower_err = np.maximum(lower_err, 0.0)
    upper_err = mse_stds.copy()

    # Create 3-row shared-x figure
    fig, axes = plt.subplots(3, 1, figsize=(6, 9), sharex=True)
    ax_mse, ax_cpu, ax_mem = axes

    # MSE (top)
    if show_errorbars:
        ax_mse.errorbar(
            x_vals_mse,
            mse_means,
            yerr=[lower_err, upper_err],
            fmt="o",
            color="black",
            markersize=6,
            capsize=5,
            linewidth=0,
            elinewidth=1.5,
        )
    else:
        ax_mse.scatter(x_vals_mse, mse_means, marker="o", color="black", s=36)
    ax_mse.set_yscale("log")
    ax_mse.set_ylabel("MSE")
    ax_mse.grid(False)

    # CPU (middle)
    ax_cpu.scatter(x_vals_metrics, cpu_arr, marker="o", color="black", s=36)
    ax_cpu.set_yscale("log")
    ax_cpu.set_ylabel("CPU time [s]")
    ax_cpu.grid(False)

    # Memory (bottom)
    ax_mem.scatter(x_vals_metrics, mem_arr, marker="o", color="black", s=36)
    ax_mem.set_yscale("log")
    ax_mem.set_ylabel("Memory [MiB]")
    ax_mem.set_xlabel("Horizon")
    ax_mem.grid(False)

    # X ticks/labels only on bottom subplot
    ax_mem.set_xticks(x_pos)
    x_labels = [str(h) for h in unique_horizons]
    ax_mem.set_xticklabels(x_labels)
    ax_mem.set_xlim(-0.5, len(unique_horizons) - 0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_mse_vs_horizon_pdf(
    mse_csv_path: Path,
    output_path: Path,
    show_errorbars: bool = True,
) -> None:
    """Plot MSE vs horizon as a separate PDF figure.

    Args:
        mse_csv_path: Path to mse_vs_horizon.csv.
        output_path: Path to save the MSE figure.
        show_errorbars: If True, show MSE errorbars; otherwise scatter only.
    """
    apply_study_plot_style()

    # Read MSE CSV
    mse_data = np.loadtxt(mse_csv_path, delimiter=",", skiprows=1)
    horizons = mse_data[:, 0].astype(int)
    mse_means = mse_data[:, 2]
    mse_stds = mse_data[:, 3]

    unique_horizons = sorted(set(horizons))
    x_pos = np.arange(len(unique_horizons))
    horizon_to_x = {h: x for x, h in zip(x_pos, unique_horizons)}
    x_vals = np.array([horizon_to_x[int(h)] for h in horizons])

    mse_means = np.asarray(mse_means, dtype=float)
    mse_stds = np.asarray(mse_stds, dtype=float)

    # Prepare MSE errorbars
    lower_err = mse_stds.copy()
    eps_frac = 1e-6
    min_allowed = np.maximum(mse_means * eps_frac, 1e-16)
    lower_err = np.minimum(lower_err, mse_means - min_allowed)
    lower_err = np.maximum(lower_err, 0.0)
    upper_err = mse_stds.copy()

    fig, ax = plt.subplots(figsize=(6, 4))
    if show_errorbars:
        ax.errorbar(
            x_vals,
            mse_means,
            yerr=[lower_err, upper_err],
            fmt="o",
            color="black",
            markersize=6,
            capsize=5,
            linewidth=0,
            elinewidth=1.5,
        )
    else:
        ax.scatter(x_vals, mse_means, marker="o", color="black", s=36)
    ax.set_yscale("log")
    ax.set_xlabel("Horizon")
    ax.set_ylabel("MSE")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(h) for h in unique_horizons])
    ax.set_xlim(-0.5, len(unique_horizons) - 0.5)
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_cpu_vs_horizon_pdf(
    metrics_csv_path: Path,
    output_path: Path,
) -> None:
    """Plot CPU time vs horizon as a separate PDF figure.

    Args:
        metrics_csv_path: Path to metrics_vs_horizon.csv.
        output_path: Path to save the CPU figure.
    """
    apply_study_plot_style()

    # Read metrics CSV
    metrics_data = np.loadtxt(metrics_csv_path, delimiter=",", skiprows=1)
    horizons = metrics_data[:, 0].astype(int)
    cpu_seconds = metrics_data[:, 1]

    unique_horizons = sorted(set(horizons))
    x_pos = np.arange(len(unique_horizons))
    horizon_to_x = {h: x for x, h in zip(x_pos, unique_horizons)}
    x_vals = np.array([horizon_to_x[int(h)] for h in horizons])

    cpu_arr = np.asarray(cpu_seconds, dtype=float)
    cpu_arr[cpu_arr <= 0] = (
        np.min(cpu_arr[cpu_arr > 0]) if np.any(cpu_arr > 0) else 1e-9
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x_vals, cpu_arr, marker="o", color="black", s=36)
    ax.set_yscale("log")
    ax.set_xlabel("Horizon")
    ax.set_ylabel("CPU time [s]")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(h) for h in unique_horizons])
    ax.set_xlim(-0.5, len(unique_horizons) - 0.5)
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_memory_vs_horizon_pdf(
    metrics_csv_path: Path,
    output_path: Path,
) -> None:
    """Plot Memory vs horizon as a separate PDF figure.

    Args:
        metrics_csv_path: Path to metrics_vs_horizon.csv.
        output_path: Path to save the Memory figure.
    """
    apply_study_plot_style()

    # Read metrics CSV
    metrics_data = np.loadtxt(metrics_csv_path, delimiter=",", skiprows=1)
    horizons = metrics_data[:, 0].astype(int)
    memory_mib = metrics_data[:, 3]

    unique_horizons = sorted(set(horizons))
    x_pos = np.arange(len(unique_horizons))
    horizon_to_x = {h: x for x, h in zip(x_pos, unique_horizons)}
    x_vals = np.array([horizon_to_x[int(h)] for h in horizons])

    mem_arr = np.asarray(memory_mib, dtype=float)
    mem_arr[mem_arr <= 0] = (
        np.min(mem_arr[mem_arr > 0]) if np.any(mem_arr > 0) else 1e-9
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x_vals, mem_arr, marker="o", color="black", s=36)
    ax.set_yscale("log")
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Memory [MiB]")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(h) for h in unique_horizons])
    ax.set_xlim(-0.5, len(unique_horizons) - 0.5)
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def convert_png_rollouts_to_pdf(output_dir: Path) -> None:
    """Convert existing PNG rollout figures to PDF format.

    Looks for files like teacher_forcing_rollout.png and horizon_*_rollout.png
    in the output directory and saves them as PDFs.

    Args:
        output_dir: Directory containing the PNG files.
    """
    import matplotlib.image as mpimg

    # Patterns for rollout PNGs
    patterns = ["teacher_forcing_rollout.png", "horizon_*_rollout.png"]

    for pattern in patterns:
        if "*" in pattern:
            # Glob for horizon files
            png_files = glob.glob(str(output_dir / pattern))
        else:
            png_path = output_dir / pattern
            png_files = [str(png_path)] if png_path.exists() else []

        for png_file in png_files:
            png_path = Path(png_file)
            if png_path.exists():
                pdf_path = png_path.with_suffix(".pdf")

                # Load PNG as image
                img = mpimg.imread(png_path)

                # Create figure and display image
                fig, ax = plt.subplots(
                    figsize=(img.shape[1] / 100, img.shape[0] / 100), dpi=100
                )
                ax.imshow(img)
                ax.axis("off")

                # Save as PDF
                fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
                print(f"Converted: {png_path} -> {pdf_path}")


def main() -> None:
    """Main entry point. Generates combined and separate PDF plots from CSV files, and converts PNG rollouts to PDFs."""
    parser = argparse.ArgumentParser(
        description="Plot Duffing horizon comparison results from CSV files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory containing the CSV files (e.g., h1-256_rk2_s96_e100_...)",
    )
    parser.add_argument(
        "--no-errorbars",
        action="store_true",
        help="If set, plot mean-only curves without error bars.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    # Check for required CSV files
    mse_csv = output_dir / "mse_vs_horizon.csv"
    metrics_csv = output_dir / "metrics_vs_horizon.csv"

    if not mse_csv.exists():
        raise FileNotFoundError(f"MSE CSV not found: {mse_csv}")
    if not metrics_csv.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {metrics_csv}")

    # Generate combined plot (MSE top, CPU middle, Memory bottom)
    print(f"Reading results from: {output_dir}")
    print("\nGenerating combined MSE/CPU/Memory plot...")
    plot_mse_cpu_mem_vs_horizon(
        mse_csv_path=mse_csv,
        metrics_csv_path=metrics_csv,
        output_path=output_dir / "mse_cpu_memory_vs_horizon_duffing.pdf",
        show_errorbars=(not args.no_errorbars),
    )

    # Generate separate PDF plots for each metric
    print("\nGenerating separate MSE vs Horizon plot...")
    plot_mse_vs_horizon_pdf(
        mse_csv_path=mse_csv,
        output_path=output_dir / "mse_vs_horizon.pdf",
        show_errorbars=(not args.no_errorbars),
    )

    print("\nGenerating separate CPU vs Horizon plot...")
    plot_cpu_vs_horizon_pdf(
        metrics_csv_path=metrics_csv,
        output_path=output_dir / "cpu_vs_horizon.pdf",
    )

    print("\nGenerating separate Memory vs Horizon plot...")
    plot_memory_vs_horizon_pdf(
        metrics_csv_path=metrics_csv,
        output_path=output_dir / "memory_vs_horizon.pdf",
    )

    # Convert existing PNG rollout figures to PDFs
    print("\nConverting PNG rollout figures to PDFs...")
    convert_png_rollouts_to_pdf(output_dir)

    print("All plots generated successfully")


if __name__ == "__main__":
    main()
