"""
AI Note: This file was written by AI (I hate matplotlib).

Layer 0 figures: CUDA vs PyTorch from layer0_results.csv.

Writes:
  figures/layer0_latency.png
  figures/layer0_slowdown.png
  figures/layer0_throughput.png
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "results" / "layer0_results.csv"
OUT_DIR = HERE / "figures"

LABELS = {
    "matmul": "Matmul",
    "matmul_coalesced": "Matmul\n(coalesced)",
    "matmul_smem": "Matmul\n(smem)",
    "batch_matmul": "Batched\nmatmul",
    "batch_matmul_coalesced": "Batched\nmatmul\n(coalesced)",
    "batch_matmul_smem": "Batched\nmatmul\n(smem)",
    "addition": "Add",
    "multi": "Mul",
    "softmax": "Softmax",
    "rmsnorm": "RMSNorm",
}

BG = "#1c1c22"
TEXT = "#F2F2F5"
GRID = "#4a4a55"
CUDA_COLOR = "#B8E0D2"
CUDA_EDGE = "#2F6F64"
TORCH_COLOR = "#F5C6AA"
TORCH_EDGE = "#8B4A32"
BEST_COLOR = "#C4B5FD"
BEST_EDGE = "#6D28D9"
ERROR_COLOR = "#E8E8ED"
BAR_EDGEWIDTH = 1.5
GROUPED_BAR_WIDTH = 0.22
SLOWDOWN_BAR_WIDTH = 0.4


def load_rows():
    with CSV_PATH.open(newline="") as f:
        return list(csv.DictReader(f))


def _kernel_family(name):
    if name.startswith("batch_matmul"):
        return "batch_matmul"
    if name.startswith("matmul"):
        return "matmul"
    return name


def _best_indices(rows):
    """Lowest slowdown in each kernel family (current best of that type)."""
    best = {}
    for i, row in enumerate(rows):
        family = _kernel_family(row["kernel"])
        slowdown = float(row["slowdown"])
        if family not in best or slowdown < best[family][1]:
            best[family] = (i, slowdown)
    return {i for i, _ in best.values()}


def _style():
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "axes.edgecolor": TEXT,
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "text.color": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "legend.labelcolor": TEXT,
            "grid.color": GRID,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": BG,
            "savefig.edgecolor": BG,
        }
    )


def plot_latency(rows):
    """Grouped bars: CUDA vs Torch latency. Log y so GEMM and tiny ops share a plot."""
    names = [LABELS[r["kernel"]] for r in rows]
    cuda_ms = np.array([float(r["cuda_ms"]) for r in rows])
    torch_ms = np.array([float(r["torch_ms"]) for r in rows])
    cuda_std = np.array([float(r["cuda_std_ms"]) for r in rows])
    torch_std = np.array([float(r["torch_std_ms"]) for r in rows])

    x = np.arange(len(names))
    width = GROUPED_BAR_WIDTH

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.bar(
        x - width / 2,
        cuda_ms,
        width,
        yerr=cuda_std,
        capsize=3,
        ecolor=ERROR_COLOR,
        label="CUDA Kernel",
        color=CUDA_COLOR,
        edgecolor=CUDA_EDGE,
        linewidth=BAR_EDGEWIDTH,
        zorder=3,
    )
    ax.bar(
        x + width / 2,
        torch_ms,
        width,
        yerr=torch_std,
        capsize=3,
        ecolor=ERROR_COLOR,
        label="Torch Kernel",
        color=TORCH_COLOR,
        edgecolor=TORCH_EDGE,
        linewidth=BAR_EDGEWIDTH,
        zorder=3,
    )
    ax.set_yscale("log")
    ax.set_ylabel("Latency (ms, log)")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_title("CUDA Kernel vs Torch Kernel Latency")
    ax.legend(frameon=False)
    ax.yaxis.grid(True, which="both", linestyle="--", alpha=0.35, zorder=0)

    def _label_bars(xs, heights, stds):
        for xpos, height, std in zip(xs, heights, stds):
            ax.text(
                xpos,
                (height + std) * 1.12,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=TEXT,
            )

    _label_bars(x - width / 2, cuda_ms, cuda_std)
    _label_bars(x + width / 2, torch_ms, torch_std)
    ax.set_ylim(top=max((cuda_ms + cuda_std).max(), (torch_ms + torch_std).max()) * 2.0)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "layer0_latency.png", dpi=160)
    plt.close(fig)


def plot_slowdown(rows):
    names = [LABELS[r["kernel"]] for r in rows]
    slowdown = np.array([float(r["slowdown"]) for r in rows])
    x = np.arange(len(names))
    best_idxs = _best_indices(rows)
    colors = [BEST_COLOR if i in best_idxs else CUDA_COLOR for i in range(len(rows))]
    edges = [BEST_EDGE if i in best_idxs else CUDA_EDGE for i in range(len(rows))]

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    bars = ax.bar(
        x,
        slowdown,
        width=SLOWDOWN_BAR_WIDTH,
        color=colors,
        edgecolor=edges,
        linewidth=BAR_EDGEWIDTH,
        zorder=3,
    )
    ax.axhline(1.0, color=TORCH_COLOR, linestyle="--", linewidth=1.4)
    ax.set_ylabel("Slowdown (CUDA ms / Torch ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_title("CUDA Kernel vs Torch Kernel Slowdown")
    ax.legend(
        handles=[
            plt.Line2D(
                [0],
                [0],
                color=TORCH_COLOR,
                linestyle="--",
                linewidth=1.4,
                label="Torch Kernel (1×)",
            ),
            Patch(
                facecolor=BEST_COLOR,
                edgecolor=BEST_EDGE,
                linewidth=BAR_EDGEWIDTH,
                label="Current best",
            ),
        ],
        frameon=False,
    )
    ax.yaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
    for bar, val in zip(bars, slowdown):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.08,
            f"{val:.1f}×",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_ylim(0, max(slowdown) * 1.18)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "layer0_slowdown.png", dpi=160)
    plt.close(fig)


def plot_throughput(rows):
    """Two panels: TFLOPS for GEMMs, GB/s for bandwidth kernels. Do not mix units."""
    gemm = [r for r in rows if r["metric"] == "TFLOPS"]
    bw = [r for r in rows if r["metric"] == "GB/s"]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    width = GROUPED_BAR_WIDTH

    def _grouped(ax, subset, ylabel, title):
        names = [LABELS[r["kernel"]] for r in subset]
        cuda_rate = np.array([float(r["cuda_rate"]) for r in subset])
        torch_rate = np.array([float(r["torch_rate"]) for r in subset])
        x = np.arange(len(names))
        ax.bar(
            x - width / 2,
            cuda_rate,
            width,
            label="CUDA Kernel",
            color=CUDA_COLOR,
            edgecolor=CUDA_EDGE,
            linewidth=BAR_EDGEWIDTH,
            zorder=3,
        )
        ax.bar(
            x + width / 2,
            torch_rate,
            width,
            label="Torch Kernel",
            color=TORCH_COLOR,
            edgecolor=TORCH_EDGE,
            linewidth=BAR_EDGEWIDTH,
            zorder=3,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.yaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
        ax.legend(frameon=False)

    _grouped(axes[0], gemm, "TFLOPS", "FLOPS")
    _grouped(axes[1], bw, "GB/s", "GB/s")
    fig.suptitle("CUDA Kernel vs Torch Kernel FLOPS and GB/s", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "layer0_throughput.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    _style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    plot_latency(rows)
    plot_slowdown(rows)
    plot_throughput(rows)
    for name in ("layer0_latency.png", "layer0_slowdown.png", "layer0_throughput.png"):
        print(f"wrote {OUT_DIR / name}")


if __name__ == "__main__":
    main()
