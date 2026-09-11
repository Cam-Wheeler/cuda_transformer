"""
AI Note: This file was written by AI (I hate matplotlib).

Layer 0 figures: CUDA vs PyTorch from layer0_results.csv.

Writes each PNG to profiling/figures/ and to figures/ (README).

Matmul / batched matmul:
  layer0_matmul_{latency,slowdown,throughput}.png

Add, mul, softmax, rmsnorm:
  layer0_other_{latency,slowdown,throughput}.png
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "results" / "layer0_results.csv"
OUT_DIR = HERE / "figures"
README_FIGURES = HERE.parent / "figures"

LABELS = {
    "matmul": "Matmul",
    "matmul_coalesced": "Matmul\n(coalesced)",
    "matmul_smem": "Matmul\n(smem)",
    "matmul_blocktiling": "Matmul\n(1D blocktiling)",
    "matmul_blocktiling_2d": "Matmul\n(2D blocktiling)",
    "matmul_vectorize": "Matmul\n(vectorised)",
    "matmul_vectorize_bk16": "Matmul\n(vectorised, BK=16)",
    "batch_matmul": "Batched\nmatmul",
    "batch_matmul_coalesced": "Batched\nmatmul\n(coalesced)",
    "batch_matmul_smem": "Batched\nmatmul\n(smem)",
    "batch_matmul_blocktiling": "Batched\nmatmul\n(1D blocktiling)",
    "batch_matmul_blocktiling_2d": "Batched\nmatmul\n(2D blocktiling)",
    "batch_matmul_blocktiling_2d_qk": "Batched\nmatmul\n(2D QK tile)",
    "batch_matmul_vectorize": "Batched\nmatmul\n(vectorised)",
    "addition": "Add",
    "addition_vectorize": "Add\n(vectorised)",
    "multi": "Mul",
    "multi_vectorize": "Mul\n(vectorised)",
    "softmax": "Softmax",
    "softmax_online": "Softmax\n(online)",
    "softmax_shuffle": "Softmax\n(shuffle)",
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
FIG_HEIGHT = 5.2
FIG_WIDTH_PER_BAR = 1.35
FIG_MIN_WIDTH = 7.0

OUT_NAMES = (
    "layer0_matmul_latency.png",
    "layer0_matmul_slowdown.png",
    "layer0_matmul_throughput.png",
    "layer0_other_latency.png",
    "layer0_other_slowdown.png",
    "layer0_other_throughput.png",
)


def _single_panel_width(n_items: int) -> float:
    return max(FIG_MIN_WIDTH, n_items * FIG_WIDTH_PER_BAR)


def load_rows():
    with CSV_PATH.open(newline="") as f:
        return list(csv.DictReader(f))


def _is_gemm(name):
    return name.startswith("matmul") or name.startswith("batch_matmul")


def split_groups(rows):
    gemm = [r for r in rows if _is_gemm(r["kernel"])]
    other = [r for r in rows if not _is_gemm(r["kernel"])]
    return gemm, other


def _gemm_panels(rows):
    matmul = [r for r in rows if r["kernel"].startswith("matmul")]
    batched = [r for r in rows if r["kernel"].startswith("batch_matmul")]
    return matmul, batched


def _kernel_family(name):
    if name.startswith("batch_matmul"):
        return "batch_matmul"
    if name.startswith("matmul"):
        return "matmul"
    if name.startswith("addition"):
        return "addition"
    if name.startswith("multi"):
        return "multi"
    if name.startswith("softmax"):
        return "softmax"
    return name


def _best_indices(rows):
    """Lowest slowdown in each optimisation ladder. Ties go to the later kernel."""
    best = {}
    counts = {}
    for i, row in enumerate(rows):
        family = _kernel_family(row["kernel"])
        counts[family] = counts.get(family, 0) + 1
        slowdown = float(row["slowdown"])
        if family not in best or slowdown <= best[family][1]:
            best[family] = (i, slowdown)
    return {idx for fam, (idx, _) in best.items() if counts[fam] > 1}


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


def _plot_latency_ax(ax, rows, title, show_legend):
    names = [LABELS[r["kernel"]] for r in rows]
    cuda_ms = np.array([float(r["cuda_ms"]) for r in rows])
    torch_ms = np.array([float(r["torch_ms"]) for r in rows])
    cuda_std = np.array([float(r["cuda_std_ms"]) for r in rows])
    torch_std = np.array([float(r["torch_std_ms"]) for r in rows])
    x = np.arange(len(names))
    width = GROUPED_BAR_WIDTH
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
    ax.set_ylabel("Latency (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_title(title)
    if show_legend:
        ax.legend(frameon=False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)

    def _label_bars(xs, heights, stds):
        for xpos, height, std in zip(xs, heights, stds):
            ax.text(
                xpos,
                height + std + max(heights) * 0.03,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=TEXT,
            )

    _label_bars(x - width / 2, cuda_ms, cuda_std)
    _label_bars(x + width / 2, torch_ms, torch_std)
    ax.set_ylim(top=max((cuda_ms + cuda_std).max(), (torch_ms + torch_std).max()) * 1.22)


def _plot_slowdown_ax(ax, rows, title, show_legend):
    names = [LABELS[r["kernel"]] for r in rows]
    slowdown = np.array([float(r["slowdown"]) for r in rows])
    x = np.arange(len(names))
    best_idxs = _best_indices(rows)
    colors = [BEST_COLOR if i in best_idxs else CUDA_COLOR for i in range(len(rows))]
    edges = [BEST_EDGE if i in best_idxs else CUDA_EDGE for i in range(len(rows))]
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
    ax.set_xticklabels(names, fontsize=9)
    ax.set_title(title)
    if show_legend:
        handles = [
            plt.Line2D(
                [0],
                [0],
                color=TORCH_COLOR,
                linestyle="--",
                linewidth=1.4,
                label="Torch Kernel (1×)",
            ),
        ]
        if best_idxs:
            handles.append(
                Patch(
                    facecolor=BEST_COLOR,
                    edgecolor=BEST_EDGE,
                    linewidth=BAR_EDGEWIDTH,
                    label="Current best",
                )
            )
        ax.legend(handles=handles, frameon=False)
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


def _plot_rate_ax(ax, rows, ylabel, title, show_legend):
    names = [LABELS[r["kernel"]] for r in rows]
    cuda_rate = np.array([float(r["cuda_rate"]) for r in rows])
    torch_rate = np.array([float(r["torch_rate"]) for r in rows])
    x = np.arange(len(names))
    width = GROUPED_BAR_WIDTH
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
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
    if show_legend:
        ax.legend(frameon=False)


def _savefig(fig, name, **kwargs):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    README_FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / name, **kwargs)
    fig.savefig(README_FIGURES / name, **kwargs)


def _two_panel_figure(left_n, right_n):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(_single_panel_width(left_n) + _single_panel_width(right_n), FIG_HEIGHT),
        gridspec_kw={"width_ratios": [left_n, right_n]},
    )
    return fig, axes


def _finish_matmul_figure(fig, title, handles, out_name):
    """Legend sits above both panels so it cannot cover bars."""
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=len(handles),
        frameon=False,
    )
    fig.suptitle(title, y=1.14)
    fig.tight_layout()
    _savefig(fig, out_name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _cuda_torch_handles():
    return [
        Patch(
            facecolor=CUDA_COLOR,
            edgecolor=CUDA_EDGE,
            linewidth=BAR_EDGEWIDTH,
            label="CUDA Kernel",
        ),
        Patch(
            facecolor=TORCH_COLOR,
            edgecolor=TORCH_EDGE,
            linewidth=BAR_EDGEWIDTH,
            label="Torch Kernel",
        ),
    ]


def _slowdown_handles():
    return [
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
    ]


def plot_matmul_latency(rows):
    matmul, batched = _gemm_panels(rows)
    fig, axes = _two_panel_figure(len(matmul), len(batched))
    _plot_latency_ax(axes[0], matmul, "Matmul", show_legend=False)
    _plot_latency_ax(axes[1], batched, "Batched matmul", show_legend=False)
    _finish_matmul_figure(
        fig,
        "CUDA Kernel vs Torch Kernel Latency",
        _cuda_torch_handles(),
        "layer0_matmul_latency.png",
    )


def plot_other_latency(rows):
    fig, ax = plt.subplots(figsize=(_single_panel_width(len(rows)), FIG_HEIGHT))
    _plot_latency_ax(ax, rows, "CUDA Kernel vs Torch Kernel Latency", show_legend=True)
    fig.tight_layout()
    _savefig(fig, "layer0_other_latency.png", dpi=160)
    plt.close(fig)


def plot_matmul_slowdown(rows):
    matmul, batched = _gemm_panels(rows)
    fig, axes = _two_panel_figure(len(matmul), len(batched))
    _plot_slowdown_ax(axes[0], matmul, "Matmul", show_legend=False)
    _plot_slowdown_ax(axes[1], batched, "Batched matmul", show_legend=False)
    _finish_matmul_figure(
        fig,
        "CUDA Kernel vs Torch Kernel Slowdown",
        _slowdown_handles(),
        "layer0_matmul_slowdown.png",
    )


def plot_other_slowdown(rows):
    fig, ax = plt.subplots(figsize=(_single_panel_width(len(rows)), FIG_HEIGHT))
    _plot_slowdown_ax(ax, rows, "CUDA Kernel vs Torch Kernel Slowdown", show_legend=True)
    fig.tight_layout()
    _savefig(fig, "layer0_other_slowdown.png", dpi=160)
    plt.close(fig)


def plot_matmul_throughput(rows):
    matmul, batched = _gemm_panels(rows)
    fig, axes = _two_panel_figure(len(matmul), len(batched))
    _plot_rate_ax(axes[0], matmul, "TFLOPS", "Matmul", show_legend=False)
    _plot_rate_ax(axes[1], batched, "TFLOPS", "Batched matmul", show_legend=False)
    _finish_matmul_figure(
        fig,
        "CUDA Kernel vs Torch Kernel FLOPS",
        _cuda_torch_handles(),
        "layer0_matmul_throughput.png",
    )


def plot_other_throughput(rows):
    fig, ax = plt.subplots(figsize=(_single_panel_width(len(rows)), FIG_HEIGHT))
    _plot_rate_ax(ax, rows, "GB/s", "CUDA Kernel vs Torch Kernel GB/s", show_legend=True)
    fig.tight_layout()
    _savefig(fig, "layer0_other_throughput.png", dpi=160)
    plt.close(fig)


def main():
    _style()
    gemm, other = split_groups(load_rows())
    plot_matmul_latency(gemm)
    plot_matmul_slowdown(gemm)
    plot_matmul_throughput(gemm)
    plot_other_latency(other)
    plot_other_slowdown(other)
    plot_other_throughput(other)
    for name in OUT_NAMES:
        print(f"wrote {OUT_DIR / name}")


if __name__ == "__main__":
    main()
