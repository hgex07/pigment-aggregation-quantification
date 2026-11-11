"""Figuras de la pipeline. Solo dibujan; no calculan nada nuevo."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pigment_pipeline import Channels

CONTOUR_COLOR = "cyan"


def od_color_limit(channels: dict[str, Channels]) -> float:
    """Tope común de la escala de OD, para que los colores sean comparables."""
    return max(np.percentile(ch.inside(ch.od), 99.5) for ch in channels.values())


def _shown_gray(ch: Channels) -> np.ndarray:
    """Canal original con el exterior del recorte en blanco (solo para dibujar)."""
    return ch.gray if ch.roi is None else np.where(ch.roi, ch.gray, 1.0)


def _shown(ch: Channels, image: np.ndarray, fill: float = 1.0) -> np.ndarray:
    return image if ch.roi is None else np.where(ch.roi, image, fill)


def plot_pipeline(name: str, ch: Channels, mask: np.ndarray, seed_thr: float, low_thr: float,
                  od_max: float, path: Path) -> None:
    """Las etapas de una imagen en fila."""
    fig, axes = plt.subplots(1, 5, figsize=(17.5, 5.2))
    panels = [
        (_shown_gray(ch), "1. Gris (canal elegido)", "gray", (0, 1)),
        (_shown(ch, ch.background), "2. Fondo estimado", "gray", (0, 1)),
        (_shown(ch, ch.transmittance), "3. Transmitancia", "gray", (0, 1)),
        (ch.od, "4. Densidad óptica", "magma", (0, od_max)),
        (_shown_gray(ch), f"5. Máscara (OD > {low_thr:.2f}, unida a OD > {seed_thr:.3f})", "gray", (0, 1)),
    ]
    for ax, (image, title, cmap, (vmin, vmax)) in zip(axes, panels):
        shown = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        if cmap == "magma":
            fig.colorbar(shown, ax=ax, fraction=0.046)
    axes[-1].contour(mask, levels=[0.5], colors=CONTOUR_COLOR, linewidths=0.6)
    fig.suptitle(f"Pipeline — {name}", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_vertical(ch: Channels, mask: np.ndarray, od_max: float, path: Path, gap_px: int = 12) -> None:
    """Las etapas apiladas en vertical, sin texto ni barras de color."""
    h, w = ch.gray.shape
    panels = [
        (_shown_gray(ch), "gray", 1.0),
        (_shown(ch, ch.background), "gray", 1.0),
        (_shown(ch, ch.transmittance), "gray", 1.0),
        (ch.od, "magma", od_max),
        (_shown_gray(ch), "gray", 1.0),  # + contorno de la máscara
    ]
    total_h = len(panels) * h + (len(panels) - 1) * gap_px
    fig = plt.figure(figsize=(w / 100, total_h / 100), dpi=200, facecolor="white")
    for i, (image, cmap, vmax) in enumerate(panels):
        bottom = 1 - (i * (h + gap_px) + h) / total_h
        ax = fig.add_axes([0, bottom, 1, h / total_h])
        ax.imshow(image, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest", aspect="auto")
        if i == len(panels) - 1:
            ax.contour(mask, levels=[0.5], colors=CONTOUR_COLOR, linewidths=0.9)
        ax.set_xlim(-0.5, w - 0.5)
        ax.set_ylim(h - 0.5, -0.5)
        ax.axis("off")
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def _threshold_lines(ax, seed_thr: float, low_thr: float) -> None:
    ax.axvline(seed_thr, color="k", ls="--", lw=0.8, label=f"semilla (Otsu) {seed_thr:.3f}")
    ax.axvline(low_thr, color="tab:green", ls="--", lw=0.8, label=f"umbral bajo {low_thr:.2f}")


def plot_comparison(names: list[str], channels: dict[str, Channels], masks: dict[str, np.ndarray],
                    tiles: dict[str, np.ndarray], sweep: pd.DataFrame, metrics: pd.DataFrame,
                    seed_thr: float, low_thr: float, path: Path) -> None:
    """OD + máscara, teselas, barrido de umbrales y teselas emparejadas con la primera imagen."""
    n = len(names)
    fig = plt.figure(figsize=(5 * n + 6, 10))
    grid = fig.add_gridspec(2, n + 1)
    od_max = od_color_limit(channels)

    for i, name in enumerate(names):
        ax = fig.add_subplot(grid[0, i])
        ax.imshow(channels[name].od, cmap="magma", vmin=0, vmax=od_max)
        ax.contour(masks[name], levels=[0.5], colors=CONTOUR_COLOR, linewidths=0.5)
        ax.set_title(f"{name}: OD + máscara\nárea pigm. = {metrics.loc[name, 'area_pigmentada_pct']:.1f}%")
        ax.axis("off")

        ax = fig.add_subplot(grid[1, i])
        shown = ax.imshow(tiles[name], cmap="viridis", vmin=0, vmax=100)
        for (row, col), value in np.ndenumerate(tiles[name]):
            if np.isnan(value):
                continue
            ax.text(col, row, f"{value:.0f}", ha="center", va="center", fontsize=8, color="w")
        ax.set_title(f"{name}: % pigmentado por tesela")
        ax.axis("off")
    fig.colorbar(shown, ax=ax, fraction=0.046)

    ax = fig.add_subplot(grid[0, n])
    for name in names:
        ax.plot(sweep["umbral_od"], sweep[name], label=name)
    _threshold_lines(ax, seed_thr, low_thr)
    ax.set_xlabel("umbral de densidad óptica")
    ax.set_ylabel("% de área por encima del umbral")
    ax.set_title("Robustez: barrido de umbrales")
    ax.legend()

    ref = names[0]
    paired = [name for name in names[1:] if tiles[name].shape == tiles[ref].shape]
    if paired:
        ax = fig.add_subplot(grid[1, n])
        for name in paired:
            ax.scatter(tiles[ref].ravel(), tiles[name].ravel(), s=18, label=name)
        ax.plot([0, 100], [0, 100], "k--", lw=0.8)
        ax.set_xlabel(f"% pigmentado tesela ({ref})")
        ax.set_ylabel(f"% pigmentado tesela ({paired[0]})" if len(paired) == 1 else "% pigmentado tesela (resto)")
        if len(paired) > 1:
            ax.legend(fontsize=8)
        ax.set_title("Teselas emparejadas (misma región)")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_od_histogram(names: list[str], channels: dict[str, Channels], seed_thr: float,
                      low_thr: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name in names:
        ax.hist(channels[name].inside(channels[name].od), bins=150, range=(0, 1.2), histtype="step",
                density=True, lw=1.4, label=name)
    _threshold_lines(ax, seed_thr, low_thr)
    ax.set_xlabel("densidad óptica por píxel")
    ax.set_ylabel("densidad de probabilidad")
    ax.set_title("Distribución del pigmento por píxel")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
