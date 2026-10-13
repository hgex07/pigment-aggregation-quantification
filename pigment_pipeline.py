"""Núcleo de la pipeline: foto -> densidad óptica (OD) -> máscara -> métricas.

El contraste (CLAHE, ecualización...) NO se usa en ningún paso de medida:
estira cada imagen de forma distinta y falsearía la comparación.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from skimage import filters, io, measure, morphology

LUMINANCE_WEIGHTS = np.array([0.299, 0.587, 0.114])  # R, G, B
MIN_TRANSMITTANCE = 1e-3  # evita log(0)


@dataclass(frozen=True)
class Params:
    blur_sigma: float = 1.0   # suavizado previo contra el ruido de cámara (px)
    bg_order: int = 2         # grado del polinomio de fondo (2 = curva suave)
    low_thr: float = 0.18     # umbral bajo de OD (extiende la máscara); elegido a ojo
    min_object_px: int = 15   # manchas/huecos menores se consideran ruido
    tile_px: int = 64         # lado de las teselas del análisis local


@dataclass
class Channels:
    gray: np.ndarray           # luminancia original, 0-1
    background: np.ndarray     # luz estimada sin pigmento
    transmittance: np.ndarray  # fracción de luz que pasa, 0-1
    od: np.ndarray             # densidad óptica = -log10(transmitancia)


# ---------------------------------------------------------------- preprocesado
def load_gray(path: Path) -> np.ndarray:
    """Lee una imagen y la devuelve como luminancia en [0, 1]."""
    img = io.imread(path)
    if img.ndim == 3:
        return (img[..., :3] / 255.0) @ LUMINANCE_WEIGHTS
    return img / (255.0 if img.max() > 1 else 1.0)


def _poly_terms(y: np.ndarray, x: np.ndarray, order: int) -> np.ndarray:
    """Columnas [1, y, x, y², yx, x², ...] de un polinomio 2D."""
    cols = [np.ones_like(y)]
    for degree in range(1, order + 1):
        for k in range(degree + 1):
            cols.append(y ** (degree - k) * x ** k)
    return np.stack(cols, axis=-1)


def estimate_background(gray: np.ndarray, order: int, n_iter: int = 4) -> np.ndarray:
    """Ajusta una superficie suave a los píxeles de fondo.

    Empieza con el 60 % más claro y en cada vuelta descarta lo claramente
    más oscuro que la superficie (pigmento), para que las células no
    arrastren el fondo hacia abajo.
    """
    h, w = gray.shape
    y, x = np.mgrid[0:h, 0:w]
    y = (y / h - 0.5) * 2
    x = (x / w - 0.5) * 2
    terms = _poly_terms(y.ravel(), x.ravel(), order)
    values = gray.ravel()

    is_background = values > np.percentile(values, 40)
    for _ in range(n_iter):
        coef, *_ = np.linalg.lstsq(terms[is_background], values[is_background], rcond=None)
        fitted = terms @ coef
        residual = values - fitted
        spread = 1.4826 * np.median(np.abs(residual[is_background] - np.median(residual[is_background])))
        is_background = residual > -2.0 * spread
    return np.clip(fitted.reshape(h, w), 1e-3, None)


def compute_channels(gray: np.ndarray, params: Params) -> Channels:
    smooth = filters.gaussian(gray, params.blur_sigma)
    background = estimate_background(smooth, params.bg_order)
    transmittance = np.clip(smooth / background, MIN_TRANSMITTANCE, 1.0)
    return Channels(gray, background, transmittance, -np.log10(transmittance))


# ---------------------------------------------------------------- máscara
def otsu_threshold(od_maps: list[np.ndarray]) -> float:
    """Umbral de Otsu calculado con los píxeles de todas las imágenes juntas."""
    return float(filters.threshold_otsu(np.concatenate([od.ravel() for od in od_maps])))


def segment(od: np.ndarray, seed_thr: float, low_thr: float, min_px: int) -> np.ndarray:
    """Máscara por histéresis: OD > low_thr, solo si conecta con OD > seed_thr."""
    mask = filters.apply_hysteresis_threshold(od, min(low_thr, seed_thr), seed_thr)
    mask = morphology.remove_small_objects(mask, max_size=min_px)
    return morphology.remove_small_holes(mask, max_size=min_px)


# ---------------------------------------------------------------- métricas
def compute_metrics(od: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    mask_px = int(mask.sum())
    total_od = od[mask].sum()
    blobs = measure.regionprops(measure.label(mask))
    areas = [b.area for b in blobs] or [0]
    solidity = [b.solidity for b in blobs] or [np.nan]
    perimeter = sum(b.perimeter for b in blobs)
    # perímetro total / perímetro de un círculo con la misma área (1 = disco)
    complexity = perimeter / (2 * np.sqrt(np.pi * mask_px)) if mask_px else np.nan

    return {
        "area_pigmentada_pct": 100 * mask.mean(),
        "od_media_pigmento": od[mask].mean() if mask_px else np.nan,
        "pigmento_total_kOD": total_od / 1000,
        "od_media_imagen": od.mean(),
        # superficie por unidad de pigmento: bajo = agregado, alto = disperso
        "indice_extension": mask_px / total_od if total_od else np.nan,
        "n_manchas": len(blobs),
        "area_mediana_mancha_px": float(np.median(areas)),
        "area_media_mancha_px": float(np.mean(areas)),
        "solidez_media": float(np.nanmean(solidity)),
        "complejidad_contorno": complexity,
    }


def threshold_sweep(od_maps: dict[str, np.ndarray], thresholds: np.ndarray) -> pd.DataFrame:
    """% de la imagen con OD por encima de cada umbral."""
    table = {"umbral_od": thresholds}
    for name, od in od_maps.items():
        table[name] = [100 * (od > t).mean() for t in thresholds]
    return pd.DataFrame(table)


def tile_percent(mask: np.ndarray, tile_px: int) -> np.ndarray:
    """% de píxeles de máscara en cada cuadrado de tile_px (descarta el borde sobrante)."""
    ny, nx = mask.shape[0] // tile_px, mask.shape[1] // tile_px
    tiles = mask[: ny * tile_px, : nx * tile_px].reshape(ny, tile_px, nx, tile_px)
    return 100 * tiles.mean(axis=(1, 3))
