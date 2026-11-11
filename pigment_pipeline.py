"""Núcleo de la pipeline: foto -> densidad óptica (OD) -> máscara -> métricas.

v2: admite recortes con canal alfa. Si el PNG tiene píxeles transparentes,
solo se mide dentro del recorte (la "roi") y el fondo se estima sobre el
tejido de dentro. Sin alfa y con los parámetros por defecto, el resultado es
idéntico a la v1.

El contraste (CLAHE, ecualización...) NO se usa en ningún paso de medida:
estira cada imagen de forma distinta y falsearía la comparación.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import filters, io, measure, morphology

CHANNEL_WEIGHTS = {
    "lum": np.array([0.299, 0.587, 0.114]),  # luminancia
    "R": np.array([1.0, 0.0, 0.0]),          # rojo: casi ciego al pigmento amarillo (xantóforos)
    "G": np.array([0.0, 1.0, 0.0]),
}
MIN_TRANSMITTANCE = 1e-3  # evita log(0)


@dataclass(frozen=True)
class Params:
    blur_sigma: float = 1.0   # suavizado previo contra el ruido de cámara (px)
    bg_order: int = 2         # grado del polinomio de fondo (2 = curva suave; 0 = constante)
    low_thr: float = 0.18     # umbral bajo de OD (extiende la máscara); elegido a ojo
    min_object_px: int = 15   # manchas/huecos menores se consideran ruido
    tile_px: int = 64         # lado de las teselas del análisis local
    channel: str = "lum"      # lum | R | G
    seed_thr: float = -1.0    # umbral semilla fijo; negativo = Otsu común a todas las imágenes
    alpha_min: int = 200      # con alfa: píxel dentro del recorte si alfa >= alpha_min
    erode_px: int = 2         # erosión de la roi (quita el borde difuminado del recorte)


@dataclass
class Channels:
    gray: np.ndarray           # canal elegido original, 0-1
    background: np.ndarray     # luz estimada sin pigmento
    transmittance: np.ndarray  # fracción de luz que pasa, 0-1
    od: np.ndarray             # densidad óptica = -log10(transmitancia); 0 fuera de la roi
    roi: np.ndarray | None = None  # región medida (None = toda la imagen)

    def inside(self, image: np.ndarray) -> np.ndarray:
        """Píxeles de `image` dentro de la roi, en 1D."""
        return image.ravel() if self.roi is None else image[self.roi]


# ---------------------------------------------------------------- preprocesado
def load_image(path: Path, params: Params = Params()) -> tuple[np.ndarray, np.ndarray | None]:
    """Lee una imagen y devuelve (canal elegido en [0, 1], roi).

    roi = píxeles opacos si la imagen tiene alfa con zonas transparentes;
    None en otro caso. OJO: los píxeles transparentes suelen guardarse como
    RGB = (0, 0, 0); sin la roi se medirían como pigmento negro.
    """
    img = io.imread(path)
    if img.ndim == 2:
        return img / (255.0 if img.max() > 1 else 1.0), None
    roi = None
    if img.shape[-1] == 4 and (img[..., 3] < params.alpha_min).any():
        roi = img[..., 3] >= params.alpha_min
        if params.erode_px > 0:
            roi = ndi.binary_erosion(roi, morphology.disk(params.erode_px))
    return (img[..., :3] / 255.0) @ CHANNEL_WEIGHTS[params.channel], roi


def _poly_terms(y: np.ndarray, x: np.ndarray, order: int) -> np.ndarray:
    """Columnas [1, y, x, y², yx, x², ...] de un polinomio 2D."""
    cols = [np.ones_like(y)]
    for degree in range(1, order + 1):
        for k in range(degree + 1):
            cols.append(y ** (degree - k) * x ** k)
    return np.stack(cols, axis=-1)


def estimate_background(gray: np.ndarray, order: int, n_iter: int = 4,
                        roi: np.ndarray | None = None) -> np.ndarray:
    """Ajusta una superficie suave a los píxeles de fondo.

    Sin roi: empieza con el 60 % más claro y en cada vuelta descarta lo
    claramente más oscuro que la superficie (pigmento), para que las células
    no arrastren el fondo hacia abajo.

    Con roi: los candidatos son solo la clase clara de Otsu dentro del
    recorte (tejido sin melanina) y las vueltas solo pueden QUITAR
    candidatos. Si no, cuando el pigmento cubre >50 % del recorte, el ajuste
    se escapa hacia abajo y absorbe el pigmento. En recortes ajustados
    conviene order=0: un polinomio se extrapola mal bajo las zonas cubiertas.
    """
    h, w = gray.shape
    y, x = np.mgrid[0:h, 0:w]
    y = (y / h - 0.5) * 2
    x = (x / w - 0.5) * 2
    terms = _poly_terms(y.ravel(), x.ravel(), order)
    values = gray.ravel()

    if roi is None:
        candidates = np.ones_like(values, bool)
        is_background = values > np.percentile(values, 40)
    else:
        inside = roi.ravel()
        candidates = inside & (values > filters.threshold_otsu(values[inside]))
        is_background = candidates.copy()
    for _ in range(n_iter):
        coef, *_ = np.linalg.lstsq(terms[is_background], values[is_background], rcond=None)
        fitted = terms @ coef
        residual = values - fitted
        spread = 1.4826 * np.median(np.abs(residual[is_background] - np.median(residual[is_background])))
        is_background = candidates & (residual > -2.0 * spread)
    return np.clip(fitted.reshape(h, w), 1e-3, None)


def compute_channels(gray: np.ndarray, params: Params, roi: np.ndarray | None = None) -> Channels:
    if roi is None:
        smooth = filters.gaussian(gray, params.blur_sigma)
    else:
        # suavizado normalizado: el negro de fuera del recorte no oscurece el borde
        weight = filters.gaussian(roi.astype(float), params.blur_sigma)
        smooth = filters.gaussian(gray * roi, params.blur_sigma) / np.maximum(weight, 1e-6)
        smooth = np.where(roi, smooth, 1.0)
    background = estimate_background(smooth, params.bg_order, roi=roi)
    transmittance = np.clip(smooth / background, MIN_TRANSMITTANCE, 1.0)
    od = -np.log10(transmittance)
    if roi is not None:
        od = np.where(roi, od, 0.0)  # fuera del recorte no hay medida
    return Channels(gray, background, transmittance, od, roi)


def load_channels(path: Path, params: Params) -> Channels:
    gray, roi = load_image(path, params)
    return compute_channels(gray, params, roi)


# ---------------------------------------------------------------- máscara
def otsu_threshold(channels: list[Channels]) -> float:
    """Umbral de Otsu con los píxeles (de dentro) de todas las imágenes juntas."""
    return float(filters.threshold_otsu(np.concatenate([ch.inside(ch.od) for ch in channels])))


def segment(od: np.ndarray, seed_thr: float, low_thr: float, min_px: int,
            roi: np.ndarray | None = None) -> np.ndarray:
    """Máscara por histéresis: OD > low_thr, solo si conecta con OD > seed_thr."""
    mask = filters.apply_hysteresis_threshold(od, min(low_thr, seed_thr), seed_thr)
    if roi is not None:
        mask &= roi
    mask = morphology.remove_small_objects(mask, max_size=min_px)
    mask = morphology.remove_small_holes(mask, max_size=min_px)
    return mask if roi is None else mask & roi


# ---------------------------------------------------------------- métricas
def compute_metrics(ch: Channels, mask: np.ndarray) -> dict[str, float]:
    """Con roi, los porcentajes son respecto al recorte, no al lienzo."""
    od = ch.od
    mask_px = int(mask.sum())
    total_od = od[mask].sum()
    blobs = measure.regionprops(measure.label(mask))
    areas = [b.area for b in blobs] or [0]
    solidity = [b.solidity for b in blobs] or [np.nan]
    perimeter = sum(b.perimeter for b in blobs)
    # perímetro total / perímetro de un círculo con la misma área (1 = disco)
    complexity = perimeter / (2 * np.sqrt(np.pi * mask_px)) if mask_px else np.nan
    roi_px = od.size if ch.roi is None else int(ch.roi.sum())

    return {
        "area_pigmentada_pct": 100 * mask_px / roi_px,
        "od_media_pigmento": od[mask].mean() if mask_px else np.nan,
        "pigmento_total_kOD": total_od / 1000,
        "od_media_imagen": ch.inside(od).mean(),
        # superficie por unidad de pigmento: bajo = agregado, alto = disperso
        "indice_extension": mask_px / total_od if total_od else np.nan,
        "n_manchas": len(blobs),
        "area_mediana_mancha_px": float(np.median(areas)),
        "area_media_mancha_px": float(np.mean(areas)),
        "solidez_media": float(np.nanmean(solidity)),
        "complejidad_contorno": complexity,
    }


def threshold_sweep(channels: dict[str, Channels], thresholds: np.ndarray) -> pd.DataFrame:
    """% de la imagen (o del recorte) con OD por encima de cada umbral."""
    table = {"umbral_od": thresholds}
    for name, ch in channels.items():
        od = ch.inside(ch.od)
        table[name] = [100 * (od > t).mean() for t in thresholds]
    return pd.DataFrame(table)


def tile_percent(mask: np.ndarray, tile_px: int, roi: np.ndarray | None = None,
                 min_cover: float = 0.8) -> np.ndarray:
    """% de píxeles de máscara en cada cuadrado de tile_px (descarta el borde sobrante).

    Con roi, solo cuentan las teselas cubiertas por el recorte en >= min_cover;
    el resto queda NaN.
    """
    ny, nx = mask.shape[0] // tile_px, mask.shape[1] // tile_px

    def blocks(a: np.ndarray) -> np.ndarray:
        return a[: ny * tile_px, : nx * tile_px].reshape(ny, tile_px, nx, tile_px).astype(float)

    if roi is None:
        return 100 * blocks(mask).mean(axis=(1, 3))
    inside = blocks(roi).sum(axis=(1, 3))
    hits = blocks(mask & roi).sum(axis=(1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = 100 * hits / inside
    out[inside < min_cover * tile_px ** 2] = np.nan
    return out
