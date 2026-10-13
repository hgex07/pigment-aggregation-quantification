#!/usr/bin/env python3
"""Ejecuta la pipeline completa sobre una o más imágenes.

    python analyze.py data/img1.png data/img2.png --labels img1 img2 --out results

Escribe en --out: metricas.csv, barrido_umbrales.csv, resumen.json y las figuras.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import pandas as pd

import figures
from pigment_pipeline import (Params, compute_channels, compute_metrics, load_gray,
                              otsu_threshold, segment, threshold_sweep, tile_percent)

SWEEP_THRESHOLDS = np.linspace(0.02, 0.8, 79)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--labels", nargs="+", help="un nombre por imagen (por defecto, el nombre del archivo)")
    parser.add_argument("--out", type=Path, default=Path("results"))
    for field in fields(Params):
        parser.add_argument(f"--{field.name.replace('_', '-')}", type=type(field.default), default=field.default)
    return parser.parse_args()


def pairwise_summary(a: str, b: str, od_maps, tiles, sweep) -> dict:
    """Comparación de dos imágenes alineadas: b respecto a a."""
    return {
        "a": a,
        "b": b,
        "correlacion_od_pixel_a_pixel": float(np.corrcoef(od_maps[a].ravel(), od_maps[b].ravel())[0, 1]),
        "teselas_b_mayor_que_a": int((tiles[b] > tiles[a]).sum()),
        "teselas_totales": int(tiles[a].size),
        "mediana_diferencia_teselas_b_menos_a_pct": float(np.median(tiles[b] - tiles[a])),
        "fraccion_umbrales_b_mayor_que_a": float((sweep[b] > sweep[a]).mean()),
    }


def main() -> None:
    args = parse_args()
    params = Params(**{f.name: getattr(args, f.name) for f in fields(Params)})
    names = args.labels or [path.stem for path in args.images]
    if len(names) != len(args.images):
        raise SystemExit("hace falta un label por imagen")
    args.out.mkdir(parents=True, exist_ok=True)

    channels = {name: compute_channels(load_gray(path), params) for name, path in zip(names, args.images)}
    od_maps = {name: ch.od for name, ch in channels.items()}
    seed_thr = otsu_threshold(list(od_maps.values()))
    masks = {name: segment(od, seed_thr, params.low_thr, params.min_object_px) for name, od in od_maps.items()}
    tiles = {name: tile_percent(mask, params.tile_px) for name, mask in masks.items()}
    sweep = threshold_sweep(od_maps, SWEEP_THRESHOLDS)

    metrics = pd.DataFrame({name: compute_metrics(od_maps[name], masks[name]) for name in names}).T
    metrics.index.name = "imagen"
    summary = {"umbral_semilla_otsu": seed_thr, "umbral_bajo": params.low_thr, "parametros": asdict(params)}
    if len(names) == 2:
        a, b = names
        summary["comparacion"] = pairwise_summary(a, b, od_maps, tiles, sweep)
        metrics.loc[f"{b} / {a}"] = metrics.loc[b] / metrics.loc[a]

    metrics.to_csv(args.out / "metricas.csv", float_format="%.4f")
    sweep.to_csv(args.out / "barrido_umbrales.csv", index=False, float_format="%.4f")
    (args.out / "resumen.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    od_max = figures.od_color_limit(channels)
    for name in names:
        figures.plot_pipeline(name, channels[name], masks[name], seed_thr, params.low_thr, od_max,
                              args.out / f"pipeline_{name}.png")
        figures.plot_vertical(channels[name], masks[name], od_max, args.out / f"{name}_vertical.png")
    figures.plot_comparison(names, channels, masks, tiles, sweep, metrics, seed_thr, params.low_thr,
                            args.out / "comparacion.png")
    figures.plot_od_histogram(names, channels, seed_thr, params.low_thr, args.out / "histograma_od.png")

    print(metrics.T.to_string(float_format=lambda x: f"{x:.3f}"))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
