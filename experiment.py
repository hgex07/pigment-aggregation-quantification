#!/usr/bin/env python3
"""Análisis por lotes de las fotos de la práctica 3 (data/p3).

    python experiment.py data/p3 --out results/p3

- Agrupa las fotos por larva según el nombre (L2 y L2' son el mismo grupo:
  mismo recorte y alineadas juntas) y las ordena según la secuencia del
  experimento (tabla SECUENCIA).
- Configuración para recortes ajustados: canal R, fondo constante
  (bg_order=0) estimado sobre el tejido de dentro del recorte, y el exterior
  transparente fuera de la medida.
- Umbral semilla: Otsu común a TODAS las imágenes.
- Repite el análisis con 24 variantes de parámetros (sensibilidad) y anota si
  la dirección de cada cambio se mantiene.
"""
from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import filters

import figures
from pigment_pipeline import Channels, Params, compute_metrics, load_channels, otsu_threshold, segment

# Secuencia de cada larva: (archivo sin .png, etapa, papel)
# papel: basal (Ringer), agregado (epinefrina), rescate (reactivo) o control.
SECUENCIA = {
    "Control": [
        ("Control_ringer", "Ringer", "basal"),
        ("Control_epinef_t0", "Epi t0", "agregado?"),
        ("Control_epinef_t15", "Epi +15", "agregado"),
        ("Control_vehic_tf", "Vehículo tf", "control"),
    ],
    "L2": [
        ("L2'_ringer", "Ringer", "basal"),
        ("L2'_epinef_10+10", "Epi 10+10", "agregado"),
        ("L2_forsk_+10", "Forsk +10", "rescate"),
    ],
    "L3": [
        ("L3_ringer", "Ringer", "basal"),
        ("L3_epinef_+10", "Epi +10", "agregado"),
        ("L3_caf_+15", "Caf +15", "rescate"),
    ],
    "L4": [
        ("L4_ringer", "Ringer", "basal"),
        ("L4_epinef_+10", "Epi +10", "agregado"),
        ("L4_ampc+epinef_+10", "DBcAMP+Epi +10", "rescate"),
        ("L4_+ampc_+10+10", "DBcAMP +10+10", "rescate"),
    ],
    "L4v2": [
        ("L4v2_ringer", "Ringer", "basal"),
        ("L4v2_epinef_+10", "Epi +10", "agregado"),
        ("L4v2_ampc_+10", "DBcAMP +10", "rescate"),
    ],
}

PARAMS = Params(channel="R", bg_order=0)

# Variantes para la sensibilidad (todas las combinaciones)
VARIANTES = {
    "channel": ["R", "lum"],
    "bg_order": [0, 1],
    "low_thr": [0.12, 0.18, 0.25],
    "umbral": ["global", "por_grupo"],
}

MIN_AGREGACION = 0.05  # caída mínima del área (5 %) para considerar que agregó

METRICAS = ["indice_extension", "area_pigmentada_pct", "od_media_pigmento", "pigmento_total_kOD"]


def cargar(img_dir: Path, params: Params) -> dict[str, tuple[Channels, str, str, str]]:
    """{archivo: (canales, grupo, etapa, papel)}"""
    return {fname: (load_channels(img_dir / f"{fname}.png", params), grupo, etapa, papel)
            for grupo, seq in SECUENCIA.items() for fname, etapa, papel in seq}


def nitidez(ch: Channels) -> float:
    """Control de calidad del enfoque: varianza del laplaciano dentro del
    recorte (sin el borde), x1e4. Más bajo = más borroso. El desenfoque
    imita la dispersión (puntos que se difuminan, velo gris)."""
    inner = ndi.binary_erosion(ch.roi, iterations=4)
    lap = filters.laplace(filters.gaussian(ch.gray, 0.7))
    return float(1e4 * lap[inner].var())


def medir(data: dict, params: Params, umbral: str = "global") -> tuple[pd.DataFrame, dict]:
    if umbral == "global":
        t = otsu_threshold([ch for ch, *_ in data.values()])
        thr = {g: t for g in SECUENCIA}
    else:
        thr = {g: otsu_threshold([ch for ch, grupo, *_ in data.values() if grupo == g]) for g in SECUENCIA}
    rows, masks = [], {}
    for fname, (ch, grupo, etapa, papel) in data.items():
        mask = segment(ch.od, thr[grupo], params.low_thr, params.min_object_px, ch.roi)
        masks[fname] = mask
        row = compute_metrics(ch, mask)
        row.update(imagen=fname, grupo=grupo, etapa=etapa, papel=papel,
                   umbral_semilla=thr[grupo], nitidez=nitidez(ch))
        rows.append(row)
    df = pd.DataFrame(rows).set_index("imagen")
    # Relativo al basal (Ringer) de cada larva
    for mtr in METRICAS:
        base = df[df.papel == "basal"].set_index("grupo")[mtr]
        df[f"{mtr}_rel_basal"] = df[mtr] / df.grupo.map(base)
    # % de rescate: 0 % = agregado, 100 % = basal. Solo si la epinefrina redujo
    # el área al menos MIN_AGREGACION (si no, el denominador es ruido).
    for mtr, nombre in [("area_pigmentada_pct", "rescate_area_pct"),
                        ("indice_extension", "rescate_IE_pct")]:
        b = df[df.papel == "basal"].set_index("grupo")[mtr]
        a = df[df.papel == "agregado"].set_index("grupo")[mtr]
        agrego = (1 - df.grupo.map(a) / df.grupo.map(b)) >= MIN_AGREGACION
        df[nombre] = np.where(
            (df.papel == "rescate") & agrego,
            100 * (df[mtr] - df.grupo.map(a)) / (df.grupo.map(b) - df.grupo.map(a)),
            np.nan)
    return df, masks


def alineamiento(data: dict) -> pd.DataFrame:
    """Correlación de OD píxel a píxel de cada imagen con el basal de su larva
    (alineamiento + cambio real; no se pueden separar)."""
    rows = []
    for g, seq in SECUENCIA.items():
        ref = data[seq[0][0]][0]
        for fname, etapa, _ in seq[1:]:
            ch = data[fname][0]
            both = ref.roi & ch.roi
            rows.append(dict(grupo=g, imagen=fname, etapa=etapa,
                             r_OD_vs_basal=np.corrcoef(ref.od[both], ch.od[both])[0, 1]))
    return pd.DataFrame(rows)


def fig_resumen(df: pd.DataFrame, out: Path) -> None:
    color = "#2a78d6"
    fig, axs = plt.subplots(2, len(SECUENCIA), figsize=(4 * len(SECUENCIA), 6.4), sharey="row")
    for j, g in enumerate(SECUENCIA):
        sub = df.loc[[s[0] for s in SECUENCIA[g]]]
        x = np.arange(len(sub))
        for i, (col, lab) in enumerate([("area_pigmentada_pct_rel_basal", "Área pigmentada\n(rel. a Ringer)"),
                                        ("indice_extension_rel_basal", "Índice de extensión (rel.)\n(poco fiable aquí)")]):
            a = axs[i, j]
            a.axhline(1, color="#b5b4ae", lw=1, zorder=0)
            a.plot(x, sub[col], color=color, lw=2, marker="o", ms=8,
                   markeredgecolor="#fcfcfb", markeredgewidth=2)
            for xi, v in zip(x, sub[col]):
                a.annotate(f"{v:.2f}", (xi, v), textcoords="offset points", xytext=(0, 9),
                           ha="center", fontsize=8, color="#52514e")
            a.set_xticks(x)
            a.set_xticklabels(sub.etapa if i == 1 else [], rotation=35, ha="right", fontsize=8)
            for s in ("top", "right"):
                a.spines[s].set_visible(False)
            a.tick_params(colors="#52514e", labelsize=8)
            a.grid(axis="y", color="#ebeae5", lw=0.6)
            if j == 0:
                a.set_ylabel(lab, fontsize=9, color="#0b0b0b")
            if i == 0:
                a.set_title(g, fontsize=11, color="#0b0b0b")
    fig.suptitle("Cambio respecto al basal (Ringer) en cada larva — "
                 ">1 = más disperso, <1 = más agregado", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=140, facecolor="#fcfcfb")
    plt.close(fig)


def fig_grupo(g: str, data: dict, masks: dict, df: pd.DataFrame, out: Path) -> None:
    seq = SECUENCIA[g]
    n = len(seq)
    vmax = max(np.percentile(data[f][0].inside(data[f][0].od), 99.5) for f, _, _ in seq)
    fig, ax = plt.subplots(2, n, figsize=(3.6 * n, 7.5))
    for j, (f, etapa, _) in enumerate(seq):
        ch = data[f][0]
        ax[0, j].imshow(np.where(ch.roi, ch.gray, 1.0), cmap="gray", vmin=0, vmax=1)
        ax[0, j].contour(masks[f], levels=[0.5], colors="cyan", linewidths=0.6)
        ax[0, j].set_title(f"{etapa}\nIE={df.loc[f, 'indice_extension']:.2f} · "
                           f"área={df.loc[f, 'area_pigmentada_pct']:.0f}%", fontsize=9)
        shown = ax[1, j].imshow(np.ma.masked_where(~ch.roi, ch.od), cmap="magma", vmin=0, vmax=vmax)
        ax[1, j].set_title("densidad óptica", fontsize=9)
        for a in ax[:, j]:
            a.axis("off")
    fig.colorbar(shown, ax=ax[1, :].tolist(), fraction=0.02)
    fig.suptitle(f"{g}: canal {PARAMS.channel}, máscara (cian) y OD", fontsize=11)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def sensibilidad(img_dir: Path) -> pd.DataFrame:
    """Cambio relativo de área e índice en cada paso de cada larva, para cada
    combinación de parámetros."""
    rows, cache = [], {}
    keys = list(VARIANTES)
    for combo in itertools.product(*VARIANTES.values()):
        v = dict(zip(keys, combo))
        params = replace(PARAMS, channel=v["channel"], bg_order=v["bg_order"], low_thr=v["low_thr"])
        key = (v["channel"], v["bg_order"])
        if key not in cache:
            cache[key] = cargar(img_dir, params)
        df, _ = medir(cache[key], params, v["umbral"])
        for mtr in ("area_pigmentada_pct", "indice_extension"):
            for g, seq in SECUENCIA.items():
                vals = df.loc[[s[0] for s in seq], mtr].values
                for (_, e0, _), (_, e1, _), a, b in zip(seq, seq[1:], vals, vals[1:]):
                    rows.append(dict(v, metrica=mtr, grupo=g, paso=f"{e0} -> {e1}", cambio_rel=b / a - 1))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("img_dir", type=Path, nargs="?", default=Path("data/p3"))
    ap.add_argument("--out", type=Path, default=Path("results/p3"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    data = cargar(args.img_dir, PARAMS)
    df, masks = medir(data, PARAMS, "global")
    df.to_csv(args.out / "metricas_experimento.csv", float_format="%.4f")
    al = alineamiento(data)
    al.to_csv(args.out / "alineamiento.csv", index=False, float_format="%.3f")

    fig_resumen(df, args.out / "resumen_area_e_IE.png")
    for g, seq in SECUENCIA.items():
        fig_grupo(g, data, masks, df, args.out / f"grupo_{g}.png")
        for f, _, _ in seq:
            ch = data[f][0]
            figures.plot_pipeline(f, ch, masks[f], df.loc[f, "umbral_semilla"], PARAMS.low_thr,
                                  figures.od_color_limit({f: ch}), args.out / f"pipeline_{f}.png")

    sens = sensibilidad(args.img_dir)
    sens.to_csv(args.out / "sensibilidad.csv", index=False, float_format="%.4f")
    tab = (sens.groupby(["metrica", "grupo", "paso"], sort=False)["cambio_rel"]
           .agg(mediana="median", minimo="min", maximo="max", frac_positivo=lambda s: (s > 0).mean()))
    tab.to_csv(args.out / "sensibilidad_resumen.csv", float_format="%.3f")

    (args.out / "parametros.json").write_text(json.dumps(
        dict(asdict(PARAMS), umbral_semilla_global=float(df.umbral_semilla.iloc[0])),
        indent=2, ensure_ascii=False))

    pd.set_option("display.width", 220)
    cols = ["grupo", "etapa", "nitidez", "area_pigmentada_pct", "od_media_pigmento", "pigmento_total_kOD",
            "indice_extension", "area_pigmentada_pct_rel_basal", "indice_extension_rel_basal",
            "rescate_area_pct", "rescate_IE_pct"]
    print(df[cols].to_string(float_format=lambda x: f"{x:.3f}"))
    print()
    print(al.to_string(float_format=lambda x: f"{x:.3f}"))
    print()
    print(tab.to_string(float_format=lambda x: f"{x:+.3f}"))


if __name__ == "__main__":
    main()
