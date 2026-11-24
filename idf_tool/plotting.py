"""
plotting.py — Generación modular de gráficas.

Cada función retorna (fig, titulo). Las figuras se cierran tras guardar.
"""

from __future__ import annotations

import os
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from .distributions import COLORES_DIST
from .series import MESES_ES


def configurar_matplotlib() -> None:
    plt.rcParams.update({
        "figure.dpi"    : 130,
        "savefig.dpi"   : 200,
        "font.size"     : 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.grid"     : True,
        "grid.alpha"    : 0.4,
        "lines.linewidth": 1.8,
    })


def guardar(fig: Figure, ruta: str) -> str:
    """Guarda figura en PNG y devuelve la ruta."""
    fig.savefig(ruta, bbox_inches="tight")
    return ruta


# ------------------------------------------------------------------------------
# Gráficas básicas
# ------------------------------------------------------------------------------

def fig_serie_diaria(df: pd.DataFrame, columna: str, codigo: str | int) -> tuple[Figure, str]:
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df.index, df[columna], color="steelblue", linewidth=0.7, alpha=0.85,
            label="Precipitación")
    mask_nan = df[columna].isna()
    if mask_nan.any():
        max_v = float(df[columna].max() or 1.0)
        ax.fill_between(df.index, 0, max_v * 1.1, where=mask_nan, color="red", alpha=0.2,
                        label="Días sin dato")
    n_total = len(df)
    n_falt = int(mask_nan.sum())
    ax.set_title(f"Serie diaria | Est. {codigo} | Presentes: {n_total - n_falt} | "
                 f"Faltantes: {n_falt} ({n_falt/n_total*100:.1f} %)")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Precipitación (mm)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    return fig, "Serie diaria con faltantes"


def fig_completitud_mensual(comp_df: pd.DataFrame, codigo: str | int) -> tuple[Figure, str]:
    pivot_pct = comp_df.pivot(index="anio", columns="mes", values="con_dato")
    pivot_tot = comp_df.pivot(index="anio", columns="mes", values="total")
    pivot = (pivot_pct / pivot_tot * 100).round(1)
    # asegurar 12 columnas
    for m in range(1, 13):
        if m not in pivot.columns:
            pivot[m] = np.nan
    pivot = pivot[list(range(1, 13))]
    pivot.columns = MESES_ES

    fig, ax = plt.subplots(figsize=(13, max(5, len(pivot) * 0.28)))
    im = ax.imshow(pivot.values.astype(float), aspect="auto", cmap="RdYlGn",
                   vmin=0, vmax=100, interpolation="nearest")
    ax.set_xticks(range(12)); ax.set_xticklabels(MESES_ES)
    ax.set_yticks(range(len(pivot))); ax.set_yticklabels(pivot.index.astype(str), fontsize=8)
    for i in range(len(pivot)):
        for j in range(12):
            v = pivot.iloc[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        fontsize=6, color="white" if v < 50 else "black")
    plt.colorbar(im, ax=ax, label="Completitud (%)", shrink=0.8)
    ax.set_title(f"Completitud mensual | Est. {codigo}")
    ax.set_xlabel("Mes"); ax.set_ylabel("Año")
    plt.tight_layout()
    return fig, "Completitud mensual"


def fig_mapa_calor_maximos(tabla: pd.DataFrame, codigo: str | int) -> tuple[Figure, str]:
    valores = tabla[MESES_ES].values.astype(float)
    fig, ax = plt.subplots(figsize=(13, max(5, len(tabla) * 0.28)))
    im = ax.imshow(valores, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xticks(range(12)); ax.set_xticklabels(MESES_ES)
    ax.set_yticks(range(len(tabla))); ax.set_yticklabels(tabla.index.astype(str), fontsize=8)
    q75 = np.nanquantile(valores, 0.75)
    for i in range(len(tabla)):
        for j in range(12):
            v = valores[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if v > q75 else "black")
    plt.colorbar(im, ax=ax, label="Pmax mensual (mm)", shrink=0.8)
    ax.set_title(f"Mapa de calor — máximos mensuales | Est. {codigo}")
    ax.set_xlabel("Mes"); ax.set_ylabel("Año")
    plt.tight_layout()
    return fig, "Máximos mensuales"


def fig_completitud_anual(meses_con_dato: pd.Series, min_meses: int,
                          n_completos: int, n_incompletos: int, n_insufic: int,
                          codigo: str | int) -> tuple[Figure, str]:
    fig, ax = plt.subplots(figsize=(14, 4))
    colores = [
        "#2ecc71" if m == 12 else "#f39c12" if m >= min_meses else "#e74c3c"
        for m in meses_con_dato.values
    ]
    ax.bar(meses_con_dato.index.astype(str), meses_con_dato.values, color=colores)
    ax.axhline(min_meses, color="black", ls="--", lw=1.0)
    ax.axhline(12, color="gray", ls=":", lw=1.0)
    leyenda = [
        Patch(color="#2ecc71", label=f"Completo: {n_completos}"),
        Patch(color="#f39c12", label=f"Incompleto: {n_incompletos}"),
        Patch(color="#e74c3c", label=f"Insuficiente: {n_insufic}"),
    ]
    ax.legend(handles=leyenda, loc="lower right", fontsize=8)
    ax.set_title(f"Completitud por año | Est. {codigo}")
    ax.set_xlabel("Año"); ax.set_ylabel("Meses con dato"); ax.set_ylim(0, 13)
    plt.xticks(rotation=90, fontsize=7); plt.tight_layout()
    return fig, "Completitud por año"


def fig_pmax24(tabla_completada: pd.DataFrame, anios_incompletos: list[int],
               codigo: str | int) -> tuple[Figure, str]:
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(tabla_completada.index.astype(str), tabla_completada["Pmax24"],
           color="steelblue", alpha=0.75)
    if anios_incompletos:
        ap = [a for a in anios_incompletos if a in tabla_completada.index]
        ax.bar([str(a) for a in ap], tabla_completada.loc[ap, "Pmax24"],
               color="#f39c12", alpha=0.95, label="Año completado")
    prom = float(tabla_completada["Pmax24"].mean())
    ax.axhline(prom, color="red", ls="--", lw=1.2, label=f"Promedio = {prom:.1f} mm")
    ax.set_title(f"Pmax24 por año | Est. {codigo}")
    ax.set_xlabel("Año"); ax.set_ylabel("Pmax24 (mm)")
    ax.legend()
    plt.xticks(rotation=90, fontsize=7); plt.tight_layout()
    return fig, "Pmax24 anual"


def fig_outliers(tabla: pd.DataFrame, info: dict[str, Any], codigo: str | int) -> tuple[Figure, str]:
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(tabla))
    anios = tabla.index.astype(str)
    vals = tabla["Pmax24"].values
    lim_sup = info.get("lim_superior", np.inf)
    lim_inf = info.get("lim_inferior", -np.inf)
    mask_n = (vals >= lim_inf) & (vals <= lim_sup)
    ax.scatter(x[mask_n], vals[mask_n], color="steelblue", s=45, zorder=3, label="Normales")
    if (~mask_n).any():
        ax.scatter(x[~mask_n], vals[~mask_n], color="red", s=70, zorder=4,
                   marker="D", label=f"Outliers ({(~mask_n).sum()})")
        for xi, anio, val in zip(x[~mask_n], anios[~mask_n], vals[~mask_n]):
            ax.annotate(f"{anio}\n{val:.0f}", xy=(xi, val), xytext=(5, 8),
                        textcoords="offset points", fontsize=8, color="red")
    ax.plot(x, vals, color="steelblue", lw=0.8, alpha=0.5, zorder=2)
    ax.axhline(lim_sup, color="#e74c3c", ls="--", lw=1.2, label=f"Sup = {lim_sup:.1f}")
    ax.axhline(lim_inf, color="#e67e22", ls="--", lw=1.2, label=f"Inf = {lim_inf:.1f}")
    ax.fill_between(x, lim_inf, lim_sup, color="#2ecc71", alpha=0.08)
    ax.set_title(f"Outliers Bulletin 17B | Est. {codigo} | n={info['n']}, Kn={info['Kn']:.3f}")
    ax.set_xlabel("Año"); ax.set_ylabel("Pmax24 (mm)")
    ax.set_xticks(x); ax.set_xticklabels(anios, rotation=90, fontsize=7)
    ax.legend(loc="upper right", fontsize=8); plt.tight_layout()
    return fig, "Outliers Bulletin 17B"


def fig_qq_plots(resultados: list[dict[str, Any]], serie: pd.Series,
                 codigo: str | int) -> tuple[Figure, str]:
    validos = [r for r in resultados if "_error" not in r]
    n_d = len(validos)
    n_cols = min(3, n_d)
    n_rows = int(np.ceil(n_d / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
    if n_d == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()

    datos_real = serie.dropna().values
    datos_log = np.log10(datos_real)
    n = len(datos_real)
    probs = (np.arange(1, n + 1) - 0.375) / (n + 0.25)  # Blom

    for i, r in enumerate(validos):
        ax = axes[i]
        nombre, do, pp, esc = r["Distribucion"], r["_dist_obj"], r["_params"], r["_escala"]
        datos = np.sort(datos_log) if esc == "log" else np.sort(datos_real)
        cuant = do.ppf(probs, *pp)
        ax.scatter(cuant, datos, color=COLORES_DIST.get(nombre, "gray"),
                   s=40, alpha=0.85, zorder=3)
        lmin = min(cuant.min(), datos.min()) * 0.98
        lmax = max(cuant.max(), datos.max()) * 1.02
        ax.plot([lmin, lmax], [lmin, lmax], "k--", lw=1.0)
        ax.set_title(f"{nombre}\nAIC={r['AIC']:.1f}  KS p={r['KS_pvalor']:.3f}  AD={r['AD_stat']:.3f}",
                     fontsize=9)
        ax.set_xlabel("Teóricos", fontsize=8); ax.set_ylabel("Observados", fontsize=8)
    for j in range(n_d, len(axes)):
        fig.delaxes(axes[j])
    fig.suptitle(f"QQ Plots | Est. {codigo}", fontsize=11)
    plt.tight_layout()
    return fig, "QQ Plots"


def fig_curvas_frecuencia(resultados: list[dict[str, Any]], serie: pd.Series,
                          Tr_diseno: int, dist_seleccionada: str,
                          codigo: str | int) -> tuple[Figure, str]:
    Tr_curva = np.logspace(np.log10(1.01), np.log10(500), 200)
    probs = 1.0 - 1.0 / Tr_curva
    datos_ord = np.sort(serie.dropna().values)
    n = len(datos_ord)
    Tr_emp = 1.0 / (1.0 - np.arange(1, n + 1) / (n + 1))

    fig, ax = plt.subplots(figsize=(12, 7))
    for r in resultados:
        if "_error" in r:
            continue
        do, pp, esc = r["_dist_obj"], r["_params"], r["_escala"]
        curva = 10 ** do.ppf(probs, *pp) if esc == "log" else do.ppf(probs, *pp)
        es_sel = r["Distribucion"] == dist_seleccionada
        ax.semilogx(Tr_curva, curva, color=COLORES_DIST.get(r["Distribucion"], "gray"),
                    lw=2.5 if es_sel else 1.4,
                    ls="-" if es_sel else "--",
                    label=r["Distribucion"] + (" [DISEÑO]" if es_sel else ""))
    ax.scatter(Tr_emp, datos_ord, color="black", s=30, zorder=6, label="Observados (Weibull)")
    ax.axvline(Tr_diseno, color="gray", ls=":", lw=1.4, label=f"Tr = {Tr_diseno}")
    ax.set_title(f"Curvas de frecuencia | Est. {codigo}")
    ax.set_xlabel("Período de retorno (años)")
    ax.set_ylabel("Pmax24 (mm)")
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.legend(loc="upper left", fontsize=8); plt.tight_layout()
    return fig, "Curvas de frecuencia"


def fig_idf(idf_df: pd.DataFrame, titulo: str, codigo: str | int) -> tuple[Figure, str]:
    colores_Tr = ["#1a9641", "#78c679", "#addd8e", "#d9f0a3",
                  "#fd8d3c", "#f03b20", "#bd0026", "#7f0000", "#49006a"]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    Tr_lista = list(idf_df.columns)
    duraciones = list(idf_df.index)
    for i, Tr in enumerate(Tr_lista):
        ax.plot(duraciones, idf_df[Tr].values, color=colores_Tr[i % len(colores_Tr)],
                marker="o", ms=4, lw=1.8, label=f"Tr = {Tr}")
    ax.set_title(f"IDF — {titulo} | Est. {codigo}")
    ax.set_xlabel("Duración (min)"); ax.set_ylabel("Intensidad (mm/h)")
    ax.set_xticks(duraciones)
    ax.legend(title="Tr (años)", loc="upper right", fontsize=8)
    ax.set_xlim(min(duraciones) - 2, max(duraciones) + 2)
    ax.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    return fig, f"IDF — {titulo}"


def fig_comparacion_idf(idfs: dict[str, pd.DataFrame], Tr_a_comparar: list[int],
                        codigo: str | int) -> tuple[Figure, str]:
    """Compara hasta 6 métodos IDF en una misma figura para los Tr seleccionados."""
    fig, axes = plt.subplots(1, len(Tr_a_comparar), figsize=(5 * len(Tr_a_comparar), 5),
                              sharey=True)
    if len(Tr_a_comparar) == 1:
        axes = [axes]
    colores = ["#2c3e50", "#e74c3c", "#27ae60", "#8e44ad", "#f39c12", "#16a085", "#7f8c8d"]
    for j, Tr in enumerate(Tr_a_comparar):
        ax = axes[j]
        for idx, (nombre, df) in enumerate(idfs.items()):
            if Tr in df.columns:
                ax.plot(df.index, df[Tr].values, color=colores[idx % len(colores)],
                        marker="o", ms=3.5, lw=1.6, label=nombre)
        ax.set_title(f"Tr = {Tr} años")
        ax.set_xlabel("Duración (min)")
        if j == 0:
            ax.set_ylabel("Intensidad (mm/h)")
        ax.grid(True, ls="--", alpha=0.5)
        if j == len(Tr_a_comparar) - 1:
            ax.legend(fontsize=7, loc="upper right")
    fig.suptitle(f"Comparación entre métodos IDF | Est. {codigo}", fontsize=11)
    plt.tight_layout()
    return fig, "Comparación métodos IDF"


def fig_correlacion_E2(serie_p: pd.Series, serie_aux: pd.Series,
                       m: float, b: float, r: float,
                       nombre_aux: str) -> tuple[Figure, str]:
    anios = sorted(set(serie_p.dropna().index.year) & set(serie_aux.dropna().index.year))
    if not anios:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "Sin años comunes", ha="center", va="center")
        return fig, f"Correlación con {nombre_aux}"
    mp = serie_p.groupby(serie_p.index.year).max().loc[anios]
    ma = serie_aux.groupby(serie_aux.index.year).max().loc[anios]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(ma.values, mp.values, color="steelblue", s=55, zorder=3)
    x_reg = np.linspace(ma.min(), ma.max(), 100)
    ax.plot(x_reg, m * x_reg + b, "r-", lw=1.8,
            label=f"y = {m:.3f}·x + {b:.2f}\nr = {r:.4f},  R² = {r*r:.4f}")
    ax.set_title(f"Correlación principal vs {nombre_aux} ({len(anios)} años comunes)")
    ax.set_xlabel(f"Pmax24 {nombre_aux} (mm)"); ax.set_ylabel("Pmax24 principal (mm)")
    ax.legend(); plt.tight_layout()
    return fig, f"Correlación con {nombre_aux}"
