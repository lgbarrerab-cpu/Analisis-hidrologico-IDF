"""
series.py — Construcción de serie de máximos anuales (AMS) y análisis de outliers.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import obtener_Kn


MESES_ES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
            "Jul", "Ago", "Set", "Oct", "Nov", "Dic"]


# ------------------------------------------------------------------------------
# Construcción de tabla mensual y AMS
# ------------------------------------------------------------------------------

def construir_tabla_mensual(
    df_diario: pd.DataFrame,
    columna: str = "precipitacion",
) -> pd.DataFrame:
    """
    Construye tabla anio × mes con el máximo diario por mes (Pmax mensual).
    Añade Pmax24 (máximo anual) y log10(Pmax24).
    """
    df = df_diario.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df_diario debe tener índice de fecha (DatetimeIndex).")

    df["_mes"]  = df.index.month
    df["_anio"] = df.index.year

    tabla = df.groupby(["_anio", "_mes"])[columna].max().unstack(level="_mes")
    # asegurar 12 columnas
    for m in range(1, 13):
        if m not in tabla.columns:
            tabla[m] = np.nan
    tabla = tabla[list(range(1, 13))]
    tabla.columns = MESES_ES
    tabla.index.name = "Año"

    tabla["Pmax24"] = tabla[MESES_ES].max(axis=1)
    tabla["Log(P24h)"] = np.log10(tabla["Pmax24"]).round(4)
    return tabla


def clasificar_anios(
    tabla_mensual: pd.DataFrame,
    min_meses: int = 6,
) -> dict[str, list[int]]:
    """Clasifica años en completos / incompletos / insuficientes."""
    meses_con_dato = tabla_mensual[MESES_ES].notna().sum(axis=1)
    return {
        "completos"    : list(meses_con_dato[meses_con_dato == 12].index.astype(int)),
        "incompletos"  : list(meses_con_dato[(meses_con_dato >= min_meses) & (meses_con_dato < 12)].index.astype(int)),
        "insuficientes": list(meses_con_dato[meses_con_dato < min_meses].index.astype(int)),
        "meses_por_anio": meses_con_dato.to_dict(),
    }


# ------------------------------------------------------------------------------
# Detección de outliers — Bulletin 17B/17C (Grubbs-Beck)
# ------------------------------------------------------------------------------

def detectar_outliers_grubbs_beck(serie_max: pd.Series) -> dict[str, Any]:
    """
    Grubbs-Beck en logaritmos base 10 (Bulletin 17B, USGS 1982; mantenido en 17C).

    Devuelve límites, outliers altos/bajos y Kn empleado.
    """
    s = pd.to_numeric(serie_max, errors="coerce").dropna()
    s = s[s > 0]  # log requiere positivos
    n = len(s)
    if n < 10:
        return {
            "n"           : n,
            "warning"     : "n < 10: análisis no recomendado por Bulletin 17C.",
            "lim_superior": float("nan"),
            "lim_inferior": float("nan"),
            "outliers_altos": [],
            "outliers_bajos": [],
            "Kn"          : float("nan"),
        }
    log_s = np.log10(s)
    media = float(log_s.mean())
    desv  = float(log_s.std(ddof=1))
    Kn    = obtener_Kn(n)

    log_sup = media + Kn * desv
    log_inf = media - Kn * desv
    lim_sup = 10 ** log_sup
    lim_inf = 10 ** log_inf

    altos = s[s > lim_sup]
    bajos = s[s < lim_inf]

    return {
        "n"             : n,
        "Kn"            : Kn,
        "media_log"     : round(media, 4),
        "desv_log"      : round(desv, 4),
        "lim_superior"  : round(float(lim_sup), 2),
        "lim_inferior"  : round(float(lim_inf), 2),
        "lim_sup_log"   : round(float(log_sup), 4),
        "lim_inf_log"   : round(float(log_inf), 4),
        "outliers_altos": [(int(a), float(v)) for a, v in altos.items()],
        "outliers_bajos": [(int(a), float(v)) for a, v in bajos.items()],
    }


def aplicar_tratamiento_outliers(
    tabla: pd.DataFrame,
    info_outliers: dict[str, Any],
    tratamiento: str = "conservar",
) -> tuple[pd.DataFrame, str]:
    """
    Aplica el tratamiento elegido a los outliers detectados.

    Devuelve (tabla_final, descripcion).
    """
    out = tabla.copy()
    if tratamiento == "conservar":
        return out, "Outliers conservados (no se modifican)."

    lim_sup = info_outliers.get("lim_superior", np.inf)
    lim_inf = info_outliers.get("lim_inferior", -np.inf)
    altos = [a for a, _ in info_outliers.get("outliers_altos", [])]
    bajos = [a for a, _ in info_outliers.get("outliers_bajos", [])]

    if tratamiento == "eliminar":
        out = out[(out["Pmax24"] >= lim_inf) & (out["Pmax24"] <= lim_sup)].copy()
        return out, f"Outliers eliminados ({len(altos) + len(bajos)} años removidos)."

    if tratamiento == "reemplazar":
        for a in altos:
            out.loc[a, "Pmax24"]    = round(lim_sup, 2)
            out.loc[a, "Log(P24h)"] = round(info_outliers["lim_sup_log"], 4)
        for a in bajos:
            out.loc[a, "Pmax24"]    = round(lim_inf, 2)
            out.loc[a, "Log(P24h)"] = round(info_outliers["lim_inf_log"], 4)
        return out, f"Outliers reemplazados por límites Bulletin 17B (altos={len(altos)}, bajos={len(bajos)})."

    return out, f"Tratamiento '{tratamiento}' no reconocido. Outliers conservados."


# ------------------------------------------------------------------------------
# Estadísticos descriptivos
# ------------------------------------------------------------------------------

def estadisticos_descriptivos(serie: pd.Series) -> dict[str, float]:
    """Estadísticos completos para la memoria de cálculo."""
    s = pd.to_numeric(serie, errors="coerce").dropna()
    n = len(s)
    if n == 0:
        return {}
    log_s = np.log10(s[s > 0])
    return {
        "N (años)"                 : n,
        "Media (mm)"               : round(float(s.mean()), 2),
        "Mediana (mm)"             : round(float(s.median()), 2),
        "Desviación estándar (mm)" : round(float(s.std(ddof=1)), 2),
        "Coef. variación (Cv)"     : round(float(s.std(ddof=1) / s.mean()), 4) if s.mean() else 0.0,
        "Coef. asimetría (Cs)"     : round(float(s.skew()), 4),
        "Curtosis (Ck)"            : round(float(s.kurtosis()), 4),
        "Mínimo (mm)"              : round(float(s.min()), 2),
        "Máximo (mm)"              : round(float(s.max()), 2),
        "Percentil 25 (mm)"        : round(float(s.quantile(0.25)), 2),
        "Percentil 75 (mm)"        : round(float(s.quantile(0.75)), 2),
        "Rango intercuartílico"    : round(float(s.quantile(0.75) - s.quantile(0.25)), 2),
        "Media log₁₀"              : round(float(log_s.mean()), 4),
        "Desv. estándar log₁₀"     : round(float(log_s.std(ddof=1)), 4),
        "Asimetría log₁₀"          : round(float(log_s.skew()), 4),
    }
