"""
tests.py — Tests hidrológicos sobre la serie de máximos anuales (AMS).

Requeridos por Bulletin 17C (USGS 2019) y WMO No. 168 (2009) antes de cualquier
ajuste de distribución probabilística:

  1. Independencia            — Wald-Wolfowitz (runs test) + autocorrelación lag-1
  2. Tendencia/Estacionariedad — Mann-Kendall (no paramétrico)
  3. Homogeneidad              — Wilcoxon rank-sum (subseries primera/segunda mitad)
  4. Punto de cambio           — Pettitt (cambio en mediana)

Si alguno rechaza, NO se puede asumir el supuesto de Identical & Independent
Distribution (i.i.d.) y el análisis paramétrico clásico pierde validez.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


# ------------------------------------------------------------------------------
# 1. Wald-Wolfowitz runs test (independencia)
# ------------------------------------------------------------------------------

def test_wald_wolfowitz(serie: pd.Series, alpha: float = 0.05) -> dict[str, Any]:
    """
    Test de rachas de Wald-Wolfowitz (1940).
    H0: la serie es independiente (rachas aleatorias respecto a la mediana).
    """
    s = pd.to_numeric(serie, errors="coerce").dropna().values
    n = len(s)
    if n < 10:
        return {"test": "Wald-Wolfowitz", "n": n, "warning": "n < 10, no aplicable"}

    md = float(np.median(s))
    # rachas: secuencia binaria sobre/bajo la mediana
    binarios = (s > md).astype(int)
    # remover empates con la mediana (s == md) puede sesgar; los omitimos
    binarios = binarios[s != md]
    n_efectivo = len(binarios)
    if n_efectivo < 10:
        return {"test": "Wald-Wolfowitz", "n": n_efectivo, "warning": "Muchos empates con la mediana"}

    n1 = int((binarios == 1).sum())
    n2 = int((binarios == 0).sum())
    rachas = 1 + int(np.sum(np.diff(binarios) != 0))

    # aproximación normal (válida para n1, n2 >= 10)
    mu = 2 * n1 * n2 / (n1 + n2) + 1
    var = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / (((n1 + n2) ** 2) * (n1 + n2 - 1))
    z = (rachas - mu) / np.sqrt(var) if var > 0 else 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return {
        "test"     : "Wald-Wolfowitz (runs)",
        "n"        : n_efectivo,
        "rachas"   : rachas,
        "z"        : round(float(z), 4),
        "p_valor"  : round(float(p), 4),
        "resultado": "ACEPTA H0 (independiente)" if p > alpha else "RECHAZA H0 (no independiente)",
        "alpha"    : alpha,
    }


def autocorrelacion_lag1(serie: pd.Series, alpha: float = 0.05) -> dict[str, Any]:
    """
    Coeficiente de autocorrelación lag-1 (Anderson, 1942).
    Para n grande: ICs ≈ +-1.96/√n (lo aceptamos si |r1| < zα/2 / √n).
    """
    s = pd.to_numeric(serie, errors="coerce").dropna().values
    n = len(s)
    if n < 10:
        return {"test": "Autocorrelación lag-1", "n": n, "warning": "n < 10"}
    r1 = float(np.corrcoef(s[:-1], s[1:])[0, 1])
    z_crit = stats.norm.ppf(1 - alpha / 2) / np.sqrt(n)
    return {
        "test"      : "Autocorrelación lag-1",
        "n"         : n,
        "r1"        : round(r1, 4),
        "limite_ic" : round(float(z_crit), 4),
        "resultado" : "ACEPTA H0 (independiente)" if abs(r1) < z_crit else "RECHAZA H0 (autocorrelación)",
        "alpha"     : alpha,
    }


# ------------------------------------------------------------------------------
# 2. Mann-Kendall (tendencia)
# ------------------------------------------------------------------------------

def test_mann_kendall(serie: pd.Series, alpha: float = 0.05) -> dict[str, Any]:
    """
    Mann-Kendall (Mann 1945, Kendall 1948) — test no paramétrico de tendencia.
    H0: no hay tendencia monotónica.
    """
    s = pd.to_numeric(serie, errors="coerce").dropna().values
    n = len(s)
    if n < 10:
        return {"test": "Mann-Kendall", "n": n, "warning": "n < 10"}

    S = 0
    for i in range(n - 1):
        S += int(np.sum(np.sign(s[i + 1:] - s[i])))

    # Varianza (sin corrección por empates para simplicidad; conservador)
    var_S = n * (n - 1) * (2 * n + 5) / 18.0
    # Z de signo (corrige cero)
    if S > 0:
        Z = (S - 1) / np.sqrt(var_S)
    elif S < 0:
        Z = (S + 1) / np.sqrt(var_S)
    else:
        Z = 0.0
    p = 2 * (1 - stats.norm.cdf(abs(Z)))

    # Pendiente de Sen (1968)
    pares = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            pares.append((s[j] - s[i]) / (j - i))
    sen = float(np.median(pares)) if pares else 0.0

    tend = "creciente" if Z > 0 else "decreciente" if Z < 0 else "nula"
    return {
        "test"          : "Mann-Kendall",
        "n"             : n,
        "S"             : int(S),
        "Z"             : round(float(Z), 4),
        "p_valor"       : round(float(p), 4),
        "pendiente_Sen" : round(sen, 4),
        "tendencia"     : tend,
        "resultado"     : "ACEPTA H0 (sin tendencia)" if p > alpha else f"RECHAZA H0 (tendencia {tend})",
        "alpha"         : alpha,
    }


# ------------------------------------------------------------------------------
# 3. Wilcoxon rank-sum (homogeneidad)
# ------------------------------------------------------------------------------

def test_wilcoxon_homogeneidad(serie: pd.Series, alpha: float = 0.05) -> dict[str, Any]:
    """
    Wilcoxon-Mann-Whitney sobre primera vs segunda mitad de la serie.
    H0: las dos submuestras provienen de la misma distribución (serie homogénea).
    """
    s = pd.to_numeric(serie, errors="coerce").dropna().values
    n = len(s)
    if n < 12:
        return {"test": "Wilcoxon rank-sum", "n": n, "warning": "n < 12"}

    mitad = n // 2
    g1 = s[:mitad]
    g2 = s[mitad:]
    stat, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
    return {
        "test"     : "Wilcoxon rank-sum (Mann-Whitney U)",
        "n1"       : len(g1),
        "n2"       : len(g2),
        "U"        : round(float(stat), 4),
        "p_valor"  : round(float(p), 4),
        "resultado": "ACEPTA H0 (homogénea)" if p > alpha else "RECHAZA H0 (no homogénea)",
        "alpha"    : alpha,
    }


# ------------------------------------------------------------------------------
# 4. Pettitt (punto de cambio)
# ------------------------------------------------------------------------------

def test_pettitt(serie: pd.Series, alpha: float = 0.05) -> dict[str, Any]:
    """
    Pettitt (1979) — test no paramétrico para detectar un punto de cambio
    en la mediana de la serie.
    """
    s = pd.to_numeric(serie, errors="coerce").dropna().values
    n = len(s)
    if n < 12:
        return {"test": "Pettitt", "n": n, "warning": "n < 12"}

    # Cálculo del estadístico U_t,n
    U = np.zeros(n)
    for t in range(n):
        suma = 0
        for i in range(t + 1):
            for j in range(t + 1, n):
                suma += int(np.sign(s[i] - s[j]))
        U[t] = suma
    K = int(np.max(np.abs(U)))
    t_cambio = int(np.argmax(np.abs(U)))
    # aproximación de p-valor (Pettitt 1979, Ec. 7)
    p = 2 * np.exp(-6 * K ** 2 / (n ** 3 + n ** 2))
    p = min(max(float(p), 0.0), 1.0)
    return {
        "test"             : "Pettitt",
        "n"                : n,
        "K"                : K,
        "t_cambio"         : t_cambio,
        "p_valor"          : round(p, 4),
        "resultado"        : "ACEPTA H0 (sin punto de cambio)" if p > alpha else f"RECHAZA H0 (cambio en posición {t_cambio})",
        "alpha"            : alpha,
    }


# ------------------------------------------------------------------------------
# Ejecución conjunta + dictamen
# ------------------------------------------------------------------------------

def ejecutar_bateria_tests(serie: pd.Series, alpha: float = 0.05) -> dict[str, Any]:
    """
    Ejecuta los cuatro tests hidrológicos y devuelve un dictamen integral.
    """
    resultados = {
        "wald_wolfowitz"   : test_wald_wolfowitz(serie, alpha),
        "autocorrelacion"  : autocorrelacion_lag1(serie, alpha),
        "mann_kendall"     : test_mann_kendall(serie, alpha),
        "wilcoxon"         : test_wilcoxon_homogeneidad(serie, alpha),
        "pettitt"          : test_pettitt(serie, alpha),
    }

    rechazos = [
        n for n, r in resultados.items()
        if isinstance(r.get("resultado"), str) and "RECHAZA" in r["resultado"]
    ]
    if not rechazos:
        dictamen = (
            "Serie aceptable como i.i.d. — apta para análisis de frecuencias paramétrico."
        )
    else:
        dictamen = (
            f"! Tests con rechazo de H0: {rechazos}. "
            "El supuesto i.i.d. está en duda — el análisis paramétrico debe interpretarse "
            "con cautela. Considerar análisis no estacionario (Salas et al. 2018) si la "
            "tendencia es severa."
        )

    resultados["dictamen"] = dictamen
    resultados["tests_rechazados"] = rechazos
    return resultados
