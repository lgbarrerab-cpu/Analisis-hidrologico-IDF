"""
distributions.py — Catálogo extendido de distribuciones probabilísticas.

Distribuciones implementadas (9):
  1. Normal               — N(μ, σ)
  2. Log-Normal           — ln(X) ~ N
  3. Gumbel (EV-I)        — Valores extremos tipo I, máximos
  4. GEV                  — Generalized Extreme Value (Jenkinson 1955, WMO recom.)
  5. Weibull              — Distribución de Weibull
  6. Gamma                — Pearson tipo III re-parametrizada
  7. Pearson III          — Original (sin logaritmo)
  8. Log-Pearson III      — Bulletin 17C estándar USGS
  9. Exponencial          — Exp(λ)

Bondad de ajuste:
  - AIC (Akaike 1974)
  - BIC (Schwarz 1978)
  - Kolmogorov-Smirnov (un-sample)
  - Anderson-Darling (un-sample) con valores críticos por distribución
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import (exponweib, expon, gamma, genextreme, gumbel_r,
                         kstest, lognorm, norm, pearson3, weibull_min)

from .config import AD_CRITICOS


# ------------------------------------------------------------------------------
# Catálogo de distribuciones
#   Estructura: nombre -> (objeto scipy, 'log' | 'real')
#   - 'log'  : el ajuste se hace sobre log10(X) y los cuantiles se "des-transforman"
#   - 'real' : ajuste y cuantiles en la escala original
# ------------------------------------------------------------------------------

CATALOGO_DISTRIBUCIONES: dict[str, tuple[Any, str]] = {
    "Normal"          : (norm,        "real"),
    "Log-Normal"      : (lognorm,     "real"),
    "Gumbel"          : (gumbel_r,    "real"),
    "GEV"             : (genextreme,  "real"),
    "Weibull"         : (weibull_min, "real"),
    "Gamma"           : (gamma,       "real"),
    "Pearson III"     : (pearson3,    "real"),
    "Log-Pearson III" : (pearson3,    "log"),
    "Exponencial"     : (expon,       "real"),
}

# Colores estables para gráficas
COLORES_DIST: dict[str, str] = {
    "Normal"          : "#8e44ad",
    "Log-Normal"      : "#27ae60",
    "Gumbel"          : "#e74c3c",
    "GEV"             : "#d35400",
    "Weibull"         : "#16a085",
    "Gamma"           : "#2980b9",
    "Pearson III"     : "#7f8c8d",
    "Log-Pearson III" : "steelblue",
    "Exponencial"     : "#c0392b",
}


# ------------------------------------------------------------------------------
# Cálculo de un cuantil P(Tr)
# ------------------------------------------------------------------------------

def cuantil(
    dist_obj: Any,
    params: tuple,
    escala: str,
    Tr: float,
) -> float:
    """Calcula P_T = ppf(1 - 1/Tr) con manejo de escala log/real."""
    p = 1.0 - 1.0 / Tr
    q = float(dist_obj.ppf(p, *params))
    if escala == "log":
        return float(10 ** q)
    return q


# ------------------------------------------------------------------------------
# Anderson-Darling (un-sample) — Stephens 1986
# ------------------------------------------------------------------------------

def anderson_darling_unsample(
    datos: np.ndarray,
    dist_obj: Any,
    params: tuple,
) -> float:
    """A² genérico para una muestra contra una distribución continua."""
    n = len(datos)
    if n < 4:
        return float("nan")
    u = dist_obj.cdf(np.sort(datos), *params)
    u = np.clip(u, 1e-10, 1.0 - 1e-10)
    i = np.arange(1, n + 1)
    A2 = -n - np.mean((2 * i - 1) * (np.log(u) + np.log(1 - u[::-1])))
    return float(A2)


# ------------------------------------------------------------------------------
# Ajuste y bondad para todas las distribuciones
# ------------------------------------------------------------------------------

def ajustar_y_evaluar(
    serie_max: pd.Series,
    distribuciones: list[str],
    Tr_lista: list[int],
    alpha: float = 0.05,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """
    Ajusta cada distribución, evalúa bondad de ajuste y devuelve:
      - lista de dicts con detalle por distribución (incluye objetos scipy)
      - DataFrame resumen ordenable
    """
    s = pd.to_numeric(serie_max, errors="coerce").dropna()
    s = s[s > 0]
    datos_real = s.values
    datos_log  = np.log10(datos_real)
    n_obs = len(datos_real)

    resultados: list[dict[str, Any]] = []
    for nombre in distribuciones:
        if nombre not in CATALOGO_DISTRIBUCIONES:
            continue
        dist_obj, escala = CATALOGO_DISTRIBUCIONES[nombre]
        datos = datos_log if escala == "log" else datos_real

        try:
            params = dist_obj.fit(datos)
        except Exception as exc:
            resultados.append({
                "Distribucion": nombre,
                "_error": f"Ajuste falló: {exc}",
            })
            continue

        k_params = len(params)
        try:
            loglik = float(np.sum(dist_obj.logpdf(datos, *params)))
        except Exception:
            loglik = float("nan")
        AIC = 2 * k_params - 2 * loglik
        BIC = k_params * np.log(n_obs) - 2 * loglik

        # Kolmogorov-Smirnov
        try:
            ks_stat, ks_p = kstest(datos, lambda x: dist_obj.cdf(x, *params))
        except Exception:
            ks_stat, ks_p = float("nan"), float("nan")
        ks_res = "OK" if (np.isfinite(ks_p) and ks_p > alpha) else "RECHAZA"

        # Anderson-Darling
        ad_stat = anderson_darling_unsample(datos, dist_obj, params)
        ad_crit = AD_CRITICOS.get(nombre, 2.492)
        ad_res = "OK" if np.isfinite(ad_stat) and ad_stat < ad_crit else "RECHAZA"

        # Cuantiles por Tr
        cuantiles = {Tr: round(cuantil(dist_obj, params, escala, Tr), 2) for Tr in Tr_lista}

        resultados.append({
            "Distribucion": nombre,
            "Escala"      : escala,
            "k_params"    : k_params,
            "loglik"      : round(loglik, 4),
            "AIC"         : round(AIC, 2),
            "BIC"         : round(BIC, 2),
            "KS_stat"     : round(float(ks_stat), 4),
            "KS_pvalor"   : round(float(ks_p), 4),
            "KS_resultado": ks_res,
            "AD_stat"     : round(float(ad_stat), 4),
            "AD_critico"  : ad_crit,
            "AD_resultado": ad_res,
            "cuantiles"   : cuantiles,
            "_dist_obj"   : dist_obj,
            "_params"     : params,
            "_escala"     : escala,
        })

    # DataFrame resumen
    cols = ["Distribucion", "AIC", "BIC", "KS_stat", "KS_pvalor", "KS_resultado",
            "AD_stat", "AD_critico", "AD_resultado"]
    rows = []
    for r in resultados:
        if "_error" in r:
            rows.append({c: (r["Distribucion"] if c == "Distribucion" else "—") for c in cols})
            continue
        d = {c: r[c] for c in cols}
        for Tr, v in r["cuantiles"].items():
            d[f"P24h_Tr{Tr}a"] = v
        rows.append(d)
    df = pd.DataFrame(rows)
    df = df.sort_values("AIC") if "AIC" in df.columns else df
    return resultados, df.reset_index(drop=True)


def seleccionar_distribucion(
    resultados: list[dict[str, Any]],
    seleccion: str = "auto",
    alpha: float = 0.05,
) -> dict[str, Any] | None:
    """
    Selecciona la distribución de diseño.

    'auto': prioriza distribuciones que no rechacen KS y AD; entre ellas, menor AIC.
            Si todas rechazan al menos un test -> menor AIC.
    nombre exacto: devuelve esa entrada si existe.
    """
    validas = [r for r in resultados if "_error" not in r]
    if not validas:
        return None

    if seleccion != "auto":
        for r in validas:
            if r["Distribucion"] == seleccion:
                return r
        return None  # no encontrada

    no_rechazan = [r for r in validas if r["KS_resultado"] == "OK" and r["AD_resultado"] == "OK"]
    if no_rechazan:
        return min(no_rechazan, key=lambda r: r["AIC"])
    return min(validas, key=lambda r: r["AIC"])


# ------------------------------------------------------------------------------
# Intervalos de confianza por bootstrap
# ------------------------------------------------------------------------------

def intervalos_confianza_bootstrap(
    serie_max: pd.Series,
    dist_obj: Any,
    escala: str,
    Tr_lista: list[int],
    n_bootstrap: int = 1000,
    nivel: float = 0.95,
    semilla: int = 42,
) -> dict[int, dict[str, float]]:
    """
    Intervalos de confianza por bootstrap no paramétrico (Efron 1979).
    Re-muestrea con reemplazo y re-ajusta la distribución en cada iteración.
    """
    rng = np.random.default_rng(semilla)
    s = pd.to_numeric(serie_max, errors="coerce").dropna()
    s = s[s > 0].values
    n = len(s)
    alfa = (1.0 - nivel) / 2.0

    ic: dict[int, dict[str, float]] = {}
    for Tr in Tr_lista:
        muestras = []
        for _ in range(n_bootstrap):
            boot = rng.choice(s, size=n, replace=True)
            try:
                datos = np.log10(boot) if escala == "log" else boot
                params_b = dist_obj.fit(datos)
                muestras.append(cuantil(dist_obj, params_b, escala, Tr))
            except Exception:
                continue
        if muestras:
            ic[Tr] = {
                "inf": float(np.percentile(muestras, alfa * 100)),
                "sup": float(np.percentile(muestras, (1 - alfa) * 100)),
            }
    return ic
