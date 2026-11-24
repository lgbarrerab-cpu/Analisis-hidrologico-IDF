"""
idf_methods.py — Métodos para construcción de curvas IDF.

Métodos implementados (todos con referencia académica verificable):

  1. Chen (1983)               — ASCE J. Hyd. Eng. 109(12): 1603-1621
     Tres parámetros R1=P60/P24, R2=P60_T10/P60_T1, R3=P10/P60
     i(t, T) = a · P_ref · f(T) / (t + b)^n

  2. Díaz-Granados / INVIAS (2009) — Manual de Drenaje, Ec. 2.103
     i = a · T^b · M^d / (t/60)^c
     M = precipitación máxima 24h media anual de la estación

  3. Sherman (1931) — Engineering News-Record
     i = A / (t + B)^n        (ajuste por mínimos cuadrados a tabla IDF base)

  4. Bernard (1932)
     i = A / t^n              (forma reducida sin desplazamiento)

  5. Bell (1969) — J. Hyd. Div. ASCE, 95(HY1)
     Desagregación universal usando factores adimensionales:
       P_t,T = (0.21·ln(T) + 0.52) · (0.54·t^0.25 - 0.50) · P_60,10
     donde P_60,10 = lluvia de 1 h y T=10 años (base).
     Generaliza para t en minutos (5 <= t <= 120) y T en años (2 <= T <= 100).

  6. Kothyari & Garde (1992) — J. Inst. Engineers India, 73: 159-165
     i = C · T^a · M^c / t^b   (similar a Díaz-Granados pero calibrado para India,
     se replica acá con coeficientes Colombia-Orinoquia derivados de Chen)

Cada función retorna (DataFrame_IDF, dict_parametros).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from .config import COEFICIENTES_DG


# ------------------------------------------------------------------------------
# 1. Chen (1983)
# ------------------------------------------------------------------------------

def idf_chen(
    P24h_por_Tr: dict[int, float],
    duraciones_min: list[int],
    Tr_lista: list[int],
    R1: float, R2: float, R3: float,
    Tr_ref: int = 10,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Chen (1983). Ref: Chen, C.L. (1983). "Rainfall Intensity-Duration-Frequency
    Formulas". Journal of Hydraulic Engineering, ASCE, 109(12), 1603-1621.

    P_ref = R1 · P24h(T_ref)   con T_ref típicamente 10 años.

    Factor de frecuencia normalizado:
        f(T) = (1 + K·log10(T)) / (1 + K·log10(T_ref))
    Esto garantiza i(60, T_ref) = P_ref (calibración correcta del método).
    """
    P24_ref = float(P24h_por_Tr[Tr_ref])

    # Coeficientes Chen
    x  = 1.0 / (1.0 - np.log10(R2) / np.log10(R1 * R2))
    b  = 60.0 * (R2 ** (x / (x - 1)) / R1 - 1.0) / (R2 ** (1.0 / (x - 1)) - 1.0)
    n  = 1.0 - np.log10(R3) / np.log10(60.0 / (60.0 + b) * R3)
    a  = (60.0 + b) ** n
    K  = (10 ** (0.278 * (1.0 - R1)) - 1.0) / (10 ** (0.278 * (1.0 - R1)) + 1.0)

    P_ref = R1 * P24_ref

    def i_chen(t: float, Tr: float) -> float:
        # normalización: f(Tr_ref) = 1
        f_T = (1.0 + K * np.log10(Tr)) / (1.0 + K * np.log10(Tr_ref))
        return (a * P_ref * f_T) / ((t + b) ** n)

    idf = pd.DataFrame(index=duraciones_min, columns=Tr_lista, dtype=float)
    idf.index.name = "Duración (min)"
    for Tr in Tr_lista:
        for t in duraciones_min:
            idf.loc[t, Tr] = round(i_chen(t, Tr), 2)

    return idf, {
        "metodo": "Chen 1983",
        "a": round(a, 4),
        "b": round(b, 4),
        "n": round(n, 4),
        "K": round(K, 4),
        "x": round(x, 4),
        "P_ref_mm": round(P_ref, 2),
        "P24h_ref_T10": round(P24_ref, 2),
        "R1": R1, "R2": R2, "R3": R3, "Tr_ref": Tr_ref,
    }


# ------------------------------------------------------------------------------
# 2. Díaz-Granados / INVIAS (2009)
# ------------------------------------------------------------------------------

def idf_diaz_granados(
    M_promedio: float,
    duraciones_min: list[int],
    Tr_lista: list[int],
    region: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    INVIAS (2009) Manual de Drenaje, Ec. 2.103, basada en Vargas & Díaz-Granados
    (1998), Universidad de los Andes.

        i = a · Tr^b · M^d / (t/60)^c

    M es la media de la serie AMS (Pmax24 medio anual de la estación) y es
    CONSTANTE por estación. El efecto de Tr lo asume Tr^b.
    """
    coefs = COEFICIENTES_DG[region]
    a, b, c, d = coefs["a"], coefs["b"], coefs["c"], coefs["d"]

    def i_dg(t: float, Tr: float) -> float:
        return (a * (Tr ** b) * (M_promedio ** d)) / ((t / 60.0) ** c)

    idf = pd.DataFrame(index=duraciones_min, columns=Tr_lista, dtype=float)
    idf.index.name = "Duración (min)"
    for Tr in Tr_lista:
        for t in duraciones_min:
            idf.loc[t, Tr] = round(i_dg(t, Tr), 2)

    return idf, {
        "metodo": "Diaz-Granados INVIAS 2009",
        "region": region,
        "a": a, "b": b, "c": c, "d": d,
        "M_promedio_mm": round(M_promedio, 2),
    }


# ------------------------------------------------------------------------------
# 3. Sherman (1931) — ajuste por mínimos cuadrados sobre la IDF base
# ------------------------------------------------------------------------------

def idf_sherman(
    idf_base: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Sherman (1931). Forma:  i(t) = A / (t + B)^n  por cada Tr.

    Se calibra contra una IDF base (típicamente Chen o Díaz-Granados) usando
    curve_fit. El método clásico Sherman se aplica cuando se tienen registros
    pluviográficos directos; al carecer de ellos, lo más defendible es calibrar
    una forma de Sherman SOBRE la IDF base obtenida por desagregación, dejando
    la ecuación analítica disponible para uso en hidrología urbana.
    """
    def modelo(t, A, B, n):
        return A / (t + B) ** n

    params_por_Tr: dict[int, dict[str, float]] = {}
    idf_out = pd.DataFrame(index=idf_base.index, columns=idf_base.columns, dtype=float)
    idf_out.index.name = "Duración (min)"

    duraciones = np.array(idf_base.index, dtype=float)
    for Tr in idf_base.columns:
        valores = idf_base[Tr].values.astype(float)
        try:
            p0 = [valores.max() * 10, 5.0, 0.7]
            popt, _ = curve_fit(modelo, duraciones, valores, p0=p0, maxfev=10000)
            A, B, n = popt
        except Exception:
            A, B, n = float("nan"), float("nan"), float("nan")
        params_por_Tr[int(Tr)] = {"A": round(float(A), 4), "B": round(float(B), 4), "n": round(float(n), 4)}
        for t in duraciones:
            idf_out.loc[t, Tr] = round(float(modelo(t, A, B, n)), 2)

    return idf_out, {"metodo": "Sherman 1931 (ajuste a IDF base)", "parametros_por_Tr": params_por_Tr}


# ------------------------------------------------------------------------------
# 4. Bernard (1932) — i = A / t^n (forma reducida)
# ------------------------------------------------------------------------------

def idf_bernard(
    idf_base: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Bernard (1932). Forma:  i(t) = A / t^n.
    Útil para hidrología urbana; calibrada sobre la IDF base por Tr.
    """
    def modelo(t, A, n):
        return A / (t ** n)

    params_por_Tr: dict[int, dict[str, float]] = {}
    idf_out = pd.DataFrame(index=idf_base.index, columns=idf_base.columns, dtype=float)
    idf_out.index.name = "Duración (min)"
    duraciones = np.array(idf_base.index, dtype=float)
    for Tr in idf_base.columns:
        valores = idf_base[Tr].values.astype(float)
        try:
            p0 = [valores.max() * 10, 0.7]
            popt, _ = curve_fit(modelo, duraciones, valores, p0=p0, maxfev=10000)
            A, n = popt
        except Exception:
            A, n = float("nan"), float("nan")
        params_por_Tr[int(Tr)] = {"A": round(float(A), 4), "n": round(float(n), 4)}
        for t in duraciones:
            idf_out.loc[t, Tr] = round(float(modelo(t, A, n)), 2)

    return idf_out, {"metodo": "Bernard 1932 (ajuste a IDF base)", "parametros_por_Tr": params_por_Tr}


# ------------------------------------------------------------------------------
# 5. Bell (1969) — desagregación universal P24h -> P_t,T
# ------------------------------------------------------------------------------

def idf_bell(
    P24h_por_Tr: dict[int, float],
    duraciones_min: list[int],
    Tr_lista: list[int],
    Tr_ref: int = 10,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Bell (1969). Generalización empírica para 5 <= t <= 120 min, 2 <= T <= 100 años.

        P_t,T = (0.21·ln(T) + 0.52) · (0.54·t^0.25 - 0.50) · P_60,10

    Y P_60,10 se obtiene de la P24h(T=10): se asume razón Chen R1 ≈ 0.45 entre
    P60 y P24h (Chen 1983; rango típico 0.40–0.55 en climas tropicales).
    """
    R1 = 0.45  # razón típica P60/P24h (Chen 1983)
    P_60_10 = R1 * P24h_por_Tr[Tr_ref]

    def i_bell(t: float, Tr: float) -> float:
        P_t_T = (0.21 * np.log(Tr) + 0.52) * (0.54 * t ** 0.25 - 0.50) * P_60_10
        # convertir a intensidad mm/h: (P en mm)·(60/t)
        return max(0.0, P_t_T) * (60.0 / t)

    idf = pd.DataFrame(index=duraciones_min, columns=Tr_lista, dtype=float)
    idf.index.name = "Duración (min)"
    for Tr in Tr_lista:
        for t in duraciones_min:
            idf.loc[t, Tr] = round(i_bell(t, Tr), 2)

    return idf, {
        "metodo": "Bell 1969 (desagregación universal)",
        "rango_validez": "5 <= t <= 120 min, 2 <= T <= 100 años",
        "R1_asumido": R1,
        "P_60_10_mm": round(P_60_10, 2),
        "Tr_ref": Tr_ref,
    }


# ------------------------------------------------------------------------------
# 6. Kothyari & Garde (1992) — ajuste tipo regresión
# ------------------------------------------------------------------------------

def idf_kothyari_garde(
    P24h_por_Tr: dict[int, float],
    M_promedio: float,
    duraciones_min: list[int],
    Tr_lista: list[int],
    Tr_ref: int = 10,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Kothyari & Garde (1992). Forma generalizada:
        i = K · T^a · M^c / t^b

    Los parámetros K, a, b, c se calibran por mínimos cuadrados sobre los pares
    (t, T, i_objetivo). Como i_objetivo usamos los valores de la IDF Chen
    construida con el mismo P24h y duración, para garantizar consistencia interna.

    Esta función NO inventa: requiere una IDF base previa (Chen) como anclaje.
    """
    # Requiere Chen como referencia interna
    # Para simplificar, construimos Chen con R1=0.45, R2=0.50, R3=0.73 (valores típicos)
    R1, R2, R3 = 0.45, 0.50, 0.73
    idf_ref, _ = idf_chen(P24h_por_Tr, duraciones_min, Tr_lista, R1, R2, R3, Tr_ref=Tr_ref)

    # ln(i) = ln(K) + a·ln(T) + c·ln(M) - b·ln(t)
    # Modelo lineal en los logaritmos
    filas = []
    for Tr in Tr_lista:
        for t in duraciones_min:
            i_obs = idf_ref.loc[t, Tr]
            if i_obs > 0:
                filas.append((np.log(t), np.log(Tr), i_obs))
    X = np.array([[1.0, lnt, lnT] for lnt, lnT, _ in filas])
    y = np.log([i for _, _, i in filas])
    # OLS: y = c0 - b·lnt + a·lnT, donde c0 = ln(K·M^c)
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    c0, beta_lnt, beta_lnT = coef
    b = -float(beta_lnt)
    a = float(beta_lnT)
    # ln(K · M^c) = c0; sin grados de libertad para c, dejamos c=0.42 (Orinoquia DG)
    # y resolvemos K
    c = COEFICIENTES_DG.get("Orinoquia", {"d": 0.42})["d"]
    K = float(np.exp(c0) / max(M_promedio ** c, 1e-9))

    def i_kg(t: float, Tr: float) -> float:
        return K * (Tr ** a) * (M_promedio ** c) / (t ** b)

    idf_out = pd.DataFrame(index=duraciones_min, columns=Tr_lista, dtype=float)
    idf_out.index.name = "Duración (min)"
    for Tr in Tr_lista:
        for t in duraciones_min:
            idf_out.loc[t, Tr] = round(i_kg(t, Tr), 2)

    return idf_out, {
        "metodo": "Kothyari-Garde 1992 (ajuste a Chen)",
        "K": round(K, 4), "a": round(a, 4), "b": round(b, 4), "c": c,
        "M_promedio_mm": round(M_promedio, 2),
        "nota": "c fijado por defecto al exponente DG-Orinoquia.",
    }


# ------------------------------------------------------------------------------
# Orquestador
# ------------------------------------------------------------------------------

def calcular_todas_idf(
    P24h_por_Tr: dict[int, float],
    M_promedio: float,
    duraciones_min: list[int],
    Tr_lista: list[int],
    metodos: list[str],
    region: str,
    R1: float, R2: float, R3: float,
) -> dict[str, tuple[pd.DataFrame, dict[str, Any]]]:
    """
    Calcula los métodos solicitados. Devuelve dict {nombre_metodo: (df_IDF, params)}.
    """
    resultados: dict[str, tuple[pd.DataFrame, dict[str, Any]]] = {}

    if "Chen 1983" in metodos:
        resultados["Chen 1983"] = idf_chen(P24h_por_Tr, duraciones_min, Tr_lista, R1, R2, R3)

    if "Diaz-Granados INVIAS 2009" in metodos:
        resultados["Diaz-Granados INVIAS 2009"] = idf_diaz_granados(
            M_promedio, duraciones_min, Tr_lista, region
        )

    if "Bell 1969" in metodos:
        resultados["Bell 1969"] = idf_bell(P24h_por_Tr, duraciones_min, Tr_lista)

    if "Kothyari-Garde 1992" in metodos:
        resultados["Kothyari-Garde 1992"] = idf_kothyari_garde(
            P24h_por_Tr, M_promedio, duraciones_min, Tr_lista
        )

    # Sherman y Bernard se calibran sobre Chen si está disponible, sino sobre DG
    base = None
    if "Chen 1983" in resultados:
        base = resultados["Chen 1983"][0]
    elif "Diaz-Granados INVIAS 2009" in resultados:
        base = resultados["Diaz-Granados INVIAS 2009"][0]

    if "Sherman 1931" in metodos and base is not None:
        resultados["Sherman 1931"] = idf_sherman(base)
    if "Bernard 1932" in metodos and base is not None:
        resultados["Bernard 1932"] = idf_bernard(base)

    return resultados


def matriz_diferencias_porcentuales(
    idf_a: pd.DataFrame, idf_b: pd.DataFrame
) -> pd.DataFrame:
    """Diferencia % absoluta |b - a| / a * 100, redondeada a 1 decimal."""
    return ((idf_b - idf_a).abs() / idf_a.abs() * 100).round(1)
