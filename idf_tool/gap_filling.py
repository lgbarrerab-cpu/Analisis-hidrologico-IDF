"""
gap_filling.py — Completamiento de datos faltantes con múltiples estaciones.

Métodos implementados (todos con base académica verificable):

  1. Razón normal (Paulhus & Kohler, 1952; WMO No. 168, sec. 3.5.1)
     P̂_x = (1/n) · Σ (N_x / N_i) · P_i

  2. Regresión lineal simple
     P̂_x = a + b · P_i, evaluada sólo con r >= r_min y sesgo aceptable

  3. Regresión lineal múltiple (OLS por mínimos cuadrados)
     P̂_x = a + Σ b_i · P_i

  4. Distancia inversa ponderada (IDW)
     P̂_x = Σ (w_i · P_i) / Σ w_i,   w_i = 1 / d_i^p
     d_i = distancia haversine (o euclidiana planar) a la estación auxiliar.

  5. Promedio mensual histórico (fallback cuando no hay auxiliares válidas)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# ------------------------------------------------------------------------------
# Evaluación de aptitud entre estaciones (correlación + sesgo)
# ------------------------------------------------------------------------------

def evaluar_aptitud_estacion(
    serie_principal: pd.Series,
    serie_auxiliar: pd.Series,
    min_anios_comunes: int = 10,
    r_minimo: float = 0.70,
    sesgo_max_pct: float = 15.0,
) -> dict[str, Any]:
    """
    Evalúa si una estación auxiliar puede usarse para completar la principal.

    Trabaja sobre los máximos anuales de cada serie (más robusto que diario
    para series IDEAM con muchos faltantes).
    """
    diag: dict[str, Any] = {"apto": False}

    # máximos anuales (más estables y comparables que diarios)
    m_p = serie_principal.groupby(serie_principal.index.year).max().dropna()
    m_a = serie_auxiliar.groupby(serie_auxiliar.index.year).max().dropna()
    comunes = sorted(set(m_p.index) & set(m_a.index))
    diag["anios_comunes"] = len(comunes)

    if len(comunes) < min_anios_comunes:
        diag["motivo"] = f"Insuficientes años comunes ({len(comunes)} < {min_anios_comunes})"
        return diag

    p = m_p.loc[comunes].values
    a = m_a.loc[comunes].values
    r = float(np.corrcoef(p, a)[0, 1])
    diag["r_pearson"] = round(r, 4)
    diag["R2"]        = round(r * r, 4)

    if r < r_minimo:
        diag["motivo"] = f"Correlación insuficiente (r={r:.4f} < {r_minimo})"
        return diag

    # regresión lineal P = m·A + b  (ordinaria)
    m, b = np.polyfit(a, p, 1)
    pred = m * a + b
    # sesgo medio relativo
    sesgo = float(np.mean((pred - p) / p) * 100)
    diag["regresion_m"] = round(m, 4)
    diag["regresion_b"] = round(b, 2)
    diag["sesgo_pct"]   = round(sesgo, 2)

    if abs(sesgo) > sesgo_max_pct:
        diag["motivo"] = f"Sesgo excesivo ({sesgo:.2f}% > +-{sesgo_max_pct}%)"
        return diag

    diag["apto"] = True
    diag["motivo"] = "Apta para completamiento"
    return diag


# ------------------------------------------------------------------------------
# Método 1: Razón normal (Paulhus & Kohler 1952)
# ------------------------------------------------------------------------------

def completar_razon_normal(
    serie_principal: pd.Series,
    series_auxiliares: list[pd.Series],
    media_principal: float | None = None,
    medias_auxiliares: list[float] | None = None,
    nombres_auxiliares: list[str] | None = None,
) -> tuple[pd.Series, dict[str, Any]]:
    """
    Método de razón normal — WMO No. 168 (2009), sec. 3.5.1.

    P̂_x(t) = (1/n) · Σ_i (N_x / N_i) · P_i(t)

    donde N_x = precipitación media anual de la estación principal
    y N_i = precipitación media anual de la auxiliar i.

    Se aplica solo donde la principal es NaN y todas las auxiliares tienen dato.
    """
    serie_out = serie_principal.copy()
    nombres_auxiliares = nombres_auxiliares or [f"AUX{i+1}" for i in range(len(series_auxiliares))]

    # Calcular medias anuales medias si no se entregan
    if media_principal is None:
        media_principal = float(
            serie_principal.dropna().groupby(serie_principal.dropna().index.year).sum().mean()
        )
    if medias_auxiliares is None:
        medias_auxiliares = []
        for s in series_auxiliares:
            ss = s.dropna()
            if len(ss):
                medias_auxiliares.append(float(ss.groupby(ss.index.year).sum().mean()))
            else:
                medias_auxiliares.append(np.nan)

    diag: dict[str, Any] = {
        "metodo": "razon_normal",
        "N_x": round(media_principal, 2),
        "N_i": [round(m, 2) for m in medias_auxiliares],
    }

    # Alinear todas las series por fecha
    df = pd.concat([serie_principal] + series_auxiliares, axis=1)
    df.columns = ["principal"] + nombres_auxiliares

    mask_falt = df["principal"].isna()
    rellenados = 0

    for fecha in df.index[mask_falt]:
        vals = df.loc[fecha, nombres_auxiliares]
        validos = vals.dropna()
        if len(validos) == 0:
            continue
        razones = []
        for nombre, v in validos.items():
            idx = nombres_auxiliares.index(nombre)
            ni = medias_auxiliares[idx]
            if ni and ni > 0:
                razones.append((media_principal / ni) * v)
        if razones:
            serie_out.loc[fecha] = float(np.mean(razones))
            rellenados += 1

    diag["rellenados"] = rellenados
    diag["restantes_faltantes"] = int(serie_out.isna().sum())
    return serie_out, diag


# ------------------------------------------------------------------------------
# Método 2: Regresión lineal simple (mejor auxiliar)
# ------------------------------------------------------------------------------

def completar_regresion_simple(
    serie_principal: pd.Series,
    serie_auxiliar: pd.Series,
    coef_m: float,
    coef_b: float,
) -> tuple[pd.Series, dict[str, Any]]:
    """
    P̂_x(t) = m · P_aux(t) + b, sólo donde P_x es NaN y P_aux tiene valor.
    Los coeficientes se calculan previamente con evaluar_aptitud_estacion().

    Alinea índices con concat para tolerar series de distinta longitud
    (caso típico en estaciones IDEAM con periodos de operación distintos).
    """
    df = pd.concat([serie_principal, serie_auxiliar], axis=1, join="outer")
    df.columns = ["principal", "aux"]
    out = df["principal"].copy()
    mask = out.isna() & df["aux"].notna()
    if mask.any():
        valores = np.maximum(0.0, coef_m * df.loc[mask, "aux"].values + coef_b)
        out.loc[mask] = valores
    # Conservar sólo el rango de la serie principal original
    out = out.reindex(serie_principal.index)
    return out, {
        "metodo"    : "regresion_simple",
        "m"         : round(float(coef_m), 4),
        "b"         : round(float(coef_b), 2),
        "rellenados": int(mask.sum()),
    }


# ------------------------------------------------------------------------------
# Método 3: Regresión lineal múltiple (OLS)
# ------------------------------------------------------------------------------

def completar_regresion_multiple(
    serie_principal: pd.Series,
    series_auxiliares: list[pd.Series],
    nombres_auxiliares: list[str] | None = None,
) -> tuple[pd.Series, dict[str, Any]]:
    """
    Regresión múltiple OLS sobre días con dato simultáneo en todas las series.
    Modelo: P_x = a + Σ b_i · P_i + ε

    Aplicable cuando hay 2+ auxiliares y suficiente solape simultáneo (>30 días).
    """
    nombres_auxiliares = nombres_auxiliares or [f"AUX{i+1}" for i in range(len(series_auxiliares))]
    df = pd.concat([serie_principal] + series_auxiliares, axis=1)
    df.columns = ["principal"] + nombres_auxiliares

    train = df.dropna()
    diag: dict[str, Any] = {
        "metodo"       : "regresion_multiple",
        "n_entrenamiento": len(train),
        "auxiliares"   : nombres_auxiliares,
    }
    if len(train) < 30:
        diag["error"] = f"Insuficiente solape ({len(train)} días) para regresión múltiple."
        return serie_principal, diag

    X = train[nombres_auxiliares].values
    y = train["principal"].values
    X_aum = np.column_stack([np.ones(len(X)), X])
    # solución por mínimos cuadrados
    coef, _, _, _ = np.linalg.lstsq(X_aum, y, rcond=None)
    a = float(coef[0])
    bs = coef[1:].astype(float)

    # bondad: R² ajustado
    pred_train = X_aum @ coef
    ss_res = float(np.sum((y - pred_train) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    k = X.shape[1]
    n = len(y)
    r2_aj = 1 - (1 - r2) * (n - 1) / max(1, (n - k - 1))
    diag["R2"]        = round(r2, 4)
    diag["R2_ajustado"] = round(r2_aj, 4)
    diag["a"]         = round(a, 4)
    diag["b"]         = [round(b, 4) for b in bs]

    # Predicción: completar donde principal=NaN y todas las auxiliares tienen dato
    mask_complete = df["principal"].isna() & df[nombres_auxiliares].notna().all(axis=1)
    out = df["principal"].copy()
    if mask_complete.any():
        Xpred = df.loc[mask_complete, nombres_auxiliares].values
        Xpred_aum = np.column_stack([np.ones(len(Xpred)), Xpred])
        ypred = Xpred_aum @ coef
        out.loc[mask_complete] = np.maximum(0.0, ypred)
    # Conservar el rango de la serie principal original
    out = out.reindex(serie_principal.index)
    diag["rellenados"] = int(mask_complete.sum())
    return out, diag


# ------------------------------------------------------------------------------
# Método 4: Distancia inversa ponderada (IDW)
# ------------------------------------------------------------------------------

def distancia_haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distancia haversine en km entre dos coordenadas (lat, lon en grados)."""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def completar_idw(
    serie_principal: pd.Series,
    series_auxiliares: list[pd.Series],
    coord_principal: tuple[float, float],
    coord_auxiliares: list[tuple[float, float]],
    nombres_auxiliares: list[str] | None = None,
    potencia: float = 2.0,
) -> tuple[pd.Series, dict[str, Any]]:
    """
    IDW — Inverse Distance Weighting.

    w_i = 1 / d_i^p ;   P̂_x(t) = Σ w_i · P_i(t) / Σ w_i

    coord_principal y coord_auxiliares son (lat, lon) en grados decimales.
    """
    nombres_auxiliares = nombres_auxiliares or [f"AUX{i+1}" for i in range(len(series_auxiliares))]
    distancias = [
        distancia_haversine(*coord_principal, *c) for c in coord_auxiliares
    ]
    # evita división por cero (estación duplicada)
    pesos_base = np.array([1.0 / max(d, 0.01) ** potencia for d in distancias])

    df = pd.concat([serie_principal] + series_auxiliares, axis=1)
    df.columns = ["principal"] + nombres_auxiliares

    out = serie_principal.copy()
    mask_falt = df["principal"].isna()
    rellenados = 0
    for fecha in df.index[mask_falt]:
        vals = df.loc[fecha, nombres_auxiliares].values.astype(float)
        validos = ~np.isnan(vals)
        if validos.sum() == 0:
            continue
        w = pesos_base[validos]
        v = vals[validos]
        out.loc[fecha] = float(np.sum(w * v) / np.sum(w))
        rellenados += 1

    return out, {
        "metodo": "idw",
        "distancias_km": [round(d, 2) for d in distancias],
        "potencia": potencia,
        "rellenados": rellenados,
    }


# ------------------------------------------------------------------------------
# Orquestador automático
# ------------------------------------------------------------------------------

def completar_automatico(
    serie_principal: pd.Series,
    auxiliares_aptas: dict[str, dict[str, Any]],
    series_aux: dict[str, pd.Series],
    coord_principal: tuple[float, float] | None = None,
    coord_aux: dict[str, tuple[float, float]] | None = None,
    metodo_forzado: str = "auto",
) -> tuple[pd.Series, list[dict[str, Any]]]:
    """
    Orquesta la selección de método según los datos disponibles.

    Estrategia:
      - Si hay 2+ auxiliares aptas y solape >= 30 días -> regresión múltiple
      - Si hay 1 auxiliar apta -> regresión simple
      - Si hay 2+ auxiliares aptas pero solape insuficiente -> razón normal
      - Si hay coordenadas -> IDW como respaldo
    """
    historial: list[dict[str, Any]] = []
    aptas = [(n, d) for n, d in auxiliares_aptas.items() if d.get("apto")]
    n_aptas = len(aptas)
    if n_aptas == 0:
        historial.append({"decision": "No hay auxiliares aptas. No se aplica completamiento multi-estación."})
        return serie_principal, historial

    if metodo_forzado != "auto":
        metodo_elegido = metodo_forzado
    elif n_aptas >= 2:
        metodo_elegido = "regresion_multiple"
    else:
        metodo_elegido = "regresion_simple"

    historial.append({"metodo_elegido": metodo_elegido, "n_auxiliares_aptas": n_aptas})

    if metodo_elegido == "regresion_simple":
        nombre, info = aptas[0]
        out, diag = completar_regresion_simple(
            serie_principal, series_aux[nombre], info["regresion_m"], info["regresion_b"]
        )
        diag["aux_usada"] = nombre
        historial.append(diag)
        return out, historial

    if metodo_elegido == "regresion_multiple":
        nombres = [n for n, _ in aptas]
        out, diag = completar_regresion_multiple(
            serie_principal, [series_aux[n] for n in nombres], nombres
        )
        if "error" in diag:
            historial.append(diag)
            # fallback a razón normal
            out, diag2 = completar_razon_normal(
                serie_principal, [series_aux[n] for n in nombres],
                nombres_auxiliares=nombres,
            )
            historial.append(diag2)
            return out, historial
        historial.append(diag)
        return out, historial

    if metodo_elegido == "razon_normal":
        nombres = [n for n, _ in aptas]
        out, diag = completar_razon_normal(
            serie_principal, [series_aux[n] for n in nombres],
            nombres_auxiliares=nombres,
        )
        historial.append(diag)
        return out, historial

    if metodo_elegido == "idw":
        if not coord_principal or not coord_aux:
            historial.append({"error": "IDW requiere coordenadas. Operación cancelada."})
            return serie_principal, historial
        nombres = [n for n, _ in aptas if n in coord_aux]
        out, diag = completar_idw(
            serie_principal, [series_aux[n] for n in nombres],
            coord_principal, [coord_aux[n] for n in nombres], nombres_auxiliares=nombres,
        )
        historial.append(diag)
        return out, historial

    historial.append({"error": f"Método no implementado: {metodo_elegido}"})
    return serie_principal, historial


# ------------------------------------------------------------------------------
# Completamiento de máximos anuales por promedio mensual histórico
# ------------------------------------------------------------------------------

def completar_mensual_por_promedio(
    tabla_mensual: pd.DataFrame,
    columnas_meses: list[str],
    anios_completos: list[int],
    anios_a_completar: list[int],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Rellena meses faltantes en años parcialmente incompletos usando el promedio
    histórico de los años completos. Método transparente y conservador.

    NO altera los años "insuficientes" — esos se descartan aparte.
    """
    tabla = tabla_mensual.copy()
    promedios = tabla.loc[anios_completos, columnas_meses].mean()
    rellenados: list[tuple[int, str, float]] = []
    for anio in anios_a_completar:
        for mes in columnas_meses:
            if pd.isna(tabla.loc[anio, mes]):
                v = round(float(promedios[mes]), 2)
                tabla.loc[anio, mes] = v
                rellenados.append((int(anio), mes, v))
    return tabla, {
        "n_celdas_rellenadas": len(rellenados),
        "detalle"            : rellenados[:50],  # primeras 50
        "promedios_mensuales": {m: round(float(promedios[m]), 2) for m in columnas_meses},
    }
