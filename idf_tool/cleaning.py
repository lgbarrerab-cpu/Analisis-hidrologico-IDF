"""
cleaning.py — Filtros técnicos y limpieza de datos IDEAM.

Aplica filtros estándar IDEAM, deduplica, elimina negativos, completa el índice
de fechas y filtra meses con baja completitud.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def aplicar_filtros_ideam(
    df: pd.DataFrame,
    codigo_estacion: int | str | None = None,
    variable: str = "PRECIPITACION",
    parametro: str = "Día Pluviométrico (convencional)",
    aprobaciones: list[str] | None = None,
    unidad: str = "mm",
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Filtra DataFrame IDEAM crudo. Devuelve df filtrado + diagnóstico.
    Solo aplica filtros sobre columnas que existan (tolerante con CSVs minimalistas).
    """
    diag: dict[str, Any] = {"registros_inicial": len(df)}
    aprobaciones = aprobaciones or ["Definitivo", "Preliminar"]

    if codigo_estacion is not None and "codigo_estacion" in df.columns:
        df = df[df["codigo_estacion"].astype(str) == str(codigo_estacion)]
        diag["tras_filtro_estacion"] = len(df)

    if "variable" in df.columns:
        df = df[df["variable"].astype(str).str.upper() == variable.upper()]
        diag["tras_filtro_variable"] = len(df)

    if "parametro" in df.columns:
        df = df[df["parametro"] == parametro]
        diag["tras_filtro_parametro"] = len(df)

    if "nivel_aprobacion" in df.columns:
        df = df[df["nivel_aprobacion"].isin(aprobaciones)]
        diag["tras_filtro_aprobacion"] = len(df)

    if "unidad" in df.columns:
        df = df[df["unidad"].astype(str).str.lower() == unidad.lower()]
        diag["tras_filtro_unidad"] = len(df)

    if verbose:
        print(f"    Filtros aplicados: {diag['registros_inicial']} -> {len(df)} registros")
    return df, diag


def limpiar_serie(
    df: pd.DataFrame,
    nombre_columna_valor: str = "valor",
    nuevo_nombre: str = "precipitacion",
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Deduplica por fecha, elimina negativos, ordena y deja índice de fecha.
    """
    diag: dict[str, Any] = {}
    df = df[["fecha", nombre_columna_valor]].copy()
    df.rename(columns={nombre_columna_valor: nuevo_nombre}, inplace=True)
    df = df.dropna(subset=["fecha"]).sort_values("fecha")

    duplicados = df.duplicated(subset=["fecha"]).sum()
    df = df.drop_duplicates(subset=["fecha"], keep="last")
    diag["duplicados_eliminados"] = int(duplicados)

    neg = (df[nuevo_nombre] < 0).sum()
    df = df[df[nuevo_nombre] >= 0]
    diag["negativos_eliminados"] = int(neg)

    df.set_index("fecha", inplace=True)
    diag["periodo_inicio"] = str(df.index.min().date()) if len(df) else None
    diag["periodo_fin"]    = str(df.index.max().date()) if len(df) else None
    diag["registros_finales"] = len(df)

    if verbose:
        print(f"    Duplicados: {duplicados} | Negativos: {neg} | "
              f"Periodo: {diag['periodo_inicio']} -> {diag['periodo_fin']}")
    return df, diag


def completar_continuidad_temporal(
    df: pd.DataFrame,
    columna: str = "precipitacion",
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Reindex con frecuencia diaria para detectar todos los días faltantes (NaN).
    """
    if len(df) == 0:
        return df, {"total_dias": 0, "faltantes": 0, "porcentaje_falt": 0.0}
    rango = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")
    df = df.reindex(rango)
    df.index.name = "fecha"
    total = len(df)
    falt = int(df[columna].isna().sum())
    diag = {
        "total_dias"     : total,
        "faltantes"      : falt,
        "con_dato"       : total - falt,
        "porcentaje_falt": round(falt / total * 100, 2),
    }
    if verbose:
        print(f"    Total días: {total} | Faltantes: {falt} ({diag['porcentaje_falt']}%)")
    return df, diag


def filtrar_meses_por_completitud(
    df: pd.DataFrame,
    columna: str = "precipitacion",
    max_dias_faltantes: int = 5,
    umbral_completitud: float = 0.85,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Marca como NaN los meses que no cumplen completitud mínima.

    Un mes se descarta si:
      - Tiene más de `max_dias_faltantes` días sin dato, O
      - Su completitud (días con dato / días del mes) < `umbral_completitud`

    Devuelve (df_filtrado, tabla_completitud, diagnostico).
    """
    df = df.copy()
    tmp = pd.DataFrame({
        "anio": df.index.year,
        "mes": df.index.month,
        "dato": df[columna].notna().astype(int),
    })
    comp = tmp.groupby(["anio", "mes"]).agg(
        con_dato=("dato", "sum"),
        total=("dato", "count"),
    ).reset_index()
    comp["faltantes"] = comp["total"] - comp["con_dato"]
    comp["completitud"] = comp["con_dato"] / comp["total"]

    malos = comp[
        (comp["faltantes"] > max_dias_faltantes) |
        (comp["completitud"] < umbral_completitud)
    ]
    for _, m in malos.iterrows():
        mask = (df.index.year == m["anio"]) & (df.index.month == m["mes"])
        df.loc[mask, columna] = np.nan

    diag = {
        "meses_descartados": len(malos),
        "umbral_completitud": umbral_completitud,
        "max_dias_faltantes": max_dias_faltantes,
    }
    if verbose:
        print(f"    Meses descartados por baja completitud: {len(malos)}")
    return df, comp, diag


def clasificar_serie_por_faltantes(porcentaje: float) -> str:
    """Clasificación WMO No. 168 / IDEAM."""
    if porcentaje <= 5:
        return "EXCELENTE (<= 5 %)"
    if porcentaje <= 10:
        return "ACEPTABLE (5–10 %)"
    if porcentaje <= 20:
        return "REGULAR (10–20 %)"
    if porcentaje <= 30:
        return "DEFICIENTE (20–30 %)"
    return "INACEPTABLE (> 30 %)"
