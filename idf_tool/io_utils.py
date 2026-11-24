"""
io_utils.py — Carga de archivos IDEAM y autodetección de unidades.

Resuelve el problema crónico: IDEAM a veces exporta precipitación en décimas
de mm pero con etiqueta 'mm'. Aquí se autodetecta antes de cualquier cálculo.
"""

from __future__ import annotations

import csv
import os
from typing import Any

import numpy as np
import pandas as pd


# ------------------------------------------------------------------------------
# Detección de codificación y delimitador
# ------------------------------------------------------------------------------

def detectar_delimitador_y_encoding(file_path: str) -> tuple[str, str]:
    """Detecta el delimitador y la codificación de un CSV."""
    with open(file_path, "rb") as f:
        raw = f.read(8192)
    encoding = "utf-8"
    decoded = ""
    for enc in ("utf-8", "latin1", "cp1252"):
        try:
            decoded = raw.decode(enc)
            encoding = enc
            break
        except UnicodeDecodeError:
            continue
    if not decoded:
        decoded = raw.decode("latin1", errors="replace")
        encoding = "latin1"
    try:
        sep = csv.Sniffer().sniff(decoded, delimiters=",;|\t").delimiter
    except csv.Error:
        sep = ";"  # IDEAM por defecto
    return sep, encoding


# ------------------------------------------------------------------------------
# Carga del CSV crudo
# ------------------------------------------------------------------------------

def cargar_csv_ideam(file_path: str, verbose: bool = True) -> pd.DataFrame:
    """
    Carga un CSV IDEAM crudo y normaliza columnas mínimas:
      fecha (datetime), valor (float)

    No filtra: solo carga y tipifica. El filtrado se hace en `cleaning`.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Archivo no encontrado: {file_path}")

    sep, enc = detectar_delimitador_y_encoding(file_path)
    if verbose:
        print(f"  · {os.path.basename(file_path)} | sep='{sep}' | enc='{enc}'")
    df = pd.read_csv(file_path, encoding=enc, sep=sep, low_memory=False)

    # normalización mínima: nombres en minúscula para tolerar variantes
    df.columns = [c.strip().lower() for c in df.columns]

    if "fecha" not in df.columns:
        raise ValueError(f"El archivo {file_path} no tiene columna 'fecha'.")
    if "valor" not in df.columns:
        raise ValueError(f"El archivo {file_path} no tiene columna 'valor'.")

    df["fecha"] = pd.to_datetime(df["fecha"], dayfirst=True, errors="coerce")
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")

    n_nan_f = df["fecha"].isna().sum()
    n_nan_v = df["valor"].isna().sum()
    if verbose and (n_nan_f or n_nan_v):
        print(f"    ! Fechas inválidas: {n_nan_f} | Valores no numéricos: {n_nan_v}")

    return df


# ------------------------------------------------------------------------------
# Autodetección de unidades (mm vs décimas de mm)
# ------------------------------------------------------------------------------

def autodetectar_factor_unidades(
    serie: pd.Series,
    rango_esperado_mm: tuple[float, float] = (30.0, 250.0),
    verbose: bool = True,
) -> tuple[float, dict[str, Any]]:
    """
    Autodetecta si una serie diaria de precipitación está en mm o décimas de mm.

    Estrategia (Bulletin 17C, sección 5 + Chow et al. 1988):
      - Calcula el máximo anual medio (M) usando los datos disponibles.
      - Si M cae dentro de `rango_esperado_mm` -> factor = 1.0 (mm)
      - Si M es ~10× superior al rango -> factor = 0.1 (décimas -> mm)
      - Caso ambiguo: emite advertencia y devuelve 1.0 (no transformar).

    Retorna (factor, diagnostico_dict).
    """
    s = pd.to_numeric(serie, errors="coerce").dropna()
    if len(s) == 0:
        return 1.0, {"diagnostico": "Serie vacía. Sin transformación."}

    # Calcular máximo anual medio si hay índice de fecha
    if isinstance(s.index, pd.DatetimeIndex):
        max_anual = s.groupby(s.index.year).max()
        media_max_anual = float(max_anual.mean()) if len(max_anual) else float(s.max())
    else:
        media_max_anual = float(s.max())

    rmin, rmax = rango_esperado_mm
    diag: dict[str, Any] = {
        "media_max_anual_original": round(media_max_anual, 2),
        "rango_esperado_mm"       : rango_esperado_mm,
    }

    if rmin <= media_max_anual <= rmax * 1.5:
        factor = 1.0
        diag["decision"] = "Datos en mm. Factor=1.0."
    elif rmin * 8 <= media_max_anual <= rmax * 15:
        factor = 0.1
        diag["decision"] = (
            f"Datos parecen estar en DÉCIMAS de mm "
            f"(media máx. anual {media_max_anual:.1f} ≈ 10× rango esperado). "
            f"Factor=0.1 aplicado."
        )
    elif media_max_anual > rmax * 15:
        factor = 0.01
        diag["decision"] = (
            f"Datos posiblemente en centésimas de mm "
            f"(media máx. anual {media_max_anual:.1f} ≈ 100× rango). "
            f"Factor=0.01 aplicado. VERIFICAR MANUALMENTE."
        )
    else:
        factor = 1.0
        diag["decision"] = (
            f"! Media máx. anual {media_max_anual:.1f} mm fuera de "
            f"rangos típicos. NO se aplica corrección. Verificar manualmente."
        )

    diag["factor_aplicado"] = factor
    diag["media_max_anual_corregida"] = round(media_max_anual * factor, 2)

    if verbose:
        print(f"    -> {diag['decision']}")
    return factor, diag


def aplicar_factor_unidades(
    df: pd.DataFrame,
    columna: str,
    factor: float | str,
    rango_esperado_mm: tuple[float, float],
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Aplica el factor de conversión. Si factor='auto', autodetecta primero.
    Asume que `df` ya tiene índice de fecha cuando es posible.
    """
    if factor == "auto":
        # Para autodetectar necesitamos serie con índice de fecha si existe
        if "fecha" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            tmp = df.set_index("fecha")[columna]
        else:
            tmp = df[columna] if not isinstance(df.index, pd.DatetimeIndex) else df[columna]
        factor_num, diag = autodetectar_factor_unidades(tmp, rango_esperado_mm, verbose)
    else:
        factor_num = float(factor)
        diag = {"factor_aplicado": factor_num, "decision": f"Factor manual={factor_num}"}
        if verbose:
            print(f"    -> Factor manual aplicado: ×{factor_num}")

    if factor_num != 1.0:
        df = df.copy()
        df[columna] = df[columna] * factor_num
    return df, diag


# ------------------------------------------------------------------------------
# Descubrimiento de archivos en carpeta de entrada
# ------------------------------------------------------------------------------

def listar_csvs_entrada(carpeta: str, archivo_principal: str | None = None) -> dict[str, list[str]]:
    """
    Lista los CSVs disponibles en la carpeta de entrada.

    Retorna:
      {'principal': ruta_o_None, 'auxiliares': [rutas]}

    Si `archivo_principal` está especificado, se busca por nombre; si no,
    se toma el primer CSV alfabético como principal y el resto como auxiliares.
    """
    if not os.path.isdir(carpeta):
        raise FileNotFoundError(f"Carpeta de entrada no existe: {carpeta}")
    csvs = sorted(
        f for f in os.listdir(carpeta)
        if f.lower().endswith(".csv") and not f.startswith(".")
    )
    rutas = {os.path.basename(f): os.path.join(carpeta, f) for f in csvs}
    principal = None
    auxiliares: list[str] = []

    if archivo_principal and archivo_principal in rutas:
        principal = rutas.pop(archivo_principal)
        auxiliares = list(rutas.values())
    elif csvs:
        primero = csvs[0]
        principal = rutas.pop(primero)
        auxiliares = list(rutas.values())

    return {"principal": principal, "auxiliares": auxiliares}
