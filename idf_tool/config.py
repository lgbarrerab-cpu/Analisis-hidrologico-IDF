"""
config.py — Configuración centralizada y catálogos normativos.

CONFIG por defecto + carga desde YAML/JSON.
Tablas y catálogos hidrológicos verificables.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

# ------------------------------------------------------------------------------
# CONFIG por defecto
# ------------------------------------------------------------------------------

CONFIG_DEFAULT: dict[str, Any] = {
    # -- Carpetas (rutas) ------------------------------------------------------
    "carpeta_entrada"   : "./entrada",
    "carpeta_salida"    : "./salida",
    "subcarpetas_salida": ["datos", "graficas", "informe"],

    # -- Identificación de la estación principal (E1) --------------------------
    "archivo_principal" : "estacion_principal.csv",  # debe estar en carpeta_entrada
    "codigo_estacion"   : None,   # si None: se detecta del archivo
    "nombre_estacion"   : "",
    "entidad"           : "",
    "proyecto"          : "Análisis pluviométrico e IDF",

    # -- Estaciones auxiliares para completamiento -----------------------------
    "archivos_auxiliares": [],    # lista de nombres dentro de carpeta_entrada
    # coordenadas opcionales para IDW: {"codigo": (lat, lon, elev_msnm)}
    "coordenadas"        : {},

    # -- Validación de unidades ------------------------------------------------
    # "auto" -> autodetectar; 1.0 -> asumir mm; 0.1 -> asumir décimas de mm
    "factor_conversion": "auto",
    "rango_Pmax24_esperado_mm": (30, 250),   # validación regional

    # -- Filtros técnicos IDEAM ------------------------------------------------
    "filtro_variable"  : "PRECIPITACION",
    "filtro_parametro" : "Día Pluviométrico (convencional)",
    "filtro_aprobacion": ["Definitivo", "Preliminar"],
    "filtro_unidad"    : "mm",

    # -- Completamiento multi-estación -----------------------------------------
    "metodo_completamiento": "auto",
    # "auto" | "razon_normal" | "regresion_simple" | "regresion_multiple" | "idw"
    "min_anios_comunes": 10,
    "r_minimo"         : 0.70,
    "sesgo_max_pct"    : 15.0,

    # -- Completitud mensual y anual -------------------------------------------
    "min_meses_anio"        : 6,
    "max_dias_faltantes_mes": 5,
    "umbral_completitud_mes": 0.85,   # >= 85 % de días con dato/mes

    # -- Análisis de frecuencias -----------------------------------------------
    "Tr_diseno"        : 25,
    "Tr_lista_IDF"     : [2, 3, 5, 10, 25, 50, 100, 200, 500],
    "nivel_significancia": 0.05,
    # subset de DISTRIBUCIONES_CATALOGO a evaluar
    "distribuciones": [
        "Log-Pearson III", "Gumbel", "Log-Normal", "Normal",
        "Gamma", "GEV", "Weibull", "Pearson III", "Exponencial",
    ],
    "dist_seleccionada": "auto",  # "auto" | nombre exacto

    # -- Outliers (Bulletin 17B) -----------------------------------------------
    "tratamiento_outliers": "conservar",  # "conservar" | "eliminar" | "reemplazar"

    # -- IDF Chen (1983) -------------------------------------------------------
    "R1": 0.45, "R2": 0.50, "R3": 0.73,

    # -- IDF Díaz-Granados (INVIAS 2009) ---------------------------------------
    "region_IDF": "Orinoquia",  # Andina | Caribe | Pacifico | Orinoquia

    # -- Métodos IDF a calcular ------------------------------------------------
    "metodos_IDF": [
        "Chen 1983",
        "Diaz-Granados INVIAS 2009",
        "Sherman 1931",
        "Bernard 1932",
        "Bell 1969",
        "Kothyari-Garde 1992",
    ],

    # -- Duraciones IDF (min) --------------------------------------------------
    "duraciones_min": [5, 10, 15, 20, 30, 45, 60, 75, 90, 120, 150, 180],

    # -- Intervalos de confianza -----------------------------------------------
    "calcular_IC"       : True,
    "nivel_confianza_IC": 0.95,
    "n_bootstrap"       : 1000,
    "semilla_aleatoria" : 42,

    # -- Tests hidrológicos ----------------------------------------------------
    "ejecutar_tests": True,
    # Mann-Kendall (tendencia), Wald-Wolfowitz (independencia),
    # Pettitt (punto de cambio), Wilcoxon (homogeneidad)

    # -- Versionado y salidas --------------------------------------------------
    "versionado_timestamp": False,
    "generar_excel" : True,
    "generar_pdf"   : True,
    "generar_word"  : True,
    "verbose"       : True,
}


# ------------------------------------------------------------------------------
# CATÁLOGOS NORMATIVOS — todos con referencia académica
# ------------------------------------------------------------------------------

# Coeficientes Díaz-Granados / Vargas-Díaz-Granados (1998), adoptados por
# INVIAS Manual de Drenaje (2009) Ec. 2.103: i = a · Tr^b · M^d / (t/60)^c
COEFICIENTES_DG: dict[str, dict[str, float]] = {
    "Andina"   : {"a": 0.94,  "b": 0.18, "c": 0.66, "d": 0.83},
    "Caribe"   : {"a": 24.85, "b": 0.22, "c": 0.50, "d": 0.10},
    "Pacifico" : {"a": 13.92, "b": 0.19, "c": 0.58, "d": 0.20},
    "Orinoquia": {"a": 5.53,  "b": 0.17, "c": 0.63, "d": 0.42},
}

# Valores críticos Anderson-Darling, α=0.05 (Stephens 1986, J.Am.Stat.Assoc.)
AD_CRITICOS: dict[str, float] = {
    "Normal"         : 0.752,
    "Log-Normal"     : 0.752,
    "Gumbel"         : 0.757,
    "Gamma"          : 0.752,
    "Pearson III"    : 2.492,
    "Log-Pearson III": 2.492,
    "GEV"            : 0.757,
    "Weibull"        : 0.757,
    "Exponencial"    : 1.341,
}

# Tabla Kn — Bulletin 17B/17C, Appendix 4. Grubbs-Beck para detección outliers.
TABLA_KN: dict[int, float] = {
    10: 2.036, 11: 2.088, 12: 2.134, 13: 2.175, 14: 2.213,
    15: 2.247, 16: 2.279, 17: 2.309, 18: 2.335, 19: 2.361,
    20: 2.385, 21: 2.408, 22: 2.429, 23: 2.448, 24: 2.467,
    25: 2.486, 26: 2.502, 27: 2.519, 28: 2.534, 29: 2.549,
    30: 2.563, 31: 2.577, 32: 2.591, 33: 2.604, 34: 2.616,
    35: 2.628, 36: 2.639, 37: 2.650, 38: 2.661, 39: 2.671,
    40: 2.682, 41: 2.692, 42: 2.700, 43: 2.710, 44: 2.719,
    45: 2.727, 46: 2.736, 47: 2.744, 48: 2.753, 49: 2.760,
    50: 2.768, 55: 2.804, 60: 2.837, 65: 2.866, 70: 2.893,
    75: 2.917, 80: 2.940, 85: 2.961, 90: 2.981, 95: 3.000,
    100: 3.017, 110: 3.049, 120: 3.078, 130: 3.104, 140: 3.129,
}


def obtener_Kn(n: int) -> float:
    """
    Kn de Bulletin 17B/17C para detección de outliers.

    Para n > 140 se extrapola logarítmicamente. Para n < 10 se usa Kn(10) = 2.036
    (no se recomienda análisis de frecuencias con n < 10 — ver USGS Bulletin 17C
    sec. 7).
    """
    if n in TABLA_KN:
        return TABLA_KN[n]
    claves = sorted(TABLA_KN.keys())
    if n < claves[0]:
        return TABLA_KN[claves[0]]
    if n > claves[-1]:
        n1, n2 = claves[-2], claves[-1]
        k1, k2 = TABLA_KN[n1], TABLA_KN[n2]
        # extrapolación log-lineal
        return k1 + (k2 - k1) * (np.log(n) - np.log(n1)) / (np.log(n2) - np.log(n1))
    # interpolación lineal entre vecinos
    n_inf = max(k for k in claves if k <= n)
    n_sup = min(k for k in claves if k >= n)
    if n_inf == n_sup:
        return TABLA_KN[n_inf]
    f = (n - n_inf) / (n_sup - n_inf)
    return TABLA_KN[n_inf] + f * (TABLA_KN[n_sup] - TABLA_KN[n_inf])


# ------------------------------------------------------------------------------
# Carga / volcado / validación
# ------------------------------------------------------------------------------

def cargar_config(ruta: str | None = None) -> dict[str, Any]:
    """
    Carga configuración. Si la ruta no se entrega o no existe, devuelve
    CONFIG_DEFAULT. Soporta YAML (si PyYAML disponible) y JSON.
    """
    cfg = dict(CONFIG_DEFAULT)  # copia profunda no necesaria: solo escalares y listas

    if not ruta:
        return cfg
    if not os.path.exists(ruta):
        print(f"! Archivo de configuración no encontrado: {ruta}. Se usan valores por defecto.")
        return cfg

    ext = os.path.splitext(ruta)[1].lower()
    try:
        if ext in (".yaml", ".yml"):
            import yaml  # opcional
            with open(ruta, "r", encoding="utf-8") as f:
                user = yaml.safe_load(f) or {}
        else:
            with open(ruta, "r", encoding="utf-8") as f:
                user = json.load(f)
    except Exception as exc:
        raise RuntimeError(f"No se pudo leer la configuración '{ruta}': {exc}") from exc

    cfg.update(user)
    validar_config(cfg)
    return cfg


def validar_config(cfg: dict[str, Any]) -> None:
    """Chequeos básicos. Lanza ValueError si algo es inconsistente."""
    if cfg["Tr_diseno"] not in cfg["Tr_lista_IDF"]:
        cfg["Tr_lista_IDF"] = sorted(set(cfg["Tr_lista_IDF"]) | {cfg["Tr_diseno"]})

    for k in ("R1", "R2", "R3"):
        v = cfg[k]
        if not (0.10 <= v <= 0.90):
            print(f"! Chen {k}={v} fuera de rango típico [0.10, 0.90]. Verificar.")

    if cfg["region_IDF"] not in COEFICIENTES_DG:
        raise ValueError(
            f"region_IDF='{cfg['region_IDF']}' no válida. Opciones: {list(COEFICIENTES_DG)}"
        )

    if cfg["metodo_completamiento"] not in (
        "auto", "razon_normal", "regresion_simple", "regresion_multiple", "idw"
    ):
        raise ValueError(f"metodo_completamiento inválido: {cfg['metodo_completamiento']}")

    if cfg["tratamiento_outliers"] not in ("conservar", "eliminar", "reemplazar"):
        raise ValueError(f"tratamiento_outliers inválido: {cfg['tratamiento_outliers']}")


def preparar_salida(cfg: dict[str, Any]) -> dict[str, str]:
    """Crea carpetas de salida y devuelve rutas absolutas por categoría."""
    base = os.path.abspath(cfg["carpeta_salida"])
    if cfg["versionado_timestamp"]:
        base = os.path.join(base, datetime.now().strftime("run_%Y%m%d_%H%M%S"))
    os.makedirs(base, exist_ok=True)

    rutas = {"base": base}
    for sub in cfg["subcarpetas_salida"]:
        full = os.path.join(base, sub)
        os.makedirs(full, exist_ok=True)
        rutas[sub] = full
    return rutas


def guardar_config_efectiva(cfg: dict[str, Any], ruta_destino: str) -> None:
    """Guarda la CONFIG efectiva usada en el run (auditoría/trazabilidad)."""
    # Filtra elementos no-serializables si los hubiera
    serial = {}
    for k, v in cfg.items():
        try:
            json.dumps(v)
            serial[k] = v
        except TypeError:
            serial[k] = str(v)
    with open(ruta_destino, "w", encoding="utf-8") as f:
        json.dump(serial, f, indent=2, ensure_ascii=False)
