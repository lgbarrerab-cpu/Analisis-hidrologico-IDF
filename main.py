"""
main.py — Orquestador CLI para análisis hidrológico e IDF.

Uso:
    python main.py --config config.yaml
    python main.py --carpeta-entrada ./entrada --carpeta-salida ./salida
    python main.py --carpeta-entrada ./entrada --principal estacion_X.csv

Pipeline (10 fases):
    1. Carga de configuración + descubrimiento de CSVs
    2. Lectura cruda + autodetección de unidades (mm vs décimas)
    3. Filtros IDEAM + limpieza + continuidad temporal
    4. Evaluación de aptitud de estaciones auxiliares
    5. Completamiento multi-estación (regresión múltiple/simple/razón normal/IDW)
    6. Tabla de máximos mensuales + tratamiento de outliers (Grubbs-Beck)
    7. Tests hidrológicos i.i.d. (Wald-Wolfowitz, Mann-Kendall, Wilcoxon, Pettitt)
    8. Ajuste de 9 distribuciones + bondad (KS + Anderson-Darling) + IC bootstrap
    9. 6 métodos IDF (Chen, Diaz-Granados, Sherman, Bernard, Bell, Kothyari-Garde)
   10. Gráficas + Excel + PDF + Word

Referencias normativas implementadas:
    - USGS Bulletin 17C (2018) — análisis de frecuencias de inundación
    - WMO-No. 168 (2009) — Guide to Hydrological Practices
    - INVIAS (2009) — Manual de Drenaje, Ec. 2.103 Díaz-Granados
    - RAS 2017 (Resol. 0330/2017 MVCT) — diseño de redes pluviales
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from datetime import datetime

import numpy as np
import pandas as pd

# permitir ejecutar desde la raíz del paquete
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ------------------------------------------------------------------------------
# Conversión automática Excel → CSV (formato IDEAM)
# ------------------------------------------------------------------------------

def _detectar_columna(df: pd.DataFrame, candidatos: list[str]) -> str | None:
    """Devuelve el primer nombre de columna (case-insensitive) que coincida."""
    mapa = {c.lower().replace(" ", "").replace("_", ""): c for c in df.columns}
    for cand in candidatos:
        key = cand.lower().replace(" ", "").replace("_", "")
        if key in mapa:
            return mapa[key]
    return None


def _diagnostico_unidades(valores: pd.Series, rango_mm: tuple[float, float] = (30.0, 350.0)) -> dict:
    """
    Analiza si los valores parecen estar en décimas de mm.

    Criterio: la media de los máximos anuales debería caer dentro de rango_mm.
    Si la media supera rango_mm[1], se asume décimas → factor = 0.1.
    Si la media supera rango_mm[1] * 10 y factor 0.1 sigue alto → factor = 0.01.
    Si la media ya está dentro del rango → factor = 1.0.

    Imprime un resumen detallado para trazabilidad.
    """
    positivos = valores[valores > 0].dropna()
    if positivos.empty:
        return {"factor_aplicado": 1.0, "decision": "Sin valores positivos — sin diagnóstico",
                "media_maximos": float("nan")}

    # Agrupar por año y tomar el máximo
    s = positivos.copy()
    s.index = pd.to_datetime(s.index)
    maximos_anuales = s.groupby(s.index.year).max()
    media = float(maximos_anuales.mean())
    lo, hi = rango_mm

    print(f"  -> Diagnóstico de unidades:")
    print(f"       Media de máximos anuales (brutos) : {media:.2f}")
    print(f"       Rango esperado región              : {lo:.1f} – {hi:.1f} mm")

    if lo <= media <= hi:
        factor = 1.0
        decision = f"Valores en mm. Media={media:.2f} dentro del rango esperado."
    elif lo * 10 <= media <= hi * 10:
        factor = 0.1
        decision = (f"Posibles décimas de mm. Media={media:.2f} ≈ 10× rango. "
                    f"Factor aplicado: 0.1 → media corregida ≈ {media*0.1:.2f} mm")
    elif lo * 100 <= media <= hi * 100:
        factor = 0.01
        decision = (f"Posibles centésimas de mm. Media={media:.2f} ≈ 100× rango. "
                    f"Factor aplicado: 0.01 → media corregida ≈ {media*0.01:.2f} mm")
    elif media < lo:
        factor = 1.0
        decision = (f"Media={media:.2f} por debajo del rango esperado. "
                    "Puede indicar serie incompleta. Factor conservado en 1.0.")
    else:
        factor = 0.1
        decision = (f"Media={media:.2f} supera ampliamente el rango. "
                    "Factor 0.1 aplicado como mejor estimación.")

    print(f"       Decisión                           : {decision}")
    if factor != 1.0:
        print(f"       ⚠  Se aplicará factor = {factor} a todos los valores.")
    else:
        print(f"       ✔  Unidades correctas. No se aplica conversión.")

    return {
        "factor_aplicado": factor,
        "decision": decision,
        "media_maximos": media,
        "media_corregida": round(media * factor, 2),
    }


def _convertir_excel_a_csv(ruta_xlsx: str, ruta_csv_destino: str,
                            rango_mm: tuple[float, float] = (30.0, 350.0)) -> str:
    """
    Lee un archivo Excel con formato de descarga IDEAM (cualquier orden de columnas),
    aplica diagnóstico de unidades, escribe un CSV estándar y devuelve su ruta.

    Columnas buscadas (tolerante a variantes de nombre):
        - Código estación : CodigoEstacion / Codigo / CodigoSensor
        - Nombre estación : NombreEstacion / Nombre / Estacion
        - Variable        : Variable
        - Parámetro       : Parametro / Parametrootros
        - Fecha           : Fecha / Date / FechaHora
        - Unidad          : Unidad / Unidades
        - Valor           : Valor / Value / Dato
        - Aprobación      : NivelAprobacion / Aprobacion / Estado
    """
    print(f"\n  [Excel→CSV] Leyendo: {os.path.basename(ruta_xlsx)}")

    try:
        import openpyxl  # noqa: F401
    except ImportError:
        raise ImportError(
            "openpyxl no está instalado. Ejecuta: pip install openpyxl"
        )

    # Leer todas las hojas; usar la primera que tenga datos
    xls = pd.ExcelFile(ruta_xlsx, engine="openpyxl")
    df_raw = None
    for hoja in xls.sheet_names:
        tmp = pd.read_excel(xls, sheet_name=hoja, header=0)
        if len(tmp) > 0:
            df_raw = tmp
            print(f"  [Excel→CSV] Hoja usada: '{hoja}' "
                  f"({len(df_raw)} filas × {len(df_raw.columns)} cols)")
            break

    if df_raw is None or df_raw.empty:
        raise ValueError(f"El archivo Excel '{ruta_xlsx}' no contiene datos.")

    print(f"  [Excel→CSV] Columnas encontradas: {list(df_raw.columns)}")

    # Mapear columnas (tolerante a mayúsculas / variantes)
    col_fecha  = _detectar_columna(df_raw, ["Fecha", "FechaHora", "Date", "FECHA"])
    col_valor  = _detectar_columna(df_raw, ["Valor", "Value", "Dato", "VALOR"])
    col_codigo = _detectar_columna(df_raw, ["CodigoEstacion", "Codigo", "CodigoSensor",
                                             "CODIGOESTACION", "CODIGO"])
    col_nombre = _detectar_columna(df_raw, ["NombreEstacion", "Nombre", "Estacion",
                                             "NOMBREESTACION", "NOMBRE"])
    col_var    = _detectar_columna(df_raw, ["Variable", "VARIABLE"])
    col_param  = _detectar_columna(df_raw, ["Parametro", "PARAMETRO"])
    col_unidad = _detectar_columna(df_raw, ["Unidad", "Unidades", "UNIDAD"])
    col_aprob  = _detectar_columna(df_raw, ["NivelAprobacion", "Aprobacion", "Estado",
                                             "NIVELAPROBACION"])

    if col_fecha is None:
        raise ValueError(
            "No se encontró columna de Fecha en el Excel. "
            f"Columnas disponibles: {list(df_raw.columns)}"
        )
    if col_valor is None:
        raise ValueError(
            "No se encontró columna de Valor en el Excel. "
            f"Columnas disponibles: {list(df_raw.columns)}"
        )

    # Construir DataFrame limpio
    df = pd.DataFrame()
    df["fecha"]  = pd.to_datetime(df_raw[col_fecha], errors="coerce")
    df["valor"]  = pd.to_numeric(df_raw[col_valor], errors="coerce")

    # Columnas opcionales (el CSV IDEAM las necesita para los filtros de Fase 3)
    df["CodigoEstacion"]   = df_raw[col_codigo].astype(str)  if col_codigo else "SIN_CODIGO"
    df["NombreEstacion"]   = df_raw[col_nombre].astype(str)  if col_nombre else "SIN_NOMBRE"
    df["Variable"]         = df_raw[col_var].astype(str)     if col_var    else "PRECIPITACION"
    df["Parametro"]        = df_raw[col_param].astype(str)   if col_param  else "Día Pluviométrico (convencional)"
    df["Unidad"]           = df_raw[col_unidad].astype(str)  if col_unidad else "mm"
    df["NivelAprobacion"]  = df_raw[col_aprob].astype(str)   if col_aprob  else "Definitivo"

    # Eliminar filas sin fecha o sin valor
    antes = len(df)
    df = df.dropna(subset=["fecha", "valor"])
    descartadas = antes - len(df)
    if descartadas:
        print(f"  [Excel→CSV] {descartadas} fila(s) descartadas por fecha/valor nulo.")

    # Diagnóstico de unidades sobre los valores brutos
    serie_tmp = df.set_index("fecha")["valor"]
    diag = _diagnostico_unidades(serie_tmp, rango_mm=rango_mm)
    factor = diag["factor_aplicado"]

    if factor != 1.0:
        df["valor"] = (df["valor"] * factor).round(2)

    # Escribir CSV en formato que io_utils.cargar_csv_ideam espera
    df_csv = pd.DataFrame({
        "CodigoEstacion" : df["CodigoEstacion"],
        "NombreEstacion" : df["NombreEstacion"],
        "Variable"       : df["Variable"],
        "Parametro"      : df["Parametro"],
        "Fecha"          : df["fecha"].dt.strftime("%Y-%m-%d"),
        "Unidad"         : df["Unidad"],
        "Valor"          : df["valor"],
        "NivelAprobacion": df["NivelAprobacion"],
    })

    df_csv.to_csv(ruta_csv_destino, index=False, encoding="utf-8")
    print(f"  [Excel→CSV] CSV generado: {os.path.basename(ruta_csv_destino)}")
    print(f"  [Excel→CSV] Filas exportadas: {len(df_csv)} | "
          f"Factor aplicado: {factor}")
    return ruta_csv_destino


def _convertir_excels_en_carpeta(carpeta: str,
                                  rango_mm: tuple[float, float] = (30.0, 350.0)) -> list[str]:
    """
    Escanea `carpeta` en busca de archivos .xlsx / .xls y los convierte
    a CSV en la misma carpeta. Devuelve lista de rutas CSV generados.
    """
    generados = []
    for fname in os.listdir(carpeta):
        ext = os.path.splitext(fname)[1].lower()
        if ext in (".xlsx", ".xls"):
            ruta_excel = os.path.join(carpeta, fname)
            ruta_csv   = os.path.join(carpeta,
                                       os.path.splitext(fname)[0] + "_convertido.csv")
            if os.path.exists(ruta_csv):
                print(f"  [Excel→CSV] CSV ya existe, reutilizando: "
                      f"{os.path.basename(ruta_csv)}")
                generados.append(ruta_csv)
                continue
            try:
                csv_path = _convertir_excel_a_csv(ruta_excel, ruta_csv, rango_mm)
                generados.append(csv_path)
            except Exception as exc:
                print(f"  ! Error convirtiendo {fname}: {exc}")
    return generados

from idf_tool import (
    cleaning,
    config as cfg_mod,
    distributions as dist_mod,
    gap_filling,
    idf_methods,
    io_utils,
    plotting,
    reports,
    series as series_mod,
    tests as tests_mod,
)


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def banner(texto: str) -> None:
    print(f"\n{'=' * 70}\n  {texto}\n{'=' * 70}")


def encabezado_fase(n: int, total: int, texto: str) -> None:
    print(f"\n[Fase {n}/{total}] {texto}\n{'-' * 70}")


# ------------------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------------------

def ejecutar_pipeline(cfg: dict) -> int:
    TOTAL_FASES = 10
    t0 = datetime.now()
    contexto: dict = {}

    # -- Fase 1: configuración + descubrimiento --------------------------------
    encabezado_fase(1, TOTAL_FASES, "Configuración y descubrimiento de archivos")
    rutas = cfg_mod.preparar_salida(cfg)
    print(f"  -> Carpeta entrada: {os.path.abspath(cfg['carpeta_entrada'])}")
    print(f"  -> Carpeta salida : {rutas['base']}")
    cfg_mod.guardar_config_efectiva(cfg, os.path.join(rutas["base"], "config_efectiva.json"))

    # Conversión automática Excel → CSV antes de buscar archivos
    # Detecta cualquier .xlsx / .xls en la carpeta de entrada,
    # aplica diagnóstico de unidades (mm vs décimas) y genera el CSV equivalente.
    rango_mm = tuple(cfg.get("rango_Pmax24_esperado_mm", [30.0, 350.0]))
    csvs_generados = _convertir_excels_en_carpeta(cfg["carpeta_entrada"], rango_mm=rango_mm)
    if csvs_generados:
        print(f"  -> Excel(s) convertidos a CSV: {[os.path.basename(p) for p in csvs_generados]}")
        # Si el usuario no especificó archivo principal y hay un solo Excel convertido,
        # usarlo automáticamente como principal
        if not cfg.get("archivo_principal") and len(csvs_generados) == 1:
            cfg["archivo_principal"] = os.path.basename(csvs_generados[0])
            print(f"  -> Archivo principal detectado automáticamente: {cfg['archivo_principal']}")

    inventario = io_utils.listar_csvs_entrada(cfg["carpeta_entrada"], cfg.get("archivo_principal"))
    if not inventario["principal"]:
        print("X No se encontró CSV principal. Verifica la carpeta y/o `archivo_principal`.")
        return 2
    print(f"  -> Principal  : {os.path.basename(inventario['principal'])}")
    print(f"  -> Auxiliares : {len(inventario['auxiliares'])} archivo(s)")
    for aux in inventario["auxiliares"]:
        print(f"      · {os.path.basename(aux)}")

    # -- Fase 2: lectura + unidades --------------------------------------------
    encabezado_fase(2, TOTAL_FASES, "Lectura cruda y autodetección de unidades (mm vs décimas)")
    df_principal_raw = io_utils.cargar_csv_ideam(inventario["principal"], verbose=True)
    codigo = cfg.get("codigo_estacion") or _detectar_codigo(df_principal_raw, inventario["principal"])
    nombre_e1 = cfg.get("nombre_estacion") or str(codigo)
    print(f"  -> Estación principal: {codigo} — {nombre_e1}")

    # Si el archivo principal fue convertido desde Excel en Fase 1,
    # el diagnóstico y corrección de unidades ya fue aplicado durante la conversión.
    # En ese caso el factor está embebido en los datos del CSV y no debe aplicarse de nuevo.
    archivo_principal_nombre = os.path.basename(inventario["principal"])
    fue_convertido_de_excel = archivo_principal_nombre.endswith("_convertido.csv")

    df_principal_raw["fecha"] = pd.to_datetime(df_principal_raw["fecha"], errors="coerce")
    df_principal_raw = df_principal_raw.dropna(subset=["fecha", "valor"]).copy()
    tmp = df_principal_raw.set_index("fecha")["valor"]

    if fue_convertido_de_excel:
        # Unidades ya corregidas en la conversión — solo documentar, no volver a aplicar
        factor = 1.0
        diag_unid = {
            "decision": "Archivo convertido desde Excel en Fase 1. "
                        "Diagnóstico y corrección de unidades ya aplicados.",
            "factor_aplicado": 1.0,
        }
        print(f"  -> Unidades: corrección ya aplicada en conversión Excel→CSV (factor histórico conservado).")
    elif cfg["factor_conversion"] == "auto":
        factor, diag_unid = io_utils.autodetectar_factor_unidades(
            tmp, tuple(cfg["rango_Pmax24_esperado_mm"]), verbose=True
        )
    else:
        factor = float(cfg["factor_conversion"])
        diag_unid = {
            "decision": f"Factor manual={cfg['factor_conversion']}",
            "factor_aplicado": float(cfg["factor_conversion"]),
        }

    if factor != 1.0:
        df_principal_raw["valor"] = df_principal_raw["valor"] * factor
    contexto["diagnostico_unidades"] = diag_unid

    # -- Fase 3: filtros IDEAM + limpieza + continuidad ------------------------
    encabezado_fase(3, TOTAL_FASES, "Filtros técnicos IDEAM, limpieza y continuidad temporal")
    df_principal, _ = cleaning.aplicar_filtros_ideam(
        df_principal_raw,
        variable=cfg["filtro_variable"],
        parametro=cfg["filtro_parametro"],
        aprobaciones=cfg["filtro_aprobacion"],
        unidad=cfg["filtro_unidad"],
        verbose=True,
    )
    df_principal, _ = cleaning.limpiar_serie(df_principal, verbose=True)
    df_principal, diag_cont = cleaning.completar_continuidad_temporal(df_principal, verbose=True)

    # df_principal ahora tiene índice fecha y columna 'precipitacion'
    serie_principal = pd.to_numeric(df_principal["precipitacion"], errors="coerce")
    serie_principal.index = pd.to_datetime(serie_principal.index)
    print(f"  -> Serie diaria: {len(serie_principal)} días | "
          f"{serie_principal.index.min().date()} -> {serie_principal.index.max().date()}")
    pct_faltantes = float(diag_cont.get("porcentaje_falt", serie_principal.isna().mean() * 100))
    print(f"  -> Faltantes globales: {pct_faltantes:.2f}% "
          f"({cleaning.clasificar_serie_por_faltantes(pct_faltantes)})")

    # -- Fase 4: evaluación de auxiliares --------------------------------------
    encabezado_fase(4, TOTAL_FASES, "Evaluación de aptitud de estaciones auxiliares")
    series_aux: dict[str, pd.Series] = {}
    aptitud: dict[str, dict] = {}
    for ruta_aux in inventario["auxiliares"]:
        try:
            df_aux_raw = io_utils.cargar_csv_ideam(ruta_aux, verbose=False)
            df_aux_raw["fecha"] = pd.to_datetime(df_aux_raw["fecha"], errors="coerce")
            df_aux_raw = df_aux_raw.dropna(subset=["fecha", "valor"]).copy()
            # mismo factor que la principal (las series IDEAM suelen tener la misma codificación)
            tmp_a = df_aux_raw.set_index("fecha")["valor"]
            fa, _ = (io_utils.autodetectar_factor_unidades(
                tmp_a, tuple(cfg["rango_Pmax24_esperado_mm"]), verbose=False
            ) if cfg["factor_conversion"] == "auto"
                else (float(cfg["factor_conversion"]), {}))
            if fa != 1.0:
                df_aux_raw["valor"] = df_aux_raw["valor"] * fa

            df_aux, _ = cleaning.aplicar_filtros_ideam(
                df_aux_raw,
                variable=cfg["filtro_variable"],
                parametro=cfg["filtro_parametro"],
                aprobaciones=cfg["filtro_aprobacion"],
                unidad=cfg["filtro_unidad"],
                verbose=False,
            )
            df_aux, _ = cleaning.limpiar_serie(df_aux, verbose=False)
            df_aux, _ = cleaning.completar_continuidad_temporal(df_aux, verbose=False)
            s_aux = pd.to_numeric(df_aux["precipitacion"], errors="coerce")
            s_aux.index = pd.to_datetime(s_aux.index)
            cod_aux = _detectar_codigo(df_aux_raw, ruta_aux)
            series_aux[str(cod_aux)] = s_aux

            diag = gap_filling.evaluar_aptitud_estacion(
                serie_principal, s_aux,
                min_anios_comunes=cfg["min_anios_comunes"],
                r_minimo=cfg["r_minimo"],
                sesgo_max_pct=cfg["sesgo_max_pct"],
            )
            aptitud[str(cod_aux)] = diag
            status = "v APTA" if diag.get("apto") else "✗ DESCARTADA"
            print(f"  {status} | {os.path.basename(ruta_aux)} -> {diag.get('motivo','-')} | "
                  f"r={diag.get('r_pearson','-')} | n_años={diag.get('anios_comunes','-')}")
        except Exception as exc:
            print(f"  ! Error procesando {os.path.basename(ruta_aux)}: {exc}")
    contexto["resumen_E2"] = aptitud

    # -- Fase 5: completamiento multi-estación ---------------------------------
    encabezado_fase(5, TOTAL_FASES, "Completamiento multi-estación")
    coord_principal = None
    coord_aux_dict = None
    coords = cfg.get("coordenadas", {}) or {}
    if str(codigo) in coords:
        v = coords[str(codigo)]
        coord_principal = (float(v[0]), float(v[1]))
    if coords:
        coord_aux_dict = {k: (float(v[0]), float(v[1])) for k, v in coords.items() if k != str(codigo)}

    serie_completada, historial_compl = gap_filling.completar_automatico(
        serie_principal,
        auxiliares_aptas=aptitud,
        series_aux=series_aux,
        coord_principal=coord_principal,
        coord_aux=coord_aux_dict,
        metodo_forzado=cfg["metodo_completamiento"],
    )
    contexto["historial_completamiento"] = historial_compl
    for h in historial_compl:
        print(f"  · {h}")

    # -- Fase 6: tabla mensual + outliers --------------------------------------
    encabezado_fase(6, TOTAL_FASES, "Tabla de máximos mensuales y outliers (Grubbs-Beck 17B)")
    df_completa = pd.DataFrame({"precipitacion": serie_completada.values},
                               index=pd.to_datetime(serie_completada.index))
    df_completa.index.name = "fecha"
    tabla_mensual = series_mod.construir_tabla_mensual(df_completa, "precipitacion")
    clasif = series_mod.clasificar_anios(tabla_mensual, min_meses=cfg["min_meses_anio"])

    # completar máximos mensuales por promedio histórico para años incompletos
    tabla_definitiva, diag_relleno = gap_filling.completar_mensual_por_promedio(
        tabla_mensual,
        columnas_meses=series_mod.MESES_ES,
        anios_completos=clasif["completos"],
        anios_a_completar=clasif["incompletos"],
    )
    # recalcular Pmax24 y Log tras el relleno mensual
    tabla_definitiva["Pmax24"]    = tabla_definitiva[series_mod.MESES_ES].max(axis=1)
    tabla_definitiva["Log(P24h)"] = np.log10(tabla_definitiva["Pmax24"]).round(4)
    # descartar años insuficientes
    tabla_definitiva = tabla_definitiva.drop(index=clasif["insuficientes"], errors="ignore")
    print(f"  -> Relleno mensual: {diag_relleno['n_celdas_rellenadas']} celdas en "
          f"{len(clasif['incompletos'])} año(s) incompleto(s); "
          f"{len(clasif['insuficientes'])} año(s) descartado(s)")

    serie_pmax_anual = tabla_definitiva["Pmax24"].dropna()
    serie_pmax_anual = serie_pmax_anual[serie_pmax_anual > 0]
    print(f"  -> Años con Pmax24 anual: {len(serie_pmax_anual)} "
          f"(completos: {len(clasif['completos'])}, completados: {len(clasif['incompletos'])})")

    info_outliers = series_mod.detectar_outliers_grubbs_beck(serie_pmax_anual)
    print(f"  -> Outliers (log10, Kn={info_outliers.get('Kn',0):.3f}): "
          f"altos={len(info_outliers.get('outliers_altos',[]))} | "
          f"bajos={len(info_outliers.get('outliers_bajos',[]))}")

    tabla_tratada, descripcion_outliers = series_mod.aplicar_tratamiento_outliers(
        tabla_definitiva, info_outliers, tratamiento=cfg["tratamiento_outliers"],
    )
    print(f"  -> {descripcion_outliers}")
    serie_pmax_tratada = tabla_tratada["Pmax24"].dropna()
    serie_pmax_tratada = serie_pmax_tratada[serie_pmax_tratada > 0]

    estadisticos = series_mod.estadisticos_descriptivos(serie_pmax_tratada)
    contexto["tabla_mensual"]        = tabla_mensual
    contexto["tabla_definitiva"]     = tabla_tratada
    contexto["estadisticos"]         = estadisticos
    contexto["outliers_info"]        = info_outliers
    contexto["tratamiento_outliers"] = cfg["tratamiento_outliers"]
    contexto["serie_pmax"]           = serie_pmax_tratada
    # actualizar tabla_definitiva para fases siguientes (gráficas)
    tabla_definitiva = tabla_tratada

    # -- Fase 7: tests hidrológicos --------------------------------------------
    encabezado_fase(7, TOTAL_FASES, "Tests hidrológicos i.i.d. (Bulletin 17C / WMO 168)")
    if cfg["ejecutar_tests"] and len(serie_pmax_tratada) >= 10:
        bateria = tests_mod.ejecutar_bateria_tests(serie_pmax_tratada, alpha=cfg["nivel_significancia"])
        contexto["tests_hidrologicos"] = bateria
        for nombre, res in bateria.items():
            if isinstance(res, dict) and "resultado" in res:
                print(f"  · {nombre:25s} -> {res['resultado']}")
            elif nombre == "dictamen":
                print(f"  -> Dictamen i.i.d.: {res}")
    else:
        print("  ! Tests omitidos (n<10 o `ejecutar_tests=False`).")
        contexto["tests_hidrologicos"] = {}

    # -- Fase 8: ajuste de distribuciones + IC ---------------------------------
    encabezado_fase(8, TOTAL_FASES, "Ajuste de distribuciones (9) + bondad + IC bootstrap")
    np.random.seed(int(cfg.get("semilla_aleatoria", 42)))
    resultados_dist, df_bondad = dist_mod.ajustar_y_evaluar(
        serie_pmax_tratada,
        distribuciones=cfg["distribuciones"],
        Tr_lista=cfg["Tr_lista_IDF"],
        alpha=cfg["nivel_significancia"],
    )
    print(df_bondad.to_string(index=False))

    elegida = dist_mod.seleccionar_distribucion(
        resultados_dist, seleccion=cfg.get("dist_seleccionada", "auto"),
        alpha=cfg["nivel_significancia"],
    )
    if elegida is None:
        print("X Ninguna distribución se pudo ajustar. Abortando antes de IDF.")
        return 3
    print(f"  -> Distribución elegida: {elegida['Distribucion']} "
          f"(AIC={elegida['AIC']}, KS={elegida['KS_resultado']}, AD={elegida['AD_resultado']})")

    # P24h por Tr según la elegida
    P24h_por_Tr = {Tr: float(elegida["cuantiles"][Tr]) for Tr in cfg["Tr_lista_IDF"]}

    # IC bootstrap sobre la elegida
    intervalos = {}
    if cfg.get("calcular_IC", True):
        intervalos = dist_mod.intervalos_confianza_bootstrap(
            serie_pmax_tratada,
            dist_obj=elegida["_dist_obj"], escala=elegida["_escala"],
            Tr_lista=cfg["Tr_lista_IDF"],
            n_bootstrap=int(cfg.get("n_bootstrap", 1000)),
            nivel=float(cfg.get("nivel_confianza_IC", 0.95)),
        )

    contexto["resultados_dist"]      = resultados_dist
    contexto["bondad"]               = df_bondad
    contexto["dist_elegida"]         = elegida["Distribucion"]
    contexto["P24h_por_distribucion"] = _df_P24h_por_dist(resultados_dist, cfg["Tr_lista_IDF"])
    contexto["P24h_diseno"]          = P24h_por_Tr
    contexto["intervalos_confianza"] = intervalos

    # -- Fase 9: métodos IDF ---------------------------------------------------
    encabezado_fase(9, TOTAL_FASES, "Construcción de curvas IDF (6 métodos)")
    M_promedio = float(serie_pmax_tratada.mean())
    print(f"  -> Pmax24 medio multianual M = {M_promedio:.2f} mm")
    idfs_dict = idf_methods.calcular_todas_idf(
        P24h_por_Tr=P24h_por_Tr,
        M_promedio=M_promedio,
        duraciones_min=cfg["duraciones_min"],
        Tr_lista=cfg["Tr_lista_IDF"],
        metodos=cfg["metodos_IDF"],
        region=cfg["region_IDF"],
        R1=cfg["R1"], R2=cfg["R2"], R3=cfg["R3"],
    )
    idfs_solo_df = {nombre: par[0] for nombre, par in idfs_dict.items()}
    parametros_idf = {nombre: par[1] for nombre, par in idfs_dict.items()}
    print(f"  -> Métodos calculados: {list(idfs_solo_df.keys())}")

    # matriz de diferencias entre pares (% absoluto)
    diferencias = {}
    nombres_idf = list(idfs_solo_df.keys())
    for i in range(len(nombres_idf)):
        for j in range(i + 1, len(nombres_idf)):
            a, b = nombres_idf[i], nombres_idf[j]
            diferencias[(a, b)] = idf_methods.matriz_diferencias_porcentuales(
                idfs_solo_df[a], idfs_solo_df[b]
            )

    contexto["idfs"]            = idfs_solo_df
    contexto["parametros_idf"]  = parametros_idf
    contexto["diferencias_idf"] = diferencias
    contexto["M_promedio"]      = M_promedio

    # -- Fase 10: gráficas + exportación ---------------------------------------
    encabezado_fase(10, TOTAL_FASES, "Gráficas, Excel, PDF y Word")
    plotting.configurar_matplotlib()
    figs: list = []            # lista (fig, titulo) para el PDF consolidado
    rutas_figuras: dict[str, str] = {}   # claves esperadas por reports.escribir_word

    def _registrar(clave: str, par):
        """Guarda PNG individual y agrega la figura al PDF."""
        fig, titulo = par
        ruta_png = os.path.join(rutas["graficas"], f"{clave}.png")
        plotting.guardar(fig, ruta_png)
        rutas_figuras[clave] = ruta_png
        figs.append((fig, titulo))

    # Serie diaria
    try:
        _registrar("serie_diaria",
                   plotting.fig_serie_diaria(df_completa, "precipitacion", codigo))
    except Exception as exc:
        print(f"  ! fig_serie_diaria: {exc}")

    # Mapa de calor de máximos mensuales
    try:
        _registrar("mapa_calor",
                   plotting.fig_mapa_calor_maximos(tabla_definitiva, codigo))
    except Exception as exc:
        print(f"  ! fig_mapa_calor_maximos: {exc}")

    # Completitud anual (cuenta meses con Pmax distinto de NaN)
    try:
        meses_con_dato = tabla_mensual[series_mod.MESES_ES].notna().sum(axis=1)
        _registrar("completitud_anual",
                   plotting.fig_completitud_anual(
                       meses_con_dato, min_meses=cfg["min_meses_anio"],
                       n_completos=len(clasif["completos"]),
                       n_incompletos=len(clasif["incompletos"]),
                       n_insufic=len(clasif["insuficientes"]),
                       codigo=codigo))
    except Exception as exc:
        print(f"  ! fig_completitud_anual: {exc}")

    # Pmax24 anual con marcado de años completados
    try:
        _registrar("pmax24",
                   plotting.fig_pmax24(tabla_definitiva, clasif["incompletos"], codigo))
    except Exception as exc:
        print(f"  ! fig_pmax24: {exc}")

    # Outliers
    try:
        _registrar("outliers",
                   plotting.fig_outliers(tabla_definitiva, info_outliers, codigo))
    except Exception as exc:
        print(f"  ! fig_outliers: {exc}")

    # QQ plots
    try:
        _registrar("qq",
                   plotting.fig_qq_plots(resultados_dist, serie_pmax_tratada, codigo))
    except Exception as exc:
        print(f"  ! fig_qq_plots: {exc}")

    # Curvas de frecuencia
    try:
        _registrar("curvas_frec",
                   plotting.fig_curvas_frecuencia(
                       resultados_dist, serie_pmax_tratada,
                       Tr_diseno=cfg["Tr_diseno"],
                       dist_seleccionada=elegida["Distribucion"],
                       codigo=codigo))
    except Exception as exc:
        print(f"  ! fig_curvas_frecuencia: {exc}")

    # IDFs por método
    for nombre, df_idf in idfs_solo_df.items():
        try:
            _registrar(f"idf_{nombre}",
                       plotting.fig_idf(df_idf, nombre, codigo))
        except Exception as exc:
            print(f"  ! fig_idf {nombre}: {exc}")

    # Comparación IDF
    if len(idfs_solo_df) >= 2:
        try:
            Tr_comp = [t for t in (10, cfg["Tr_diseno"], 100) if t in cfg["Tr_lista_IDF"]]
            _registrar("comparacion_idf",
                       plotting.fig_comparacion_idf(idfs_solo_df, Tr_comp, codigo))
        except Exception as exc:
            print(f"  ! fig_comparacion_idf: {exc}")

    # Correlación con cada auxiliar apta
    for cod_aux, s_aux in series_aux.items():
        info_apta = aptitud.get(cod_aux, {})
        if info_apta.get("apto"):
            try:
                _registrar(f"corr_{cod_aux}",
                           plotting.fig_correlacion_E2(
                               serie_principal, s_aux,
                               m=info_apta["regresion_m"],
                               b=info_apta["regresion_b"],
                               r=info_apta["r_pearson"],
                               nombre_aux=str(cod_aux)))
            except Exception as exc:
                print(f"  ! fig_correlacion_E2 {cod_aux}: {exc}")

    # Identificación común para reports
    contexto["identificacion"] = {
        "Proyecto"      : cfg.get("proyecto", ""),
        "Entidad"       : cfg.get("entidad", ""),
        "Estación (E1)" : f"{codigo} — {nombre_e1}",
        "Región IDF"    : cfg["region_IDF"],
        "Periodo"       : f"{serie_principal.index.min().date()} -> {serie_principal.index.max().date()}",
        "Distribución"  : elegida["Distribucion"],
        "Tr de diseño"  : cfg["Tr_diseno"],
        "Fecha del run" : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Factor unidades": diag_unid.get("factor_aplicado", 1.0),
    }

    # Conclusiones automáticas para la memoria
    contexto["conclusiones"] = _generar_conclusiones(cfg, contexto, elegida, P24h_por_Tr)

    # Excel
    if cfg["generar_excel"]:
        ruta_xlsx = os.path.join(rutas["datos"], "resultados.xlsx")
        try:
            reports.escribir_excel(ruta_xlsx, contexto)
            print(f"  v Excel  -> {ruta_xlsx}")
        except Exception as exc:
            print(f"  ! Error generando Excel: {exc}")

    # PDF
    if cfg["generar_pdf"]:
        ruta_pdf = os.path.join(rutas["graficas"], "graficas.pdf")
        try:
            reports.escribir_pdf(ruta_pdf, figs, codigo=str(codigo))
            print(f"  v PDF    -> {ruta_pdf}")
        except Exception as exc:
            print(f"  ! Error generando PDF: {exc}")

    # Word
    if cfg["generar_word"]:
        try:
            ruta_docx = os.path.join(rutas["informe"], "memoria_calculo.docx")
            reports.escribir_word(ruta_docx, contexto, rutas_figuras)
            print(f"  v Word   -> {ruta_docx}")
        except ImportError:
            print("  ! python-docx no instalado. Memoria Word omitida.")
        except Exception as exc:
            print(f"  ! Error generando Word: {exc}")

    # -- Resumen final ---------------------------------------------------------
    dt = (datetime.now() - t0).total_seconds()
    banner(f"PIPELINE COMPLETO — Duración: {dt:.1f} s")
    print(f"  Estación      : {codigo} — {nombre_e1}")
    print(f"  Años útiles   : {len(serie_pmax_tratada)}")
    print(f"  Distribución  : {elegida['Distribucion']}")
    print(f"  Pmax24 Tr={cfg['Tr_diseno']}a: {P24h_por_Tr[cfg['Tr_diseno']]:.2f} mm")
    print(f"  Salidas en    : {rutas['base']}\n")
    return 0


# ------------------------------------------------------------------------------
# Utilitarios internos
# ------------------------------------------------------------------------------

def _detectar_codigo(df: pd.DataFrame, ruta_archivo: str) -> str:
    """
    Detecta el código de la estación buscando columnas frecuentes en CSVs IDEAM:
    'codigoestacion', 'codigo', 'estacion'. Si no, deriva del nombre de archivo.
    """
    for cand in ("codigoestacion", "codigo", "estacion", "codigosensor"):
        if cand in df.columns:
            v = df[cand].dropna()
            if len(v):
                return str(v.iloc[0]).strip()
    base = os.path.splitext(os.path.basename(ruta_archivo))[0]
    return base


def _df_P24h_por_dist(resultados: list[dict], Tr_lista: list[int]) -> pd.DataFrame:
    """Tabla compacta: filas = distribución, columnas = Tr (años)."""
    filas = []
    for r in resultados:
        if "_error" in r:
            continue
        fila = {"Distribucion": r["Distribucion"]}
        for Tr in Tr_lista:
            fila[f"Tr_{Tr}a"] = r["cuantiles"].get(Tr)
        filas.append(fila)
    return pd.DataFrame(filas)


def _generar_conclusiones(cfg: dict, contexto: dict, elegida: dict, P24h: dict) -> list[str]:
    """
    Conclusiones automáticas basadas en los resultados del pipeline.
    No inventa números: solo formula afirmaciones soportadas por el contexto.
    """
    conclusiones: list[str] = []
    n = len(contexto.get("serie_pmax", []))
    Tr_d = cfg["Tr_diseno"]

    conclusiones.append(
        f"Se construyó una serie de máximos anuales (AMS) con {n} años de registro "
        f"utilizables, suficientes para análisis de frecuencias paramétrico "
        f"(Bulletin 17C recomienda n >= 10, con n >= 25 deseable)."
        if n >= 10 else
        f"! La serie AMS solo cuenta con {n} años. Por debajo de 10 años el "
        "análisis de frecuencias paramétrico es altamente incierto (Bulletin 17C)."
    )

    tests = contexto.get("tests_hidrologicos", {})
    if "dictamen" in tests:
        conclusiones.append(f"Tests hidrológicos i.i.d.: {tests['dictamen']}")

    info_out = contexto.get("outliers_info", {})
    if info_out:
        alt = len(info_out.get("outliers_altos", []))
        baj = len(info_out.get("outliers_bajos", []))
        conclusiones.append(
            f"Análisis de outliers (Grubbs-Beck, Bulletin 17B): {alt} altos / "
            f"{baj} bajos detectados. Tratamiento aplicado: "
            f"{contexto.get('tratamiento_outliers', 'conservar')}."
        )

    conclusiones.append(
        f"Distribución seleccionada para diseño: {elegida['Distribucion']} "
        f"(AIC={elegida['AIC']}, KS={elegida['KS_resultado']}, "
        f"AD={elegida['AD_resultado']})."
    )

    if Tr_d in P24h:
        conclusiones.append(
            f"P24h de diseño para Tr={Tr_d} años: {P24h[Tr_d]:.2f} mm."
        )

    idfs = contexto.get("idfs", {})
    if len(idfs) >= 2:
        conclusiones.append(
            f"Se construyeron {len(idfs)} curvas IDF independientes "
            f"({', '.join(idfs.keys())}). Las diferencias porcentuales entre métodos "
            "permiten acotar la incertidumbre por modelo."
        )

    ic = contexto.get("intervalos_confianza", {})
    if Tr_d in ic and ic[Tr_d]:
        d = ic[Tr_d]
        conclusiones.append(
            f"Intervalo de confianza bootstrap al "
            f"{int(cfg.get('nivel_confianza_IC', 0.95)*100)}% para Tr={Tr_d}a: "
            f"[{d.get('inf', float('nan')):.2f}, {d.get('sup', float('nan')):.2f}] mm."
        )

    conclusiones.append(
        "Las curvas IDF entregadas son aplicables a la región hidrológica indicada "
        f"({cfg['region_IDF']}). Para diseño definitivo de obras civiles se recomienda "
        "comparar contra IDFs regionales oficiales del IDEAM, validar con eventos "
        "registrados localmente y consultar RAS 2017 / INVIAS 2009 según el tipo de obra."
    )

    return conclusiones


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

def parsear_argumentos() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="idf_tool",
        description="Análisis hidrológico pluviométrico + curvas IDF (6 métodos, 9 distribuciones).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Ejemplos:\n"
               "  python main.py --config config.yaml\n"
               "  python main.py --carpeta-entrada ./entrada --carpeta-salida ./salida\n"
               "  python main.py --carpeta-entrada ./entrada --principal estacion_X.csv --region Orinoquia",
    )
    p.add_argument("--config", type=str, help="Ruta al archivo YAML/JSON de configuración.")
    p.add_argument("--carpeta-entrada", type=str, help="Carpeta con CSVs IDEAM.")
    p.add_argument("--carpeta-salida",  type=str, help="Carpeta destino para resultados.")
    p.add_argument("--principal", type=str, help="Nombre del CSV principal (E1).")
    p.add_argument("--region",    type=str, choices=["Andina", "Caribe", "Pacifico", "Orinoquia"],
                   help="Región para IDF Díaz-Granados.")
    p.add_argument("--tr-diseno", type=int, help="Periodo de retorno de diseño (años).")
    p.add_argument("--factor",    type=str, help="Factor de unidades: 'auto', 1.0, 0.1, 0.01.")
    p.add_argument("--no-tests",  action="store_true", help="Omitir tests hidrológicos.")
    p.add_argument("--no-ic",     action="store_true", help="Omitir IC bootstrap.")
    p.add_argument("--no-word",   action="store_true", help="No generar memoria Word.")
    p.add_argument("--versionado", action="store_true", help="Versionar carpeta de salida con timestamp.")
    return p.parse_args()


def aplicar_overrides_cli(cfg: dict, args: argparse.Namespace) -> dict:
    """Los argumentos CLI sobrescriben la configuración."""
    if args.carpeta_entrada: cfg["carpeta_entrada"] = args.carpeta_entrada
    if args.carpeta_salida:  cfg["carpeta_salida"]  = args.carpeta_salida
    if args.principal:       cfg["archivo_principal"] = args.principal
    if args.region:          cfg["region_IDF"]      = args.region
    if args.tr_diseno:       cfg["Tr_diseno"]       = args.tr_diseno
    if args.factor:          cfg["factor_conversion"] = args.factor
    if args.no_tests:        cfg["ejecutar_tests"]  = False
    if args.no_ic:           cfg["calcular_IC"]     = False
    if args.no_word:         cfg["generar_word"]    = False
    if args.versionado:      cfg["versionado_timestamp"] = True
    return cfg


def main() -> int:
    args = parsear_argumentos()
    banner("idf_tool — Análisis pluviométrico + IDF (Colombia / mundial)")

    try:
        cfg = cfg_mod.cargar_config(args.config)
        cfg = aplicar_overrides_cli(cfg, args)
        cfg_mod.validar_config(cfg)
        return ejecutar_pipeline(cfg)
    except KeyboardInterrupt:
        print("\nX Interrumpido por el usuario.")
        return 130
    except FileNotFoundError as exc:
        print(f"\nX {exc}")
        return 2
    except Exception as exc:
        print(f"\nX Error inesperado: {exc}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
