"""
reports.py — Exportación consolidada de resultados.

Genera tres entregables sincronizados:
  - resultados.xlsx  (hojas con series, bondad, IDF por método, IC, estadísticos)
  - graficas.pdf     (todas las figuras consolidadas)
  - memoria.docx     (memoria de cálculo profesional)
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure


# ------------------------------------------------------------------------------
# Excel
# ------------------------------------------------------------------------------

def escribir_excel(
    ruta: str,
    contexto: dict[str, Any],
) -> str:
    """
    Escribe el archivo Excel consolidado.

    El `contexto` agrupa todos los DataFrames y diccionarios producidos.
    Llaves esperadas (opcionales todas excepto identificacion):
      - identificacion        : dict con metadata
      - tabla_mensual         : DataFrame
      - tabla_definitiva      : DataFrame
      - estadisticos          : dict
      - bondad                : DataFrame
      - P24h_por_distribucion : DataFrame
      - intervalos_confianza  : dict {Tr: {inf, sup, central}}
      - idfs                  : dict {nombre_metodo: DataFrame}
      - diferencias_idf       : dict {(a, b): DataFrame}
      - resumen_E2            : dict
      - tests_hidrologicos    : dict
      - diagnostico_unidades  : dict
    """
    with pd.ExcelWriter(ruta, engine="openpyxl") as writer:
        # Hoja 1: identificación
        meta = contexto.get("identificacion", {})
        df_meta = pd.DataFrame(list(meta.items()), columns=["Parámetro", "Valor"])
        df_meta.to_excel(writer, sheet_name="00_Identificacion", index=False)

        if "diagnostico_unidades" in contexto:
            d = contexto["diagnostico_unidades"]
            pd.DataFrame(list(d.items()), columns=["Parámetro", "Valor"]).to_excel(
                writer, sheet_name="01_Diagnostico_Unidades", index=False
            )

        if "resumen_E2" in contexto and contexto["resumen_E2"]:
            rows = []
            for nombre, info in contexto["resumen_E2"].items():
                for k, v in info.items():
                    rows.append({"Auxiliar": nombre, "Parámetro": k, "Valor": str(v)})
            pd.DataFrame(rows).to_excel(writer, sheet_name="02_Estaciones_Auxiliares", index=False)

        if "tabla_mensual" in contexto:
            contexto["tabla_mensual"].to_excel(writer, sheet_name="03_Maximos_Mensuales")

        if "tabla_definitiva" in contexto:
            contexto["tabla_definitiva"].to_excel(writer, sheet_name="04_Serie_Definitiva")

        if "estadisticos" in contexto:
            pd.DataFrame(list(contexto["estadisticos"].items()),
                         columns=["Estadístico", "Valor"]).to_excel(
                writer, sheet_name="05_Estadisticos", index=False
            )

        if "tests_hidrologicos" in contexto:
            t = contexto["tests_hidrologicos"]
            rows = []
            for nombre, info in t.items():
                if isinstance(info, dict):
                    for k, v in info.items():
                        rows.append({"Test": nombre, "Parámetro": k, "Valor": str(v)})
                else:
                    rows.append({"Test": nombre, "Parámetro": "valor", "Valor": str(info)})
            pd.DataFrame(rows).to_excel(writer, sheet_name="06_Tests_Hidrologicos", index=False)

        if "bondad" in contexto:
            contexto["bondad"].to_excel(writer, sheet_name="07_Bondad_Ajuste", index=False)

        if "P24h_por_distribucion" in contexto:
            contexto["P24h_por_distribucion"].to_excel(
                writer, sheet_name="08_P24h_por_Distribucion", index=False
            )

        if "intervalos_confianza" in contexto and contexto["intervalos_confianza"]:
            rows = []
            for Tr, ic in contexto["intervalos_confianza"].items():
                rows.append({
                    "Tr (años)"   : Tr,
                    "P24h_inf"    : round(ic.get("inf", float("nan")), 2),
                    "P24h_central": round(ic.get("central", float("nan")), 2),
                    "P24h_sup"    : round(ic.get("sup", float("nan")), 2),
                })
            pd.DataFrame(rows).to_excel(writer, sheet_name="09_Intervalos_Confianza", index=False)

        # IDFs por método (una hoja por método)
        idfs = contexto.get("idfs", {})
        for nombre, df_idf in idfs.items():
            hoja = f"IDF_{nombre[:25]}".replace(" ", "_").replace("/", "_")
            df_idf.to_excel(writer, sheet_name=hoja)

        # Diferencias
        difs = contexto.get("diferencias_idf", {})
        for (a, b), df_dif in difs.items():
            hoja = f"DIF_{a[:10]}_vs_{b[:10]}".replace(" ", "_")[:31]
            df_dif.to_excel(writer, sheet_name=hoja)

    return ruta


# ------------------------------------------------------------------------------
# PDF de gráficas
# ------------------------------------------------------------------------------

def escribir_pdf(ruta: str, figuras: list[tuple[Figure, str]],
                 codigo: str | int) -> str:
    with PdfPages(ruta) as pdf:
        for fig, _titulo in figuras:
            pdf.savefig(fig, bbox_inches="tight")
        info = pdf.infodict()
        info["Title"] = f"Análisis hidrológico — Estación {codigo}"
        info["CreationDate"] = datetime.now()
    return ruta


# ------------------------------------------------------------------------------
# Word
# ------------------------------------------------------------------------------

def escribir_word(
    ruta: str,
    contexto: dict[str, Any],
    rutas_figuras: dict[str, str],
) -> str:
    """
    Genera memoria de cálculo en .docx con estructura profesional.

    rutas_figuras es un dict {clave: ruta_png} con las figuras a embeber.
    Claves esperadas: 'serie_diaria', 'completitud_mensual', 'mapa_calor',
                      'completitud_anual', 'pmax24', 'outliers',
                      'qq', 'curvas_frec', 'idfs' (dict por método),
                      'comparacion_idf', 'correlacion_E2' (dict por auxiliar)
    """
    try:
        from docx import Document
        from docx.shared import Pt, Inches, Cm
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.enum.table import WD_TABLE_ALIGNMENT
    except ImportError:
        print("! python-docx no instalado. Se omite la memoria Word.")
        return ""

    doc = Document()

    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)
    style.paragraph_format.space_after = Pt(6)

    def parrafo(texto: str, justificado: bool = True, negrita: bool = False, size: int | None = None):
        p = doc.add_paragraph()
        run = p.add_run(texto)
        if negrita:
            run.bold = True
        if size:
            run.font.size = Pt(size)
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY if justificado else WD_ALIGN_PARAGRAPH.LEFT
        return p

    def centrado(texto: str, negrita: bool = False, size: int | None = None):
        p = doc.add_paragraph()
        run = p.add_run(texto)
        if negrita:
            run.bold = True
        if size:
            run.font.size = Pt(size)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        return p

    def tabla_dict(data: dict[str, list], titulo: str | None = None):
        if titulo:
            doc.add_heading(titulo, level=3)
        cols = list(data.keys())
        n_rows = len(next(iter(data.values())))
        t = doc.add_table(rows=n_rows + 1, cols=len(cols))
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, c in enumerate(cols):
            t.rows[0].cells[i].text = str(c)
            for r in t.rows[0].cells[i].paragraphs:
                for run in r.runs:
                    run.bold = True
        for i in range(n_rows):
            for j, c in enumerate(cols):
                v = data[c][i]
                if isinstance(v, float):
                    t.rows[i + 1].cells[j].text = f"{v:.2f}"
                else:
                    t.rows[i + 1].cells[j].text = str(v)
        doc.add_paragraph()
        return t

    def tabla_df(df: pd.DataFrame, titulo: str | None = None, max_rows: int = 60):
        if df is None or df.empty:
            return None
        df_ = df.head(max_rows)
        data = {str(c): df_[c].tolist() for c in df_.columns}
        return tabla_dict(data, titulo)

    def img(path: str | None, caption: str, width_in: float = 5.7):
        if path and os.path.exists(path):
            doc.add_picture(path, width=Inches(width_in))
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            centrado(caption, negrita=True, size=9)
            doc.add_paragraph()

    ident = contexto.get("identificacion", {})

    # -- Portada --
    for _ in range(4): doc.add_paragraph()
    centrado("MEMORIA DE CÁLCULO", negrita=True, size=20)
    doc.add_paragraph()
    centrado("Análisis de frecuencias de precipitación máxima en 24 h", size=14)
    centrado("y construcción de curvas IDF", size=14)
    doc.add_paragraph(); doc.add_paragraph()
    for k, v in ident.items():
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(f"{k}: ")
        run.bold = True
        p.add_run(str(v))
    centrado("\nNormas: USGS Bulletin 17C · WMO No. 168 · INVIAS 2009 · RAS 2017")
    doc.add_page_break()

    # -- 1. Objetivo y alcance --
    doc.add_heading("1. OBJETIVO Y ALCANCE", level=1)
    parrafo(
        "Esta memoria documenta de forma completa, trazable y reproducible el análisis "
        "de frecuencias de precipitación máxima en 24 horas (Pmax24) y la construcción "
        "de curvas IDF para la estación principal indicada. El procedimiento aplica el "
        "marco normativo del USGS Bulletin 17C, la WMO No. 168 y los lineamientos del "
        "Manual de Drenaje del INVIAS (2009)."
    )

    # -- 2. Marco normativo --
    doc.add_heading("2. MARCO NORMATIVO", level=1)
    refs = [
        ("USGS Bulletin 17C (2019)", "Análisis de frecuencias de crecientes y precipitaciones extremas."),
        ("WMO No. 168 (2009)",       "Guía de prácticas hidrológicas (completamiento, tests)."),
        ("INVIAS — Manual de Drenaje (2009)", "Curvas IDF, ecuación Díaz-Granados (Ec. 2.103)."),
        ("RAS 2017 (Res. 0330)",     "Reglamento técnico de agua potable y saneamiento."),
        ("IDEAM",                    "Protocolo y catálogo de estaciones pluviométricas."),
    ]
    tabla_dict({"Referencia": [r[0] for r in refs], "Alcance": [r[1] for r in refs]},
               "Tabla 2.1 — Marco normativo aplicado")

    # -- 3. Datos de entrada --
    doc.add_heading("3. DATOS DE ENTRADA", level=1)
    if "diagnostico_unidades" in contexto:
        d = contexto["diagnostico_unidades"]
        doc.add_heading("3.1 Validación de unidades", level=2)
        parrafo(str(d.get("decision", "")))
        tabla_dict(
            {"Parámetro": list(d.keys()), "Valor": [str(v) for v in d.values()]},
            "Tabla 3.1 — Diagnóstico de unidades",
        )

    if "resumen_E2" in contexto and contexto["resumen_E2"]:
        doc.add_heading("3.2 Estaciones auxiliares evaluadas", level=2)
        for nombre, info in contexto["resumen_E2"].items():
            doc.add_heading(f"3.2.{list(contexto['resumen_E2']).index(nombre)+1} {nombre}",
                            level=3)
            tabla_dict(
                {"Parámetro": list(info.keys()), "Valor": [str(v) for v in info.values()]},
                f"Tabla — Evaluación de aptitud para {nombre}",
            )
            img(rutas_figuras.get(f"corr_{nombre}"), f"Figura — Correlación principal vs {nombre}")

    # -- 4. Procesamiento --
    doc.add_heading("4. PROCESAMIENTO Y DEPURACIÓN", level=1)
    img(rutas_figuras.get("serie_diaria"), "Figura — Serie diaria con días sin dato resaltados")
    img(rutas_figuras.get("completitud_mensual"), "Figura — Completitud mensual (%)")
    img(rutas_figuras.get("mapa_calor"), "Figura — Mapa de calor de máximos mensuales")

    # -- 5. Serie AMS --
    doc.add_heading("5. SERIE DE MÁXIMOS ANUALES (AMS)", level=1)
    img(rutas_figuras.get("completitud_anual"), "Figura — Completitud por año")
    img(rutas_figuras.get("pmax24"), "Figura — Pmax24 anual (con completamiento marcado)")

    if "estadisticos" in contexto:
        doc.add_heading("5.1 Estadísticos descriptivos", level=2)
        tabla_dict(
            {"Estadístico": list(contexto["estadisticos"].keys()),
             "Valor"      : list(contexto["estadisticos"].values())},
            "Tabla 5.1 — Estadísticos descriptivos de la serie definitiva",
        )

    # -- 6. Tests hidrológicos --
    if "tests_hidrologicos" in contexto:
        doc.add_heading("6. TESTS HIDROLÓGICOS (BULLETIN 17C / WMO 168)", level=1)
        for nombre, info in contexto["tests_hidrologicos"].items():
            if nombre in ("dictamen", "tests_rechazados"):
                continue
            if isinstance(info, dict):
                doc.add_heading(f"6.{list(contexto['tests_hidrologicos']).index(nombre)+1} "
                                f"{info.get('test', nombre)}", level=2)
                tabla_dict(
                    {"Parámetro": list(info.keys()), "Valor": [str(v) for v in info.values()]},
                    f"Tabla — {info.get('test', nombre)}",
                )
        dictamen = contexto["tests_hidrologicos"].get("dictamen", "")
        if dictamen:
            parrafo(f"Dictamen integral: {dictamen}", negrita=True)

    # -- 7. Outliers --
    if "outliers_info" in contexto:
        doc.add_heading("7. ANÁLISIS DE OUTLIERS (BULLETIN 17B)", level=1)
        info = contexto["outliers_info"]
        parrafo(
            f"n = {info.get('n')}, Kn = {info.get('Kn'):.4f}. "
            f"Límite superior = {info.get('lim_superior'):.2f} mm, "
            f"límite inferior = {info.get('lim_inferior'):.2f} mm. "
            f"Outliers altos: {len(info.get('outliers_altos', []))}, "
            f"outliers bajos: {len(info.get('outliers_bajos', []))}. "
            f"Tratamiento aplicado: {contexto.get('tratamiento_outliers')}."
        )
        img(rutas_figuras.get("outliers"), "Figura — Detección de outliers (Bulletin 17B)")

    # -- 8. Ajuste y bondad --
    doc.add_heading("8. AJUSTE DE DISTRIBUCIONES Y BONDAD DE AJUSTE", level=1)
    if "bondad" in contexto:
        cols = [c for c in contexto["bondad"].columns if c in
                ("Distribucion", "AIC", "BIC", "KS_stat", "KS_pvalor",
                 "KS_resultado", "AD_stat", "AD_resultado")]
        tabla_df(contexto["bondad"][cols], "Tabla 8.1 — Bondad de ajuste")
    img(rutas_figuras.get("qq"), "Figura — QQ Plots")
    img(rutas_figuras.get("curvas_frec"), "Figura — Curvas de frecuencia")

    if "P24h_por_distribucion" in contexto:
        doc.add_heading("8.1 P24h por distribución", level=2)
        tabla_df(contexto["P24h_por_distribucion"], "Tabla 8.2 — P24h (mm) por distribución y Tr")

    if "intervalos_confianza" in contexto and contexto["intervalos_confianza"]:
        doc.add_heading("8.2 Intervalos de confianza (bootstrap)", level=2)
        ic = contexto["intervalos_confianza"]
        tabla_dict(
            {"Tr (años)"   : list(ic.keys()),
             "Lím. inferior": [round(v.get("inf", float("nan")), 2) for v in ic.values()],
             "Central"     : [round(v.get("central", float("nan")), 2) for v in ic.values()],
             "Lím. superior": [round(v.get("sup", float("nan")), 2) for v in ic.values()]},
            "Tabla 8.3 — Intervalos de confianza P24h",
        )

    # -- 9. IDF --
    doc.add_heading("9. CURVAS IDF", level=1)
    idfs = contexto.get("idfs", {})
    params_idf = contexto.get("parametros_idf", {})
    for nombre, df_idf in idfs.items():
        doc.add_heading(f"9.{list(idfs).index(nombre)+1} {nombre}", level=2)
        pars = params_idf.get(nombre, {})
        if pars:
            tabla_dict(
                {"Parámetro": [k for k in pars if not str(k).startswith("_")],
                 "Valor"    : [str(v) for k, v in pars.items() if not str(k).startswith("_")]},
                "Tabla — Parámetros de la ecuación",
            )
        df_idf_view = df_idf.reset_index().rename(columns={"Duración (min)": "t (min)"})
        tabla_df(df_idf_view, "Tabla — Intensidad (mm/h)")
        img(rutas_figuras.get(f"idf_{nombre}"), f"Figura — IDF {nombre}")
    img(rutas_figuras.get("comparacion_idf"), "Figura — Comparación entre métodos IDF")

    # -- 10. Conclusiones --
    doc.add_heading("10. CONCLUSIONES", level=1)
    concs = contexto.get("conclusiones", [])
    for c in concs:
        doc.add_paragraph(c, style="List Bullet")

    # -- 11. Referencias --
    doc.add_heading("11. REFERENCIAS BIBLIOGRÁFICAS", level=1)
    referencias = [
        "Bell, F.C. (1969). Generalized rainfall-duration-frequency relationships. J. Hyd. Div., ASCE, 95(HY1), 311-327.",
        "Bernard, M.M. (1932). Formulas for rainfall intensities of long duration. Trans. ASCE, 96, 592-624.",
        "Chen, C.L. (1983). Rainfall intensity-duration-frequency formulas. J. Hyd. Eng., ASCE, 109(12), 1603-1621.",
        "Chow, V.T., Maidment, D.R., & Mays, L.W. (1988). Applied Hydrology. McGraw-Hill.",
        "England, J.F. et al. (2019). Guidelines for determining flood flow frequency — Bulletin 17C. USGS.",
        "IDEAM. Catálogo nacional de estaciones. http://www.ideam.gov.co",
        "INVIAS (2009). Manual de Drenaje para Carreteras.",
        "Kothyari, U.C., & Garde, R.J. (1992). Rainfall intensity-duration-frequency formula for India. J. Inst. Engineers, 73, 159-165.",
        "Kendall, M.G. (1948). Rank Correlation Methods. Griffin.",
        "Mann, H.B. (1945). Nonparametric tests against trend. Econometrica, 13, 245-259.",
        "Paulhus, J.L.H., & Kohler, M.A. (1952). Interpolation of missing precipitation records. Monthly Weather Review, 80(8), 129-133.",
        "Pettitt, A.N. (1979). A non-parametric approach to the change-point problem. J. R. Stat. Soc. C, 28(2), 126-135.",
        "Sherman, C.W. (1931). Frequency and intensity of excessive rainfalls at Boston. Trans. ASCE, 95, 951-960.",
        "Stephens, M.A. (1986). Tests based on EDF statistics. En D'Agostino & Stephens, Goodness-of-fit techniques.",
        "Vargas, R., & Díaz-Granados, M. (1998). Curvas sintéticas regionalizadas de IDF para Colombia. Universidad de los Andes.",
        "WMO (2009). Guide to Hydrological Practices, No. 168, Vol. II.",
        "Wald, A., & Wolfowitz, J. (1940). On a test whether two samples are from the same population. Ann. Math. Stat., 11(2), 147-162.",
    ]
    for r in referencias:
        doc.add_paragraph(r, style="List Bullet")

    # -- 12. Firma --
    doc.add_paragraph(); doc.add_paragraph()
    parrafo("________________________________________", justificado=False)
    parrafo("XXX", justificado=False)
    parrafo(f"Fecha de generación: {datetime.now().strftime('%d/%m/%Y %H:%M')}", justificado=False)

    doc.save(ruta)
    return ruta
