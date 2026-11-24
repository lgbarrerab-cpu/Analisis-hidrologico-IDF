"""
idf_tool — Herramienta de análisis pluviométrico e IDF.

Pipeline:
  1. Carga de datos IDEAM (CSV) — estación principal + auxiliares
  2. Validación de unidades (mm vs décimas de mm)
  3. Filtros técnicos y limpieza
  4. Completamiento de datos faltantes (multi-estación)
  5. Construcción de serie de máximos anuales (AMS)
  6. Tests hidrológicos (independencia, estacionariedad, homogeneidad)
  7. Detección de outliers (Bulletin 17B)
  8. Ajuste de distribuciones (9 candidatas) + bondad de ajuste
  9. Curvas IDF (6 métodos)
  10. Exportación consolidada (Excel + PDF + Word)

Referencias normativas:
  - USGS Bulletin 17C (England et al., 2019)
  - WMO No. 168 — Guide to Hydrological Practices (2009)
  - INVIAS — Manual de Drenaje para Carreteras (2009)
  - RAS 2017 — Reglamento Técnico del Sector de Agua Potable (Res. 0330)
  - IDEAM — Protocolo para análisis de información hidrometeorológica
"""

__version__ = "1.0.0"
__author__ = "XXX"
