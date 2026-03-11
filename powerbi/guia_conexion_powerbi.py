# -*- coding: utf-8 -*-
"""
Guía de Conexión: Power BI + Python + Datos del Modelo
======================================================
Proyecto: Adopción Digital — Entidad Financiera
Autora: Claudia Rocha

ARCHIVOS DISPONIBLES PARA POWER BI:
  powerbi/coeficientes_modelo.csv   → Impacto de cada variable
  powerbi/feature_importance.csv    → Importancia de features (Random Forest)
  powerbi/metricas_modelos.csv      → Comparación de 3 modelos
  powerbi/resumen_segmentos.csv     → Resumen por ciudad/nómina/digital
  data/scoring_clientes.csv         → Score individual de cada cliente
"""

# ================================================================
# OPCIÓN 1: SCRIPT PYTHON DENTRO DE POWER BI
# ================================================================
# En Power BI Desktop:
# 1. Obtener datos → Python Script
# 2. Pegar este código
# 3. Power BI creará una tabla con los resultados

# --- Script para cargar scoring de clientes ---
import pandas as pd
dataset = pd.read_csv(r'C:\Claudia\proyecto_adopcion_digital\data\scoring_clientes.csv')


# ================================================================
# OPCIÓN 2: VISUAL PYTHON EN POWER BI (para gráficos)
# ================================================================
# En Power BI:
# 1. Insertar → Visual de Python
# 2. Arrastrar campos al visual
# 3. Pegar este código en el editor de script

# --- Gráfico: Distribución de scores de propensión ---
# (Pegar en visual Python de Power BI)
"""
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(10, 6))

# 'dataset' es la tabla que Power BI pasa al script Python
for digital, color, label in [(0, '#ff6b6b', 'NO Digital'), (1, '#51cf66', 'SÍ Digital')]:
    subset = dataset[dataset['es_digital'] == digital]
    ax.hist(subset['score_propension_digital'], bins=30, alpha=0.6, color=color, label=label)

ax.set_xlabel('Score de Propensión Digital', fontsize=13)
ax.set_ylabel('Cantidad de Clientes', fontsize=13)
ax.set_title('Distribución de Scores de Propensión Digital', fontsize=14, fontweight='bold')
ax.legend(fontsize=12)
plt.tight_layout()
plt.show()
"""


# ================================================================
# OPCIÓN 3: MEDIDAS DAX SUGERIDAS
# ================================================================
"""
MEDIDAS DAX para el dashboard de adopción digital:

// --- KPIs principales ---

Tasa Adopción Digital =
DIVIDE(
    CALCULATE(COUNTROWS(scoring_clientes), scoring_clientes[es_digital] = 1),
    COUNTROWS(scoring_clientes)
) * 100

Clientes No Digitales Alto Potencial =
CALCULATE(
    COUNTROWS(scoring_clientes),
    scoring_clientes[es_digital] = 0,
    scoring_clientes[score_propension_digital] >= 0.6
)

Score Promedio Propensión =
AVERAGE(scoring_clientes[score_propension_digital])

// --- Segmentación ---

Tasa Digital Con Nómina =
DIVIDE(
    CALCULATE(COUNTROWS(scoring_clientes),
              scoring_clientes[es_digital] = 1,
              scoring_clientes[tiene_nomina] = 1),
    CALCULATE(COUNTROWS(scoring_clientes),
              scoring_clientes[tiene_nomina] = 1)
) * 100

Tasa Digital Sin Nómina =
DIVIDE(
    CALCULATE(COUNTROWS(scoring_clientes),
              scoring_clientes[es_digital] = 1,
              scoring_clientes[tiene_nomina] = 0),
    CALCULATE(COUNTROWS(scoring_clientes),
              scoring_clientes[tiene_nomina] = 0)
) * 100

Brecha Digital Nómina =
[Tasa Digital Con Nómina] - [Tasa Digital Sin Nómina]

// --- Oportunidad por segmento ---

Oportunidad Conversión =
VAR ClientesNoDig =
    CALCULATE(
        COUNTROWS(scoring_clientes),
        scoring_clientes[es_digital] = 0,
        scoring_clientes[score_propension_digital] >= 0.6
    )
RETURN
    ClientesNoDig

Impacto Estimado Digitalización =
CALCULATE(
    COUNTROWS(scoring_clientes),
    scoring_clientes[es_digital] = 0,
    scoring_clientes[score_propension_digital] >= 0.5
)
"""


# ================================================================
# ESTRUCTURA SUGERIDA DEL DASHBOARD EN POWER BI
# ================================================================
"""
PÁGINA 1: "Panorama General"
┌──────────────────┬──────────────────┬──────────────────┐
│  KPI: Tasa       │  KPI: Total      │  KPI: Brecha     │
│  Adopción %      │  Clientes        │  Digital Nómina  │
├──────────────────┴──────────────────┴──────────────────┤
│                                                        │
│  [Gráfico de dona: Digital vs No Digital]              │
│                                                        │
├────────────────────────┬───────────────────────────────┤
│  Tabla: Tasa por       │  Mapa: Tasa por ciudad        │
│  segmento              │  (Colombia)                   │
│  - Con nómina          │                               │
│  - Sin nómina          │                               │
│  - Solo préstamo       │                               │
└────────────────────────┴───────────────────────────────┘

PÁGINA 2: "Modelo Predictivo"
┌──────────────────┬──────────────────┬──────────────────┐
│  KPI: AUC-ROC    │  KPI: Accuracy   │  KPI: Candidatos │
│  del modelo      │                  │  alto potencial  │
├──────────────────┴──────────────────┴──────────────────┤
│                                                        │
│  [Barras horizontales: Feature Importance / Coefs.]    │
│  (Archivo: coeficientes_modelo.csv)                    │
│                                                        │
├────────────────────────┬───────────────────────────────┤
│  Comparación modelos   │  Distribución de scores       │
│  (metricas_modelos.csv)│  (Visual Python)              │
│                        │                               │
└────────────────────────┴───────────────────────────────┘

PÁGINA 3: "Oportunidad de Negocio"
┌──────────────────┬──────────────────┬──────────────────┐
│  KPI: Clientes   │  KPI: Tasa       │  KPI: Impacto    │
│  oportunidad     │  conversión est. │  estimado        │
├──────────────────┴──────────────────┴──────────────────┤
│                                                        │
│  [Tabla: Top clientes no digitales con alto score]     │
│  (Filtrable por ciudad, nómina, tipo producto)         │
│                                                        │
├────────────────────────┬───────────────────────────────┤
│  Segmentos de acción:  │  Recomendaciones:             │
│  - Muy alto potencial  │  Campañas personalizadas por  │
│  - Alto potencial      │  segmento                     │
│  - Potencial medio     │                               │
└────────────────────────┴───────────────────────────────┘

CONEXIÓN DE DATOS:
1. Obtener datos → Texto/CSV
2. Seleccionar cada archivo de powerbi/ y data/
3. Crear relaciones entre tablas por cliente_id
4. Aplicar medidas DAX arriba
"""
