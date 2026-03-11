# Proyecto: Modelo de Adopción Digital — Banca Colombiana
## Claudia Rocha | ADL Digital Lab

---

## Objetivo
Identificar **por qué los clientes con productos bancarios no están registrados como clientes digitales**, y construir un modelo predictivo que permita:
- Cuantificar el impacto de cada variable (nómina, edad, tipo de producto, etc.)
- Generar un score de propensión digital por cliente
- Dirigir campañas de digitalización al segmento con mayor oportunidad

---

## Estructura del Proyecto

```
proyecto_adopcion_digital/
├── notebooks/
│   ├── 01_datos_sinteticos_y_eda.py      ← EMPEZAR AQUÍ
│   ├── 02_modelo_clasificacion.py        ← Modelos ML
│   └── 03_ia_generativa_gemini.py        ← IA Generativa
├── sql/
│   └── 01_bigquery_ml_modelo_adopcion.sql ← Modelo en SQL puro
├── powerbi/
│   ├── guia_conexion_powerbi.py          ← Guía + DAX + layout
│   ├── coeficientes_modelo.csv           ← Datos para dashboard
│   ├── feature_importance.csv
│   ├── metricas_modelos.csv
│   └── resumen_segmentos.csv
├── prompts/
│   └── toolkit_prompts_claudia.md        ← 25+ plantillas de prompts
├── data/
│   ├── dataset_adopcion_digital.csv      ← Dataset consolidado
│   ├── scoring_clientes.csv              ← Score por cliente
│   └── reporte_ejecutivo_ia.txt          ← Reporte generado por IA
└── README.md                             ← Este archivo
```

---

## Cómo Empezar (Paso a Paso)

### Paso 1: Ejecutar el EDA con datos sintéticos
1. Abrir [Google Colab](https://colab.research.google.com/)
2. Subir `notebooks/01_datos_sinteticos_y_eda.py`
3. Ejecutar celda por celda
4. **Resultado**: Datos sintéticos + análisis exploratorio + validación de hipótesis

### Paso 2: Entrenar los modelos de clasificación
1. En Colab, subir `notebooks/02_modelo_clasificacion.py`
2. Subir también `data/dataset_adopcion_digital.csv` (generado en paso 1)
3. Ejecutar celda por celda
4. **Resultado**: 3 modelos entrenados + scoring de clientes + gráficos

### Paso 3 (Opción SQL): Modelo en BigQuery ML
1. Abrir [BigQuery Console](https://console.cloud.google.com/bigquery)
2. Subir el CSV como tabla a BigQuery
3. Ejecutar los queries de `sql/01_bigquery_ml_modelo_adopcion.sql` en orden
4. **Resultado**: Modelo creado directamente en SQL, sin Python

### Paso 4: Conectar a Power BI
1. Abrir Power BI Desktop
2. Obtener datos → Texto/CSV → seleccionar archivos de `powerbi/`
3. Crear medidas DAX (ver `powerbi/guia_conexion_powerbi.py`)
4. **Resultado**: Dashboard interactivo con resultados del modelo

### Paso 5: Activar IA Generativa
1. Configurar Vertex AI en Google Cloud (ver notebook 03)
2. Ejecutar `notebooks/03_ia_generativa_gemini.py`
3. **Resultado**: Reportes automáticos, SQL desde español, documentación de ETLs

---

## Roadmap Completo (16 semanas)

### Fase 1: Fundamentos (Semanas 1-4)
| Semana | Actividad | Entregable |
|--------|-----------|------------|
| 1-2 | Python esencial: Pandas, NumPy, Google Colab | Notebook de práctica |
| 2-3 | Estadística aplicada: correlación, datos categóricos | Análisis de las 2 tablas |
| 3-4 | BigQuery ML intro: CREATE MODEL con SQL | Primer modelo en SQL |

**Recursos gratuitos:**
- [Google Cloud Skills Boost: BigQuery ML](https://www.cloudskillsboost.google/)
- [Kaggle Learn: Intro to Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning)
- [StatQuest: Regresión Logística](https://www.youtube.com/c/joshstarmer)

### Fase 2: Caso de Uso — Modelo (Semanas 5-8)
| Semana | Actividad | Entregable |
|--------|-----------|------------|
| 5 | Feature engineering + preparación de datos | Dataset consolidado |
| 6 | EDA + validación de hipótesis (nómina) | Gráficos + hallazgos |
| 7 | Modelo: regresión logística + árbol de decisión | Modelo entrenado |
| 8 | Scoring + dashboard Power BI | Dashboard + reporte |

### Fase 3: IA Generativa (Semanas 9-12)
| Semana | Actividad | Entregable |
|--------|-----------|------------|
| 9 | Copilot para Power BI + DAX automático | Dashboard mejorado |
| 10 | Vertex AI / Gemini: reportes automáticos | Pipeline de reportes |
| 11 | SQL desde español + documentación ETLs | Herramientas internas |
| 12 | Prompt engineering aplicado | Toolkit de prompts |

### Fase 4: Escalamiento (Semanas 13-16)
| Semana | Actividad | Entregable |
|--------|-----------|------------|
| 13 | Nuevos casos: churn, propensión, anomalías | Modelos adicionales |
| 14 | MLOps básico: re-entrenamiento programado | Pipeline automatizado |
| 15-16 | Caso de éxito + presentación de resultados | Propuesta de valor |

---

## Decisiones Técnicas

| Decisión | Justificación |
|----------|---------------|
| **BigQuery ML como punto de entrada** | Permite hacer ML con SQL (la fortaleza de Claudia), sin Python |
| **Regresión logística como primer modelo** | Interpretable, regulatoriamente aceptable (SFC), responde el "por qué" |
| **Google Cloud como plataforma** | Ya es su ecosistema de trabajo |
| **Datos sintéticos primero** | Permite practicar sin riesgos antes de conectar datos reales |
| **Progresión SQL → Python → ML → IA** | Curva de aprendizaje gradual |

---

## Conexión a Datos Reales

Cuando Claudia esté lista para usar datos reales del banco, solo debe cambiar la fuente de datos:

**En Python (Colab):**
```python
from google.colab import auth
auth.authenticate_user()
df = pd.read_gbq("SELECT * FROM proyecto.tabla_productos", project_id="mi-proyecto")
```

**En BigQuery ML:**
Reemplazar `tu-proyecto.adopcion_digital.dataset_adopcion_digital` por la tabla real.

**En Power BI:**
Cambiar la fuente de CSV a conexión directa a BigQuery.

---

## Herramientas

| Herramienta | Para qué | Costo |
|---|---|---|
| Google Colab | Notebooks Python en la nube | Gratis |
| BigQuery ML | Modelos ML con SQL | Gratis (1 TB/mes) |
| Vertex AI / Gemini | IA generativa | Pay-per-use |
| Power BI Desktop | Dashboards | Ya disponible |
| scikit-learn | ML en Python | Gratis (open source) |

---

## Contacto y Soporte

Para dudas sobre la implementación, usar el toolkit de prompts en `prompts/toolkit_prompts_claudia.md` — contiene plantillas listas para cada tarea del día a día.
