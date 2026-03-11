# -*- coding: utf-8 -*-
"""
03 - IA Generativa: Automatización con Vertex AI (Gemini)
=========================================================
Proyecto: Adopción Digital — Entidad Financiera
Autora: Claudia Rocha

Este notebook muestra cómo usar IA generativa (Gemini de Google Cloud)
para automatizar tareas del día a día:
  1. Generar reportes ejecutivos a partir de datos
  2. Explicar resultados del modelo en lenguaje de negocio
  3. Generar queries SQL desde preguntas en español
  4. Documentar ETLs automáticamente

REQUISITOS:
  - Proyecto en Google Cloud con Vertex AI habilitado
  - pip install google-cloud-aiplatform
  - Autenticación: gcloud auth application-default login
"""

# ============================================================
# CELDA 1: Configuración
# ============================================================
# !pip install google-cloud-aiplatform pandas --quiet

import pandas as pd
import json

# --- Configuración de Google Cloud ---
PROJECT_ID = "tu-proyecto-gcp"        # ← CAMBIAR
LOCATION = "us-central1"              # ← Cambiar si tu proyecto está en otra región

# Importar Vertex AI
try:
    import vertexai
    from vertexai.generative_models import GenerativeModel, Part

    vertexai.init(project=PROJECT_ID, location=LOCATION)
    model = GenerativeModel("gemini-1.5-flash")  # Modelo rápido y económico
    GEMINI_DISPONIBLE = True
    print("✅ Vertex AI (Gemini) configurado correctamente")
except ImportError:
    GEMINI_DISPONIBLE = False
    print("⚠️ Vertex AI no disponible. Usando modo demo con respuestas simuladas.")
    print("   Para habilitar: pip install google-cloud-aiplatform")

def llamar_gemini(prompt, temperatura=0.3):
    """Llama a Gemini y retorna la respuesta como texto."""
    if GEMINI_DISPONIBLE:
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": temperatura,
                "max_output_tokens": 2048,
            }
        )
        return response.text
    else:
        return f"[MODO DEMO] Prompt enviado ({len(prompt)} caracteres). Instalar Vertex AI para respuesta real."

print("✅ Configuración lista")

# ============================================================
# CELDA 2: Cargar datos del modelo
# ============================================================
df = pd.read_csv('c:/Claudia/proyecto_adopcion_digital/data/scoring_clientes.csv')
coeficientes = pd.read_csv('c:/Claudia/proyecto_adopcion_digital/powerbi/coeficientes_modelo.csv')
metricas = pd.read_csv('c:/Claudia/proyecto_adopcion_digital/powerbi/metricas_modelos.csv')

# Calcular estadísticas clave
total_clientes = len(df)
digitales = df['es_digital'].sum()
no_digitales = total_clientes - digitales
tasa_digital = digitales / total_clientes * 100
oportunidad = len(df[(df['es_digital'] == 0) & (df['score_propension_digital'] >= 0.6)])
tasa_con_nomina = df[df['tiene_nomina'] == 1]['es_digital'].mean() * 100
tasa_sin_nomina = df[df['tiene_nomina'] == 0]['es_digital'].mean() * 100

print(f"✅ Datos cargados: {total_clientes:,} clientes")

# ============================================================
# CASO DE USO 1: Generar reporte ejecutivo automático
# ============================================================
"""
Este es el caso más valioso para Claudia:
Tomar los resultados del modelo → generar un reporte para gerencia automáticamente.
"""

prompt_reporte = f"""
Eres un analista senior de datos en una entidad financiera colombiana.
Genera un REPORTE EJECUTIVO profesional en español dirigido al Comité Directivo,
basado en los siguientes resultados del modelo de adopción digital.

DATOS DEL ANÁLISIS:
- Total de clientes con productos bancarios: {total_clientes:,}
- Clientes registrados como digitales: {digitales:,} ({tasa_digital:.1f}%)
- Clientes NO digitales: {no_digitales:,} ({100-tasa_digital:.1f}%)
- Tasa de adopción digital CON nómina: {tasa_con_nomina:.1f}%
- Tasa de adopción digital SIN nómina: {tasa_sin_nomina:.1f}%
- Clientes no digitales con alto potencial de conversión: {oportunidad:,}

VARIABLES MÁS IMPORTANTES DEL MODELO (ordenadas por impacto):
{coeficientes.to_string(index=False)}

MÉTRICAS DEL MODELO:
{metricas.to_string(index=False)}

FORMATO DEL REPORTE:
1. Resumen ejecutivo (3-4 líneas)
2. Hallazgos principales (bullet points)
3. Oportunidad de negocio (cuantificada)
4. Recomendaciones (acciones concretas)
5. Siguiente paso

Tono: profesional, orientado a decisiones de negocio, evitar jerga técnica excesiva.
"""

print("=" * 70)
print("📝 GENERANDO REPORTE EJECUTIVO CON IA...")
print("=" * 70)
reporte = llamar_gemini(prompt_reporte, temperatura=0.4)
print(reporte)

# Guardar reporte
with open('c:/Claudia/proyecto_adopcion_digital/data/reporte_ejecutivo_ia.txt', 'w', encoding='utf-8') as f:
    f.write(reporte)
print("\n✅ Reporte guardado en data/reporte_ejecutivo_ia.txt")


# ============================================================
# CASO DE USO 2: Explicar resultados del modelo para no técnicos
# ============================================================

prompt_explicacion = f"""
Eres un científico de datos que presenta a un gerente de banca retail en Colombia.
Explica los siguientes coeficientes de una regresión logística que predice
si un cliente se registrará como cliente digital.

COEFICIENTES:
{coeficientes.to_string(index=False)}

EXPLICACIÓN REQUERIDA:
- Qué significa cada coeficiente en lenguaje simple
- Cuáles son los 3 factores más importantes
- Una analogía fácil de entender para el odds_ratio
- Implicación para el negocio de cada hallazgo

No uses terminología estadística. Imagina que le explicas a alguien que
nunca ha visto un modelo de machine learning. Usa ejemplos concretos del
sector bancario colombiano.
"""

print("\n" + "=" * 70)
print("🎓 EXPLICACIÓN DEL MODELO PARA NO TÉCNICOS")
print("=" * 70)
explicacion = llamar_gemini(prompt_explicacion, temperatura=0.5)
print(explicacion)


# ============================================================
# CASO DE USO 3: Generar queries SQL desde preguntas en español
# ============================================================
"""
Claudia puede preguntar en español y obtener el SQL listo para BigQuery.
Esto acelera enormemente su trabajo diario.
"""

preguntas_negocio = [
    "¿Cuántos clientes de Bogotá con nómina no son digitales pero tienen más de 3 productos?",
    "¿Cuál es la tasa de adopción digital por ciudad, ordenada de mayor a menor?",
    "¿Cuáles son los 100 clientes con mayor score de propensión que no son digitales y tienen préstamo?",
]

print("\n" + "=" * 70)
print("🔍 GENERADOR SQL DESDE PREGUNTAS EN ESPAÑOL")
print("=" * 70)

esquema_tabla = """
TABLA: adopcion_digital.scoring_clientes
COLUMNAS:
  - cliente_id (STRING): ID único del cliente
  - edad (INT): Edad del cliente
  - ciudad (STRING): Ciudad del cliente (Bogota, Medellin, Cali, etc.)
  - canal_vinculacion (STRING): Canal por el que se vinculó (oficina, web, call_center, app_movil, corresponsal)
  - antiguedad_meses (INT): Meses desde que es cliente
  - num_productos (INT): Número de productos bancarios
  - tiene_nomina (INT): 1 si tiene nómina, 0 si no
  - tiene_prestamo (INT): 1/0
  - tiene_cuenta_ahorro (INT): 1/0
  - tiene_tarjeta_credito (INT): 1/0
  - tiene_cdt (INT): 1/0
  - monto_total_productos (FLOAT): Monto total en COP
  - es_digital (INT): 1 si es cliente digital, 0 si no
  - score_propension_digital (FLOAT): Score del modelo (0 a 1)
  - segmento_accion (STRING): Bajo potencial, Potencial medio, Alto potencial, Muy alto potencial
"""

for pregunta in preguntas_negocio:
    prompt_sql = f"""
Genera un query de BigQuery (SQL estándar de Google) para responder esta pregunta:

PREGUNTA: {pregunta}

ESQUEMA DE LA TABLA:
{esquema_tabla}

Responde SOLO con el query SQL, sin explicaciones adicionales.
Usa alias descriptivos en español para las columnas del resultado.
"""
    print(f"\n❓ {pregunta}")
    print("-" * 50)
    sql = llamar_gemini(prompt_sql, temperatura=0.1)
    print(sql)


# ============================================================
# CASO DE USO 4: Documentar ETL automáticamente
# ============================================================
"""
Claudia trabaja con ETLs complejos. La IA puede documentar automáticamente.
"""

ejemplo_etl = """
SELECT
    c.cliente_id,
    c.nombre,
    c.fecha_nacimiento,
    DATEDIFF(CURRENT_DATE(), c.fecha_nacimiento, YEAR) as edad,
    p.tipo_producto,
    p.monto,
    p.fecha_desembolso,
    CASE
        WHEN d.cliente_id IS NOT NULL THEN 1
        ELSE 0
    END as es_digital,
    n.valor_nomina,
    CASE
        WHEN n.valor_nomina IS NOT NULL THEN 1
        ELSE 0
    END as tiene_nomina
FROM clientes c
LEFT JOIN productos p ON c.cliente_id = p.cliente_id
LEFT JOIN clientes_digitales d ON c.cliente_id = d.cliente_id
LEFT JOIN nomina n ON c.cliente_id = n.cliente_id
WHERE p.estado = 'ACTIVO'
    AND c.estado_cliente != 'RETIRADO'
"""

prompt_doc = f"""
Eres un ingeniero de datos senior. Documenta el siguiente ETL/query SQL
generando:

1. Descripción general (2-3 líneas)
2. Tablas involucradas y su propósito
3. Lógica de negocio (qué hace cada JOIN y cada CASE)
4. Campos calculados y su significado
5. Filtros aplicados y su justificación
6. Posibles mejoras o riesgos

QUERY:
```sql
{ejemplo_etl}
```

Formato: documentación técnica en español, clara y concisa.
"""

print("\n" + "=" * 70)
print("📄 DOCUMENTACIÓN AUTOMÁTICA DE ETL")
print("=" * 70)
documentacion = llamar_gemini(prompt_doc, temperatura=0.3)
print(documentacion)


# ============================================================
# CASO DE USO 5: Generar narrativas para Power BI
# ============================================================

prompt_narrativa = f"""
Genera 3 narrativas cortas (2-3 oraciones cada una) para insertar como
texto dinámico en un dashboard de Power BI sobre adopción digital bancaria.

DATOS:
- Tasa de adopción digital general: {tasa_digital:.1f}%
- Tasa con nómina: {tasa_con_nomina:.1f}%
- Tasa sin nómina: {tasa_sin_nomina:.1f}%
- Clientes con oportunidad de conversión: {oportunidad:,}
- Total clientes: {total_clientes:,}

FORMATO:
Narrativa 1: Para el KPI principal (tasa general)
Narrativa 2: Para el hallazgo de nómina
Narrativa 3: Para la oportunidad de negocio

Tono: ejecutivo, directo, con datos específicos.
Contexto: entidad financiera colombiana.
"""

print("\n" + "=" * 70)
print("💬 NARRATIVAS PARA POWER BI")
print("=" * 70)
narrativas = llamar_gemini(prompt_narrativa, temperatura=0.5)
print(narrativas)


# ============================================================
# CELDA FINAL: Pipeline completo automatizado
# ============================================================
"""
PIPELINE COMPLETO:
  Datos (BigQuery) → Modelo (BigQuery ML / scikit-learn) → Scoring
  → IA Generativa (Gemini) → Reporte + Narrativas → Power BI

Para automatizar con Cloud Functions o Cloud Scheduler:
1. Subir este notebook como Cloud Function
2. Programar ejecución diaria/semanal
3. El reporte se genera y envía automáticamente
"""

print("\n" + "=" * 70)
print("🏁 PIPELINE COMPLETO — Resumen de archivos generados")
print("=" * 70)
print("""
📁 proyecto_adopcion_digital/
├── 📁 notebooks/
│   ├── 01_datos_sinteticos_y_eda.py    → Datos + Análisis Exploratorio
│   ├── 02_modelo_clasificacion.py      → Modelos ML (Logística, Árbol, RF)
│   └── 03_ia_generativa_gemini.py      → IA Generativa (este archivo)
├── 📁 sql/
│   └── 01_bigquery_ml_modelo_adopcion.sql → Modelo completo en SQL (BigQuery ML)
├── 📁 powerbi/
│   ├── guia_conexion_powerbi.py        → Guía + DAX + estructura dashboard
│   ├── coeficientes_modelo.csv         → Para visualizar en Power BI
│   ├── feature_importance.csv          → Importancia de variables
│   ├── metricas_modelos.csv            → Comparación de modelos
│   └── resumen_segmentos.csv           → Datos por segmento
├── 📁 prompts/
│   └── toolkit_prompts_claudia.md      → Plantillas de prompts diarios
├── 📁 data/
│   ├── dataset_adopcion_digital.csv    → Dataset consolidado
│   ├── scoring_clientes.csv            → Score por cliente
│   └── reporte_ejecutivo_ia.txt        → Reporte generado por IA
└── README.md                           → Guía del proyecto + Roadmap
""")
