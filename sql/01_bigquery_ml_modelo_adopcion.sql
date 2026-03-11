-- ============================================================
-- BIGQUERY ML — Modelo de Adopción Digital
-- ============================================================
-- Proyecto: Identificar por qué clientes con productos bancarios
--           no están registrados como clientes digitales
-- Autora: Claudia Rocha
-- Plataforma: Google Cloud BigQuery
--
-- INSTRUCCIONES:
-- 1. Abrir BigQuery Console: https://console.cloud.google.com/bigquery
-- 2. Crear un dataset: CREATE SCHEMA IF NOT EXISTS `tu-proyecto.adopcion_digital`;
-- 3. Subir el CSV generado por el notebook 01 como tabla
-- 4. Ejecutar estos scripts en orden
-- ============================================================


-- ============================================================
-- PASO 0: Crear el dataset (ejecutar una sola vez)
-- ============================================================
CREATE SCHEMA IF NOT EXISTS `tu-proyecto.adopcion_digital`
OPTIONS(
  description = 'Proyecto de modelo de adopción digital - Claudia Rocha',
  location = 'us-central1'  -- Cambiar según tu región
);


-- ============================================================
-- PASO 1: Exploración de datos (equivalente al EDA en Python)
-- ============================================================

-- 1.1 Distribución general: digitales vs no digitales
SELECT
  es_digital,
  COUNT(*) AS cantidad,
  ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) AS porcentaje
FROM `tu-proyecto.adopcion_digital.dataset_adopcion_digital`
GROUP BY es_digital
ORDER BY es_digital;

-- 1.2 Validar hipótesis: nómina como predictor
SELECT
  tiene_nomina,
  COUNT(*) AS total_clientes,
  SUM(es_digital) AS digitales,
  ROUND(AVG(es_digital) * 100, 1) AS tasa_digital_pct
FROM `tu-proyecto.adopcion_digital.dataset_adopcion_digital`
GROUP BY tiene_nomina
ORDER BY tiene_nomina;

-- 1.3 Caso específico: solo préstamo, sin nómina
SELECT
  CASE
    WHEN tiene_nomina = 1 AND tiene_cuenta_ahorro = 1 THEN 'Nómina + Cuenta Ahorro'
    WHEN tiene_nomina = 1 THEN 'Con Nómina (otros)'
    WHEN tiene_nomina = 0 AND tiene_cuenta_ahorro = 1 THEN 'Sin Nómina + Cuenta Ahorro'
    WHEN tiene_nomina = 0 AND tiene_prestamo = 1 AND num_productos = 1 THEN '⚠️ Solo Préstamo (sin nómina)'
    ELSE 'Sin Nómina (otros)'
  END AS segmento,
  COUNT(*) AS total,
  SUM(es_digital) AS digitales,
  ROUND(AVG(es_digital) * 100, 1) AS tasa_digital_pct
FROM `tu-proyecto.adopcion_digital.dataset_adopcion_digital`
GROUP BY segmento
ORDER BY tasa_digital_pct;

-- 1.4 Tasa de adopción por ciudad
SELECT
  ciudad,
  COUNT(*) AS total,
  ROUND(AVG(es_digital) * 100, 1) AS tasa_digital_pct
FROM `tu-proyecto.adopcion_digital.dataset_adopcion_digital`
GROUP BY ciudad
ORDER BY tasa_digital_pct DESC;

-- 1.5 Tasa de adopción por rango de edad
SELECT
  CASE
    WHEN edad < 25 THEN '18-24'
    WHEN edad < 35 THEN '25-34'
    WHEN edad < 45 THEN '35-44'
    WHEN edad < 55 THEN '45-54'
    WHEN edad < 65 THEN '55-64'
    ELSE '65+'
  END AS rango_edad,
  COUNT(*) AS total,
  ROUND(AVG(es_digital) * 100, 1) AS tasa_digital_pct
FROM `tu-proyecto.adopcion_digital.dataset_adopcion_digital`
GROUP BY rango_edad
ORDER BY rango_edad;


-- ============================================================
-- PASO 2: CREAR MODELO DE REGRESIÓN LOGÍSTICA
-- ============================================================
-- Esta es la MAGIA de BigQuery ML: crear modelos con SQL puro
-- No necesitas Python, no necesitas instalar nada
-- ============================================================

CREATE OR REPLACE MODEL `tu-proyecto.adopcion_digital.modelo_regresion_logistica`
OPTIONS(
  model_type = 'LOGISTIC_REG',
  input_label_cols = ['es_digital'],
  auto_class_weights = TRUE,          -- Balancea clases (importante si hay desbalance)
  enable_global_explain = TRUE,        -- Habilita feature importance
  data_split_method = 'AUTO_SPLIT',    -- BigQuery divide automáticamente train/test
  max_iterations = 20
) AS
SELECT
  -- Features (variables predictoras)
  edad,
  ciudad,                              -- BigQuery ML maneja variables categóricas automáticamente
  canal_vinculacion,
  antiguedad_meses,
  num_productos,
  tiene_nomina,
  tiene_prestamo,
  tiene_cuenta_ahorro,
  tiene_tarjeta_credito,
  tiene_cdt,
  LOG(monto_total_productos + 1) AS log_monto,  -- Transformar monto (distribución sesgada)

  -- Variable objetivo
  es_digital
FROM `tu-proyecto.adopcion_digital.dataset_adopcion_digital`;

-- Tiempo estimado: 1-3 minutos


-- ============================================================
-- PASO 3: EVALUAR EL MODELO
-- ============================================================

-- 3.1 Métricas generales del modelo
SELECT
  *
FROM ML.EVALUATE(
  MODEL `tu-proyecto.adopcion_digital.modelo_regresion_logistica`
);
-- Buscar: precision, recall, accuracy, f1_score, log_loss, roc_auc

-- 3.2 Matriz de confusión
SELECT
  *
FROM ML.CONFUSION_MATRIX(
  MODEL `tu-proyecto.adopcion_digital.modelo_regresion_logistica`
);

-- 3.3 Curva ROC
SELECT
  *
FROM ML.ROC_CURVE(
  MODEL `tu-proyecto.adopcion_digital.modelo_regresion_logistica`
);


-- ============================================================
-- PASO 4: FEATURE IMPORTANCE (¿Qué variables importan más?)
-- ============================================================
-- Esto responde DIRECTAMENTE la pregunta de Claudia:
-- "¿Por qué los clientes no se digitalizan?"

-- 4.1 Importancia global de features
SELECT
  *
FROM ML.GLOBAL_EXPLAIN(
  MODEL `tu-proyecto.adopcion_digital.modelo_regresion_logistica`
)
ORDER BY attribution DESC;

-- 4.2 Coeficientes del modelo (pesos de cada variable)
SELECT
  *
FROM ML.WEIGHTS(
  MODEL `tu-proyecto.adopcion_digital.modelo_regresion_logistica`
)
ORDER BY ABS(weight) DESC;


-- ============================================================
-- PASO 5: PREDICCIONES — Scoring de clientes
-- ============================================================

-- 5.1 Predecir probabilidad de ser digital para TODOS los clientes
CREATE OR REPLACE TABLE `tu-proyecto.adopcion_digital.predicciones_clientes` AS
SELECT
  cliente_id,
  es_digital AS es_digital_real,
  predicted_es_digital AS prediccion,
  predicted_es_digital_probs
FROM ML.PREDICT(
  MODEL `tu-proyecto.adopcion_digital.modelo_regresion_logistica`,
  (SELECT * FROM `tu-proyecto.adopcion_digital.dataset_adopcion_digital`)
);

-- 5.2 Identificar clientes NO digitales con ALTA probabilidad de serlo
-- (Estos son los candidatos ideales para campañas de digitalización)
SELECT
  d.cliente_id,
  d.edad,
  d.ciudad,
  d.canal_vinculacion,
  d.num_productos,
  d.tiene_nomina,
  d.tiene_prestamo,
  p.prob AS probabilidad_digital
FROM `tu-proyecto.adopcion_digital.dataset_adopcion_digital` d
JOIN (
  SELECT
    cliente_id,
    prob.prob AS prob
  FROM `tu-proyecto.adopcion_digital.predicciones_clientes`,
  UNNEST(predicted_es_digital_probs) AS prob
  WHERE prob.label = 1
) p ON d.cliente_id = p.cliente_id
WHERE d.es_digital = 0           -- Solo no digitales actuales
  AND p.prob > 0.6               -- Con alta probabilidad de ser digital
ORDER BY p.prob DESC
LIMIT 1000;

-- 5.3 Resumen: oportunidad por segmento
SELECT
  CASE
    WHEN tiene_nomina = 1 THEN 'Con Nómina'
    WHEN tiene_prestamo = 1 AND num_productos = 1 THEN 'Solo Préstamo'
    ELSE 'Otros sin Nómina'
  END AS segmento,
  COUNT(*) AS clientes_no_digitales,
  ROUND(AVG(prob), 3) AS prob_promedio_digital,
  SUM(CASE WHEN prob > 0.6 THEN 1 ELSE 0 END) AS candidatos_alta_prob
FROM (
  SELECT
    d.*,
    p.prob
  FROM `tu-proyecto.adopcion_digital.dataset_adopcion_digital` d
  JOIN (
    SELECT
      cliente_id,
      prob.prob AS prob
    FROM `tu-proyecto.adopcion_digital.predicciones_clientes`,
    UNNEST(predicted_es_digital_probs) AS prob
    WHERE prob.label = 1
  ) p ON d.cliente_id = p.cliente_id
  WHERE d.es_digital = 0
)
GROUP BY segmento
ORDER BY prob_promedio_digital DESC;


-- ============================================================
-- PASO 6: MODELO COMPLEMENTARIO — ÁRBOL DE DECISIÓN (BOOSTED TREE)
-- ============================================================
-- Ventaja: genera reglas legibles, ideal para presentar a gerencia
-- "SI tiene_nomina=0 Y edad>50 Y num_productos=1 → 90% NO digital"

CREATE OR REPLACE MODEL `tu-proyecto.adopcion_digital.modelo_arbol_decision`
OPTIONS(
  model_type = 'BOOSTED_TREE_CLASSIFIER',
  input_label_cols = ['es_digital'],
  auto_class_weights = TRUE,
  num_parallel_tree = 1,        -- Un solo árbol para interpretabilidad
  max_tree_depth = 5,           -- Profundidad limitada para reglas legibles
  enable_global_explain = TRUE,
  data_split_method = 'AUTO_SPLIT'
) AS
SELECT
  edad,
  ciudad,
  canal_vinculacion,
  antiguedad_meses,
  num_productos,
  tiene_nomina,
  tiene_prestamo,
  tiene_cuenta_ahorro,
  tiene_tarjeta_credito,
  tiene_cdt,
  LOG(monto_total_productos + 1) AS log_monto,
  es_digital
FROM `tu-proyecto.adopcion_digital.dataset_adopcion_digital`;

-- Evaluar árbol
SELECT * FROM ML.EVALUATE(MODEL `tu-proyecto.adopcion_digital.modelo_arbol_decision`);

-- Feature importance del árbol
SELECT * FROM ML.GLOBAL_EXPLAIN(MODEL `tu-proyecto.adopcion_digital.modelo_arbol_decision`)
ORDER BY attribution DESC;

-- Comparar ambos modelos
SELECT 'Regresión Logística' AS modelo, * FROM ML.EVALUATE(MODEL `tu-proyecto.adopcion_digital.modelo_regresion_logistica`)
UNION ALL
SELECT 'Árbol de Decisión' AS modelo, * FROM ML.EVALUATE(MODEL `tu-proyecto.adopcion_digital.modelo_arbol_decision`);
