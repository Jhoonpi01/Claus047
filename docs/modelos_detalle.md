# Detalle de Modelos — Adopción Digital Bancaria

## Contexto del Problema

El caso de uso es **clasificación binaria**: predecir si un cliente con productos bancarios será (`1`) o no (`0`) cliente digital.

Se implementan **3 modelos con propósitos distintos** — cada uno responde una pregunta diferente del negocio.

---

## Modelo 1: Regresión Logística

**Archivos:** `notebooks/02_modelo_clasificacion.py` · `sql/01_bigquery_ml_modelo_adopcion.sql`

### ¿Qué hace?
Calcula la **probabilidad** de que cada cliente sea digital, asignando un peso (coeficiente) a cada variable.

**Fórmula:**
```
P(digital) = 1 / (1 + e^-(β0 + β1·nomina + β2·edad + β3·num_productos + ...))
```

Ejemplo: si el coeficiente de `tiene_nomina` es `1.8`, el odds ratio es `e^1.8 ≈ 6x` —
tener nómina **multiplica por 6** la probabilidad de ser cliente digital.

### Por qué usarlo

| Razón | Detalle |
|---|---|
| **Interpretabilidad** | Cada coeficiente tiene una traducción directa en lenguaje de negocio |
| **Regulación SFC** | La Superfinanciera exige modelos explicables y auditables en entidades financieras |
| **Responde el "por qué"** | No solo predice — cuantifica el impacto de cada variable |
| **Robusto con pocos datos** | No necesita millones de registros para funcionar bien |

### Output clave
Tabla de coeficientes → base del dashboard en Power BI.

---

## Modelo 2: Árbol de Decisión

**Archivos:** `notebooks/02_modelo_clasificacion.py` · `sql/01_bigquery_ml_modelo_adopcion.sql`

### ¿Qué hace?
Genera un conjunto de **reglas binarias anidadas** que segmentan a los clientes:

```
¿tiene_nomina = SI?
├── SÍ → ¿tiene_cuenta_ahorro = SI?
│         ├── SÍ → 87% digital ✅
│         └── NO → 61% digital ⚠️
└── NO → ¿edad < 35?
          ├── SÍ → 28% digital ❌
          └── NO → 11% digital ❌
```

### Por qué usarlo

| Razón | Detalle |
|---|---|
| **Reglas legibles** | El gerente entiende "SI no tiene nómina Y tiene solo un préstamo → 90% no se digitaliza" sin saber nada de ML |
| **Segmentos accionables** | Cada rama del árbol es un segmento al que se puede dirigir una campaña específica |
| **Captura relaciones no lineales** | La regresión logística asume relaciones lineales; el árbol no |
| **Visual en Power BI** | El árbol se puede ilustrar como diagrama de flujo en el dashboard |

### Output clave
Reglas de negocio en texto plano + visualización del árbol para presentaciones a gerencia.

---

## Modelo 3: Random Forest

**Archivo:** `notebooks/02_modelo_clasificacion.py`

### ¿Qué hace?
Entrena **100 árboles de decisión** en paralelo sobre subconjuntos aleatorios de los datos y promedia sus predicciones (método ensemble).

### Por qué usarlo

| Razón | Detalle |
|---|---|
| **Mejor precisión predictiva** | Casi siempre supera en AUC a los modelos anteriores |
| **Feature Importance robusta** | Al promediar 100 árboles, la importancia de variables es más confiable que en un solo árbol |
| **Resistente al overfitting** | El árbol individual puede memorizar los datos de entrenamiento; el forest generaliza mejor |
| **Scoring operativo** | Genera el `score_propension_digital` (0 a 1) por cliente, insumo directo para campañas |

### Output clave
Archivo `data/scoring_clientes.csv` con la probabilidad de digitalización de cada cliente.

---

## ¿Por qué los tres modelos juntos?

No se elige uno y se descartan los otros — cada modelo **responde una pregunta diferente del negocio**:

| Objetivo | Modelo Recomendado | Razón |
|---|---|---|
| Presentar a gerencia / comité directivo | **Regresión Logística** | Coeficientes = impacto claro en lenguaje de negocio |
| Segmentar clientes para campañas | **Árbol de Decisión** | Reglas legibles por segmento |
| Generar score de propensión | **Random Forest** | Mejor predicción numérica |

---

## Comparación de Métricas Esperadas

| Modelo | AUC-ROC esperado | Interpretabilidad | Uso principal |
|---|---|---|---|
| Regresión Logística | ~0.82 | ★★★★★ | Explicar a gerencia |
| Árbol de Decisión | ~0.78 | ★★★★★ | Reglas por segmento |
| Random Forest | ~0.88 | ★★ | Scoring operativo |

### ¿Qué es el AUC-ROC?
Mide qué tan bien separa el modelo los clientes digitales de los no digitales.
- `0.5` = el modelo no sabe (equivale a tirar una moneda)
- `0.88` = en el 88% de los casos, el modelo le asigna mayor probabilidad a un cliente digital real que a uno no digital

---

## Flujo de Implementación

```
Datos (BigQuery / CSV)
        │
        ▼
Feature Engineering
(unir tablas, crear variable es_digital, calcular features)
        │
        ├──► Regresión Logística ──► Coeficientes → Power BI (informe ejecutivo)
        │
        ├──► Árbol de Decisión ───► Reglas legibles → Segmentación de campañas
        │
        └──► Random Forest ────────► Score 0-1 por cliente → CRM / campañas dirigidas
```

---

## Variables del Modelo (Features)

| Variable | Tipo | Hipótesis | Hallazgo esperado |
|---|---|---|---|
| `tiene_nomina` | Binaria | Principal predictor (hipótesis Claudia) | Mayor impacto positivo |
| `edad` | Numérica | Clientes jóvenes más digitales | Coeficiente negativo para edad alta |
| `num_productos` | Numérica | Más productos → más interacción con el banco | Impacto positivo moderado |
| `tiene_cuenta_ahorro` | Binaria | Cuenta ahorro implica uso habitual de la app | Impacto positivo |
| `tiene_tarjeta_credito` | Binaria | Tarjeta crédito se gestiona digitalmente | Impacto positivo |
| `canal_vinculacion` | Categórica | Vinculados por web/app ya tenían intención digital | Impacto positivo en web/app |
| `antiguedad_meses` | Numérica | Clientes muy nuevos no han completado onboarding | Impacto mixto |
| `monto_total_productos` | Numérica (log) | Clientes con más saldo tienen más incentivo para gestionar digitalmente | Impacto positivo bajo |

---

## Consideraciones para Datos Reales

Antes de usar con datos reales del banco, revisar:

1. **Balance de clases**: Si hay muy pocos clientes digitales vs no digitales, usar `class_weight='balanced'` (ya incluido en el código)
2. **Variables sensibles**: No incluir variables como género, etnia o religión (regulación y ética)
3. **Data leakage**: Asegurar que las features se calculen con datos disponibles ANTES del momento de predicción
4. **Ventana temporal**: Definir claramente el período de observación (ej: clientes con productos al corte de [fecha])
5. **Monitoreo**: Re-entrenar el modelo cada trimestre o cuando el AUC baje más de 5 puntos
