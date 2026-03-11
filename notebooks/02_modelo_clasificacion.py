# -*- coding: utf-8 -*-
"""
02 - Modelo de Clasificación: Adopción Digital
===============================================
Proyecto: Modelo de Adopción Digital — Entidad Financiera
Autora: Claudia Rocha

Este notebook implementa el modelo de clasificación en Python/scikit-learn.
Es complementario al modelo en BigQuery ML (SQL).

CUÁNDO USAR CADA UNO:
- BigQuery ML (SQL): cuando los datos ya están en BigQuery, rápido, sin Python
- scikit-learn (este notebook): más control, más visualizaciones, más algoritmos

INSTRUCCIONES PARA GOOGLE COLAB:
1. Subir este archivo a Google Colab
2. Subir el archivo data/dataset_adopcion_digital.csv (generado por notebook 01)
3. Ejecutar celda por celda
"""

# ============================================================
# CELDA 1: Importar librerías
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, ConfusionMatrixDisplay
)

import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid")
print("✅ Librerías cargadas")

# ============================================================
# CELDA 2: Cargar datos
# ============================================================
"""
OPCIÓN A: Desde archivo local (datos sintéticos del notebook 01)
OPCIÓN B: Desde BigQuery (datos reales del banco)

Para Colab con BigQuery:
    from google.colab import auth
    auth.authenticate_user()
    df = pd.read_gbq("SELECT * FROM tu-proyecto.adopcion_digital.dataset_adopcion_digital",
                       project_id="tu-proyecto")
"""

# OPCIÓN A: Desde CSV local
df = pd.read_csv('c:/Claudia/proyecto_adopcion_digital/data/dataset_adopcion_digital.csv')

# Para Colab, cambiar la ruta o usar:
# from google.colab import files
# uploaded = files.upload()
# df = pd.read_csv('dataset_adopcion_digital.csv')

print(f"✅ Dataset cargado: {df.shape[0]:,} filas, {df.shape[1]} columnas")
print(f"   Distribución target: {df['es_digital'].value_counts().to_dict()}")
print(f"   % Digital: {df['es_digital'].mean()*100:.1f}%")
df.head()

# ============================================================
# CELDA 3: Preparación de datos para ML
# ============================================================

# Separar features y target
TARGET = 'es_digital'
EXCLUIR = ['cliente_id', 'es_digital']

# Variables categóricas que necesitan encoding
CATEGORICAS = ['ciudad', 'canal_vinculacion']

# Encoding de variables categóricas
le_dict = {}
df_ml = df.copy()
for col in CATEGORICAS:
    le = LabelEncoder()
    df_ml[col] = le.fit_transform(df_ml[col])
    le_dict[col] = le

# Feature: log del monto (distribución más normal)
df_ml['log_monto'] = np.log1p(df_ml['monto_total_productos'])

# Features finales
FEATURES = [c for c in df_ml.columns if c not in EXCLUIR + ['monto_total_productos']]

X = df_ml[FEATURES]
y = df_ml[TARGET]

# Dividir en train/test (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Datos preparados")
print(f"   Features ({len(FEATURES)}): {FEATURES}")
print(f"   Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")
print(f"   Balance train: {y_train.mean()*100:.1f}% digital")

# ============================================================
# CELDA 4: MODELO 1 — Regresión Logística
# ============================================================
"""
REGRESIÓN LOGÍSTICA: El modelo más interpretable para clasificación binaria.
Ideal para el sector financiero porque:
- Da coeficientes que explican el "por qué"
- Fácil de auditar (requerimiento regulatorio SFC)
- Robusto, no sobreajusta fácil
"""

# Escalar features (importante para regresión logística)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenar modelo
log_reg = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',  # Maneja desbalance de clases
    random_state=42
)
log_reg.fit(X_train_scaled, y_train)

# Predicciones
y_pred = log_reg.predict(X_test_scaled)
y_prob = log_reg.predict_proba(X_test_scaled)[:, 1]

# Métricas
print("=" * 60)
print("📊 REGRESIÓN LOGÍSTICA — Resultados")
print("=" * 60)
print(f"\nAccuracy:  {log_reg.score(X_test_scaled, y_test):.4f}")
print(f"AUC-ROC:   {roc_auc_score(y_test, y_prob):.4f}")
print(f"\n{classification_report(y_test, y_pred, target_names=['NO Digital', 'SÍ Digital'])}")

# Cross-validation (más robusto)
cv_scores = cross_val_score(log_reg, X_train_scaled, y_train, cv=5, scoring='roc_auc')
print(f"AUC-ROC Cross-Validation (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ============================================================
# CELDA 5: Interpretar coeficientes (EL VALOR DEL MODELO)
# ============================================================
"""
ESTO ES LO MÁS IMPORTANTE PARA EL NEGOCIO:
Los coeficientes dicen CUÁNTO contribuye cada variable a la adopción digital.
"""

# Coeficientes
coef_df = pd.DataFrame({
    'variable': FEATURES,
    'coeficiente': log_reg.coef_[0],
    'odds_ratio': np.exp(log_reg.coef_[0])
}).sort_values('coeficiente', ascending=True)

print("\n" + "=" * 60)
print("🔍 COEFICIENTES DEL MODELO (Interpretación de negocio)")
print("=" * 60)
print("\nOdds Ratio > 1: AUMENTA la probabilidad de ser digital")
print("Odds Ratio < 1: DISMINUYE la probabilidad de ser digital\n")

for _, row in coef_df.iterrows():
    direction = "↑" if row['coeficiente'] > 0 else "↓"
    print(f"   {direction} {row['variable']:<25} coef={row['coeficiente']:>7.3f}  "
          f"odds_ratio={row['odds_ratio']:>6.3f}")

# Visualizar
fig, ax = plt.subplots(figsize=(10, 8))
colors = ['#ff6b6b' if c < 0 else '#51cf66' for c in coef_df['coeficiente']]
bars = ax.barh(coef_df['variable'], coef_df['coeficiente'], color=colors, edgecolor='white')
ax.set_xlabel('Coeficiente (impacto en adopción digital)', fontsize=13)
ax.set_title('🔍 ¿Qué variables predicen la adopción digital?\n(Regresión Logística)',
             fontsize=14, fontweight='bold')
ax.axvline(x=0, color='black', linewidth=0.5)

# Anotar valores
for bar, (_, row) in zip(bars, coef_df.iterrows()):
    x = bar.get_width()
    ax.text(x + (0.02 if x >= 0 else -0.02), bar.get_y() + bar.get_height()/2,
            f'{row["odds_ratio"]:.2f}x',
            ha='left' if x >= 0 else 'right', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('c:/Claudia/proyecto_adopcion_digital/data/coeficientes_regresion_logistica.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("✅ Gráfico guardado")

# ============================================================
# CELDA 6: MODELO 2 — Árbol de Decisión
# ============================================================
"""
ÁRBOL DE DECISIÓN: Genera reglas legibles tipo:
"SI tiene_nomina = 0 Y edad > 55 Y num_productos = 1 → 90% NO digital"
Ideal para presentar a gerencia sin conocimiento técnico.
"""

tree = DecisionTreeClassifier(
    max_depth=4,                # Limitar profundidad para legibilidad
    min_samples_leaf=100,       # Mínimo de muestras por hoja
    class_weight='balanced',
    random_state=42
)
tree.fit(X_train, y_train)

y_pred_tree = tree.predict(X_test)
y_prob_tree = tree.predict_proba(X_test)[:, 1]

print("=" * 60)
print("🌳 ÁRBOL DE DECISIÓN — Resultados")
print("=" * 60)
print(f"\nAccuracy:  {tree.score(X_test, y_test):.4f}")
print(f"AUC-ROC:   {roc_auc_score(y_test, y_prob_tree):.4f}")
print(f"\n{classification_report(y_test, y_pred_tree, target_names=['NO Digital', 'SÍ Digital'])}")

# Reglas del árbol (ESTO ES ORO para presentar a gerencia)
print("\n" + "=" * 60)
print("📋 REGLAS DEL ÁRBOL DE DECISIÓN")
print("=" * 60)
rules = export_text(tree, feature_names=FEATURES, max_depth=4)
print(rules)

# Visualizar árbol
fig, ax = plt.subplots(figsize=(24, 12))
plot_tree(tree, feature_names=FEATURES, class_names=['NO Digital', 'SÍ Digital'],
          filled=True, rounded=True, fontsize=9, ax=ax, impurity=False, proportion=True)
ax.set_title('🌳 Árbol de Decisión — Adopción Digital\n(Para presentar a gerencia)',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('c:/Claudia/proyecto_adopcion_digital/data/arbol_decision.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("✅ Gráfico del árbol guardado")

# ============================================================
# CELDA 7: MODELO 3 — Random Forest (para mejor performance)
# ============================================================

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    min_samples_leaf=50,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

y_prob_rf = rf.predict_proba(X_test)[:, 1]

print("=" * 60)
print("🌲 RANDOM FOREST — Resultados")
print("=" * 60)
print(f"AUC-ROC: {roc_auc_score(y_test, y_prob_rf):.4f}")

# Feature importance
fi_df = pd.DataFrame({
    'variable': FEATURES,
    'importancia': rf.feature_importances_
}).sort_values('importancia', ascending=True)

fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(fi_df['variable'], fi_df['importancia'], color='#4dabf7', edgecolor='white')
ax.set_xlabel('Importancia', fontsize=13)
ax.set_title('🌲 Feature Importance (Random Forest)\n¿Qué variables son más importantes?',
             fontsize=14, fontweight='bold')
for i, (_, row) in enumerate(fi_df.iterrows()):
    ax.text(row['importancia'] + 0.005, i, f'{row["importancia"]:.3f}',
            va='center', fontsize=10)
plt.tight_layout()
plt.savefig('c:/Claudia/proyecto_adopcion_digital/data/feature_importance_rf.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("✅ Gráfico guardado")

# ============================================================
# CELDA 8: Comparación de modelos
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 8.1 Curvas ROC comparativas
modelos = {
    'Regresión Logística': y_prob,
    'Árbol de Decisión': y_prob_tree,
    'Random Forest': y_prob_rf
}

for nombre, probs in modelos.items():
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)
    axes[0].plot(fpr, tpr, label=f'{nombre} (AUC={auc:.3f})', linewidth=2)

axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('Curvas ROC — Comparación de Modelos', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)

# 8.2 Matriz de confusión (mejor modelo)
best_model_name = max(modelos, key=lambda m: roc_auc_score(y_test, modelos[m]))
best_probs = modelos[best_model_name]
best_pred = (best_probs >= 0.5).astype(int)

cm = confusion_matrix(y_test, best_pred)
ConfusionMatrixDisplay(cm, display_labels=['NO Digital', 'SÍ Digital']).plot(
    ax=axes[1], cmap='Blues', values_format='d'
)
axes[1].set_title(f'Matriz de Confusión — {best_model_name}', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('c:/Claudia/proyecto_adopcion_digital/data/comparacion_modelos.png',
            dpi=150, bbox_inches='tight')
plt.show()

print(f"\n🏆 Mejor modelo: {best_model_name} (AUC={roc_auc_score(y_test, best_probs):.4f})")

# ============================================================
# CELDA 9: Generar scoring de clientes (predicciones)
# ============================================================
"""
Score de propensión: probabilidad de que cada cliente NO digital
se convierta en digital. Útil para:
- Campañas dirigidas
- Priorización de esfuerzos
- Reporte a gerencia
"""

# Usar el mejor modelo para scoring
if best_model_name == 'Regresión Logística':
    all_probs = log_reg.predict_proba(scaler.transform(X))[:, 1]
elif best_model_name == 'Random Forest':
    all_probs = rf.predict_proba(X)[:, 1]
else:
    all_probs = tree.predict_proba(X)[:, 1]

# Agregar score al dataset
df_scoring = df.copy()
df_scoring['score_propension_digital'] = all_probs
df_scoring['segmento_accion'] = pd.cut(
    all_probs,
    bins=[0, 0.3, 0.5, 0.7, 1.0],
    labels=['Bajo potencial', 'Potencial medio', 'Alto potencial', 'Muy alto potencial']
)

# Clientes NO digitales con alto potencial (OPORTUNIDAD)
oportunidad = df_scoring[
    (df_scoring['es_digital'] == 0) &
    (df_scoring['score_propension_digital'] >= 0.6)
].sort_values('score_propension_digital', ascending=False)

print("=" * 60)
print("🎯 OPORTUNIDAD DE NEGOCIO: Clientes No Digitales con Alto Potencial")
print("=" * 60)
print(f"\n   Total clientes NO digitales: {(df_scoring['es_digital'] == 0).sum():,}")
print(f"   Con alto potencial (score ≥ 0.6): {len(oportunidad):,}")
print(f"   Porcentaje de oportunidad: {len(oportunidad) / (df_scoring['es_digital'] == 0).sum() * 100:.1f}%")

print(f"\n📊 Distribución por segmento de acción (clientes NO digitales):")
no_digitales = df_scoring[df_scoring['es_digital'] == 0]
for seg in ['Muy alto potencial', 'Alto potencial', 'Potencial medio', 'Bajo potencial']:
    n = (no_digitales['segmento_accion'] == seg).sum()
    pct = n / len(no_digitales) * 100
    bar = "█" * int(pct / 2)
    print(f"   {seg:<22} {n:>5,} ({pct:5.1f}%) {bar}")

# Guardar scoring
df_scoring.to_csv('c:/Claudia/proyecto_adopcion_digital/data/scoring_clientes.csv', index=False)
print(f"\n✅ Scoring guardado: data/scoring_clientes.csv")

# Top 20 candidatos
print(f"\n📋 Top 20 candidatos para digitalización:")
print(oportunidad[['cliente_id', 'edad', 'ciudad', 'tiene_nomina', 'num_productos',
                     'score_propension_digital']].head(20).to_string(index=False))

# ============================================================
# CELDA 10: Exportar para Power BI
# ============================================================
"""
Archivos que Claudia puede importar directamente a Power BI:
1. scoring_clientes.csv — Score de cada cliente
2. coeficientes.csv — Para visualizar el impacto de cada variable
3. metricas_modelos.csv — Comparación de modelos
"""

# Exportar coeficientes para Power BI
coef_df.to_csv('c:/Claudia/proyecto_adopcion_digital/powerbi/coeficientes_modelo.csv', index=False)

# Exportar feature importance
fi_df.to_csv('c:/Claudia/proyecto_adopcion_digital/powerbi/feature_importance.csv', index=False)

# Exportar métricas comparativas
metricas = []
for nombre, probs in modelos.items():
    pred = (probs >= 0.5).astype(int)
    metricas.append({
        'modelo': nombre,
        'accuracy': (pred == y_test).mean(),
        'auc_roc': roc_auc_score(y_test, probs),
        'precision_digital': classification_report(y_test, pred, output_dict=True)['1']['precision'],
        'recall_digital': classification_report(y_test, pred, output_dict=True)['1']['recall'],
    })
pd.DataFrame(metricas).to_csv('c:/Claudia/proyecto_adopcion_digital/powerbi/metricas_modelos.csv', index=False)

# Exportar resumen por segmento para Power BI
resumen_seg = df_scoring.groupby(['ciudad', 'tiene_nomina', 'es_digital']).agg(
    cantidad=('cliente_id', 'count'),
    score_promedio=('score_propension_digital', 'mean'),
    edad_promedio=('edad', 'mean'),
).reset_index()
resumen_seg.to_csv('c:/Claudia/proyecto_adopcion_digital/powerbi/resumen_segmentos.csv', index=False)

print("✅ Archivos exportados para Power BI:")
print("   powerbi/coeficientes_modelo.csv")
print("   powerbi/feature_importance.csv")
print("   powerbi/metricas_modelos.csv")
print("   powerbi/resumen_segmentos.csv")

# ============================================================
# CELDA 11: Resumen ejecutivo
# ============================================================
print("\n" + "=" * 70)
print("📝 RESUMEN EJECUTIVO — MODELO DE ADOPCIÓN DIGITAL")
print("=" * 70)
print(f"""
OBJETIVO: Identificar por qué los clientes con productos bancarios
no están registrados como clientes digitales.

MODELOS ENTRENADOS:
  • Regresión Logística (interpretable) — AUC: {roc_auc_score(y_test, y_prob):.4f}
  • Árbol de Decisión (reglas legibles)   — AUC: {roc_auc_score(y_test, y_prob_tree):.4f}
  • Random Forest (mejor rendimiento)     — AUC: {roc_auc_score(y_test, y_prob_rf):.4f}

HALLAZGOS CLAVE:
  1. ✅ HIPÓTESIS CONFIRMADA: La nómina es el principal predictor
  2. Variables más importantes: nómina > edad > cuenta ahorro > canal > num_productos
  3. Segmento crítico: clientes con solo préstamo y sin nómina ({len(oportunidad):,} candidatos)

RECOMENDACIÓN:
  • Usar Regresión Logística para explicar a gerencia (coeficientes claros)
  • Usar Random Forest para el scoring operativo (mejor predicción)
  • Campañas dirigidas al segmento de alto potencial

SIGUIENTE PASO:
  • Conectar a datos reales (BigQuery) reemplazando la lectura de CSV
  • Visualizar en Power BI (archivos ya exportados en powerbi/)
  • Integrar IA generativa para generar reportes automáticos (notebook 03)
""")
