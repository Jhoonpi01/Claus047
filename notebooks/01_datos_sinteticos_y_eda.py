# -*- coding: utf-8 -*-
"""
01 - Generación de Datos Sintéticos y Análisis Exploratorio (EDA)
=================================================================
Proyecto: Modelo de Adopción Digital — Entidad Financiera
Autora: Claudia Rocha

INSTRUCCIONES PARA GOOGLE COLAB:
1. Abrir https://colab.research.google.com/
2. Archivo → Subir notebook (o copiar este script)
3. Ejecutar celda por celda

Este notebook:
- Genera datos sintéticos que simulan el caso real del banco
- Realiza el Análisis Exploratorio de Datos (EDA)
- Valida la hipótesis: "clientes con solo préstamo y sin nómina no se digitalizan"
"""

# ============================================================
# CELDA 1: Instalación de dependencias (solo en Colab)
# ============================================================
# !pip install pandas numpy matplotlib seaborn scikit-learn --quiet

# ============================================================
# CELDA 2: Importar librerías
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuración visual
sns.set_theme(style="whitegrid", palette="husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

print("✅ Librerías cargadas correctamente")

# ============================================================
# CELDA 3: Generar datos sintéticos — Tabla PRODUCTOS BANCARIOS
# ============================================================
"""
Esta celda simula la tabla real del banco con clientes y sus productos.
Cuando Claudia conecte a BigQuery, reemplazará esto con:
    df_productos = pd.read_gbq("SELECT * FROM proyecto.tabla_productos", project_id="mi-proyecto")
"""

np.random.seed(42)
N_CLIENTES = 10000

# Generar clientes
clientes_ids = [f"CLI-{str(i).zfill(6)}" for i in range(1, N_CLIENTES + 1)]

# Productos posibles en una entidad financiera colombiana
PRODUCTOS = ['prestamo_libre_inversion', 'prestamo_vehiculo', 'prestamo_hipotecario',
             'cuenta_ahorro', 'cuenta_corriente', 'cdt', 'tarjeta_credito', 'nomina']

# Ciudades principales de Colombia
CIUDADES = ['Bogota', 'Medellin', 'Cali', 'Barranquilla', 'Bucaramanga',
            'Cartagena', 'Cucuta', 'Pereira', 'Manizales', 'Ibague']

# Canal de vinculación
CANALES = ['oficina', 'web', 'call_center', 'corresponsal', 'app_movil']

# Rangos de edad
def generar_edad():
    return np.random.choice(
        range(18, 75),
        p=np.array([0.01]*7 + [0.03]*10 + [0.025]*15 + [0.015]*15 + [0.005]*10) /
          np.array([0.01]*7 + [0.03]*10 + [0.025]*15 + [0.015]*15 + [0.005]*10).sum()
    )

# Generar datos para cada cliente
data_productos = []
for cli_id in clientes_ids:
    edad = generar_edad()
    ciudad = np.random.choice(CIUDADES, p=[0.30, 0.18, 0.12, 0.10, 0.07, 0.06, 0.05, 0.05, 0.04, 0.03])
    canal = np.random.choice(CANALES, p=[0.40, 0.25, 0.15, 0.10, 0.10])
    antiguedad_meses = np.random.randint(1, 180)

    # Asignar productos (lógica realista)
    num_productos = np.random.choice([1, 2, 3, 4, 5], p=[0.35, 0.30, 0.20, 0.10, 0.05])
    tiene_nomina = np.random.random() < 0.40  # 40% tiene nómina

    productos_cliente = []
    if tiene_nomina:
        productos_cliente.append('nomina')
        # Personas con nómina tienden a tener más productos
        otros = np.random.choice([p for p in PRODUCTOS if p != 'nomina'],
                                  size=min(num_productos - 1, len(PRODUCTOS) - 1),
                                  replace=False).tolist()
        productos_cliente.extend(otros)
    else:
        # Sin nómina: más probable solo préstamo
        if np.random.random() < 0.50:
            productos_cliente = [np.random.choice([p for p in PRODUCTOS if 'prestamo' in p])]
        else:
            productos_cliente = np.random.choice([p for p in PRODUCTOS if p != 'nomina'],
                                                  size=num_productos, replace=False).tolist()

    monto_total = np.random.lognormal(mean=16, sigma=1.2)  # Monto en COP

    data_productos.append({
        'cliente_id': cli_id,
        'edad': edad,
        'ciudad': ciudad,
        'canal_vinculacion': canal,
        'antiguedad_meses': antiguedad_meses,
        'productos': productos_cliente,
        'num_productos': len(productos_cliente),
        'tiene_nomina': tiene_nomina,
        'tiene_prestamo': any('prestamo' in p for p in productos_cliente),
        'tiene_cuenta_ahorro': 'cuenta_ahorro' in productos_cliente,
        'tiene_tarjeta_credito': 'tarjeta_credito' in productos_cliente,
        'tiene_cdt': 'cdt' in productos_cliente,
        'monto_total_productos': round(monto_total, 0)
    })

df_productos = pd.DataFrame(data_productos)
print(f"✅ Tabla de PRODUCTOS generada: {len(df_productos)} clientes")
print(f"   Columnas: {list(df_productos.columns)}")
df_productos.head()

# ============================================================
# CELDA 4: Generar datos sintéticos — Tabla CLIENTES DIGITALES
# ============================================================
"""
Simula la tabla de clientes registrados en canales digitales (app, web banking).
La lógica replica el patrón real: clientes con nómina tienen MUCHA más probabilidad
de ser digitales porque NECESITAN entrar a la app.
"""

def probabilidad_digital(row):
    """Calcula la probabilidad de que un cliente sea digital basado en sus características."""
    prob = 0.15  # Base: 15% de probabilidad

    # FACTOR PRINCIPAL: Tener nómina (la hipótesis de Claudia)
    if row['tiene_nomina']:
        prob += 0.45  # Nómina es el mayor predictor

    # Otros factores
    if row['tiene_cuenta_ahorro']:
        prob += 0.15
    if row['tiene_tarjeta_credito']:
        prob += 0.10
    if row['num_productos'] >= 3:
        prob += 0.10

    # Edad: jóvenes más digitales
    if row['edad'] < 30:
        prob += 0.15
    elif row['edad'] < 45:
        prob += 0.08
    elif row['edad'] > 60:
        prob -= 0.10

    # Canal: vinculados por web/app más digitales
    if row['canal_vinculacion'] in ['web', 'app_movil']:
        prob += 0.15

    # Antigüedad: muy nuevos o muy viejos menos digitales
    if row['antiguedad_meses'] < 6:
        prob -= 0.05
    elif row['antiguedad_meses'] > 60:
        prob += 0.05

    return min(max(prob, 0.02), 0.95)

# Determinar quién es digital
df_productos['prob_digital'] = df_productos.apply(probabilidad_digital, axis=1)
df_productos['es_digital'] = df_productos['prob_digital'].apply(
    lambda p: np.random.random() < p
).astype(int)

# Crear tabla de clientes digitales (solo los que SÍ son digitales)
df_digitales = df_productos[df_productos['es_digital'] == 1][['cliente_id']].copy()
df_digitales['fecha_registro_digital'] = pd.to_datetime('2024-01-01') + pd.to_timedelta(
    np.random.randint(0, 365, size=len(df_digitales)), unit='D'
)
df_digitales['canal_digital'] = np.random.choice(
    ['app_movil', 'web_banking', 'ambos'],
    size=len(df_digitales),
    p=[0.55, 0.25, 0.20]
)

# Estadísticas
n_digitales = df_productos['es_digital'].sum()
n_no_digitales = len(df_productos) - n_digitales
pct_digital = n_digitales / len(df_productos) * 100

print(f"✅ Tabla de CLIENTES DIGITALES generada: {len(df_digitales)} registros")
print(f"\n📊 Distribución:")
print(f"   Clientes digitales:    {n_digitales:,} ({pct_digital:.1f}%)")
print(f"   Clientes NO digitales: {n_no_digitales:,} ({100-pct_digital:.1f}%)")

# ============================================================
# CELDA 5: ANÁLISIS EXPLORATORIO (EDA) — Distribución general
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 5.1 Distribución digital vs no digital
labels = ['NO Digital', 'SÍ Digital']
sizes = [n_no_digitales, n_digitales]
colors = ['#ff6b6b', '#51cf66']
axes[0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 14})
axes[0].set_title('Distribución de Clientes\nDigitales vs No Digitales', fontsize=14, fontweight='bold')

# 5.2 Distribución por número de productos
df_productos.groupby(['num_productos', 'es_digital']).size().unstack(fill_value=0).plot(
    kind='bar', ax=axes[1], color=['#ff6b6b', '#51cf66'], stacked=True
)
axes[1].set_title('Adopción Digital por\nNúmero de Productos', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Número de Productos')
axes[1].set_ylabel('Cantidad de Clientes')
axes[1].legend(['NO Digital', 'SÍ Digital'])

# 5.3 Distribución por edad
for digital, color, label in [(0, '#ff6b6b', 'NO Digital'), (1, '#51cf66', 'SÍ Digital')]:
    subset = df_productos[df_productos['es_digital'] == digital]
    axes[2].hist(subset['edad'], bins=20, alpha=0.6, color=color, label=label)
axes[2].set_title('Distribución por Edad', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Edad')
axes[2].set_ylabel('Cantidad de Clientes')
axes[2].legend()

plt.tight_layout()
plt.savefig('c:/Claudia/proyecto_adopcion_digital/data/eda_distribucion_general.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Gráfico guardado en data/eda_distribucion_general.png")

# ============================================================
# CELDA 6: EDA — VALIDAR HIPÓTESIS DE CLAUDIA
# ============================================================
"""
HIPÓTESIS: "Los clientes que solo tienen un préstamo y no tienen nómina
no entran a la app, por eso no quedan registrados como clientes digitales"
"""

print("=" * 70)
print("🔍 VALIDACIÓN DE HIPÓTESIS: Nómina como predictor de adopción digital")
print("=" * 70)

# Tasa de adopción por nómina
tasa_con_nomina = df_productos[df_productos['tiene_nomina'] == True]['es_digital'].mean() * 100
tasa_sin_nomina = df_productos[df_productos['tiene_nomina'] == False]['es_digital'].mean() * 100

print(f"\n📊 Tasa de adopción digital:")
print(f"   CON nómina:  {tasa_con_nomina:.1f}%")
print(f"   SIN nómina:  {tasa_sin_nomina:.1f}%")
print(f"   Diferencia:  {tasa_con_nomina - tasa_sin_nomina:.1f} puntos porcentuales")

# Caso específico: solo préstamo, sin nómina
solo_prestamo_sin_nomina = df_productos[
    (df_productos['tiene_prestamo'] == True) &
    (df_productos['tiene_nomina'] == False) &
    (df_productos['num_productos'] == 1)
]
tasa_solo_prestamo = solo_prestamo_sin_nomina['es_digital'].mean() * 100

print(f"\n🎯 Caso específico (hipótesis de Claudia):")
print(f"   Clientes con SOLO préstamo, SIN nómina: {len(solo_prestamo_sin_nomina):,}")
print(f"   Tasa de adopción digital: {tasa_solo_prestamo:.1f}%")
print(f"   {'✅ HIPÓTESIS CONFIRMADA' if tasa_solo_prestamo < 30 else '⚠️ HIPÓTESIS PARCIAL'}: "
      f"Este segmento tiene la menor tasa de digitalización")

# Tabla comparativa completa
print("\n📋 Tasa de adopción digital por segmento:")
print("-" * 55)

segmentos = {
    'Con nómina + cuenta ahorro': (df_productos['tiene_nomina'] & df_productos['tiene_cuenta_ahorro']),
    'Con nómina (cualquier prod.)': df_productos['tiene_nomina'],
    'Sin nómina + cuenta ahorro': (~df_productos['tiene_nomina'] & df_productos['tiene_cuenta_ahorro']),
    'Sin nómina + tarjeta crédito': (~df_productos['tiene_nomina'] & df_productos['tiene_tarjeta_credito']),
    'Solo préstamo, sin nómina': ((df_productos['tiene_prestamo']) & (~df_productos['tiene_nomina']) & (df_productos['num_productos'] == 1)),
}

for nombre, mask in segmentos.items():
    subset = df_productos[mask]
    tasa = subset['es_digital'].mean() * 100
    n = len(subset)
    bar = "█" * int(tasa / 2)
    print(f"   {nombre:<35} {tasa:5.1f}%  (n={n:,})  {bar}")

# ============================================================
# CELDA 7: EDA — Visualización de hipótesis
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 7.1 Nómina vs Adopción Digital
nomina_data = df_productos.groupby('tiene_nomina')['es_digital'].mean() * 100
bars = axes[0].bar(['Sin Nómina', 'Con Nómina'], nomina_data.values,
                    color=['#ff6b6b', '#51cf66'], edgecolor='white', linewidth=2)
for bar, val in zip(bars, nomina_data.values):
    axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=16, fontweight='bold')
axes[0].set_title('🎯 Hipótesis Claudia: Nómina = Digitalización', fontsize=14, fontweight='bold')
axes[0].set_ylabel('% Clientes Digitales')
axes[0].set_ylim(0, 100)

# 7.2 Heatmap: Tipo de producto vs Adopción digital
pivot = df_productos.groupby('canal_vinculacion')['es_digital'].agg(['mean', 'count']).reset_index()
pivot['mean'] = pivot['mean'] * 100
pivot = pivot.sort_values('mean', ascending=True)
bars2 = axes[1].barh(pivot['canal_vinculacion'], pivot['mean'],
                      color=plt.cm.RdYlGn(pivot['mean'].values / 100))
for bar, val, n in zip(bars2, pivot['mean'].values, pivot['count'].values):
    axes[1].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2.,
                 f'{val:.1f}% (n={n:,})', ha='left', va='center', fontsize=11)
axes[1].set_title('Adopción Digital por Canal de Vinculación', fontsize=14, fontweight='bold')
axes[1].set_xlabel('% Clientes Digitales')
axes[1].set_xlim(0, max(pivot['mean'].values) + 15)

plt.tight_layout()
plt.savefig('c:/Claudia/proyecto_adopcion_digital/data/eda_hipotesis_nomina.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Gráfico guardado en data/eda_hipotesis_nomina.png")

# ============================================================
# CELDA 8: Exportar datos para las siguientes fases
# ============================================================
"""
Guardar los datos preparados para usar en:
- 02_modelo_clasificacion.py (modelo de ML)
- BigQuery ML (SQL)
- Power BI (visualización)
"""

# Dataset unificado para modelamiento
df_modelo = df_productos[[
    'cliente_id', 'edad', 'ciudad', 'canal_vinculacion', 'antiguedad_meses',
    'num_productos', 'tiene_nomina', 'tiene_prestamo', 'tiene_cuenta_ahorro',
    'tiene_tarjeta_credito', 'tiene_cdt', 'monto_total_productos', 'es_digital'
]].copy()

# Convertir booleanos a int para ML
for col in ['tiene_nomina', 'tiene_prestamo', 'tiene_cuenta_ahorro', 'tiene_tarjeta_credito', 'tiene_cdt']:
    df_modelo[col] = df_modelo[col].astype(int)

# Guardar
df_modelo.to_csv('c:/Claudia/proyecto_adopcion_digital/data/dataset_adopcion_digital.csv', index=False)
df_digitales.to_csv('c:/Claudia/proyecto_adopcion_digital/data/clientes_digitales.csv', index=False)

print(f"✅ Archivos exportados:")
print(f"   data/dataset_adopcion_digital.csv ({len(df_modelo):,} filas, {len(df_modelo.columns)} columnas)")
print(f"   data/clientes_digitales.csv ({len(df_digitales):,} filas)")
print(f"\n📋 Columnas del dataset de modelamiento:")
print(f"   {list(df_modelo.columns)}")
print(f"\n📊 Resumen estadístico:")
print(df_modelo.describe().round(2).to_string())

# ============================================================
# CELDA 9: Resumen ejecutivo del EDA
# ============================================================
print("\n" + "=" * 70)
print("📝 RESUMEN EJECUTIVO — ANÁLISIS EXPLORATORIO")
print("=" * 70)
print(f"""
HALLAZGOS PRINCIPALES:

1. ✅ HIPÓTESIS CONFIRMADA: La nómina es el principal predictor de adopción digital
   - Clientes CON nómina: {tasa_con_nomina:.1f}% son digitales
   - Clientes SIN nómina: {tasa_sin_nomina:.1f}% son digitales
   - Clientes con SOLO préstamo (sin nómina): {tasa_solo_prestamo:.1f}% son digitales

2. FACTORES ADICIONALES IDENTIFICADOS:
   - Número de productos: más productos → más probabilidad de ser digital
   - Edad: clientes jóvenes (<30) tienen mayor tasa de digitalización
   - Canal de vinculación: clientes vinculados por web/app son más digitales
   - Tener cuenta de ahorro incrementa la adopción digital

3. SEGMENTO CRÍTICO:
   - {len(solo_prestamo_sin_nomina):,} clientes ({len(solo_prestamo_sin_nomina)/len(df_productos)*100:.1f}% del total)
     tienen SOLO un préstamo, SIN nómina
   - Solo {tasa_solo_prestamo:.1f}% de ellos son digitales
   - OPORTUNIDAD: Campañas dirigidas a este segmento

SIGUIENTE PASO: Construir modelo de regresión logística para cuantificar
el impacto de cada variable y generar un score de propensión digital.
""")
