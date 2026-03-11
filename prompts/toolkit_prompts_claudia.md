# Toolkit de Prompts — Claudia Rocha
## IA Generativa aplicada al día a día en Banca Digital (Colombia)

---

## 1. ANÁLISIS DE DATOS

### 1.1 Exploración rápida de una tabla
```
Actúa como analista de datos senior en banca colombiana.
Tengo una tabla con las siguientes columnas: [PEGAR COLUMNAS]
Los primeros 5 registros son: [PEGAR SAMPLE]

Genera:
1. Resumen de la tabla (qué representa, posibles usos)
2. Tipos de datos sugeridos para cada columna
3. Posibles problemas de calidad de datos
4. 5 preguntas de negocio que se podrían responder con estos datos
```

### 1.2 Validar calidad de datos
```
Actúa como ingeniero de datos especializado en ETL.
Genera un checklist de validación de calidad para una tabla con estas columnas:
[COLUMNAS]

Incluye:
- Verificaciones de nulos y duplicados
- Rangos válidos para campos numéricos
- Consistencia de formatos (fechas, IDs, montos en COP)
- Queries SQL de validación (BigQuery)
```

### 1.3 Análisis de distribución
```
Tengo un dataset de [N] clientes bancarios con las siguientes estadísticas:
[PEGAR df.describe()]

Interpreta estas estadísticas desde el punto de vista de negocio bancario:
- ¿Hay outliers?
- ¿Los rangos tienen sentido para una entidad financiera colombiana?
- ¿Qué transformaciones recomendarías antes de modelar?
```

---

## 2. SQL / BIGQUERY

### 2.1 Generar SQL desde pregunta de negocio
```
Genera un query de BigQuery (SQL estándar de Google) para responder:
"[TU PREGUNTA EN ESPAÑOL]"

Tabla: [NOMBRE_TABLA]
Columnas:
- [col1] (tipo): descripción
- [col2] (tipo): descripción

Requisitos:
- Usa aliases en español
- Agrega comentarios explicativos
- Optimiza para BigQuery (evitar SELECT *, usar particiones si aplica)
```

### 2.2 Optimizar un query lento
```
El siguiente query en BigQuery toma [X] segundos y procesa [Y] GB:

```sql
[PEGAR QUERY]
```

Optimízalo considerando:
1. ¿Se puede reducir el escaneo de datos?
2. ¿Hay JOINs que se puedan mejorar?
3. ¿Se debería crear una tabla intermedia o vista materializada?
4. ¿Hay funciones que se puedan reemplazar por alternativas más eficientes?
```

### 2.3 Documentar un query/ETL existente
```
Documenta el siguiente query SQL generando:
1. Descripción general (2-3 líneas)
2. Tablas involucradas y su propósito
3. Lógica de negocio de cada JOIN y CASE
4. Campos calculados y su significado
5. Filtros y su justificación
6. Posibles riesgos o mejoras

```sql
[PEGAR QUERY]
```
```

---

## 3. POWER BI / DAX

### 3.1 Generar medidas DAX
```
Actúa como experto en Power BI y DAX.
Necesito crear las siguientes medidas para un modelo de datos con estas tablas:
- [Tabla1]: columnas [...]
- [Tabla2]: columnas [...]

Medidas necesarias:
1. [Descripción de la medida 1]
2. [Descripción de la medida 2]
3. [Descripción de la medida 3]

Para cada medida genera:
- Código DAX completo
- Explicación de la lógica
- Formato sugerido (%, moneda COP, número)
```

### 3.2 Debugging de DAX
```
La siguiente medida DAX me da [ERROR/RESULTADO INCORRECTO]:

```dax
[PEGAR MEDIDA]
```

Contexto del modelo:
- Tabla de hechos: [nombre] con [N] filas
- Dimensiones: [tablas relacionadas]
- Relación: [tipo de relación]

Resultado esperado: [QUÉ DEBERÍA DAR]
Resultado actual: [QUÉ ESTÁ DANDO]

Diagnostica el error y corrige la medida.
```

### 3.3 Diseñar estructura de dashboard
```
Necesito diseñar un dashboard en Power BI para [AUDIENCIA] sobre [TEMA].

Datos disponibles:
- [Tabla/archivo 1]: [descripción]
- [Tabla/archivo 2]: [descripción]

Genera:
1. Estructura de páginas (3-4 páginas máximo)
2. KPIs principales para cada página
3. Tipo de visualización recomendado para cada insight
4. Filtros sugeridos (slicers)
5. Diseño visual (layout) en formato ASCII
```

---

## 4. MACHINE LEARNING

### 4.1 Seleccionar modelo adecuado
```
Tengo el siguiente problema de negocio en banca colombiana:
[DESCRIBIR EL PROBLEMA]

Datos disponibles:
- [N] registros
- Variable a predecir: [VARIABLE] (tipo: [continua/binaria/multiclase])
- Features disponibles: [LISTAR]

Consideraciones:
- El modelo debe ser interpretable (requisito regulatorio SFC)
- Los datos están en BigQuery
- Necesito explicar los resultados a gerencia no técnica

Recomienda:
1. Tipo de modelo más adecuado y por qué
2. Features engineering sugerido
3. Métricas de evaluación relevantes
4. Cómo implementarlo en BigQuery ML (con SQL)
```

### 4.2 Interpretar resultados de un modelo
```
Tengo un modelo de [TIPO] con estos resultados:

Métricas:
- Accuracy: [X]
- AUC-ROC: [X]
- Precision: [X]
- Recall: [X]

Feature importance (top 5):
1. [variable1]: [importancia]
2. [variable2]: [importancia]
...

Coeficientes (si regresión logística):
[PEGAR COEFICIENTES]

Explica estos resultados para:
A) Un informe técnico (equipo de datos)
B) Un resumen ejecutivo (gerencia, sin jerga técnica)
C) Una presentación de 3 slides (comité directivo)
```

---

## 5. REPORTES Y PRESENTACIONES

### 5.1 Generar resumen ejecutivo desde datos
```
Genera un resumen ejecutivo profesional (máximo 1 página) basado en:

DATOS:
[PEGAR estadísticas clave, resultados del análisis]

AUDIENCIA: [Comité directivo / Gerencia de Banca Digital / etc.]

FORMATO:
1. Titular con hallazgo principal
2. Contexto (2 líneas)
3. Hallazgos clave (3-5 bullets con datos)
4. Impacto en el negocio (cuantificado)
5. Recomendaciones (3 acciones concretas)
6. Siguiente paso

Tono: ejecutivo, basado en datos, orientado a acción.
Contexto: entidad financiera colombiana regulada por SFC.
```

### 5.2 Generar narrativa para Power BI
```
Genera una narrativa corta (2-3 oraciones) para insertar como texto
dinámico en un dashboard de Power BI.

KPI: [NOMBRE DEL KPI]
Valor actual: [VALOR]
Valor período anterior: [VALOR_ANTERIOR]
Contexto: [BREVE DESCRIPCIÓN]

La narrativa debe:
- Ser directa y ejecutiva
- Incluir el dato numérico
- Mencionar la tendencia (subió/bajó)
- Cerrar con una implicación de negocio
```

---

## 6. DOCUMENTACIÓN TÉCNICA

### 6.1 Documentar un proceso ETL
```
Documenta el siguiente proceso ETL para el diccionario de datos:

Fuentes: [LISTAR fuentes de datos]
Destino: [TABLA destino]
Frecuencia: [diario/semanal/mensual]
Herramienta: [Integration Services / Dataflow / etc.]

Pasos del ETL:
[DESCRIBIR pasos o pegar código]

Genera:
1. Diagrama de flujo (en texto)
2. Descripción de cada paso
3. Reglas de transformación
4. Validaciones aplicadas
5. Manejo de errores
6. SLA y dependencias
```

### 6.2 Crear diccionario de datos
```
Genera un diccionario de datos profesional para la tabla [NOMBRE]:

Columnas:
[PEGAR resultado de INFORMATION_SCHEMA o df.dtypes]

Muestra de datos:
[PEGAR primeros 3-5 registros]

Para cada columna incluir:
- Nombre técnico
- Nombre de negocio (en español)
- Tipo de dato
- Descripción
- Valores posibles / rango
- Regla de negocio asociada
- ¿Puede ser nulo?

Formato: tabla markdown para documentación interna.
```

---

## 7. USO DIARIO RÁPIDO

### 7.1 Debugear un error
```
Tengo este error en [BigQuery/Python/Power BI/SQL Server]:

Error: [PEGAR MENSAJE DE ERROR]

Código que lo produce:
[PEGAR CÓDIGO]

¿Qué lo causa y cómo lo soluciono?
```

### 7.2 Traducir entre herramientas
```
Convierte el siguiente [SQL Server/Oracle/PostgreSQL] a BigQuery (SQL estándar):

```sql
[PEGAR QUERY]
```

Nota las diferencias clave en sintaxis y funciones.
```

### 7.3 Revisar código antes de producción
```
Revisa este [SQL/Python/DAX] antes de desplegarlo a producción:

[PEGAR CÓDIGO]

Verifica:
1. ¿Hay errores lógicos?
2. ¿Es eficiente? ¿Se puede optimizar?
3. ¿Maneja casos borde (nulos, duplicados)?
4. ¿Cumple buenas prácticas?
5. ¿Hay riesgos de seguridad?
```

---

## TIPS GENERALES DE PROMPT ENGINEERING PARA CLAUDIA

1. **Siempre dar contexto bancario**: "en una entidad financiera colombiana regulada por SFC"
2. **Especificar la audiencia**: "para gerencia no técnica" vs "para el equipo de datos"
3. **Pedir formato específico**: "en tabla markdown", "en bullets", "en DAX"
4. **Incluir datos reales**: Pegar estadísticas, esquemas, errores exactos
5. **Iterar**: Si la primera respuesta no es perfecta, pedir ajustes específicos
6. **Cadena de prompts**: Usar la salida de un prompt como entrada del siguiente
