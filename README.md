# Réplica y extensión — Credit Scoring con Machine Learning (LendingClub)

**Curso:** Ciencia de Datos para la Economía — Profesor Luis Cuevas Parra
**Trabajo:** Portafolio semestral (réplica y extensión de un paper de Machine Learning)
**Integrantes:** Carla Aguilera, Joaquín Flores, Sebastián Gutiérrez, Benjamín Romero

## Paper replicado

> Moscato, V., Picariello, A., & Sperlí, G. (2021). *A benchmark of machine learning
> approaches for credit score prediction.* Expert Systems with Applications, 165, 113986.
> https://doi.org/10.1016/j.eswa.2020.113986

Revista Q1 (Artificial Intelligence / Computer Science Applications, SCImago 2024).

## Descripción del proyecto

Los autores comparan Regresión Logística, Random Forest y MLP, cada uno combinado con
distintas técnicas de balanceo de clases (submuestreo, sobremuestreo, híbridas), para
predecir si un préstamo P2P de LendingClub será pagado o caerá en default. Su
combinación ganadora es **Random Forest + Random Under-Sampling (RUS)**.

## Este repositorio:

## Pregunta de investigación e implementación de nuevo modelo

¿Qué tan bien predicen distintos algoritmos de ML el riesgo crediticio (default) de 
solicitantes de préstamos P2P bajo distintas técnicas de balanceo de clases, y qué 
combinación modelo + muestreo ofrece el mejor desempeño?

Se replica el diseño experimental del paper (misma fuente de datos, mismo filtro temporal 
2016-2017, misma variable objetivo) para **2 de sus modelos** (Regresión Logística + RUS, Random Forest + RUS/IHT).

Propone e implementa un **modelo adicional no usado por los autores**: XGBoost.

Compara el desempeño de todos los modelos entre sí y contra la Tabla 4 del paper
   original, usando las mismas métricas (AUC, Sensitivity, Specificity, G-Mean).

## Estructura del repositorio

```
.
├── README.md                      <- este archivo
├── requirements.txt                <- dependencias (pip install -r requirements.txt)
├── e-portafolio_ciencia_de_datos_2026_final_version.ipynb   <- cuaderno principal estrucuta informe
└── data/                           <- NO versionado en git; se genera al correr la
                                        Sección 2 del cuaderno (descarga desde Kaggle)
```

## Cómo ejecutar

### 1. Requisitos

- Python 3.10 o superior.
- Instalar dependencias:

  ```bash
  pip install -r requirements.txt
  ```

  (El propio cuaderno también instala lo que falte en su primera celda de código,
  vía `%pip install`.)

### 2. Datos

El dataset **"All Lending Club loan data"** se descarga automáticamente desde Kaggle al
correr la Sección 2 del cuaderno (no requiere cuenta de Kaggle para este dataset en
particular, ya que es público). Si la descarga automática falla, hay una opción manual
documentada en la misma sección: descargar el `.zip` desde
[kaggle.com/datasets/wordsforthewise/lending-club](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
y colocarlo en `data/`.

- **Filtro aplicado (igual que el paper):** solo préstamos originados en **2016–2017**.
- **Tamaño logrado:** 877.986 registros vs. 877.956 del paper (diferencia de 30 filas, ~0,003%) → réplica del filtrado prácticamente exacta.
- **Variable objetivo:** binaria — `Fully Paid` vs. `Charged Off`/equivalentes (default).
- **Variables predictoras:** se ampliaron de ~18 a **85** (77 numéricas + 8 categóricas, de 102 candidatas), usando solo información disponible *al momento de originación* del préstamo (sin fuga de información). El paper usa 151 variables no publicadas en detalle, por lo que no se pudo igualar exactamente.

### 3. Ejecutar el cuaderno

Correr con **Kernel → Restart & Run All**, de principio a fin. Notas de tiempo:

- El dataset crudo tiene ~2,3 millones de filas antes del filtro temporal — la carga
  inicial (Sección 2) puede tardar algunos minutos.
- Las celdas de `GridSearchCV` (Sección 4.5) y de validación cruzada (Sección 6.1)
  entrenan decenas de modelos — son las más lentas del cuaderno (varios minutos cada
  una).
- Todas las celdas con muestreo (imputación, VIF) usan una muestra fija de filas por
  costo computacional; esto es intencional y está documentado en cada sección, no es un
  atajo sin justificar.

## 4. Metodología

- Pipeline: imputación → escalamiento/codificación → balanceo de clases → clasificador (todo dentro de un `imblearn.Pipeline`, sin fuga de información train→test).
- **Modelos replicados del paper:** Regresión Logística + RUS, Random Forest + RUS (la combinación ganadora del paper), y Random Forest + IHT como comparación de técnicas de submuestreo.
- **Modelo propuesto (extensión):** **XGBoost**, con `scale_pos_weight` en vez de submuestreo, aprovechando su manejo nativo de valores faltantes.
- **Métricas:** AUC-ROC, Sensitivity/Recall, Specificity, G-Mean — igual que la Tabla 4 del paper.
- **Rigor estadístico agregado (más allá del paper):** validación cruzada estratificada (K=5) e intervalos de confianza bootstrap (2.000 réplicas) para determinar si las diferencias con el paper son ruido de muestreo o brechas reales.

  
## 5. Resumen de resultados

| Modelo | AUC (nuestro) | AUC (paper) | G-Mean (nuestro) | G-Mean (paper) |
|---|---|---|---|---|
| Regresión Logística + RUS | 0,720 | 0,710 | 0,659 | 0,650 |
| **Random Forest + RUS** (referente del paper) | 0,718 | **0,717** | 0,656 | **0,656** |
| Random Forest + IHT | 0,687 | 0,694 | 0,562 | 0,580 |
| **XGBoost** (modelo propuesto) | **0,733** | No aplica | **0,669** | No aplica |

Random Forest + RUS **prácticamente iguala** el AUC y G-Mean que el paper reporta como
su mejor combinación. XGBoost, no evaluado por los autores, supera a todos los modelos
del paper en toda métrica. El detalle completo, incluyendo las hipótesis puestas a
prueba para explicar los matices de la réplica, está en la Sección 7 del cuaderno.

## 6. Conclusiones

- La réplica es **fiel a nivel agregado** (AUC, G-Mean) pero revela diferencias sistemáticas a nivel de composición recall/specificity, bien documentadas y no atribuibles a ruido estadístico.
- El modelo propuesto (**XGBoost**) es el mejor de todos los evaluados, validando la intuición de que boosting suele superar a bagging en datos tabulares.
- Principales limitaciones: no se igualaron las 151 variables del paper (no publicadas), discrepancia notable en el % de faltantes de `emp_length` (93,3% paper vs. ~5,8% nuestro), y `GridSearchCV` no se extendió a RF+IHT ni XGBoost por costo computacional.

---
