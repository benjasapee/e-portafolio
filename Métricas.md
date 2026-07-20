---
title: "Conceptos clave: métricas de clasificación en clases desbalanceadas"
subtitle: "Guía de bolsillo — Réplica Moscato, Picariello & Sperlí (2021)"
author: "Carla Aguilera · Sebastián Gutiérrez · Benjamín Romero · Joaquín Flores"
geometry: margin=1.7cm
fontsize: 10pt
mainfont: "Helvetica"
---

## 1. La matriz de confusión (base de todo lo demás)

Para un umbral de decisión dado, cada predicción cae en una de cuatro celdas:

|                     | Predicho: paga (0) | Predicho: default (1) |
|---------------------|:---:|:---:|
| **Real: paga (0)**    | TN (verdadero negativo) | FP (falso positivo) |
| **Real: default (1)** | FN (falso negativo)     | TP (verdadero positivo) |

```python
from sklearn.metrics import confusion_matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
```

## 2. Error tipo I y tipo II

Vienen de la estadística inferencial clásica (contraste de hipótesis), donde
H0 = "el cliente paga". Se trasladan directo a la matriz de confusión:

- **Error tipo I** (falso positivo, *FP*): rechazar H0 siendo verdadera -> aquí, marcar
  como default a un buen pagador. Costo: oportunidad (interés no cobrado, cliente
  perdido).
- **Error tipo II** (falso negativo, *FN*): no rechazar H0 siendo falsa -> aquí, no
  detectar un default real. Costo: pérdida directa del capital prestado.

En este problema el error tipo II suele ser más costoso, por eso se prioriza
**Recall** (minimizar FN) sin descuidar del todo la Specificity (no disparar FP sin
control).

## 3. Métricas usadas en el trabajo

**Recall / Sensitivity / TPR** — de los defaults reales, ¿cuántos detecta el modelo?

$$Recall = \dfrac{TP}{TP + FN}$$

```python
from sklearn.metrics import recall_score
recall = recall_score(y_test, y_pred)          # equivalente a TP/(TP+FN)
```

**Specificity / TNR** — de los buenos pagadores reales, ¿a cuántos identifica bien?
(sklearn no la tiene directa; se calcula desde la matriz de confusión)

$$Specificity = \dfrac{TN}{TN + FP}$$

```python
specificity = tn / (tn + fp)
```

**Precision** — de todo lo que el modelo marcó como default, ¿cuánto era default real?

$$Precision = \dfrac{TP}{TP + FP}$$

**Accuracy** — % de aciertos totales. **Engañosa acá**: con ~23% de default, predecir
"todos pagan" ya da ~77% de accuracy sin detectar un solo caso real.

$$Accuracy = \dfrac{TP + TN}{TP + TN + FP + FN}$$

**G-Mean (media geométrica)** — la métrica principal del paper y de este trabajo.
Combina Recall y Specificity de forma que **ambas deben ser razonablemente altas**
para que el número final sea alto; si una es baja, el G-Mean cae fuerte (a diferencia
de un promedio simple, que "esconde" el desbalance).

$$G\text{-}Mean = \sqrt{Recall \times Specificity}$$

```python
from imblearn.metrics import geometric_mean_score
g_mean = geometric_mean_score(y_test, y_pred)   # = sqrt(recall * specificity)
```

**AUC-ROC** — mide qué tan bien el modelo *ordena* casos por riesgo, evaluando todos
los umbrales de decisión a la vez (no solo 0,5). 0,5 = azar, 1,0 = separación perfecta.

```python
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_test, y_proba[:, 1])      # y_proba: probabilidad de la clase 1
```

**F1-score** — media armónica de Precision y Recall; útil cuando ambas importan, pero
en este trabajo se prefiere G-Mean porque F1 ignora del todo la Specificity.

## 4. ¿Por qué no basta con Accuracy aquí?

Con clases desbalanceadas (~23% default / 77% paga), un modelo trivial que siempre
predice "paga" logra accuracy alta pero Recall = 0 (no detecta ningún default). Por eso
el paper —y este trabajo— reportan **Recall, Specificity y G-Mean**, no Accuracy, como
criterio principal de evaluación y de selección de umbral / hiperparámetros.

## 5. Resumen relación Recall <-> Precision <-> Errores

| Situación | Qué implica |
|---|---|
| Recall alto, Precision baja | Detecta casi todos los defaults reales, pero también marca a muchos buenos pagadores (más FP -> más error tipo I) |
| Recall bajo, Precision alta | Solo marca como default a casos muy claros, pero se le escapan muchos defaults reales (más FN -> más error tipo II) |
| G-Mean alto | Recall y Specificity son altos **simultáneamente** — el modelo no está sacrificando una clase por la otra |
