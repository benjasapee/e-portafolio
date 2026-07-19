# Guión de exposición oral

Réplica y extensión de Moscato, Picariello & Sperlí (2021), *A benchmark of machine
learning approaches for credit score prediction*. Duración objetivo: **12-15 minutos**.

Las referencias entre paréntesis (ej. "Sección 7.2") apuntan a las secciones del
cuaderno `e-portafolio_ciencia_de_datos_2026_final_version.ipynb` — úsenlas para ir
mostrando pantalla mientras hablan.

---

## 1. Paper (abstract) + introducción al código (3-4 min) — *Carla*

En este proyecto decidimos abordar uno de los problemas más críticos y costosos para
cualquier institución financiera hoy en día: la predicción del riesgo crediticio o *Credit
Scoring*. Para esto, nos basamos en un artículo publicado en 2021 en la revista *Expert
Systems with Applications*, titulado "A benchmark of machine learning approaches for
credit score prediction", de Moscato y su equipo.

El objetivo de los autores en este paper es muy claro: determinar qué algoritmo de
machine learning es el más exacto y confiable para predecir si un cliente de un
préstamo peer-to-peer va a pagar su deuda o va a caer en impago, lo que en finanzas
conocemos como default.

El gran desafío que plantea el artículo es que los datos reales están profundamente
desbalanceados. Afortunadamente, la inmensa mayoría de las personas paga sus
créditos. Sin embargo, si le entregamos estos datos crudos a un modelo predictivo, el
algoritmo se vuelve perezoso: predecirá que todo el mundo va a pagar, logrando una
exactitud que aparenta ser altísima, pero siendo completamente inútil para detectar los
verdaderos impagos.

Para solucionar esto, los autores evalúan distintas técnicas y concluyen que el modelo
ganador como benchmark es un Random Forest combinado con una técnica de
submuestreo llamada RUS (Random Under-Sampling), que recorta la clase mayoritaria
para equilibrar el entrenamiento.

Ahora bien, ¿qué es lo que van a ver en este código y cuál es nuestra propuesta de
valor? En este cuaderno no solo replicamos fielmente el estudio original utilizando casi
880.000 registros reales de la plataforma LendingClub de los años 2016 y 2017, sino
que fuimos más allá para darle un mayor rigor financiero y estadístico. A lo largo del
código verán tres grandes hitos:

Primero, un procesamiento exhaustivo donde filtramos y mantuvimos cerca de 85
variables que son estrictamente conocidas al momento de originar el crédito. Esto es
vital para evitar cualquier fuga de información y simular una decisión financiera real.

Segundo, nuestra extensión al paper: implementamos un modelo de vanguardia que los
autores no utilizaron, XGBoost, el cual maneja el desbalance de clases y los valores
nulos de forma nativa.

Y tercero, no nos quedamos con una simple métrica de precisión. Sometimos nuestros
resultados a validación cruzada y a 2.000 réplicas bootstrap para construir intervalos
de confianza al 95%. Con esto, demostramos matemáticamente que nuestras
conclusiones no son simple ruido estadístico.

Para mostrarles exactamente cómo construimos la base de este modelo, cómo lidiamos
con los valores nulos y por qué descartamos ciertas variables que los autores sí
usaron, los dejo con mi compañero, quien les explicará el tratamiento de los datos.

## 2. Datos: LendingClub, estructura, imputación y criterio de variables (3-4 min) — *Seba*

Gracias, Carla. Para replicar este estudio, lo primero era conseguir los mismos datos
que usaron los autores. Trabajamos con el dataset público de LendingClub, disponible
en Kaggle, y aplicamos exactamente el mismo filtro temporal que el paper: préstamos
originados en 2016 y 2017 (Sección 2.9). Esto nos dejó con casi 878 mil registros —
para ser exactos, 877.986, contra los 877.956 que reportan los autores. Una diferencia
de apenas 30 préstamos sobre casi 880 mil, lo que nos da mucha confianza de que
estamos replicando la misma población de datos que ellos.

La variable que queremos predecir es binaria: si el préstamo fue pagado en su
totalidad, o si cayó en default (Sección 3.1). Y aquí ya se ve el problema que
mencionaba Carla: solo un 23% de los casos son default, el resto son buenos
pagadores.

En cuanto a la estructura, sin entrar en el detalle de columna por columna, los datos se
agrupan en tres grandes bloques: las condiciones del préstamo en sí, el perfil del
solicitante, y su historial en el buró de crédito (Sección 2.7). Y dentro de eso nos
encontramos con un hallazgo interesante: había un grupo de columnas, las que
describen a un co-solicitante, que tenían entre un 97 y un 99% de valores nulos. Es
decir, prácticamente vacías. En vez de imputarlas con cualquier valor o simplemente
borrarlas, nos dimos cuenta de que la ausencia misma del dato era la señal útil: si el
campo está vacío, es porque la solicitud fue individual. Así que condensamos las 14
columnas en un solo indicador binario que dice si la solicitud fue conjunta o no
(Sección 3.2.1).

Para el resto de los valores faltantes, probamos tres estrategias de imputación: la
mediana, un método basado en vecinos más cercanos, y uno más sofisticado que
entrena un Random Forest para predecir cada valor faltante a partir de las demás
variables (Sección 3.6). Este último ganó en precisión, pero decidimos usar la mediana
en el pipeline final, porque con casi 370 mil filas de entrenamiento, el método más
sofisticado se volvía demasiado lento para ser práctico (Sección 3.7). Fue una decisión
consciente, priorizando que el cuaderno corriera en un tiempo razonable.

Ahora, un punto importante: el paper original trabaja con 151 variables, y nosotros no
llegamos a ese número exacto, porque los autores nunca publican la lista completa de
columnas que usaron. Lo que sí hicimos fue aplicar un criterio muy estricto y
replicable: solo incluimos información que estaba disponible al momento en que se
originó el préstamo (Sección 2.7). Esto significa que descartamos explícitamente todo
lo que reflejara el resultado del pago —cosas como cuánto se recuperó de un préstamo
en default—, porque usar esa información sería, en la práctica, hacerle trampa al
modelo: son datos que uno nunca tendría disponibles al momento real de decidir si
prestar o no. También sacamos identificadores, texto libre, y columnas redundantes.
Con ese criterio, terminamos con 85 variables finales: 77 numéricas y 8 categóricas
(Sección 3.4). No son las 151 del paper, pero es un número defendible y bien
justificado, no una decisión arbitraria.

Con los datos ya limpios y las variables definidas, le paso la palabra a Benjamín, que
les va a contar cómo entrenamos los modelos.

## 3. Train/test, modelos del paper y XGBoost (3-4 min) — *Benja*

Gracias. Con los datos listos, el primer paso fue separarlos en entrenamiento y
prueba, en una proporción 80-20, cuidando que la proporción de default se mantuviera
igual en ambos conjuntos, dado que es solo un 23% de los casos (Sección 3.5). Todo el
preprocesamiento —imputar, escalar, codificar las variables categóricas— se ajusta
únicamente con los datos de entrenamiento, para evitar que información del set de
prueba se filtre hacia el modelo antes de tiempo (Sección 3.7).

Ahora, el problema del desbalance que mencionaba Carla hay que resolverlo en el
entrenamiento. El paper prueba varias técnicas, pero la que gana es RUS, Random
Under-Sampling, que simplemente recorta la clase mayoritaria hasta igualarla con la
minoritaria. Nosotros elegimos replicar exactamente eso, en vez de usar SMOTE, que
es la alternativa más popular y que genera préstamos sintéticos para la clase
minoritaria. Y lo hicimos por dos razones: primero, fidelidad al paper, porque su
combinación ganadora usa RUS, no sobremuestreo, y queríamos comparar peras con
peras. Y segundo, por eficiencia: SMOTE termina generando millones de filas
sintéticas, mientras que RUS achica el set de entrenamiento, lo que lo hace mucho más
rápido de entrenar.

Con esa base, implementamos los dos modelos del paper: una Regresión Logística con
RUS, como modelo base e interpretable (Sección 4.2), y un Random Forest con RUS,
que es la combinación que los autores destacan como la mejor de todas (Sección 4.3).
Para este Random Forest, usamos 250 árboles, una profundidad máxima de 25, y un
mínimo de 2 observaciones por hoja. Estos números no los elegimos al azar: partimos
de una configuración más conservadora, y al ver que el modelo no mejoraba aunque le
diéramos más variables, la relajamos deliberadamente para que pudiera aprovechar
mejor la información disponible. Después confirmamos esta decisión con una
búsqueda formal de hiperparámetros (Sección 4.5, `GridSearchCV`), que no encontró
nada mejor que lo que ya teníamos.

Y finalmente, nuestro aporte: proponemos XGBoost como modelo adicional, uno que
los autores no evaluaron (Sección 5). La diferencia clave es que, en vez de construir
muchos árboles independientes como hace Random Forest, XGBoost construye
árboles de forma secuencial, donde cada uno corrige los errores del anterior. Esto suele
capturar mejor las interacciones no lineales entre variables. Además, maneja valores
nulos de forma nativa, así que no necesitamos imputar nada para este modelo, y en vez
de usar RUS, compensamos el desbalance con un parámetro propio de XGBoost que
pondera automáticamente la clase minoritaria.

Con los tres modelos entrenados, le paso la palabra a Joaquín, que les va a mostrar
qué tan bien les fue realmente, y cómo se comparan contra el paper.

## 4. Hallazgos, comparación con el paper y conclusiones (3-4 min) — *Joaquín*

Gracias. Esta es la parte donde ponemos a prueba todo lo que se construyó (Sección
6, comparación de resultados). Y el resultado más importante es este: nuestro Random
Forest con RUS prácticamente iguala al del paper. Ellos reportan un AUC de 0,717 y
nosotros obtuvimos 0,718; ellos reportan un G-Mean de 0,656 y nosotros llegamos
exactamente al mismo número (Sección 7.2, comparación final con la Tabla 4 del
paper). Es una réplica notablemente fiel.

Pero cuando miramos el detalle, encontramos un matiz interesante: nuestro modelo
detecta más defaults que el de los autores, pero a cambio rechaza menos buenos
pagadores que ellos (también en la Sección 7.2). Es decir, el balance interno entre
ambos errores es distinto, aunque el resultado agregado sea casi idéntico. Y en vez de
dejarlo ahí, decidimos investigar por qué pasaba esto (Sección 7.3). Probamos cuatro
explicaciones posibles, una por una: primero, si el umbral de decisión por defecto
estaba mal calibrado, y no era el caso (Sección 5.5, calibración del umbral). Segundo,
si nuestros hiperparámetros no eran los óptimos, y una búsqueda formal confirmó que
sí lo eran (Sección 4.5, `GridSearchCV`). Tercero, probamos entrenar con los
hiperparámetros exactos que reportan los autores, y el resultado empeoró en vez de
mejorar (Sección 4.3.1). Y cuarto, comparamos nuestra técnica de balanceo contra una
alternativa que no descarta datos, y ahí encontramos algo notable: el Random Forest es
mucho más sensible a la técnica de balanceo que usemos que la Regresión Logística,
y el número del paper queda casi exactamente en el punto medio de nuestros dos
extremos (Anexo A.2 y A.3). Esto nos dice que la brecha probablemente no viene de un
error nuestro, sino de diferencias no documentadas en el estudio original, como el set
exacto de variables que usaron.

Y para no quedarnos solo con un número de una sola corrida, validamos todo esto con
validación cruzada de 5 particiones (Sección 6.1) y con 2.000 réplicas bootstrap
(Sección 6.2), que nos permiten decir con confianza estadística que estas diferencias
son reales y consistentes, no ruido de muestreo.

¿Y cuál es el mejor modelo, al final? Según nuestras propias métricas, es XGBoost,
nuestro modelo propuesto: tiene el mejor AUC y el mejor G-Mean de todos, de forma
consistente (Sección 7.4, conclusión sobre el mejor modelo). Pero también
confirmamos el hallazgo central del paper: Random Forest con submuestreo es una
combinación sólida y reproducible, incluso con una fuente de variables distinta a la
original.

Como limitación honesta, no llegamos a las 151 variables del paper, porque nunca
publican la lista completa (Sección 7.5). Y como mejora futura, nos hubiese gustado
aplicar herramientas de explicabilidad como SHAP, que el paper sí usa, para entender
mejor qué variables pesan más en cada predicción individual — lo evaluamos, pero
decidimos priorizar el rigor estadístico de las comparaciones antes que la
explicabilidad (Sección 7.6).

Con esto cerramos la presentación. Muchas gracias, y quedamos abiertos a sus
preguntas.
