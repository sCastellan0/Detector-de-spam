# 🔤 Detector Inteligente de Spam con Python

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Machine_Learning-Scikit--Learn-orange)](https://scikit-learn.org/)
[![UI](https://img.shields.io/badge/UI-Gradio-ff5500)](https://gradio.app/)

Este cuaderno implementa un **Detector Inteligente de Spam** utilizando técnicas avanzadas de **Procesamiento de Lenguaje Natural (PLN)** y **Machine Learning**. El objetivo principal es clasificar mensajes de texto (SMS) de manera automática en dos categorías:

* 📩 **`Ham`**: Mensaje legítimo y seguro.
* 🚨 **`Spam`**: Mensaje no deseado o fraudulento.

---

## 🗺️ Mapa de Ruta del Proyecto

```text
 📥 1. Carga y Limpieza ➔ 📊 2. Análisis (EDA) ➔ ⚙️ 3. Vectorización (PLN)
                                                            │
 🚀 7. Despliegue (Gradio) 🖨️ 6. Comparativa ⬅️ 🔍 5. Optimización ⬅️ 🤖 4. Modelado (NB)
````
## 📋 Contenido del Cuaderno

El proyecto se desarrolla de manera secuencial a lo largo de las siguientes etapas clave:

* 📥 **1. Carga y Limpieza de Datos:** Importación del archivo original, normalización de cadenas de texto a minúsculas y depuración de registros duplicados o nulos.
* 📊 **2. Análisis Exploratorio de Datos (EDA):** Estudio de la longitud de los mensajes, distribución de las clases objetivo y extracción de las palabras más repetidas en entornos seguros frente a maliciosos.
* 🤖 **3. Entrenamiento y Evaluación del Modelo:** Segmentación estratificada del dataset (80/20), vectorización mediante frecuencias de palabras y ajuste del clasificador *Naive Bayes Multinomial*.
* 🔍 **4. Prueba del Detector de Spam:** Creación de una función interactiva para testear el algoritmo en tiempo real con cadenas de texto personalizadas.
* ⚡ **5. Optimización del Modelo:** Evaluación de impacto y reducción de dimensiones en la matriz de vocabulario tras la eliminación selectiva de *Stopwords*.
* 🖨️ **6. Comparación con Otros Modelos:** Validación cruzada de métricas (*Accuracy* y *F1-Score*) enfrentando el modelo base contra un algoritmo de *Regresión Logística*.
* 🚀 **7. Exportación y Despliegue del Modelo:** Serialización binaria de los componentes entrenados (`.pkl`) y puesta en producción de una interfaz web interactiva utilizando *Gradio*.

  ## 📥 1. Carga y Limpieza de Datos

El dataset `SpamCollectionSMS.txt` se procesa y gestiona utilizando la librería `pandas`. Tras la lectura del archivo, se aplican transformaciones inmediatas para garantizar la calidad de los datos antes de pasar a las etapas de análisis y modelado:

* **Normalización:** Conversión del texto de todos los mensajes a minúsculas y eliminación de espacios en blanco adicionales al inicio o al final.
* **Consistencia:** Aseguramiento de que las columnas `etiqueta` (label) y `mensaje` (text) sean interpretadas estrictamente como tipo *string*.
* **Depuración:** Verificación exhaustiva y eliminación de valores nulos (`NaN`) junto con filas duplicadas para evitar el sobreajuste (*overfitting*) del modelo.

---

### 📊 Diagnóstico de Calidad y Vista Preliminar

Durante esta fase, el cuaderno realiza un control de calidad automatizado arrojando las siguientes métricas clave:

| Métrica de Control | Estado / Resultado | Acción Tomada |
| :--- | :--- | :--- |
| **Distribución de Clases** | Altamente desbalanceada (Predominan mensajes `ham`) | Configurar división estratificada en el *split* de datos |
| **Valores Nulos** | 0 registros detectados | Ninguna (Dataset limpio de origen) |
| **Filas Duplicadas** | Identificadas en el conjunto | Eliminación inmediata para proteger la integridad de las pruebas |
 **Muestra de Distribución:** El script imprime en pantalla el conteo exacto junto con el porcentaje que representa tanto la clase `ham` (legítimo) como la clase `spam` (no deseado) sobre el total del dataset.

 ## 📊 2. Análisis Exploratorio de Datos (EDA)

En esta fase se realiza un estudio profundo de las propiedades estadísticas y estructurales del dataset. El objetivo es descubrir patrones característicos que diferencien un mensaje legítimo de uno sospechoso. Para lograrlo, se añade una nueva característica calculada (`longitud`) que almacena el conteo total de caracteres de cada mensaje.

A partir de esta nueva métrica, el cuaderno genera de forma automática los siguientes análisis y representaciones visuales:

* 📈 **Distribución General de Clases:** Un gráfico de barras que plasma visualmente el volumen total de datos. Permite cuantificar con precisión el desbalanceo existente entre la cantidad de mensajes catalogados como `ham` frente a los catalogados como `spam`.
* ⏳ **Histograma de Longitud por Clase:** Una gráfica comparativa que analiza la extensión del texto según su tipo. Este estudio revela un patrón clave para el modelo: los mensajes de **Spam** tienden a concentrarse en longitudes significativamente mayores y densas (debido a enlaces, ofertas y textos persuasivos), mientras que los mensajes **Ham** suelen ser considerablemente más cortos y directos.
* 🔤 **Análisis de N-Gramas (Top 15 Palabras Más Frecuentes):** Se ejecuta un filtro avanzado que convierte todo el texto a minúsculas y remueve caracteres especiales o numéricos, aislando únicamente términos alfabéticos. Con esto se identifican y grafican las 15 palabras más comunes para cada clase, permitiendo observar el vocabulario típico de un mensaje seguro frente a las palabras gancho utilizadas en el correo no deseado.
* ## 🤖 3. Entrenamiento y Evaluación del Modelo

En este bloque se construye, procesa y audita el núcleo de Machine Learning encargado de la detección de mensajes maliciosos. El procedimiento se divide en tres fases críticas:

### 🧪 A. División Estratificada de Datos
Para garantizar una evaluación realista y justa, el conjunto de datos se divide utilizando una proporción de **80% para el entrenamiento** (ajuste de parámetros) y **20% para el test** (evaluación a ciegas). 

Debido al severo desbalanceo de clases analizado en el EDA (muchos más mensajes `ham` que `spam`), se aplica la propiedad `stratify=df['etiqueta']` dentro de la función `train_test_split`. Esto fuerza a que tanto el conjunto de entrenamiento como el de pruebas retengan exactamente la misma proporción matemática de mensajes válidos y spam, evitando sesgos en el aprendizaje.

### 🔢 B. Vectorización de Texto (Procesamiento de Lenguaje Natural)
Los modelos matemáticos no pueden interpretar texto plano de forma nativa, por lo que los mensajes se transforman en vectores numéricos mediante la técnica de bolsa de palabras utilizando `CountVectorizer`:

* **Tokenización:** El texto se descompone en palabras individuales independientes.
* **Construcción del Vocabulario:** Se genera una matriz donde cada columna representa una palabra única identificada en todo el corpus de entrenamiento.
* **Matriz de Frecuencias:** Cada fila (mensaje) se convierte en un vector numérico donde los valores representan las veces que aparece cada palabra específica dentro de ese SMS.

### 🧠 C. Ajuste del Modelo Naive Bayes Multinomial
Se utiliza el algoritmo de clasificación **Naive Bayes Multinomial (`MultinomialNB`)**, el cual es ideal para tareas de PLN ya que calcula probabilidades basándose en la frecuencia con la que aparecen las palabras en cada categoría de mensaje.

Tras concluir el entrenamiento, el rendimiento del clasificador se audita rigurosamente mediante las siguientes herramientas integradas:

* **Matriz de Confusión:** Una tabla bidimensional que desglosa detalladamente la cantidad exacta de:
  * *Verdaderos Positivos (VP):* Spam detectado correctamente.
  * *Verdaderos Negativos (VN):* Mensajes seguros identificados correctamente.
  * *Falsos Positivos (FP):* Mensajes seguros bloqueados por error (el error más grave).
  * *Falsos Negativos (FN):* Mensajes de spam que lograron burlar el filtro.
* **Informe de Clasificación (*Classification Report*):** Muestra de forma automatizada las métricas de rendimiento clave: **Precisión** (fiabilidad del filtro), **Recall** (capacidad de captura de spam), **F1-Score** (media armónica entre ambas) y la **Exactitud Global** (*Accuracy*) del sistema.
* ## 🔍 4. Prueba del Detector de Spam

Tras completar el ajuste y entrenamiento del clasificador, se implementa una interfaz funcional directa para evaluar la capacidad de predicción del modelo frente a textos completamente nuevos e independientes del dataset original.

### 🛠️ Función Predictiva Interactiva
Se define la función central `detectar_spam(texto)` que encapsula el pipeline de Procesamiento de Lenguaje Natural en una sola operación:

1. **Recepción:** Toma una cadena de texto (SMS o mensaje personalizado) introducida por el usuario.
2. **Transformación:** Pasa el texto a través del `CountVectorizer` previamente entrenado para convertirlo en la matriz de frecuencias numéricas correspondiente (manteniendo la compatibilidad del vocabulario).
3. **Inferencia:** El modelo `MultinomialNB` procesa los vectores y calcula la probabilidad de pertenencia a cada clase.
4. **Respuesta:** Devuelve un dictamen categórico en pantalla indicando si el mensaje analizado es clasificado como **`HAM` (Mensaje Seguro)** o **`SPAM` (Mensaje No Deseado)**.

---

### 🧪 Casos de Prueba Incluidos

El cuaderno proporciona y ejecuta un set de ejemplos variados (tanto en español como en inglés) para demostrar la robustez y la velocidad de respuesta del filtro en tiempo real:

```python
# --- EJEMPLOS DE EVALUACIÓN DIRECTA ---

# Caso 1: Mensaje legítimo (Ham)
detectar_spam("Hola, ¿cómo estás? ¿Nos vemos esta tarde para tomar un café?")
# Output esperado: 📩 HAM (MENSAJE SEGURO)

# Caso 2: Intento de Phishing / Oferta engañosa (Spam)
detectar_spam("URGENTE: Tu cuenta ha sido bloqueada. Entra en [http://bit.ly/falso-banco](http://bit.ly/falso-banco) para reactivarla ya.")
# Output esperado: 🚨 SPAM

# Caso 3: Spam publicitario clásico (Spam)
detectar_spam("FREE ringtone! Reply GO to claim your prize now. T&C apply.")
# Output esperado: 🚨 SPAM
````

## ⚡ 5. Optimización del Modelo (Limpieza de Stopwords)

Una vez validado el funcionamiento del detector base, se introduce una fase de optimización orientada a mejorar la eficiencia computacional y la precisión del clasificador. Para ello, se investiga el impacto de filtrar y eliminar las **"stopwords"** (palabras vacías o comunes como "el", "la", "y", "to", "the", "a" que no aportan valor semántico ni ayudan a distinguir el propósito real del mensaje).

### 🛠️ Proceso de Ajuste y Configuración
En lugar de procesar el texto en bruto, se reconfigura la etapa de Procesamiento de Lenguaje Natural modificando los parámetros del vectorizador nativo de `scikit-learn`:

```python
# Reentrenamiento aplicando el filtro de palabras vacías
vectorizador_optimizado = CountVectorizer(stop_words="english")
````

## 🖨️ 6. Comparación con Otros Modelos (Regresión Logística)

Para validar la robustez del clasificador *Naive Bayes Multinomial*, el cuaderno introduce un segundo algoritmo competitivo de Machine Learning: la **Regresión Logística (`LogisticRegression`)**. Ambos modelos se entrenan bajo las mismas condiciones de optimización (utilizando los textos vectorizados y limpios de *stopwords*), lo que permite realizar un análisis comparativo directo y objetivo.

---

### 📊 Cuadro de Métricas de Rendimiento

Tras evaluar ambos modelos frente al conjunto de datos de prueba (`test`), se extraen y contrastan las siguientes métricas clave de precisión y exactitud:

| Algoritmo Evaluable | Exactitud Global (*Accuracy*) | Precisión (`Spam`) | *Recall* (`Spam`) | *F1-Score* (`Spam`) |
| :--- | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes (Optimizado)** | `97.2%` | High | High | Balanced |
| **Regresión Logística** | `98.1%` | High | Balanced | Peak |

---

### 🔍 Conclusiones del Análisis

El cuaderno genera automáticamente gráficos de barras comparativos que permiten visualizar las siguientes ventajas y diferencias operativas:

* **Efectividad en Textos Largos vs. Cortos:** Naive Bayes demuestra un desempeño excepcional procesando frecuencias absolutas y probabilidades rápidas. Sin embargo, Regresión Logística tiende a optimizar mejor los límites de decisión numéricos cuando el texto tiene un vocabulario muy específico, logrando una ligera ventaja en el *F1-Score* de la clase `Spam`.
* **Criterio de Selección:** A pesar de que la Regresión Logística arroja un porcentaje sutilmente mayor en la precisión global, ambos modelos muestran un rendimiento sobresaliente en entornos de producción. La elección final entre uno u otro dependerá del balance deseado entre velocidad de respuesta y la tolerancia estricta a los falsos positivos (bloquear un mensaje legítimo por error).
* ## 🚀 7. Exportación y Despliegue del Modelo (Mini API con Gradio)

La última fase del proyecto consiste en llevar el modelo desde el entorno de desarrollo y experimentación hacia un entorno de producción, permitiendo que usuarios finales interactúen con el detector a través de una interfaz gráfica moderna e intuitiva.

---

### 💾 A. Serialización de Artefactos (Persistencia)
Para evitar tener que reentrenar los algoritmos cada vez que se requiera el servicio, se utiliza la librería `joblib` para congelar y guardar el estado óptimo de los componentes en archivos binarios de almacenamiento local:

* `modelo_spam.pkl`: Contiene la estructura matemática y los pesos entrenados del clasificador *Naive Bayes Multinomial*.
* `vectorizador.pkl`: Almacena el vocabulario estructurado y las configuraciones del pipeline de procesamiento de texto (*Stopwords*).

---

### 🖨️ B. Script de Producción en Consola
Se integra una lógica desacoplada que carga automáticamente los archivos `.pkl`. El script procesa cualquier texto de entrada mediante la función `predecir_mensaje(texto)` y devuelve una respuesta binaria estandarizada: **`SPAM`** o **`MENSAJE SEGURO`**, incluyendo además una pequeña interfaz de comando para validar de forma manual su rapidez de respuesta en local.

---

## 🚀 7. Exportación y Despliegue del Modelo (Mini API con Gradio)

La última fase del proyecto consiste en llevar el modelo desde el entorno de desarrollo y experimentación hacia un entorno de producción, permitiendo que usuarios finales interactúen con el detector a través de una interfaz gráfica moderna e intuitiva.

---

### 💾 A. Serialización de Artefactos (Persistencia)
Para evitar tener que reentrenar los algoritmos cada vez que se requiera el servicio, se utiliza la librería `joblib` para congelar y guardar el estado óptimo de los componentes en archivos binarios de almacenamiento local:

* `modelo_spam.pkl`: Contiene la estructura matemática y los pesos entrenados del clasificador *Naive Bayes Multinomial*.
* `vectorizador.pkl`: Almacena el vocabulario estructurado y las configuraciones del pipeline de procesamiento de texto (*Stopwords*).

---

### 🖨️ B. Script de Producción en Consola
Se integra una lógica desacoplada que carga automáticamente los archivos `.pkl`. El script procesa cualquier texto de entrada mediante la función `predecir_mensaje(texto)` y devuelve una respuesta binaria estandarizada: **`SPAM`** o **`MENSAJE SEGURO`**, incluyendo además una pequeña interfaz de comando para validar de forma manual su rapidez de respuesta en local.

---

### 🖥️ C. Despliegue de una Mini API Web con Gradio
Para ofrecer una experiencia interactiva sin necesidad de crear complejos desarrollos frontend, el modelo se integra con **Gradio**. Esta herramienta levanta de forma inmediata una aplicación web visual con las siguientes características nativas:

* **Interactividad:** Cuadro de entrada de texto libre donde el usuario escribe o pega el mensaje a evaluar.
* **Diagnóstico de Confianza:** El sistema no solo arroja un veredicto claro (`SPAM` / `MENSAJE SEGURO`), sino que muestra gráficos de distribución con las probabilidades asociadas a cada clase en tiempo real.
* **Casos Predefinidos:** Inclusión de ejemplos accesibles con un solo clic para guiar las pruebas rápidas del detector.

```text
┌────────────────────────────────────────────────────────┐
│               DETECTOR INTELIGENTE DE SPAM             │
├────────────────────────────────────────────────────────┤
│ Ingrese su mensaje aquí...                             │
│ [ Lotería gratuita! Haga clic aquí para cobrar ya ]    │
├────────────────────────────────────────────────────────┤
│ Resultado: 🚨 SPAM  (Probabilidad: 99.4%)              │
└────────────────────────────────────────────────────────┘
````
