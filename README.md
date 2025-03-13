<b><a href="#ENGLISH">ENGLISH</a>/<a href="#ESPAÑOL">ESPAÑOL</a></b>

<h3 id = "ENGLISH"><b>ENGLISH</b></h3>

<h1><b>League of Legends Predictions Model</b></h1>

<h2>📌 Table of Contents</h2>
<ul>
  <li><a href="#introduction">Introduction</a></li>
  <li><a href="#aws-architecture">AWS Deployment Architecture</a></li>
  <li><a href="#repository-structure">Repository Structure</a></li>
  <ul>
    <li><a href="#code-folder">Code Folder</a></li>
    <li><a href="#webpage">Webpage</a></li>
    <li><a href="#pytorch-model">Pytorch Model</a></li>
  </ul>
  <li><a href="#limitations">Limitations, Possible Improvements, and Future Work</a></li>
</ul>

<h2 id="introduction">Introduction:</h2>
This repository showcases a fully deployed machine learning solution on AWS for predicting outcomes of League of Legends professional matches. It is organized into three primary folders:

<ol>
<li> <b>code:</b> Contains all files required to implement the prediction model on <b>AWS SageMaker</b>.
  <ul>
    <li>Configuration scripts <em>(requirements.txt)</em></li>
    <li>Deployment files <em>(inference.py, lambda_handler.py, and model.tar.gz)</em></li>
  </ul>
</li>
<br>
<li>
  <b>Pytorch Model:</b> This folder contains all the files and subfolders needed to train, evaluate, and generate predictions for the League of Legends model using PyTorch. Broadly, it includes:
    <ul>
      <li><b>Data and References:</b> A dataset used for training and evaluation and JSON files with champion and player information (IDs, ratings).</li>
      <li><b>Model Architecture and Training:</b> Scripts defining the PyTorch model structure and training logic and saved model weights for quick loading and inference.</li>
      <li><b>Utilities:</b> Helper scripts for data preprocessing, rating calculations, and prediction workflows and serializedd pipelines or preprocessor files to ensure consistent data transformations.</li>
    </ul>
</li>
<br>
<li>
    <b>Webpage:</b> Serves as the static website deployed to an <b>Amazon S3 bucket</b>
            <ul>
                    <li>HTML/CSS/JS Files</li>
                    <li>Front-end logic</li>
                    <li>Basic user interface</li>
            </ul>
</li>
<br>
</ol>

<h2 id="aws-architecture">AWS Architecture:</h2>
<br>
<img src="https://github.com/user-attachments/assets/bc64eb63-3bcf-4f7e-a12b-4980bc3028c7" alt="Arquitectura de AWS" />
<br>

<h2 id="repository-structure"><p>Repository Structure:</p></h2>

<h3 id="code-folder">code:</h3>
<p>
  This folder contains everything needed to implement the AWS SageMaker model and the AWS Lambda function that will call the SageMaker Endpoint (which is linked to a REST API).
  There are two groups of files inside this folder:
</p>

<b>Inference Model files</b>: Files stored in an S3 bucket to serve an inference model created in SageMaker
<ul>
  <li>
    <b>requirements.txt:</b> All the libraries that need to be installed to run the inference model.
  </li>
  <li>
    <b>model.tar.gz:</b> This archive contains all the files from the "PytorchModel" folder that are required to deploy the model on AWS (all of them will be explained in detail in the PytorchModel folder section):
    <ul>
      <li>inference.py</li>
      <li>match_predictor_model.py</li>
      <li>model.pth</li>
      <li>data_preprocessing.py</li>
      <li>preprocessor.joblib</li>
      <li>champions_ids.json</li>
      <li>player_glicko_ratings.json</li>
      <li>players_ids.json</li>
      <li>requirements.txt</li>
    </ul>
    <p>
      The command used in the console to create <em>model.tar.gz</em> is:
    </p>
    <em>
      tar -czvf model.tar.gz code/inference.py PytorchModel/model.pth PytorchModel/models/match_predictor_model.py PytorchModel/utils/data_preprocessing.py PytorchModel/preprocessor.joblib PytorchModel/info/champions_ids.json PytorchModel/glicko.py PytorchModel/info/player_glicko_ratings.json PytorchModel/info/players_ids.json code/requirements.txt
    </em>
  </li>
  <li>
    <b>inference.py:</b> A script designed for hosting the PyTorch model on SageMaker. It handles four main tasks required by SageMaker for model serving:
    <ul>
      <li><b>Model Loading (model_fn):</b> Loads the trained PyTorch model from the specified <em>model_dir</em>, which in this case is simply “model.pth.”</li>
      <li><b>Data Preprocessing (input_fn):</b> Parses the JSON input and converts it into feature tensors (Champion IDs and team Glicko ratings).</li>
      <li><b>Model Inference (predict_fn):</b> Transforms the input tensors, runs them through the model, and outputs a raw probability for the predicted winner.</li>
      <li><b>Post-processing Output (output_fn):</b> Formats this probability into a JSON response.</li>
    </ul>
  </li>
</ul>
<br>
<b>Lambda Handler</b> (lambda_handler.py):  This is a function that takes a JSON event, sends it to a specified SageMaker endpoint for inference, and returns the prediction in the response body. The idea is to use this as a bridge between teh REST API and the SageMaker Endpoint.
<br>
<h3 id="webpage">Webpage</h3>
<p>
  This folder contains the files for a simple static website hosted in an S3 bucket and served by CloudFront. The website interacts with a REST API to display predictions from the predictions model, based on user input.
</p>

<h4>Main Functionality</h4>
<p>
  This webpage provides an interactive interface for League of Legends draft predictions. Users can view all available champions in a scrollable table and filter them with a search bar. Each champion is shown with an image and name for easy identification. The table is automatically populated from <code>champions_ids.json</code>, and any champion already selected for a team becomes disabled to avoid multiple selections of the same champion.
</p>

<p>
  To select champions, users drag a champion’s image from the table and drop it into one of ten “boxes”—five for the blue team and five for the red team, ordered based on roles (Top, Jungle, Mid, Adc, Support). A grayed-out effect indicates that the champion is no longer available. Each box includes a remove button, which, when clicked, clears that champion and re-enables it in the table for future selection.
</p>

<p>
  Alongside champion picks, there are inputs for player names. These fields provide real-time suggestions and validate entries against a locally fetched <code>players_ids.json</code> file. Any invalid name is highlighted with a red border and an error message, disabling the prediction buttons until it is corrected.
</p>

<p>
  Once all ten boxes (champions and player names) are validly filled, users can request a win-probability prediction. Depending on the button clicked—<strong>Predict</strong> or <strong>Predict Draft</strong>—the system either returns a single probability or calculates a comparative probability by swapping champion compositions between teams. The result displays the favored team and includes the percentage chance of winning.
</p>

<p>
  To streamline adjustments, users can:
  <ul>
    <li><strong>Reset Champions</strong> to clear all selected champions,</li>
    <li><strong>Reset Players</strong> to clear all player inputs, and</li>
    <li><strong>Swap Players</strong> or <strong>Swap Champions</strong> to exchange players or picks between teams.</li>
  </ul>
  These convenience features allow quick iterations of different combinations. Once a prediction is requested, the result appears in an overlay that can be closed anytime to continue refining the draft.
</p>

<h4>Note on <em>config.js</em> and the const <em>API_URL</em> </h4>
<p>
  For security reasons, the file named <code>config.js</code> containing the API URL as a constant is not included in this repository. You can create your own 
  <code>config.js</code> file locally, define <code>API_URL</code> within it, and reference it in your code to enable API requests without exposing sensitive information.
</p>

<h3 id="pytorch-model">PytorchModel</h3>

<p>This folder contains the core machine learning components of the model, structured as follows:</p>

<ul>
  <li><b>data/</b>
    <ul>
      <li><b>Datasheet.csv:</b> Dataset used for training and evaluation, sourced from <a href="https://oracleselixir.com/">Oracle's Elixir</a> and preprocessed to extract relevant data.</li>
    </ul>
  </li>
  
  <li><b>info/</b>
    <ul>
      <li><b>champions_ids.json:</b> Champion ID mappings.</li>
      <li><b>player_glicko_ratings.json:</b> Player skill ratings using the Glicko system.</li>
      <li><b>players_ids.json:</b> Maps player names to unique IDs.</li>
    </ul>
  </li>

  <li><b>models/</b>
    <ul>
      <li><b>match_predictor_model.py:</b> Defines the model architecture, embedding champion IDs and processing them alongside team Glicko ratings through a multi-layer feed-forward network.</li>
    </ul>
  </li>

  <li><b>utils/</b>
    <ul>
      <li><b>data_preprocessing.py:</b> Handles data preprocessing, feature engineering, and dataset preparation. It computes Glicko ratings, normalizes numerical features, and splits data into training, validation, and test sets, ensuring consistency with a saved preprocessing pipeline.</li>
    </ul>
  </li>

  <li><b>glicko.py:</b> Computes and updates player Glicko ratings based on match outcomes, adjusting skill ratings and confidence levels, then saves them for training and inference.</li>

  <li><b>train.py:</b> Trains, evaluates, and saves the model using champion selections and team Glicko ratings. It tracks training progress, optimizes with Adam, plots accuracy/loss metrics, and stores trained weights (<code>model.pth</code>).</li>

  <li><b>predict.py:</b> Loads the trained model, preprocesses input data, and generates win probability predictions. It evaluates both original and swapped team compositions, calculating confidence levels and composition win rates to assess draft impact.</li>
</ul>

<h2 id="limitations">Limitations, Possible Improvements, and Future Work</h2>

<h3>Limitations</h3>
<ul>
  <li>
    <b>Limited to regional matches:</b> The model does not work for international competitions where teams from different regions face each other. This is due to how Glicko Ratings are calculated, as the region is not factored in. 
    <br><br>
    <i>Example:</i> In the MSI 2024 match between <b>T1</b> and <b>Estral Esports</b>, the model incorrectly predicted Estral as the winner because they had a more dominant performance in the <b>LLA</b> compared to T1 in the <b>LCK</b>. However, since LCK is a much stronger region, this prediction was unrealistic.
  </li>

  <li>
    <b>Struggles with predicting upsets:</b> The model heavily weighs the average Glicko rating as the most important feature, making it less effective at predicting unexpected results (upsets). Lower-ranked teams that perform exceptionally well in a match are often underestimated.
  </li>

  <li>
    <b>Match date is not considered:</b> The training data does not account for the date of each match, meaning older matches (e.g., from 8 months ago) are treated the same as recent ones. This prevents the model from adapting to shifts in the meta and current gameplay trends
    (e.g., if <b>Red Side</b> becomes stronger than <b>Blue Side</b> in a particular patch).
  </li>
</ul>

<h3>Possible Improvements and Future Work</h3>
<ul>
  <li>
    <b>Improve draft evaluation:</b> Instead of simply swapping compositions and recalculating probabilities, develop a more advanced method for assessing draft advantages.
  </li>

  <li>
    <b>Incorporate regional differences:</b> Some champions and team compositions perform better in specific regions but worse in others. Adding regional data as an input feature could enhance prediction accuracy.
  </li>

  <li>
    <b>Adapt the model for international tournaments:</b> Introducing a Glicko rating system for regions could help quantify the relative strength of different leagues when teams from different regions face off.
  </li>

  <li>
    <b>Expand and refine the dataset:</b> Include matches from previous years while adding the match date as a parameter to account for shifts in the meta over time.
  </li>

  <li>
    <b>Enhance champion embeddings:</b> Improve the model's understanding of champion interactions by introducing direct champion-to-champion synergy within each team's composition.
  </li>
</ul>


<h3 id="ESPAÑOL"><b>ESPAÑOL</b></h3>

<h1><b>Modelo de Predicciones de League of Legends</b></h1>

<h2>📌 Tabla de Contenidos</h2>
<ul>
  <li><a href="#introduccion">Introducción</a></li>
  <li><a href="#arquitectura-aws">Arquitectura de Despliegue en AWS</a></li>
  <li><a href="#estructura-del-repositorio">Estructura del Repositorio</a></li>
  <ul>
    <li><a href="#carpeta-code">Carpeta Code</a></li>
    <li><a href="#pagina-web">Página Web</a></li>
    <li><a href="#modelo-pytorch">Modelo Pytorch</a></li>
  </ul>
  <li><a href="#limitaciones">Limitaciones, Posibles Mejoras y Trabajo Futuro</a></li>
</ul>

<h2 id="introduccion">Introducción:</h2>
Este repositorio presenta una solución de aprendizaje automático completamente desplegada en AWS para predecir los resultados de partidos profesionales League of Legends. Está organizado en tres carpetas principales:

<ol>
  <li>
    <b>code:</b> Contiene todos los archivos necesarios para implementar el modelo de predicción en <b>AWS SageMaker</b>.
    <ul>
      <li>Scripts de configuración (<em>requirements.txt</em>)</li>
      <li>Archivos de despliegue (<em>inference.py, lambda_handler.py</em> y <em>model.tar.gz</em>)</li>
    </ul>
  </li>
  <br>
  <li>
    <b>Pytorch Model:</b> Esta carpeta contiene todos los archivos y subcarpetas necesarios para entrenar, evaluar y generar predicciones para el modelo de League of Legends utilizando PyTorch. De manera general, incluye:
    <ul>
      <li><b>Datos y Referencias:</b> Un conjunto de datos utilizado para el entrenamiento y la evaluación, y archivos JSON con información de campeones y jugadores (IDs, calificaciones).</li>
      <li><b>Arquitectura del Modelo y Entrenamiento:</b> Scripts que definen la estructura del modelo en PyTorch y la lógica de entrenamiento, además de los pesos del modelo guardados para su rápida carga e inferencia.</li>
      <li><b>Utilidades:</b> Scripts auxiliares para el preprocesamiento de datos, cálculos de calificaciones y flujos de trabajo de predicción, así como <em>pipelines</em> o preprocesadores serializados para asegurar transformaciones de datos coherentes.</li>
    </ul>
  </li>
  <br>
  <li>
    <b>Weebpage:</b> Sitio web estático desplegado en un <b>Amazon S3 bucket</b>
    <ul>
      <li>Archivos HTML/CSS/JS</li>
      <li>Lógica de front-end</li>
      <li>Interfaz de usuario básica</li>
    </ul>
  </li>
  <br>
</ol>

<h2 id="arquitectura-aws">Arquitectura AWS</h2>
<br>
<img src="https://github.com/user-attachments/assets/bc64eb63-3bcf-4f7e-a12b-4980bc3028c7" alt="Arquitectura de AWS" />
<br>

<h2 id="estructura-del-repositorio">Estructura del repositorio:</h2>

<h3 id="carpeta-code">code:</h3>
<p>
  Esta carpeta contiene todo lo necesario para implementar el modelo de AWS SageMaker y la función de AWS Lambda que llama al Endpoint de SageMaker (el cual está vinculado a una API REST).
  Hay dos grupos de archivos dentro de esta carpeta:
</p>

<b>Archivos del modelo de inferencia</b>: Archivos almacenados en un bucket de S3 para servir un modelo de inferencia creado en SageMaker
<ul>
  <li>
    <b>requirements.txt:</b> Todas las librerías que deben instalarse para ejecutar el modelo de inferencia.
  </li>
  <li>
    <b>model.tar.gz:</b> Este archivo contiene todos los archivos de la carpeta "PytorchModel" que se requieren para implementar el modelo en AWS (todos se explicarán en detalle en la sección de la carpeta "PytorchModel"):
    <ul>
      <li>inference.py</li>
      <li>match_predictor_model.py</li>
      <li>model.pth</li>
      <li>data_preprocessing.py</li>
      <li>preprocessor.joblib</li>
      <li>champions_ids.json</li>
      <li>player_glicko_ratings.json</li>
      <li>players_ids.json</li>
      <li>requirements.txt</li>
    </ul>
    <p>
      El comando utilizado en la consola para crear <em>model.tar.gz</em> es:
    </p>
    <em>
      tar -czvf model.tar.gz code/inference.py PytorchModel/model.pth PytorchModel/models/match_predictor_model.py
      PytorchModel/utils/data_preprocessing.py PytorchModel/preprocessor.joblib PytorchModel/info/champions_ids.json
      PytorchModel/glicko.py PytorchModel/info/player_glicko_ratings.json PytorchModel/info/players_ids.json
      code/requirements.txt
    </em>
  </li>
  <li>
    <b>inference.py:</b> Un script diseñado para alojar el modelo de PyTorch en SageMaker. Maneja cuatro tareas principales requeridas por SageMaker para el servicio de modelos:
    <ul>
      <li><b>Carga del modelo (model_fn):</b> Carga el modelo de PyTorch entrenado desde el <em>model_dir</em> especificado, que en este caso es simplemente “model.pth”.</li>
      <li><b>Preprocesamiento de datos (input_fn):</b> Analiza la entrada en formato JSON y la convierte en tensores de características (IDs de campeones y calificaciones Glicko de los equipos).</li>
      <li><b>Inferencia del modelo (predict_fn):</b> Transforma los tensores de entrada, los ejecuta a través del modelo y produce una probabilidad para el ganador predicho.</li>
      <li><b>Procesamiento de la salida (output_fn):</b> Da formato a esta probabilidad en una respuesta JSON.</li>
    </ul>
  </li>
</ul>

<br>

<b>Lambda Handler</b> (lambda_handler.py):
Esta es una función que recibe un evento JSON, lo envía a un endpoint específico de SageMaker para realizar la inferencia y devuelve la predicción en el cuerpo de la respuesta. La idea es utilizarla como un puente entre la API REST y el Endpoint de SageMaker.

<h3 id="pagina-web">Webpage</h3>
<p>
  Esta carpeta contiene los archivos de un sitio web estático simple alojado en un bucket de S3 y servido a través de CloudFront. El sitio web interactúa con una API REST para mostrar las predicciones del modelo de predicciones, basadas en la información proporcionada por el usuario.
</p>

<h4>Funcionalidad Principal</h4>
<p>
  Esta página web ofrece una interfaz interactiva para predicciones de drafts de League of Legends. Los usuarios pueden ver todos los campeones disponibles en una tabla desplazable y filtrarlos mediante una barra de búsqueda. Cada campeón se muestra con una imagen y un nombre para facilitar su identificación. La tabla se completa automáticamente a partir del archivo <code>champions_ids.json</code>, y cualquier campeón que ya haya sido seleccionado para un equipo se deshabilita para evitar duplicados.
</p>

<p>
  Para seleccionar campeones, los usuarios arrastran la imagen de un campeón desde la tabla y la sueltan en uno de los diez “cuadros”: cinco para el equipo azul y cinco para el equipo rojo ordenados basados en los roles del equipo (Top, Jungla, Mid, Adc, Support). Un efecto atenuado indica que el campeón ya no está disponible. Cada cuadro incluye un botón de eliminación que, al hacer clic, libera ese campeón y lo vuelve a habilitar en la tabla para futuras selecciones.
</p>

<p>
  Además de los campeones, hay campos de texto para el nombre de los jugadores. Estos campos ofrecen sugerencias en tiempo real y validan las entradas contra el archivo local <code>players_ids.json</code>. Cualquier nombre no válido se resalta con un borde rojo y un mensaje de error, deshabilitando los botones de predicción hasta que se corrija.
</p>

<p>
  Una vez que los diez cuadros (campeones y nombres de jugadores) están correctamente completados, el usuario puede solicitar una predicción de la probabilidad de victoria. Dependiendo del botón seleccionado—<strong>Predict</strong> o <strong>Predict Draft</strong>—el sistema muestra una única probabilidad o calcula una probabilidad comparativa intercambiando las composiciones de campeones entre ambos equipos. El resultado indica cuál equipo es el favorito y la probabilidad porcentual de que gane.
</p>

<p>
  Para agilizar los ajustes, los usuarios pueden:
  <ul>
    <li><strong>Reiniciar Campeones</strong> para borrar todos los campeones seleccionados,</li>
    <li><strong>Reiniciar Jugadores</strong> para borrar todos los nombres ingresados, y</li>
    <li><strong>Intercambiar Jugadores</strong> o <strong>Intercambiar Campeones</strong> para cambiar los jugadores o los picks entre los dos equipos.</li>
  </ul>
  Estas funciones permiten iterar rápidamente diferentes configuraciones. Una vez solicitada una predicción, el resultado aparece en una ventana superpuesta que se puede cerrar en cualquier momento para seguir ajustando el draft.
</p>

<h4>Nota sobre <em>config.js</em> y la constante <em>API_URL</em></h4>
<p>
  Por razones de seguridad, el archivo <code>config.js</code> que contiene la constante <code>API_URL</code> no está incluido en este repositorio. Puedes crear tu propio archivo
  <code>config.js</code> de forma local, definir <code>API_URL</code> en él y referenciarlo en tu código para habilitar las solicitudes a la API sin exponer información sensible.
</p>

<h3 id="modelo-pytorch">PytorchModel</h3>

<p>Esta carpeta contiene los componentes principales del modelo de machine learning, estructurados de la siguiente manera:</p>

<ul>
  <li><b>data/</b>
    <ul>
      <li><b>Datasheet.csv:</b> Conjunto de datos utilizado para el entrenamiento y evaluación, obtenido de <a href="https://oracleselixir.com/">Oracle's Elixir</a> y preprocesado para extraer la información relevante.</li>
    </ul>
  </li>
  
  <li><b>info/</b>
    <ul>
      <li><b>champions_ids.json:</b> Mapeo de IDs de campeones.</li>
      <li><b>player_glicko_ratings.json:</b> Calificaciones de habilidad de los jugadores basadas en el sistema Glicko.</li>
      <li><b>players_ids.json:</b> Asocia los nombres de los jugadores con sus IDs únicos.</li>
    </ul>
  </li>

  <li><b>models/</b>
    <ul>
      <li><b>match_predictor_model.py:</b> Define la arquitectura del modelo, incorporando los IDs de los campeones y procesándolos junto con las calificaciones Glicko del equipo a través de una red neuronal de múltiples capas.</li>
    </ul>
  </li>

  <li><b>utils/</b>
    <ul>
      <li><b>data_preprocessing.py:</b> Maneja el preprocesamiento de datos, la ingeniería de características y la preparación del dataset. Calcula calificaciones Glicko, normaliza características numéricas y divide los datos en conjuntos de entrenamiento, validación y prueba, garantizando la consistencia con un pipeline de preprocesamiento guardado.</li>
    </ul>
  </li>

  <li><b>glicko.py:</b> Calcula y actualiza las calificaciones Glicko de los jugadores según los resultados de los partidos, ajustando sus niveles de habilidad y confianza, y los guarda para el entrenamiento y la inferencia.</li>

  <li><b>train.py:</b> Entrena, evalúa y guarda el modelo utilizando selecciones de campeones y calificaciones Glicko de los equipos. Realiza un seguimiento del progreso del entrenamiento, optimiza con Adam, grafica métricas de precisión/pérdida y almacena los pesos entrenados (<code>model.pth</code>).</li>

  <li><b>predict.py:</b> Carga el modelo entrenado, preprocesa los datos de entrada y genera predicciones de probabilidad de victoria. Evalúa tanto la composición original como una versión intercambiada de los equipos, calculando niveles de confianza y tasas de victoria de la composición para analizar el impacto del draft.</li>
</ul>

<h2 id="limitaciones">Limitaciones, Posibles Mejoras y Trabajo Futuro</h2>

<h3>Limitaciones</h3>
<ul>
  <li>
    <b>Limitado a competiciones regionales:</b> El modelo no es preciso en torneos internacionales donde equipos de distintas regiones se enfrentan. Esto se debe a que las calificaciones Glicko no consideran la región de origen. 
    <br><br>
    <i>Ejemplo:</i> En el partido de MSI 2024 entre <b>T1</b> y <b>Estral Esports</b>, el modelo predijo incorrectamente que Estral ganaría, ya que dominó la <b>LLA</b> más que T1 en la <b>LCK</b>. Sin embargo, dado que la LCK es una región mucho más fuerte, este resultado no tenía sentido.
  </li>

  <li>
    <b>Dificultad para predecir sorpresas (upsets):</b> El modelo da una importancia excesiva a la calificación promedio de Glicko, lo que reduce la probabilidad de predecir resultados inesperados. Equipos de menor ranking que juegan excepcionalmente bien suelen ser subestimados.
  </li>

  <li>
    <b>Ignora la fecha del partido:</b> El modelo no considera cuándo se jugó un partido, por lo que trata los encuentros de hace <b>8 meses</b> como si fueran recientes. Esto impide que el modelo detecte cambios en el meta o tendencias del juego 
    (por ejemplo, si el <b>Equipo Rojo</b> es más fuerte que el <b>Equipo Azul</b> en un parche específico).
  </li>
</ul>

<h3>Posibles Mejoras y Trabajo Futuro</h3>
<ul>
  <li>
    <b>Mejorar la evaluación de drafts:</b> En lugar de solo intercambiar composiciones y recalcular probabilidades, desarrollar un método más avanzado para medir ventajas de draft.
  </li>

  <li>
    <b>Incluir diferencias entre regiones:</b> Algunos campeones y estrategias funcionan mejor en ciertas regiones y peor en otras. Agregar esta información como un parámetro de entrada podría mejorar la precisión del modelo.
  </li>

  <li>
    <b>Adaptar el modelo a torneos internacionales:</b> Incorporar un sistema de calificación Glicko para regiones ayudaría a medir la fuerza relativa de cada liga cuando compiten entre sí.
  </li>

  <li>
    <b>Ampliar y mejorar el dataset:</b> Incluir partidos de años anteriores y agregar la fecha como parámetro para que el modelo pueda diferenciar entre encuentros antiguos y actuales.
  </li>

  <li>
    <b>Optimizar las representaciones de campeones:</b> Mejorar la forma en que el modelo entiende la interacción entre campeones dentro de una misma composición de equipo.
  </li>
</ul>




