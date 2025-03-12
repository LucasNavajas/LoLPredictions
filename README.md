<b>ENGLISH/ESPAÑOL</b>

<h2><b>ENGLISH</b></h2>

<h3><b>League of Legends Predictions Model</b></h3>

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

Below is a diagram illustrating the overall AWS deployment architecture:
<br>
<img src="https://github.com/user-attachments/assets/bc64eb63-3bcf-4f7e-a12b-4980bc3028c7" alt="Arquitectura de AWS" />
<br>

<h2><p>Below is a more detailed explanation of each folder and its purpose in this project:</p></h2>

<h3>code:</h3>
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
<h3>Webpage</h3>
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

<h3>PytorchModel</h3>

<p> This folder contains the core machine-learning components of the model with the actual model used for predictions. It's structured as follows:</p>
<ul>
  <li>data/
    <ul>
      <li>Datasheet.csv: The dataset used to train and evaluate the model. All the data is scraped from https://oracleselixir.com/ and transformed in another excel file to get only the desired data for training</li>
    </ul>
  </li>
  <li>info/
    <ul>
      <li>champions_ids.json: Stores the champion ID mappings</li>
      <li>player_glicko_ratings.json: Stores players skill ratings using a Glicko rating system approach</li>
      <li>players_ids.json: File that maps player names to their unirque IDs</li>
    </ul>
  </li>
  <li>models/
    <ul>
      <li>match_predictor_model.py: Script that defines the model architecture used for predictions. The model embeds champion IDs into dense vectors, processes them through a multi-layer feed-forward network alongside their team's Glicko rating, and then concatenates all processed representations. </li>
    </ul>
  </li>
  <li>utils/
    <ul>
      <li>data_preprocessing.py: This script handles data preprocessing, feature engineering, and dataset preparation for training the League of Legends match prediction model. It calculates Glicko ratings for players based on historical match data, averages them to obtain team-wide ratings, and normalizes numerical features (the Glicko scores) while passing champion IDs as categorical inputs. The dataset is then split into training, validation, and test sets, converting them into PyTorch tensors for model training. Additionally, a preprocessing pipeline is saved using joblib to ensure consistency during inference.</li>
    </ul>
  </li>
</ul>

<h2><b>ESPAÑOL</b></h2>

<h3><b>Modelo de Predicciones de League of Legends</b></h3>

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

A continuación, se muestra un diagrama que ilustra la arquitectura general de despliegue en AWS:
<br>
<img src="https://github.com/user-attachments/assets/bc64eb63-3bcf-4f7e-a12b-4980bc3028c7" alt="Arquitectura de AWS" />
<br>

<h2>
  <p>A continuación se presenta una explicación más detallada de cada carpeta y su propósito en este proyecto:</p>
</h2>

<h3>code:</h3>
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

<h3>Webpage</h3>
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



