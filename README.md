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

Cada carpeta incluye su propio <b>README.md</b> con explicaciones detalladas de su contenido e instrucciones de uso.

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


