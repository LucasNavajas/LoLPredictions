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

Each folder includes its own <b>README.md</b> with detailed explanations of its contents and usage instructions.

Below is a diagram illustrating the overall AWS deployment architecture:
<br>
![Untitled (1)](https://github.com/user-attachments/assets/bc64eb63-3bcf-4f7e-a12b-4980bc3028c7)

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

