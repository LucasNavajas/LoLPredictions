<b>ENGLISH</b> - This repository contains a Python project that predicts match outcomes in League of Legends using Glicko rating updates and team compositions. It preprocesses data, trains a PyTorch model, and evaluates the results to determine which team is more likely to win.

Features
<ul>
<li><strong>Glicko Rating System:</strong> Updates player ratings dynamically based on match results.</li>

<li><strong>Team Composition Analysis:</strong> Incorporates champions and player Glicko ratings to predict outcomes.</li>

<li><strong>PyTorch Model:</strong> A neural network built with PyTorch to process match features and predict results.</li>

<li><strong>Data Preprocessing:</strong> Cleans and transforms input data for training using Scikit-learn.</li>

<li><strong>Safe Winner Check:</strong> Evaluates if the predicted winner is "safe" under swapped team compositions. This "safe" winner prediction means that the model considers the predicted winner to also win in terms of team composition.</li>

</ul>
<br>
<br>
<b>ESPAÑOL</b> - Este repositorio contiene un proyecto en Python que predice los resultados de las partidas en League of Legends utilizando actualizaciones de las calificaciones Glicko y composiciones de equipo. Preprocesa datos, entrena un modelo en PyTorch y evalúa los resultados para determinar qué equipo tiene más probabilidades de ganar.

Características

<ul> <li><strong>Sistema de Calificación Glicko:</strong> Actualiza las calificaciones de los jugadores de forma dinámica según los resultados de las partidas.
  
</li> <li><strong>Análisis de Composición de Equipo:</strong> Incorpora campeones y las calificaciones Glicko de los jugadores para predecir los resultados.
  
</li> <li><strong>Modelo en PyTorch:</strong> Una red neuronal construida con PyTorch que procesa las características de las partidas y predice los resultados.
  
</li> <li><strong>Preprocesamiento de Datos:</strong> Limpia y transforma los datos de entrada para el entrenamiento utilizando Scikit-learn.</li> 

  <li><strong>Verificación de Ganador Seguro:</strong> Evalúa si el ganador predicho es "seguro" al intercambiar las composiciones de equipo. Esta predicción de "ganador seguro" significa que el modelo considera que el ganador predicho también ganaría en términos de composición del equipo.</li> </ul>
