# TFG-RecommenderSystem
TFG - Sistema de recomendación basado en el dato comportamental de los usuarios de un portal web

Es un sencillo sistema de recomendación construido con el dataset MovieLens(Harper & Konstan, 2016)[1] 
El sistema nos arroja como resultado las matrices con las predicciones de las valoraciones de películas en base a  5 lugares más acordes de acuerdo a un venue_id proporcionado. Para ellos usamos la métrica del la similitud del coseno y el el Índice de Jaccard como métrica de distancia para medir la similaridad en el Espacio Vectorial.

PASOS PARA GENERAR MATRICES DE PREDICCIONES DE RATINGS + MAE Y R POR CONSOLA:
Lanzar el archivo model.py es el prototipo que deberá ser lanzado en el intérprete, o en el IDE correspondiente (ej:PyCharm)

Una vez lanzado el prototipo, se generarán varios archivos en formato .csv, con las predicciones de las calificaciones:
user_user_predictionUX.csv
item_item_predictionUX.csv
...

También se generarán dos ficheros que poseeran dos columnas, la primera con el valor REAL, y la segunda con el valor PREDICHO
df_user.csv
df_item.csv


# Referencias
[1]Herlocker, J., Konstan, J., Borchers, A., Riedl, J .. Un algoritmo
Marco para realizar el filtrado colaborativo. Actas de la
Conferencia de 1999 sobre Investigación y Desarrollo en Información
Recuperación. Ago. 1999.
