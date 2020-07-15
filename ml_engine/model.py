import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances


# Función para realizar predicciones basadas en las anteriores similitudes calculadas (tipo='user' para el algoritmo usuario-usuario) (tipo='item' para el algoritmo item-item)
def prediceValoraciones(valoraciones, similitud, tipo='user'):
    # calcular la media únicamente de las películas que hayan sido calificadas
    media_valoraciones_usuario = []
    for i in range(len(valoraciones)):
        contador = 0
        suma = 0
        media = 0
        for j in range(len(valoraciones[i])):
            if valoraciones[i][j] != 0:
                suma = suma + valoraciones[i][j]
                contador = contador + 1
        media = suma / contador
        media_valoraciones_usuario.append(media)
    media_valoraciones_usuario = np.array(media_valoraciones_usuario)

    if tipo == 'user':
        #  Una buena elección para llenar los valores que faltan podría ser la valoración media de cada usuario (sumando media_valoraciones_usuario[:, np.newaxis])
        pred = media_valoraciones_usuario[:, np.newaxis] + similitud.dot(valoraciones) / np.array([np.abs(similitud).sum(axis=1)]).T
    elif tipo == 'item':
        pred = media_valoraciones_usuario[:, np.newaxis] + valoraciones.dot(similitud) / np.array([np.abs(similitud).sum(axis=1)])
    return pred


# Datos de los usuarios (USUARIOS)
columnas_usuarios = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
usuarios = pd.read_csv('file://localhost/E:/TFG/TFG-RecommenderSystem/data/ml-100k/u.user', sep='|', names=columnas_usuarios)


# Datos de las películas (ITEMS)
columnas_items = ['movie id', 'movie title' , 'release date', 'video release date',
                  'IMDb URL', 'unknown', 'Action', 'Adventure','Animation',
                  'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                  'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                  'Thriller', 'War', 'Western']
items = pd.read_csv('file://localhost/E:/TFG/TFG-RecommenderSystem/data/ml-100k/u.item', sep='|', names=columnas_items,
                    encoding='latin-1')

# Datos de las calificaciones otorgadas por los usuarios a las películas (CALIFICACIONES)
columnas_calificaciones = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
calificaciones = pd.read_csv('file://localhost/E:/TFG/TFG-RecommenderSystem/data/ml-100k/u.data', sep='\t', names=columnas_calificaciones)
'''
Partiendo de la información proporcionada por el fichero u.info:
943 users
1682 items
100000 ratings
'''
# Se obtiene la matriz con todas las CALIFICACIONES otorgadas por los usuarios a las películas
matriz_calificaciones = np.zeros((943, 1682))
for fila in calificaciones.itertuples():
    matriz_calificaciones[fila[1] - 1, fila[2] - 1] = fila[3]

# Se vuelca la matriz de CALIFICACIONES sobre un fichero .csv
D = pd.DataFrame(data=matriz_calificaciones)
D.to_csv('matriz_calificacionescsv', sep=';', header=False, float_format='%.2f', index=False)



# ///////////////////////////////////////////////// EVALUACIÓN (Offline) //////////////////////////////////////////////////////////////
# Este sistema de recomendación posee como tarea de recomendación, la predicción de voto / valoración de los usuarios sobre las películas que no hayan votado / valorado / calificado
# Se utiliza la técnica cross-validation para la validación del modelo. En este SR, se parte de un conjunto de datos u.data que se ha dividido en K=5 particiones para realizar K iteraciones, utilizando K-1 particiones para el conjunto de TRAIN y 1 para el conjunto TEST

###################################### u1  (Primer Experimento / Iteración) ###########################################
# TRAIN
# Datos con la información de las calificaciones_u1base (CALIFICACIONES)
columnas_calificaciones_u1base = ['user_id2', 'movie_id2', 'rating2', 'unix_timestamp2']
calificaciones_u1base = pd.read_csv('file://localhost/E:/TFG/TFG-RecommenderSystem/data/ml-100k/u1.base', sep='\t', names=columnas_calificaciones_u1base)

# TEST
# Datos con la información de las calificaciones_u1test (CALIFICACIONES)
columnas_calificaciones_u1test = ['user_id2', 'movie_id2', 'rating2', 'unix_timestamp2']
calificaciones_u1test = pd.read_csv('file://localhost/E:/TFG/TFG-RecommenderSystem/data/ml-100k/u1.test', sep='\t', names=columnas_calificaciones_u1test)

# Se obtiene la matriz con las CALIFICACIONES del TRAIN para la primera iteración otorgadas por los usuarios a las películas
matriz_calificaciones_u1base = np.zeros((943, 1682))
for fila in calificaciones_u1base.itertuples():
    matriz_calificaciones_u1base[fila[1] - 1, fila[2] - 1] = fila[3]

# Se vuelca la matriz_calificaciones_u1base sobre un fichero .csv
L1B = pd.DataFrame(data=matriz_calificaciones_u1base)
L1B.to_csv('matriz_calificaciones_u1base.csv', sep=';', header=False, float_format='%.2f', index=False)

# Se obtiene la matriz con las CALIFICACIONES del TEST para la primera iteración otorgadas por los usuarios a las películas
matriz_calificaciones_u1test = np.zeros((943, 1682))
for fila in calificaciones_u1test.itertuples():
    matriz_calificaciones_u1test[fila[1] - 1, fila[2] - 1] = fila[3]

# Se vuelca la matriz_calificaciones_u1test sobre un fichero .csv
L1T = pd.DataFrame(data=matriz_calificaciones_u1test)
L1T.to_csv('matriz_calificaciones_u1test.csv', sep=';', header=False, float_format='%.2f', index=False)

# Se obtienen las matrices con las similitudes entre ususarios y entre películas
user_similarity_u1base = pairwise_distances(matriz_calificaciones_u1base, metric='cosine') #(user-user)
item_similarity_u1base = pairwise_distances(matriz_calificaciones_u1base.T, metric='cosine') #(item-item)

# Se vuelcan sobre dos ficheros .csv
L1US = pd.DataFrame(data=user_similarity_u1base)
L1US.to_csv('user_similarity_u1base.csv', sep=';', header=False, float_format='%.2f', index=False)
L1IS = pd.DataFrame(data=item_similarity_u1base)
L1IS.to_csv('item_similarity_u1base.csv', sep=';', header=False, float_format='%.2f', index=False)

# Se obtienen  las matrices con las predicciones de las valoraciones en base al filtrado colaborativo basado en los algoritmos (user-user) y (item-item)
user_prediction_u1base = prediceValoraciones(matriz_calificaciones_u1base, user_similarity_u1base, tipo='user')
item_prediction_u1base = prediceValoraciones(matriz_calificaciones_u1base, item_similarity_u1base, tipo='item')

# Se vuelcan sobre dos ficheros .csv
L1UP = pd.DataFrame(data=user_prediction_u1base)
L1UP.to_csv('user_prediction_u1base.csv', sep=';', header=False, float_format='%.2f', index=False)
L1IP = pd.DataFrame(data=item_prediction_u1base)
L1IP.to_csv('item_prediction_u1base.csv', sep=';', header=False, float_format='%.2f', index=False)

#####################################################################################

###################################### u2 (segundo Experimiento / Iteración) ###########################################
# TRAIN
# Datos con la información de las calificaciones_u2base (CALIFICACIONES)
columnas_calificaciones_u2base = ['user_id2', 'movie_id2', 'rating2', 'unix_timestamp2']
calificaciones_u2base = pd.read_csv('file://localhost/E:/TFG/TFG-RecommenderSystem/data/ml-100k/u2.base', sep='\t', names=columnas_calificaciones_u2base)

# TEST
# Datos con la información de las calificaciones_u2test (CALIFICACIONES)
columnas_calificaciones_u2test = ['user_id2', 'movie_id2', 'rating2', 'unix_timestamp2']
calificaciones_u2test = pd.read_csv('file://localhost/E:/TFG/TFG-RecommenderSystem/data/ml-100k/u2.test', sep='\t', names=columnas_calificaciones_u2test)

# Se obtiene la matriz con las CALIFICACIONES del TRAIN para la segunda iteración otorgadas por los usuarios a las películas
matriz_calificaciones_u2base = np.zeros((943, 1682))
for fila in calificaciones_u2base.itertuples():
    matriz_calificaciones_u2base[fila[1] - 1, fila[2] - 1] = fila[3]

# Se vuelca la matriz_calificaciones_u2base sobre un fichero .csv
L2B = pd.DataFrame(data=matriz_calificaciones_u2base)
L2B.to_csv('matriz_calificaciones_u2base.csv', sep=';', header=False, float_format='%.2f', index=False)

# Se obtiene la matriz con las CALIFICACIONES del TEST para la segunda iteración otorgadas por los usuarios a las películas
matriz_calificaciones_u2test = np.zeros((943, 1682))
for fila in calificaciones_u2test.itertuples():
    matriz_calificaciones_u2test[fila[1] - 1, fila[2] - 1] = fila[3]

# Se vuelca la matriz_calificaciones_u2test sobre un fichero .csv
L2T = pd.DataFrame(data=matriz_calificaciones_u2test)
L2T.to_csv('matriz_calificaciones_u2test.csv', sep=';', header=False, float_format='%.2f', index=False)

# Se obtienen las matrices con las similitudes entre ususarios y entre películas
user_similarity_u2base = pairwise_distances(matriz_calificaciones_u2base, metric='cosine') #(user-user)
item_similarity_u2base = pairwise_distances(matriz_calificaciones_u2base.T, metric='cosine') #(item-item)

# Se vuelcan sobre dos ficheros .csv
L2US = pd.DataFrame(data=user_similarity_u2base)
L2US.to_csv('user_similarity_u2base.csv', sep=';', header=False, float_format='%.2f', index=False)
L2IS = pd.DataFrame(data=item_similarity_u2base)
L2IS.to_csv('item_similarity_u2base.csv', sep=';', header=False, float_format='%.2f', index=False)

# Se obtienen  las matrices con las predicciones de las valoraciones en base al filtrado colaborativo basado en los algoritmos (user-user) y (item-item)
user_prediction_u2base = prediceValoraciones(matriz_calificaciones_u2base, user_similarity_u2base, tipo='user')
item_prediction_u2base = prediceValoraciones(matriz_calificaciones_u2base, item_similarity_u2base, tipo='item')

# Se vuelcan sobre dos ficheros .csv
L2UP = pd.DataFrame(data=user_prediction_u2base)
L2UP.to_csv('user_prediction_u2base.csv', sep=';', header=False, float_format='%.2f', index=False)
L2IP = pd.DataFrame(data=item_prediction_u2base)
L2IP.to_csv('item_prediction_u2base.csv', sep=';', header=False, float_format='%.2f', index=False)

#####################################################################################

###################################### u3 (tercer Experimiento / Iteración) ###########################################
# TRAIN
# Datos con la información de las calificaciones_u3base (CALIFICACIONES)
columnas_calificaciones_u3base = ['user_id2', 'movie_id2', 'rating2', 'unix_timestamp2']
calificaciones_u3base = pd.read_csv('file://localhost/E:/TFG/TFG-RecommenderSystem/data/ml-100k/u3.base', sep='\t', names=columnas_calificaciones_u3base)

# TEST
# Datos con la información de las calificaciones_u3test (CALIFICACIONES)
columnas_calificaciones_u3test = ['user_id2', 'movie_id2', 'rating2', 'unix_timestamp2']
calificaciones_u3test = pd.read_csv('file://localhost/E:/TFG/TFG-RecommenderSystem/data/ml-100k/u3.test', sep='\t', names=columnas_calificaciones_u3test)

# Se obtiene la matriz con las CALIFICACIONES del TRAIN para la tercera iteración otorgadas por los usuarios a las películas
matriz_calificaciones_u3base = np.zeros((943, 1682))
for fila in calificaciones_u3base.itertuples():
    matriz_calificaciones_u3base[fila[1] - 1, fila[2] - 1] = fila[3]

# Se vuelca la matriz_calificaciones_u3base sobre un fichero .csv
L3B = pd.DataFrame(data=matriz_calificaciones_u3base)
L3B.to_csv('matriz_calificaciones_u3base.csv', sep=';', header=False, float_format='%.2f', index=False)

# Se obtiene la matriz con las CALIFICACIONES del TEST para la tercera iteración otorgadas por los usuarios a las películas
matriz_calificaciones_u3test = np.zeros((943, 1682))
for fila in calificaciones_u3test.itertuples():
    matriz_calificaciones_u3test[fila[1] - 1, fila[2] - 1] = fila[3]

# Se vuelca la matriz_calificaciones_u3test sobre un fichero .csv
L3T = pd.DataFrame(data=matriz_calificaciones_u2test)
L3T.to_csv('matriz_calificaciones_u3test.csv', sep=';', header=False, float_format='%.2f', index=False)

# Se obtienen las matrices con las similitudes entre ususarios y entre películas
user_similarity_u3base = pairwise_distances(matriz_calificaciones_u3base, metric='cosine') #(user-user)
item_similarity_u3base = pairwise_distances(matriz_calificaciones_u3base.T, metric='cosine') #(item-item)

# Se vuelcan sobre dos ficheros .csv
L3US = pd.DataFrame(data=user_similarity_u3base)
L3US.to_csv('user_similarity_u3base.csv', sep=';', header=False, float_format='%.2f', index=False)
L3IS = pd.DataFrame(data=item_similarity_u3base)
L3IS.to_csv('item_similarity_u3base.csv', sep=';', header=False, float_format='%.2f', index=False)

# Se obtienen  las matrices con las predicciones de las valoraciones en base al filtrado colaborativo basado en los algoritmos (user-user) y (item-item)
user_prediction_u3base = prediceValoraciones(matriz_calificaciones_u3base, user_similarity_u3base, tipo='user')
item_prediction_u3base = prediceValoraciones(matriz_calificaciones_u3base, item_similarity_u3base, tipo='item')

# Se vuelcan sobre dos ficheros .csv
L3UP = pd.DataFrame(data=user_prediction_u3base)
L3UP.to_csv('user_prediction_u3base.csv', sep=';', header=False, float_format='%.2f', index=False)
L3IP = pd.DataFrame(data=item_prediction_u3base)
L3IP.to_csv('item_prediction_u3base.csv', sep=';', header=False, float_format='%.2f', index=False)

#####################################################################################

###################################### u4 (cuarto Experimiento / Iteración) ###########################################
# TRAIN
# Datos con la información de las calificaciones_u4base (CALIFICACIONES)
columnas_calificaciones_u4base = ['user_id2', 'movie_id2', 'rating2', 'unix_timestamp2']
calificaciones_u4base = pd.read_csv('file://localhost/E:/TFG/TFG-RecommenderSystem/data/ml-100k/u4.base', sep='\t', names=columnas_calificaciones_u4base)

# TEST
# Datos con la información de las calificaciones_u4test (CALIFICACIONES)
columnas_calificaciones_u4test = ['user_id2', 'movie_id2', 'rating2', 'unix_timestamp2']
calificaciones_u4test = pd.read_csv('file://localhost/E:/TFG/TFG-RecommenderSystem/data/ml-100k/u4.test', sep='\t', names=columnas_calificaciones_u4test)

# Se obtiene la matriz con las CALIFICACIONES del TRAIN para la cuarta iteración otorgadas por los usuarios a las películas
matriz_calificaciones_u4base = np.zeros((943, 1682))
for fila in calificaciones_u4base.itertuples():
    matriz_calificaciones_u4base[fila[1] - 1, fila[2] - 1] = fila[3]

# Se vuelca la matriz_calificaciones_u4base sobre un fichero .csv
L4B = pd.DataFrame(data=matriz_calificaciones_u4base)
L4B.to_csv('matriz_calificaciones_u4base.csv', sep=';', header=False, float_format='%.2f', index=False)

# Se obtiene la matriz con las CALIFICACIONES del TEST para la cuarta iteración otorgadas por los usuarios a las películas
matriz_calificaciones_u4test = np.zeros((943, 1682))
for fila in calificaciones_u4test.itertuples():
    matriz_calificaciones_u4test[fila[1] - 1, fila[2] - 1] = fila[3]

# Se vuelca la matriz_calificaciones_u4test sobre un fichero .csv
L4T = pd.DataFrame(data=matriz_calificaciones_u4test)
L4T.to_csv('matriz_calificaciones_u4test.csv', sep=';', header=False, float_format='%.2f', index=False)

# Se obtienen las matrices con las similitudes entre ususarios y entre películas
user_similarity_u4base = pairwise_distances(matriz_calificaciones_u4base, metric='cosine') #(user-user)
item_similarity_u4base = pairwise_distances(matriz_calificaciones_u4base.T, metric='cosine') #(item-item)

# Se vuelcan sobre dos ficheros .csv
L4US = pd.DataFrame(data=user_similarity_u4base)
L4US.to_csv('user_similarity_u4base.csv', sep=';', header=False, float_format='%.2f', index=False)
L4IS = pd.DataFrame(data=item_similarity_u4base)
L4IS.to_csv('item_similarity_u4base.csv', sep=';', header=False, float_format='%.2f', index=False)

# Se obtienen  las matrices con las predicciones de las valoraciones en base al filtrado colaborativo basado en los algoritmos (user-user) y (item-item)
user_prediction_u4base = prediceValoraciones(matriz_calificaciones_u4base, user_similarity_u4base, tipo='user')
item_prediction_u4base = prediceValoraciones(matriz_calificaciones_u4base, item_similarity_u4base, tipo='item')

# Se vuelcan sobre dos ficheros .csv
L4UP = pd.DataFrame(data=user_prediction_u4base)
L4UP.to_csv('user_prediction_u4base.csv', sep=';', header=False, float_format='%.2f', index=False)
L4IP = pd.DataFrame(data=item_prediction_u4base)
L4IP.to_csv('item_prediction_u4base.csv', sep=';', header=False, float_format='%.2f', index=False)

#####################################################################################

###################################### u5 (quinto Experimiento / Iteración) ###########################################
# TRAIN
# Datos con la información de las calificaciones_u5base (CALIFICACIONES)
columnas_calificaciones_u5base = ['user_id2', 'movie_id2', 'rating2', 'unix_timestamp2']
calificaciones_u5base = pd.read_csv('file://localhost/E:/TFG/TFG-RecommenderSystem/data/ml-100k/u5.base', sep='\t', names=columnas_calificaciones_u5base)

# TEST
# Datos con la información de las calificaciones_u5test (CALIFICACIONES)
columnas_calificaciones_u5test = ['user_id2', 'movie_id2', 'rating2', 'unix_timestamp2']
calificaciones_u5test = pd.read_csv('file://localhost/E:/TFG/TFG-RecommenderSystem/data/ml-100k/u5.test', sep='\t', names=columnas_calificaciones_u5test)

# Se obtiene la matriz con las CALIFICACIONES del TRAIN para la quinta iteración otorgadas por los usuarios a las películas
matriz_calificaciones_u5base = np.zeros((943, 1682))
for fila in calificaciones_u5base.itertuples():
    matriz_calificaciones_u5base[fila[1] - 1, fila[2] - 1] = fila[3]

# Se vuelca la matriz_calificaciones_u5base sobre un fichero .csv
L5B = pd.DataFrame(data=matriz_calificaciones_u5base)
L5B.to_csv('matriz_calificaciones_u5base.csv', sep=';', header=False, float_format='%.2f', index=False)

# Se obtiene la matriz con las CALIFICACIONES del TEST para la quinta iteración otorgadas por los usuarios a las películas
matriz_calificaciones_u5test = np.zeros((943, 1682))
for fila in calificaciones_u5test.itertuples():
    matriz_calificaciones_u5test[fila[1] - 1, fila[2] - 1] = fila[3]

# Se vuelca la matriz_calificaciones_u5test sobre un fichero .csv
L5T = pd.DataFrame(data=matriz_calificaciones_u5test)
L5T.to_csv('matriz_calificaciones_u5test.csv', sep=';', header=False, float_format='%.2f', index=False)

# Se obtienen las matrices con las similitudes entre ususarios y entre películas
user_similarity_u5base = pairwise_distances(matriz_calificaciones_u5base, metric='cosine') #(user-user)
item_similarity_u5base = pairwise_distances(matriz_calificaciones_u5base.T, metric='cosine') #(item-item)

# Se vuelcan sobre dos ficheros .csv
L5US = pd.DataFrame(data=user_similarity_u5base)
L5US.to_csv('user_similarity_u5base.csv', sep=';', header=False, float_format='%.2f', index=False)
L5IS = pd.DataFrame(data=item_similarity_u5base)
L5IS.to_csv('item_similarity_u5base.csv', sep=';', header=False, float_format='%.2f', index=False)

# Se obtienen  las matrices con las predicciones de las valoraciones en base al filtrado colaborativo basado en los algoritmos (user-user) y (item-item)
user_prediction_u5base = prediceValoraciones(matriz_calificaciones_u5base, user_similarity_u5base, tipo='user')
item_prediction_u5base = prediceValoraciones(matriz_calificaciones_u5base, item_similarity_u5base, tipo='item')

# Se vuelcan sobre dos ficheros .csv
L5UP = pd.DataFrame(data=user_prediction_u5base)
L5UP.to_csv('user_prediction_u5base.csv', sep=';', header=False, float_format='%.2f', index=False)
L5IP = pd.DataFrame(data=item_prediction_u5base)
L5IP.to_csv('item_prediction_u5base.csv', sep=';', header=False, float_format='%.2f', index=False)

#####################################################################################

###################################### u1 (primer Experimiento / Iteración) ###########################################
# Cálculo de ERRORES (MAE Y RMSE)

# Se recorren los únicos elementos de matriz_calificaciones de TEST calculando la diferencia entre el valor real y el valor predicho (ERROR ABSOLUTO)
tuplaErroresTotal = []
tuplaErroresU1 = []

for i in range(len(matriz_calificaciones_u1test)):
    for j in range(len(matriz_calificaciones_u1test[i])):
        if matriz_calificaciones_u1test[i][j] != 0:
            tuplaErroresU1.append(np.abs(matriz_calificaciones_u1test[i][j] - user_prediction_u1base[i][j]))

EAM_U1 = (np.array(tuplaErroresU1)).mean()
tuplaErroresTotal.append(EAM_U1)
print("EAM_U1")
print(EAM_U1)

EAU1 = pd.DataFrame(data=tuplaErroresU1)
EAU1.to_csv('EAU1.csv', sep=';', header=False, float_format='%.2f', index=False)
#######################################################################################################################

###################################### u2 (segundo Experimiento / Iteración) ###########################################
# Cálculo de ERRORES (MAE Y RMSE)
tuplaErroresU2 = []

# Se recorren los únicos elementos de matriz_calificaciones de TEST calculando la diferencia entre el valor real y el valor predicho (ERROR ABSOLUTO)
for i in range(len(matriz_calificaciones_u2test)):
    for j in range(len(matriz_calificaciones_u2test[i])):
        if matriz_calificaciones_u2test[i][j] != 0:
            tuplaErroresU2.append(np.abs(matriz_calificaciones_u2test[i][j] - user_prediction_u2base[i][j]))

EAM_U2 = (np.array(tuplaErroresU2)).mean()
tuplaErroresTotal.append(EAM_U2)
print("EAM_U2")
print(EAM_U2)

EAU2 = pd.DataFrame(data=tuplaErroresU2)
EAU2.to_csv('EAU2.csv', sep=';', header=False, float_format='%.2f', index=False)
#######################################################################################################################

###################################### u3 (tercer Experimiento / Iteración) ###########################################
# Cálculo de ERRORES (MAE Y RMSE)
tuplaErroresU3 = []

# Se recorren los únicos elementos de matriz_calificaciones de TEST calculando la diferencia entre el valor real y el valor predicho (ERROR ABSOLUTO)
for i in range(len(matriz_calificaciones_u3test)):
    for j in range(len(matriz_calificaciones_u3test[i])):
        if matriz_calificaciones_u3test[i][j] != 0:
            tuplaErroresU3.append(np.abs(matriz_calificaciones_u3test[i][j] - user_prediction_u3base[i][j]))

EAM_U3 = (np.array(tuplaErroresU3)).mean()
tuplaErroresTotal.append(EAM_U3)
print("EAM_U3")
print(EAM_U3)

EAU3 = pd.DataFrame(data=tuplaErroresU3)
EAU3.to_csv('EAU3.csv', sep=';', header=False, float_format='%.2f', index=False)
######################################################################################################################

###################################### u4 (Cuarto Experimiento / Iteración) ###########################################
# Cálculo de ERRORES (MAE Y RMSE)
tuplaErroresU4 = []

# Se recorren los únicos elementos de matriz_calificaciones de TEST calculando la diferencia entre el valor real y el valor predicho (ERROR ABSOLUTO)
for i in range(len(matriz_calificaciones_u4test)):
    for j in range(len(matriz_calificaciones_u4test[i])):
        if matriz_calificaciones_u4test[i][j] != 0:
            tuplaErroresU4.append(np.abs(matriz_calificaciones_u4test[i][j] - user_prediction_u4base[i][j]))

EAM_U4 = (np.array(tuplaErroresU4)).mean()
tuplaErroresTotal.append(EAM_U4)
print("EAM_U4")
print(EAM_U4)

EAU4 = pd.DataFrame(data=tuplaErroresU4)
EAU4.to_csv('EAU4.csv', sep=';', header=False, float_format='%.2f', index=False)
######################################################################################################################

###################################### u5 (quinto Experimiento / Iteración) ###########################################
# Cálculo de ERRORES (MAE Y RMSE)
tuplaErroresU5 = []

# Se recorren los únicos elementos de matriz_calificaciones de TEST calculando la diferencia entre el valor real y el valor predicho (ERROR ABSOLUTO)
for i in range(len(matriz_calificaciones_u5test)):
    for j in range(len(matriz_calificaciones_u5test[i])):
        if matriz_calificaciones_u5test[i][j] != 0:
            tuplaErroresU5.append(np.abs(matriz_calificaciones_u5test[i][j] - user_prediction_u5base[i][j]))

EAM_U5 = (np.array(tuplaErroresU5)).mean()
tuplaErroresTotal.append(EAM_U5)
print("EAM_U5")
print(EAM_U5)

EAU5 = pd.DataFrame(data=tuplaErroresU5)
EAU5.to_csv('EAU5.csv', sep=';', header=False, float_format='%.2f', index=False)

# Se calcula la media de los EAM de los 5 Experimentos / Iteraciones
EAM_media = (np.array(tuplaErroresTotal)).mean()
print("EAM_media")
print(EAM_media)
######################################################################################################################

'''
# Cogeremos todas las calificaciones y calcularemos su error

a = np.array(matriz_calificaciones)
b = np.array(user_prediction)
print(a)
print(b)
c = a-b
print("c")
print(c)
# Recorreremos los únicos elementos de matriz_calificaciones que hubieran existido
arrayTemp = []
for i in range(len(matriz_calificaciones)):
    for j in range(len(matriz_calificaciones[i])):
        if matriz_calificaciones[i][j] != 0:
            arrayTemp.append(matriz_calificaciones[i][j] - user_prediction[i][j])


L = pd.DataFrame(data=arrayTemp)
L.to_csv('ArrayTemp.csv', sep=';', header=False, float_format='%.2f', index=False)
'''