import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
from pandas import ExcelWriter

# Función para realizar predicciones basadas en las anteriores similitudes calculadas
def prediceValoraciones(valoraciones, similitud, tipo='user'):
    # calcular la media únicamente de las películas
    # mean_user_rating = valoraciones.sum(axis=1)/20
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
    print("(media_valoraciones_usuario)")
    print(media_valoraciones_usuario)
    if tipo == 'user':
        #  Una buena elección para llenar los valores que faltan podría ser la valoración media de cada usuario
        pred = media_valoraciones_usuario[:, np.newaxis] + similitud.dot(valoraciones) / np.array([np.abs(similitud).sum(axis=1)]).T
    elif tipo == 'item':
        pred = media_valoraciones_usuario[:, np.newaxis] + valoraciones.dot(similitud) / np.array([np.abs(similitud).sum(axis=1)])
    return pred


# Datos con la lista de los usuarios (USUARIOS)
columnas_usuarios = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
usuarios = pd.read_csv('file://localhost/E:/TFG/TFG-RecommenderSystem/data/ml-100k/u.user', sep='|', names=columnas_usuarios)

# Datos con la lista de las películas (ITEMS)
columnas_items = ['movie id', 'movie title' , 'release date', 'video release date',
                  'IMDb URL', 'unknown', 'Action', 'Adventure','Animation',
                  'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                  'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                  'Thriller', 'War', 'Western']
items = pd.read_csv('file://localhost/E:/TFG/TFG-RecommenderSystem/data/ml-100k/u.item', sep='|', names=columnas_items,
                    encoding='latin-1')

# DATASET COMPLETO
# Datos con la lista de las calificaciones otorgadas por los usuarios a las películas (CALIFICACIONES) total
columnas_calificaciones = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
calificaciones = pd.read_csv('file://localhost/E:/TFG/TFG-RecommenderSystem/data/ml-100k/u.data', sep='\t', names=columnas_calificaciones)

'''
# u.data
matriz_calificaciones = np.zeros((943, 1682))
for row in calificaciones.itertuples():
    matriz_calificaciones[row[1] - 1, row[2] - 1] = row[3]
print("MATRIZ DE CALIFICACIONES (matriz_calificaciones) Usuarios-Ítems:")
print(matriz_calificaciones)
print()

user_similarity = pairwise_distances(matriz_calificaciones, metric='cosine')
item_similarity = pairwise_distances(matriz_calificaciones.T, metric='cosine')

user_prediction = predict(matriz_calificaciones, user_similarity, tipo='user')
item_prediction = predict(matriz_calificaciones, item_similarity, tipo='item')

'''

# TRAIN
# Datos con la información de las calificaciones_u1base (CALIFICACIONES)
columnas_calificaciones_u1base = ['user_id2', 'movie_id2', 'rating2', 'unix_timestamp2']
calificaciones_u1base = pd.read_csv('file://localhost/E:/TFG/TFG-RecommenderSystem/data/ml-100k/u1.base', sep='\t', names=columnas_calificaciones_u1base)

# TEST
# Datos con la información de las calificaciones_u1base (CALIFICACIONES)
columnas_calificaciones_u1test = ['user_id2', 'movie_id2', 'rating2', 'unix_timestamp2']
calificaciones_u1test = pd.read_csv('file://localhost/E:/TFG/TFG-RecommenderSystem/data/ml-100k/u1.test', sep='\t', names=columnas_calificaciones_u1test)


################################### u1.base ###########################################
matriz_calificaciones_u1test = np.zeros((943, 1682))
for fila in calificaciones_u1test.itertuples():
    matriz_calificaciones_u1test[fila[1] - 1, fila[2] - 1] = fila[3]

# Volcamos la matriz_calificaciones_u1base sobre un fichero .csv
L0 = pd.DataFrame(data=matriz_calificaciones_u1test)
L0.to_csv('matriz_calificaciones_u1test.csv', sep=';', header=False, float_format='%.2f', index=False)

matriz_calificaciones_u1base = np.zeros((943, 1682))
for fila in calificaciones_u1base.itertuples():
    matriz_calificaciones_u1base[fila[1] - 1, fila[2] - 1] = fila[3]

# Volcamos la matriz_calificaciones_u1base sobre un fichero .csv
L = pd.DataFrame(data=matriz_calificaciones_u1base)
L.to_csv('matriz_calificaciones_u1base.csv', sep=';', header=False, float_format='%.2f', index=False)

# Obtenemos las matrices con las similitudes entre ususarios y entre películas
user_similarity_u1base = pairwise_distances(matriz_calificaciones_u1base, metric='cosine') # (user-user)
item_similarity_u1base = pairwise_distances(matriz_calificaciones_u1base.T, metric='cosine') # (item-item

# Las volcamos sobre dos ficheros .csv
L1 = pd.DataFrame(data=user_similarity_u1base)
L1.to_csv('user_similarity_u1base.csv', sep=';', header=False, float_format='%.2f', index=False)
L2 = pd.DataFrame(data=item_similarity_u1base)
L2.to_csv('item_similarity_u1base.csv', sep=';', header=False, float_format='%.2f', index=False)

# Obtenemos las matrices con las predicciones de las valoraciones en base al filtrado colaborativo basado en los algoritmos (usuario-usuario) y (pelicula-pelicula)
user_prediction_u1base = prediceValoraciones(matriz_calificaciones_u1base, user_similarity_u1base, tipo='user')
item_prediction_u1base = prediceValoraciones(matriz_calificaciones_u1base, item_similarity_u1base, tipo='item')

# Las volcamos sobre dos ficheros .csv
L3 = pd.DataFrame(data=user_prediction_u1base)
L3.to_csv('user_prediction_u1base.csv', sep=';', header=False, float_format='%.2f', index=False)
L4 = pd.DataFrame(data=item_prediction_u1base)
L4.to_csv('item_prediction_u1base.csv', sep=';', header=False, float_format='%.2f', index=False)

###############################################################################################

# Cogeremos todas las calificaciones y calcularemos su error
'''
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