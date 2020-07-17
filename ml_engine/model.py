import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import random as rd

'''
Prepara los arrays con los ratings reales y los ratings predichos por el modelo
'''
def getYPredTrue(y_pred_ux,y_true_ux,y_pred,y_true,ratings_pred,ratings_true):
    #Serecorrenlosúnicoselementosdematriz_calificacionesdeTESTcalculandoladiferenciaentreelvalorrealyelvalorpredicho(ERRORABSOLUTO)
    for i in range(len(ratings_true)):
        for j in range(len(ratings_true[i])):
            if ratings_true[i][j] != 0:
                y_true_ux.append(ratings_true[i][j])
                y_pred_ux.append(ratings_pred[i][j])
                y_true.append(ratings_true[i][j])
                y_pred.append(ratings_pred[i][j])
    return y_pred_ux, y_true_ux, y_pred, y_true

'''
Formato inicial de los elementos de ratings  [-2, -1, 0, 1, 2] a los valores iniciales [1,
'''
def formatoInicial(ratings_prediction):
    for i in range(len(ratings_prediction)):
        for j in range(len(ratings_prediction[i])):
            ratings_prediction[i][j] = ratings_prediction[i][j] + 3
    return ratings_prediction

def calculaPromedioValoracionesUsuario(valoraciones_train):
    # calcular la media únicamente de las películas que hayan sido calificadas
    media_valoraciones_usuario = []
    for i in range(len(valoraciones_train)):
        contador = 0
        suma = 0
        for j in range(len(valoraciones_train[i])):
            if valoraciones_train[i][j] != 0:
                suma = suma + valoraciones_train[i][j]
                contador = contador + 1
        media = suma / contador
        media_valoraciones_usuario.append(media)
    media_valoraciones_usuario = np.array(media_valoraciones_usuario)
    return media_valoraciones_usuario

def getMatriz(calificaciones):
    matriz_calificaciones = np.zeros((943, 1682))
    for fila in calificaciones.itertuples():
        matriz_calificaciones[fila[1] - 1, fila[2] - 1] = fila[3]
    return matriz_calificaciones

'''
(Usuario-Usuario)
Devuelve las predicciones de los ratings de los ususarios sobre las películas
'''
def prediceUU(ratings, similarity):
    calificacion_media_usuario = calculaPromedioValoracionesUsuario((ratings))
    # eliminamos SESGOS de los usuarios, para llevar a todos los usuarios al mismo nivel
    rattings_NoSesgados = (ratings - calificacion_media_usuario[:, np.newaxis])
    pedicciones = calificacion_media_usuario[:, np.newaxis] + similarity.dot(rattings_NoSesgados) / np.array([np.abs(similarity).sum(axis=1)]).T
    return pedicciones

'''
(Item-Item)
Devuelve las predicciones de los ratings de los ususarios sobre las películas
'''
def prediceII(ratings, similarity):
    calificacion_media_usuario = calculaPromedioValoracionesUsuario((ratings))
    # eliminamos SESGOS de los usuarios, para llevar a todos los usuarios al mismo nivel
    rattings_NoSesgados = (ratings - calificacion_media_usuario[:, np.newaxis])
    pedicciones = calificacion_media_usuario[:, np.newaxis] + rattings_NoSesgados.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pedicciones

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
matriz_calificaciones = getMatriz(calificaciones)

# Se vuelca la matriz de CALIFICACIONES sobre un fichero .csv
# D = pd.DataFrame(data=matriz_calificaciones)
# D.to_csv('matriz_calificacionescsv', sep=';', header=False, float_format='%.2f', index=False)


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
matriz_calificaciones_u1base = getMatriz(calificaciones_u1base)

# Se obtienen las matrices con las similitudes entre ususarios y entre items respectivamente
user_similarity_u1base = pairwise_distances(matriz_calificaciones_u1base, metric='cosine') #(user-user)
item_similarity_u1base = pairwise_distances(matriz_calificaciones_u1base.T, metric='cosine') #(item-item)

# Se obtienen las predicciones de los ratings de los usuarios sobre las películas
user_prediction_u1base = prediceUU(matriz_calificaciones_u1base, user_similarity_u1base)#(user-user)
item_prediction_u1base = prediceII(matriz_calificaciones_u1base, item_similarity_u1base)#(item-item)
user_prediction_u1base = formatoInicial(user_prediction_u1base)
item_prediction_u1base = formatoInicial(item_prediction_u1base)

# Se obtiene la matriz con las CALIFICACIONES del TEST para la primera iteración otorgadas por los usuarios a las películas
matriz_calificaciones_u1test = getMatriz(calificaciones_u1test)

############################################## Cálculo de MAE y R2 para u1 ##################################################################
# Estos primeros arrays se volcarán sbre un fichero .csv para posteriormente mostrar las gráficas
y_usertrue = []
y_userpred = []
y_itemtrue = []
y_itempred = []

y_usertrue_u1 = []
y_userpred_u1 = []
y_itemtrue_u1 = []
y_itempred_u1 = []

y_userpred_u1, y_usertrue_u1, y_userpred, y_usertrue = getYPredTrue(y_userpred_u1, y_usertrue_u1, y_userpred, y_usertrue, user_prediction_u1base, matriz_calificaciones_u1test)
mae_useru1 = mean_absolute_error(y_usertrue_u1, y_userpred_u1)
y_itempred_u1, y_itemtrue_u1, y_itempred, y_itemtrue = getYPredTrue(y_itempred_u1, y_itemtrue_u1, y_itempred, y_itemtrue, item_prediction_u1base, matriz_calificaciones_u1test)
mae_itemu1 = mean_absolute_error(y_itemtrue_u1, y_itempred_u1)
print("********************* u1 *************************")
print("mae_u1 (user-user)")
print(mae_useru1)
print()
print("u1 - Coeficiente de correlación de PEARSON (user-user)")
print(np.corrcoef(y_usertrue_u1, y_userpred_u1)[0, 1])
print()
print("mae_u1 (item-item)")
print(mae_itemu1)
print()
print("u1 - Coeficiente de correlación de PEARSON (item-item)")
print(np.corrcoef(y_itemtrue_u1, y_itempred_u1)[0, 1])
print("**************************************************")
print()
################################################################################################################################################

###################################### u2  (Segundo Experimento / Iteración) ###########################################
# TRAIN
# Datos con la información de las calificaciones_u2base (CALIFICACIONES)
columnas_calificaciones_u2base = ['user_id2', 'movie_id2', 'rating2', 'unix_timestamp2']
calificaciones_u2base = pd.read_csv('file://localhost/E:/TFG/TFG-RecommenderSystem/data/ml-100k/u2.base', sep='\t', names=columnas_calificaciones_u2base)

# TEST
# Datos con la información de las calificaciones_u2test (CALIFICACIONES)
columnas_calificaciones_u2test = ['user_id2', 'movie_id2', 'rating2', 'unix_timestamp2']
calificaciones_u2test = pd.read_csv('file://localhost/E:/TFG/TFG-RecommenderSystem/data/ml-100k/u2.test', sep='\t', names=columnas_calificaciones_u2test)

# Se obtiene la matriz con las CALIFICACIONES del TRAIN para la primera iteración otorgadas por los usuarios a las películas
matriz_calificaciones_u2base = getMatriz(calificaciones_u2base)

# Se obtienen las matrices con las similitudes entre ususarios y entre items respectivamente
user_similarity_u2base = pairwise_distances(matriz_calificaciones_u2base, metric='cosine') #(user-user)
item_similarity_u2base = pairwise_distances(matriz_calificaciones_u2base.T, metric='cosine') #(item-item)

# Se obtienen las predicciones de los ratings de los usuarios sobre las películas
user_prediction_u2base = prediceUU(matriz_calificaciones_u2base, user_similarity_u2base)#(user-user)
item_prediction_u2base = prediceII(matriz_calificaciones_u2base, item_similarity_u2base)#(item-item)
user_prediction_u2base = formatoInicial(user_prediction_u2base)
item_prediction_u2base = formatoInicial(item_prediction_u2base)

# Se obtiene la matriz con las CALIFICACIONES del TEST para la primera iteración otorgadas por los usuarios a las películas
matriz_calificaciones_u2test = getMatriz(calificaciones_u2test)

############################################## Cálculo de MAE y R2 para u2 ##################################################################
y_usertrue_u2 = []
y_userpred_u2 = []
y_itemtrue_u2 = []
y_itempred_u2 = []

y_userpred_u2, y_usertrue_u2, y_userpred, y_usertrue = getYPredTrue(y_userpred_u2, y_usertrue_u2, y_userpred, y_usertrue, user_prediction_u2base, matriz_calificaciones_u2test)
mae_useru2 = mean_absolute_error(y_usertrue_u2, y_userpred_u2)
y_itempred_u2, y_itemtrue_u2, y_itempred, y_itemtrue = getYPredTrue(y_itempred_u2, y_itemtrue_u2, y_itempred, y_itemtrue, item_prediction_u2base, matriz_calificaciones_u2test)
mae_itemu2 = mean_absolute_error(y_itemtrue_u2, y_itempred_u2)
print("********************* u2 *************************")
print("mae_u2 (user-user)")
print(mae_useru2)
print()
print("u2 - Coeficiente de correlación de PEARSON (user-user)")
print(np.corrcoef(y_usertrue_u2, y_userpred_u2)[0, 1])
print()
print("mae_u2 (item-item)")
print(mae_itemu2)
print()
print("u2 - Coeficiente de correlación de PEARSON (item-item)")
print(np.corrcoef(y_itemtrue_u2, y_itempred_u2)[0, 1])
print("*************************************************")
print()
################################################################################################################################################


###################################### u3  (Tercer Experimento / Iteración) ###########################################
# TRAIN
# Datos con la información de las calificaciones_u3base (CALIFICACIONES)
columnas_calificaciones_u3base = ['user_id2', 'movie_id2', 'rating2', 'unix_timestamp2']
calificaciones_u3base = pd.read_csv('file://localhost/E:/TFG/TFG-RecommenderSystem/data/ml-100k/u3.base', sep='\t', names=columnas_calificaciones_u3base)

# TEST
# Datos con la información de las calificaciones_u3test (CALIFICACIONES)
columnas_calificaciones_u3test = ['user_id2', 'movie_id2', 'rating2', 'unix_timestamp2']
calificaciones_u3test = pd.read_csv('file://localhost/E:/TFG/TFG-RecommenderSystem/data/ml-100k/u3.test', sep='\t', names=columnas_calificaciones_u3test)

# Se obtiene la matriz con las CALIFICACIONES del TRAIN para la primera iteración otorgadas por los usuarios a las películas
matriz_calificaciones_u3base = getMatriz(calificaciones_u3base)

# Se obtienen las matrices con las similitudes entre ususarios y entre items respectivamente
user_similarity_u3base = pairwise_distances(matriz_calificaciones_u3base, metric='cosine') #(user-user)
item_similarity_u3base = pairwise_distances(matriz_calificaciones_u3base.T, metric='cosine') #(item-item)

# Se obtienen las predicciones de los ratings de los usuarios sobre las películas
user_prediction_u3base = prediceUU(matriz_calificaciones_u3base, user_similarity_u3base)#(user-user)
item_prediction_u3base = prediceII(matriz_calificaciones_u3base, item_similarity_u3base)#(item-item)
user_prediction_u3base = formatoInicial(user_prediction_u3base)
item_prediction_u3base = formatoInicial(item_prediction_u3base)

# Se obtiene la matriz con las CALIFICACIONES del TEST para la primera iteración otorgadas por los usuarios a las películas
matriz_calificaciones_u3test = getMatriz(calificaciones_u3test)

############################################## Cálculo de MAE y R2 para u3 ##################################################################
y_usertrue_u3 = []
y_userpred_u3 = []
y_itemtrue_u3 = []
y_itempred_u3 = []

y_userpred_u3, y_usertrue_u3, y_userpred, y_usertrue = getYPredTrue(y_userpred_u3, y_usertrue_u3, y_userpred, y_usertrue, user_prediction_u3base, matriz_calificaciones_u3test)
mae_useru3 = mean_absolute_error(y_usertrue_u3, y_userpred_u3)
y_itempred_u3, y_itemtrue_u3, y_itempred, y_itemtrue = getYPredTrue(y_itempred_u3, y_itemtrue_u3, y_itempred, y_itemtrue, item_prediction_u3base, matriz_calificaciones_u3test)
mae_itemu3 = mean_absolute_error(y_itemtrue_u3, y_itempred_u3)
print("********************* u3 *************************")
print("mae_u3 (user-user)")
print(mae_useru3)
print()
print("u3 - Coeficiente de correlación de PEARSON (user-user)")
print(np.corrcoef(y_usertrue_u3, y_userpred_u3)[0, 1])
print()
print("mae_u3 (item-item)")
print(mae_itemu3)
print()
print("u3 - Coeficiente de correlación de PEARSON (item-item)")
print(np.corrcoef(y_itemtrue_u3, y_itempred_u3)[0, 1])
print("*************************************************")
print()
####################################################################################################################################################################################

###################################### u4  (Cuarto Experimento / Iteración) ###########################################
# TRAIN
# Datos con la información de las calificaciones_u4base (CALIFICACIONES)
columnas_calificaciones_u4base = ['user_id2', 'movie_id2', 'rating2', 'unix_timestamp2']
calificaciones_u4base = pd.read_csv('file://localhost/E:/TFG/TFG-RecommenderSystem/data/ml-100k/u4.base', sep='\t', names=columnas_calificaciones_u4base)

# TEST
# Datos con la información de las calificaciones_u4test (CALIFICACIONES)
columnas_calificaciones_u4test = ['user_id2', 'movie_id2', 'rating2', 'unix_timestamp2']
calificaciones_u4test = pd.read_csv('file://localhost/E:/TFG/TFG-RecommenderSystem/data/ml-100k/u4.test', sep='\t', names=columnas_calificaciones_u4test)

# Se obtiene la matriz con las CALIFICACIONES del TRAIN para la primera iteración otorgadas por los usuarios a las películas
matriz_calificaciones_u4base = getMatriz(calificaciones_u4base)

# Se obtienen las matrices con las similitudes entre ususarios y entre items respectivamente
user_similarity_u4base = pairwise_distances(matriz_calificaciones_u4base, metric='cosine') #(user-user)
item_similarity_u4base = pairwise_distances(matriz_calificaciones_u4base.T, metric='cosine') #(item-item)

# Se obtienen las predicciones de los ratings de los usuarios sobre las películas
user_prediction_u4base = prediceUU(matriz_calificaciones_u4base, user_similarity_u4base)#(user-user)
item_prediction_u4base = prediceII(matriz_calificaciones_u4base, item_similarity_u4base)#(item-item)
user_prediction_u4base = formatoInicial(user_prediction_u4base)
item_prediction_u4base = formatoInicial(item_prediction_u4base)

# Se obtiene la matriz con las CALIFICACIONES del TEST para la primera iteración otorgadas por los usuarios a las películas
matriz_calificaciones_u4test = getMatriz(calificaciones_u4test)

############################################## Cálculo de MAE y R2 para u4 ##################################################################
y_usertrue_u4 = []
y_userpred_u4 = []
y_itemtrue_u4 = []
y_itempred_u4 = []

y_userpred_u4, y_usertrue_u4, y_userpred, y_usertrue = getYPredTrue(y_userpred_u4, y_usertrue_u4, y_userpred, y_usertrue, user_prediction_u4base, matriz_calificaciones_u4test)
mae_useru4 = mean_absolute_error(y_usertrue_u4, y_userpred_u4)
y_itempred_u4, y_itemtrue_u4, y_itempred, y_itemtrue = getYPredTrue(y_itempred_u4, y_itemtrue_u4, y_itempred, y_itemtrue, item_prediction_u4base, matriz_calificaciones_u4test)
mae_itemu4 = mean_absolute_error(y_itemtrue_u4, y_itempred_u4)
print("********************* u4 *************************")
print("mae_u4 (user-user)")
print(mae_useru4)
print()
print("u4 - Coeficiente de correlación de PEARSON (user-user)")
print(np.corrcoef(y_usertrue_u4, y_userpred_u4)[0, 1])
print()
print("mae_u4 (item-item)")
print(mae_itemu4)
print()
print("u4 - Coeficiente de correlación de PEARSON (item-item)")
print(np.corrcoef(y_itemtrue_u4, y_itempred_u4)[0, 1])
print("*************************************************")
print()
####################################################################################################################################################################################

###################################### u5  (Quinto Experimento / Iteración) ###########################################
# TRAIN
# Datos con la información de las calificaciones_u5base (CALIFICACIONES)
columnas_calificaciones_u5base = ['user_id2', 'movie_id2', 'rating2', 'unix_timestamp2']
calificaciones_u5base = pd.read_csv('file://localhost/E:/TFG/TFG-RecommenderSystem/data/ml-100k/u5.base', sep='\t', names=columnas_calificaciones_u5base)

# TEST
# Datos con la información de las calificaciones_u5test (CALIFICACIONES)
columnas_calificaciones_u5test = ['user_id2', 'movie_id2', 'rating2', 'unix_timestamp2']
calificaciones_u5test = pd.read_csv('file://localhost/E:/TFG/TFG-RecommenderSystem/data/ml-100k/u5.test', sep='\t', names=columnas_calificaciones_u5test)

# Se obtiene la matriz con las CALIFICACIONES del TRAIN para la primera iteración otorgadas por los usuarios a las películas
matriz_calificaciones_u5base = getMatriz(calificaciones_u5base)

# Se obtienen las matrices con las similitudes entre ususarios y entre items respectivamente
user_similarity_u5base = pairwise_distances(matriz_calificaciones_u5base, metric='cosine') #(user-user)
item_similarity_u5base = pairwise_distances(matriz_calificaciones_u5base.T, metric='cosine') #(item-item)

# Se obtienen las predicciones de los ratings de los usuarios sobre las películas
user_prediction_u5base = prediceUU(matriz_calificaciones_u5base, user_similarity_u5base)#(user-user)
item_prediction_u5base = prediceII(matriz_calificaciones_u5base, item_similarity_u5base)#(item-item)
user_prediction_u5base = formatoInicial(user_prediction_u5base)
item_prediction_u5base = formatoInicial(item_prediction_u5base)

# Se obtiene la matriz con las CALIFICACIONES del TEST para la primera iteración otorgadas por los usuarios a las películas
matriz_calificaciones_u5test = getMatriz(calificaciones_u5test)

############################################## Cálculo de MAE y R2 para u5 ##################################################################
y_usertrue_u5 = []
y_userpred_u5 = []
y_itemtrue_u5 = []
y_itempred_u5 = []

y_userpred_u5, y_usertrue_u5, y_userpred, y_usertrue = getYPredTrue(y_userpred_u5, y_usertrue_u5, y_userpred, y_usertrue, user_prediction_u5base, matriz_calificaciones_u5test)
mae_useru5 = mean_absolute_error(y_usertrue_u5, y_userpred_u5)
y_itempred_u5, y_itemtrue_u5, y_itempred, y_itemtrue = getYPredTrue(y_itempred_u5, y_itemtrue_u5, y_itempred, y_itemtrue, item_prediction_u5base, matriz_calificaciones_u5test)
mae_itemu5 = mean_absolute_error(y_itemtrue_u5, y_itempred_u5)
print("********************* u5 *************************")
print("mae_u5 (user-user)")
print(mae_useru5)
print()
print("u5 - Coeficiente de correlación de PEARSON (user-user)")
print(np.corrcoef(y_usertrue_u5, y_userpred_u5)[0, 1])
print()
print("mae_u5 (item-item)")
print(mae_itemu5)
print()
print("u5 - Coeficiente de correlación de PEARSON (item-item)")
print(np.corrcoef(y_itemtrue_u5, y_itempred_u5)[0, 1])
print("*************************************************")
print()
####################################################################################################################################################################################

# Media de los MAE de los 5 experimentos
# Se calcula la media de los EAM de los 5 Experimentos / Iteraciones
mae_usermedio = mean_absolute_error(y_usertrue, y_userpred)
mae_itemmedio = mean_absolute_error(y_itemtrue, y_itempred)
print("************************** MAE medio de los 5 Experimentos realizados *************************")
print("mae_medio (user-user)")
print(mae_usermedio)
print()
print("Coeficiente de correlación de PEARSON (user-user)")
print(np.corrcoef(y_usertrue, y_userpred)[0, 1])
print()
print("mae_medio (item-item)")
print(mae_itemmedio)
print()
print("Coeficiente de correlación de PEARSON (item-item)")
print(np.corrcoef(y_itemtrue, y_itempred)[0, 1])
print("***********************************************************************************************")
print()

df_user = pd.DataFrame({'y_usertrue': y_usertrue, 'y_userpred': y_userpred},
                  columns=['y_usertrue', 'y_userpred'])
df_item = pd.DataFrame({'y_itemtrue': y_itemtrue, 'y_itempred': y_itempred},
                  columns=['y_itemtrue', 'y_itempred'])

# se vuelcan los resultados de las predicciones obtenidos mediante los dos algoritmos (user-user) and (item_item)
# Se vuelcan sobre dos ficheros .csv
LUS = pd.DataFrame(data=df_user)
LUS.to_csv('df_user.csv', sep=';', header=False, float_format='%.2f', index=False)
LIS = pd.DataFrame(data=df_item)
LIS.to_csv('df_item.csv', sep=';', header=False, float_format='%.2f', index=False)

######################################################################################################################
