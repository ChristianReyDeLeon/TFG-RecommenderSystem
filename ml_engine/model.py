import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
from pandas import ExcelWriter

# Datos de usuarios (USER)
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('file://localhost/E:/TFG/TFG-RecommenderSystem/data/ml-100k/u.user', sep='|', names=u_cols)

# Datos de valiraciones/calificaciones (DATA)
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('file://localhost/E:/TFG/TFG-RecommenderSystem/data/ml-100k/u.data', sep='\t', names=r_cols)

# Datos de películas (ITEM)
i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
items = pd.read_csv('file://localhost/E:/TFG/TFG-RecommenderSystem/data/ml-100k/u.item', sep='|', names=i_cols,
                    encoding='latin-1')

# Ahora podemos revisar la data
print("USUARIOS")
print(users.shape)
print(users.head())
print()

print("CALIFICACIONES")
print(ratings.shape)
print(ratings.head())
print()

print("ITEMS")
print(items.shape)
print(items.head())
print()

n_users = ratings.user_id.unique().shape[0]
n_items = ratings.movie_id.unique().shape[0]

data_matrix = np.zeros((n_users, n_items))
for line in ratings.itertuples():
    data_matrix[line[1]-1, line[2]-1] = line[3]
print("Data matrix (MATRIZ DE CALIFICACIONES) Usuarios-Ítems:")
print(data_matrix)
print()


user_similarity = pairwise_distances(data_matrix, metric='cosine')
item_similarity = pairwise_distances(data_matrix.T, metric='cosine')

print("User similarity:")
print(user_similarity)
print()
print("Item similarity:")
print(item_similarity)
print()



# Función para realizar predicciones basadas en las anteriores similitudes calculadas
def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        # ELIMINAMOS EL SESGO de las calificaciones
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred


user_prediction = predict(data_matrix, user_similarity, type='user')
item_prediction = predict(data_matrix, item_similarity, type='item')


print("USER PREDICTION:")
print(user_prediction)
print()
print("ITEM PREDICTION:")
print(item_prediction)
print()

# Cogeremos todas las calificaciones y calcularemos su error

a = np.array(data_matrix)
b = np.array(user_prediction)
print(a)
print(b)
c = a-b
print("c")
print(c)
# Recorreremos los únicos elementos de data_matrix que hubieran existido
arrayTemp = []
for i in range(len(data_matrix)):
    for j in range(len(data_matrix[i])):
        if data_matrix[i][j] != 0:
            arrayTemp.append(data_matrix[i][j] - user_prediction[i][j])


L = pd.DataFrame(data=arrayTemp)
L.to_csv('ArrayTemp.csv', sep=';', header=False, float_format='%.2f', index=False)




