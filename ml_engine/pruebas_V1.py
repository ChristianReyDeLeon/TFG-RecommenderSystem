import numpy as np
import sklearn

 # Definición de diccionarios
diccionario = {
    "clave1":234,
    "clave2":True,
    "clave3":"Valor 1",
    "clave4":[1,2,3,4]
}
versiones = {'python':2.7, 'zope':2.13, 'plone':None}

# Pruebas de diccionarios
'''
print (diccionario, tipo(diccionario))
print("")
print(versiones['plone'])
print(versiones['python'])
print(versiones['zope'])
print("")


print("Hola me llamo Christian y este es un diccionario de Python: ")
for key, value in diccionario.items() :
    print(key, value)
'''

data_matrix = np.array([
    [1,2,0,5,3],
    [3,2,0,5,3],
    [5,1,4,2,2]])

print("Data matrix (MATRIZ DE CALIFICACIONES) Usuarios-Ítems:")
print(data_matrix)
print()
us = sklearn.metrics.pairwise_distances(data_matrix, metric='cosine')
#user_similarity = pairwise_distances(matriz_calificaciones, metric='cosine')




