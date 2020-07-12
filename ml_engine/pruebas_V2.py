from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def calculateSimilarity(array1,array2,key1,key2):
        vec1 = np.array([array1])
        vec2 = np.array([array2])
        result = cosine_similarity(vec1,vec2)[0][0]
        rounded = round(result,2)
        print(key1+"-"+key2+": "+str(rounded))
        return rounded

history = {
    'c1':['i1','i2','i5'],
    'c2':['i2','i3','i5'],
    'c3':['i1','i2','i3','i5'],
    'c4':['i1','i3','i5'],
    'c5':['i3','i4','i5'],
    'c6':['i1','i2','i4'],
    'c7':['i2','i4','i5'],
    'c8': ['i1', 'i4'],
    'c9': ['i2', 'i3', 'i5'],
    'c10': ['i2', 'i3', 'i5'],
}


listProducts = {
    'i1' : 'Niña',
    'i2' : 'Chloé',
    'i3' : 'Hamburguesa',
    'i4' : 'Pizza',
    'i5' : 'Perrito caliente'
}


def createDynamicObjectPreferences(maxValue):
    products = {}
    print("///////////// DATOS DE CONSUMO DE USUARIOS /////////////////////")
    for i in range(maxValue):
        productID = "i" + str(i+1)
        arrayTemp = []
        for customer in history:
                try:
                    history[customer].index(productID)
                    arrayTemp.append(1)
                except:
                    arrayTemp.append(0)
                products[productID] = arrayTemp
        print("-- "+productID+" --")
        print(arrayTemp)
    return products

def compareBetweenProducts(products,maxValue):
    resultTable = {}
    for i in range(maxValue):
        productID = "i"+str(i+1)
        resultTable[productID] = {}
    print("///////////////// RESULTADOS DEL CONSENO DE SIMILITUD //////////////////")
    for i in range(maxValue):
        productID1 = "i"+str(i+1)
        for j in range(i,maxValue-1):
            productID2 = "i"+str(j+2)
            valSimilarity = calculateSimilarity(products[productID1], products[productID2], productID1, productID2)
            resultTable[productID1][productID2] = valSimilarity
            resultTable[productID2][productID1] = valSimilarity
        print("//////////////////// MATRIZ DE SIMILITUD DE LOS ITEMS /////////////////////")
        for i in range(maxValue):
            productID = "i"+str(i+1)
            print(" -- "+productID+ " --")
            print(resultTable[productID])
        print("////////////////////////////////////")
        return resultTable



print("-------- MENU ---------")
for i in range(5):
    productID = "i"+str(i+1)
    print("código: "+ productID + "      --> "+listProducts[productID])
value = input("¿Qué deseas pedir? (Ingrese el código exacto) ")
value = value.lower()
if len(value) < 2:
    value = "i"+value
productName = listProducts[value]
dynamycMatrixProducts = createDynamicObjectPreferences(5)
tableResult = compareBetweenProducts(dynamycMatrixProducts,5)
itemSelected = tableResult[value]
maxVal = -1
maxId = "-"
for son in itemSelected:
    if maxId == "-":
        maxId = son
        maxValue = itemSelected[son]
    else:
        if maxVal < itemSelected[son]:
            maxId = son
            maxVal = itemSelected[son]
productNameRecommend = listProducts[maxId]
print(maxId)
print()
print()
print()
print(" Gracias por comprar " + productName + "       :D")
print("*************** TAMBIÉN PODRÍA INTERESARTE " + productNameRecommend)
print("*************** TAMBIÉN PODRÍA INTERESARTE " + productNameRecommend_2)

print()
print(tableResult)