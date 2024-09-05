import numpy as np
import pandas as pd
import math

# Función para calcular la distancia euclidiana entre dos puntos
def distancia_euclidiana(point1, point2):
    distance = 0.0
    for i in range(len(point1)):
        distance += (point1[i] - point2[i]) ** 2
    return math.sqrt(distance)

def distancia_manhattan(point1, point2):
    distance = 0.0
    for i in range(len(point1)):
        distance += (point1[i] - point2[i])
    return distance

# Función para cargar el archivo de texto plano
def cargardatos(archivo, delimitador):
    data=pd.read_csv(archivo, delimiter=delimitador)
    return data

def normalizar_datos(datos):
    # Obtener solo las columnas numéricas
    columnas_numericas = datos.select_dtypes(include=['float64', 'int64'])

    # Normalizar cada columna numérica en el rango [0, 1]
    for columna in columnas_numericas.columns:
        min_valor = datos[columna].min()
        max_valor = datos[columna].max()
        datos[columna] = (datos[columna] - min_valor) / (max_valor - min_valor)

    return datos

# Función para mostrar el promedio y la desviación estándar por clase y atributo
def estadisticas_por_clase(datos, atributo_clase):
    grupos = datos.groupby(atributo_clase)
    for atributo in datos.columns:
        if atributo != atributo_clase:
            print(f"\nAtributo: {atributo}")
            promedios = grupos[atributo].mean()
            desviacion = grupos[atributo].std()
            print(f"\nEl promedio de los datos en el atributo {atributo} es:")
            print(promedios)
            print(f"\nLa desviacion estandar de los datos en el atributo {atributo} es:")
            print(desviacion)
            
# Función para detectar valores atípicos
def detectar_atipicos(datos):
    columnas_numericas = datos.select_dtypes(include=['float64', 'int64'])
    Q1 = columnas_numericas.quantile(0.25)
    Q3 = columnas_numericas.quantile(0.75)
    IQR = Q3 - Q1
    atipicos = ((columnas_numericas < (Q1 - 1.5 * IQR)) | (columnas_numericas > (Q3 + 1.5 * IQR)))
    return atipicos

def describir_atributos(datos):
    # Obtener tipos de datos de cada atributo
    tipos_de_datos = datos.dtypes
    
    # Contadores para atributos numéricos y categóricos
    count_numericos = 0
    count_categoricos = 0
    
    for columna, tipo in tipos_de_datos.items():
        if count_numericos + count_categoricos >= 10:
            break  # Si ya se describieron 10 atributos, se detiene
            
        if tipo == 'object':  # Si es tipo 'object', se considera categórico
            count_categoricos += 1
            categorias = datos[columna].unique()  # Obtener las categorías únicas
            print(f"Atributo '{columna}' es de tipo categórico.")
            print(f"Categorías: {', '.join(map(str, categorias))}")
            
        else:  # Si no es tipo 'object', se considera numérico
            count_numericos += 1
            print(f"Atributo '{columna}' es de tipo numérico.")
            print(f"Min: {datos[columna].min()} - Max: {datos[columna].max()}")
            print(f"Promedio: {datos[columna].mean()} - Desviación estándar: {datos[columna].std()}")
        
        print("------")
    
    # Informar si hay más atributos que no se describieron
    if count_numericos + count_categoricos < len(tipos_de_datos):
        print(f"Se han descrito {count_numericos + count_categoricos} atributos.")
        print(f"Hay {len(tipos_de_datos) - count_numericos - count_categoricos} atributos más.")


def main():
    archivo=input("Escriba el nombre del archivo de donde obtendremos la informacion: ")
    delimitador=input("Seleccione cual es el signo delimitador del archivo: ") 
    datos=cargardatos(archivo,delimitador)
    # Normalizar los datos
    opc2=int(input("Desea normalizar sus datos? 1.Si 2.No "))
    if(opc2==1):
        datos_normalizados = normalizar_datos(datos)
        print("\nDatos normalizados:")
        print(datos_normalizados)
        datos=datos_normalizados
    else:
        print("Ok, sus datos no seran normalizados!\n")
        
    #Describir dataset
    describir_atributos(datos)
    tipos_de_datos = datos.dtypes
    print(tipos_de_datos)
    
    num_filas, num_columnas = datos.shape
    print(f"El DataFrame tiene {num_filas} patrones y {num_columnas} atributos.")

    #Seleccionamos atributos para nuestro vector
    z=(str(input("Escriba el nombre de la columna que quiere predecir: ")))
    #Estadisticas por clase
    atributo_clase = z
    estadisticas_por_clase(datos, atributo_clase)
    
    #Primera verificacion
    x = datos.drop(z, axis=1).values
    y = np.array(datos[z])
    columnas_seleccionadas = input("Escriba los nombres de las columnas que desea seleccionar, separados por coma: ").split(',')

    # Convertir los nombres de las columnas a índices numéricos
    indices_columnas = [datos.columns.get_loc(col) for col in columnas_seleccionadas]

    # Generar la matriz de patrones
    matriz_patrones = x[:, indices_columnas]
    print("\n\n//////////////Verificacion General////////////////////\n\n")
    opc1=0
    while(opc1!=1 and opc1!=2):
        opc1=int(input("Ingrese el tipo de clasificador que desee usar:\n 1.Clasificador Knn 2.Clasificador distancia minima\n"))
        if(opc1==1):
            clas_knn(matriz_patrones,y)
        elif(opc1==2):
            class_min(matriz_patrones,y)
        else:
            print("Seleccione una opcion correcta")
            
    
    #Verificacion sin la primera columna
    print(tipos_de_datos)
    copia_datos=datos
            
    vector_elim_colum=[]
    dato_2=str(input(f"\nIngrese el nombre de la primer columna que desea aliminar: "))
    vector_elim_colum.append(dato_2)
    datos = datos.drop(columns=vector_elim_colum)
    num_filas, num_columnas = datos.shape
    print(f"\nEl DataFrame ahora tiene {num_filas} patrones y {num_columnas} atributos.")
    
    x = datos.drop(z, axis=1).values
    matriz_patrones=x
    
    print("\n\n//////////////Verificacion sin la primera columna////////////////////\n\n")
    opc1=0
    while(opc1!=1 and opc1!=2):
        opc1=int(input("Ingrese el tipo de clasificador que desee usar:\n 1.Clasificador Knn 2.Clasificador distancia minima\n"))
        if(opc1==1):
            clas_knn(matriz_patrones,y)
        elif(opc1==2):
            class_min(matriz_patrones,y)
        else:
            print("Seleccione una opcion correcta")
            
    #Verificacion sin la segunda columna
    tipos_de_datos = datos.dtypes
    print(tipos_de_datos)
    
    vector_elim_colum_2=[]
    dato_3=str(input(f"\nIngrese el nombre de la segunda columna que desea aliminar: "))
    vector_elim_colum_2.append(dato_3)
    datos = datos.drop(columns=vector_elim_colum_2)
    num_filas, num_columnas = datos.shape
    print(f"\nEl DataFrame ahora tiene {num_filas} patrones y {num_columnas} atributos.")
    
    #Se conserva el dataframe, datos ya no tiene la primer columna, pero se esta trabajando con una copia
    x = copia_datos.drop(z, axis=1).values
    matriz_patrones=x
    
    print("\n\n//////////////Verificacion sin la segunda columna////////////////////\n\n")
    opc1=0
    while(opc1!=1 and opc1!=2):
        opc1=int(input("Ingrese el tipo de clasificador que desee usar:\n 1.Clasificador Knn 2.Clasificador distancia minima\n"))
        if(opc1==1):
            clas_knn(matriz_patrones,y)
        elif(opc1==2):
            class_min(matriz_patrones,y)
        else:
            print("Seleccione una opcion correcta")
    
    #Se deja de trabajar con la copia, el dataframe original ya no tiene las dos columnas
    x = datos.drop(z, axis=1).values
    matriz_patrones=x
    tipos_de_datos = datos.dtypes
    
    print("\n\n//////////////Verificacion sin las dos columnas////////////////////\n\n")
    opc1=0
    while(opc1!=1 and opc1!=2):
        opc1=int(input("Ingrese el tipo de clasificador que desee usar:\n 1.Clasificador Knn 2.Clasificador distancia minima\n"))
        if(opc1==1):
            clas_knn(matriz_patrones,y)
        elif(opc1==2):
            class_min(matriz_patrones,y)
        else:
            print("Seleccione una opcion correcta")
            
    # Detectar valores atípicos
    valores_atipicos = detectar_atipicos(datos)
    print("\nValores atípicos:")
    print(valores_atipicos[valores_atipicos.any(axis=1)])
    
    opc3=int(input("\nDesea eliminar algun renglon del dataset? 1.Si 2.No "))
    if(opc3==1):
        vector_elim_renglon=[]
        opc3=int(input(f"\nCuantos renglones desea eliminar (valores entre 1 y {num_filas})?"))
        for i in range (0,opc3):
            dato_1=int(input(f"\nIngrese el indice de renglon que desea aliminar (valores entre 0 y {num_filas-1}): "))
            vector_elim_renglon.append(dato_1)
        datos = datos.drop(vector_elim_renglon)
        num_filas, num_columnas = datos.shape
        print(f"\nEl DataFrame ahora tiene {num_filas} patrones y {num_columnas} atributos.")
        
    else:
        print("Ok, no se ha eliminado ningun renglon!")
        
    x = datos.drop(z, axis=1).values
    matriz_patrones=x
    y = np.array(datos[z]) #Como eliminamos renglones hay que arreglar los indices
    tipos_de_datos = datos.dtypes
    
    print("\n\n//////////////Verificacion sin los renglones////////////////////\n\n")
    opc1=0
    while(opc1!=1 and opc1!=2):
        opc1=int(input("Ingrese el tipo de clasificador que desee usar:\n 1.Clasificador Knn 2.Clasificador distancia minima\n"))
        if(opc1==1):
            clas_knn(matriz_patrones,y)
        elif(opc1==2):
            class_min(matriz_patrones,y)
        else:
            print("Seleccione una opcion correcta")
    

def clas_knn(x,y):
    # Clasificador KNN
    class ClasificadorKNN:
        def __init__(self, n_neighbors):
            self.n_neighbors = n_neighbors

        def fit(self, X, y):
            self.X_train = X
            self.y_train = y

        def predict(self, X):
            y_pred = []
            for sample in X:
                distances = []
                for i, train_sample in enumerate(self.X_train):
                    distance = distancia_euclidiana(sample, train_sample)
                    distances.append((distance, self.y_train[i]))
              
                distances.sort(key=lambda x: x[0])
                neighbors = distances[:self.n_neighbors]
                neighbor_labels = [neighbor[1] for neighbor in neighbors]
                prediction = max(set(neighbor_labels), key=neighbor_labels.count)
                y_pred.append(prediction)
            return y_pred

    # Pedir al usuario el número de vecinos a considerar
    n_neighbors_input = int(input("Introduce el número de vecinos a considerar: "))

    # Crear una instancia del clasificador KNN con el número de vecinos especificado
    knn_classifier = ClasificadorKNN(n_neighbors=n_neighbors_input)
    
    #Creamos un arreglo para los resultados generales de las 3 pruebas
    #Para imprimirlos en un archivo txt y sea mejor su visualizacion
    resultados_generales=[]

    #Train and test
    x_train, x_test, y_train, y_test = train_test(x, y)
    knn_classifier.fit(x_train, y_train)
    y_pred = knn_classifier.predict(x_test)
    accuracy = np.mean(y_pred == y_test) *100
    error = 100 - accuracy
    print("\n////////////Precicion Train and Test//////////////\n")
    print(f"Porcentaje de precisión de la clasificación de distancia mínima: {accuracy:.2f}%")
    print(f"Precisión de error en la clasificación de distancia mínima: {error:.2f}%")
    
    resultados_generales.append(f"Train and Test: {accuracy:.2f}%")
            
    #K fold
    k = int(input("\nIngrese la cantidad de grupos (K) para la validación cruzada: "))
    n_muestras = len(x)
    tamano_grupo = n_muestras // k

    accuracy_scores = []
    error_scores = []

    print("\n////////////Precision K fold//////////////\n")

    for i in range(k):
        inicio = i * tamano_grupo
        fin = (i + 1) * tamano_grupo if i < k - 1 else n_muestras

        x_test = x[inicio:fin]
        y_test = y[inicio:fin]

        x_train = np.concatenate([x[:inicio], x[fin:]])
        y_train = np.concatenate([y[:inicio], y[fin:]])

        knn_classifier.fit(x_train, y_train)
        y_pred = knn_classifier.predict(x_test)

        accuracy = np.mean(y_pred == y_test) * 100
        error = np.mean(y_pred != y_test) * 100

        accuracy_scores.append(accuracy)
        error_scores.append(error)

        print(f"Porcentaje de precisión para el experimento {i + 1}: {accuracy:.2f}%")
        print(f"Porcentaje de error para el experimento {i + 1}: {error:.2f}%")

    avg_accuracy = np.mean(accuracy_scores)
    avg_error = np.mean(error_scores)
    std_accuracy = np.std(accuracy_scores)
    std_error = np.std(error_scores)

    print("\nResultados generales:")
    print(f"Porcentaje de precisión promedio: {avg_accuracy:.2f}% ± {std_accuracy:.2f}")
    print(f"Porcentaje de error promedio: {avg_error:.2f}% ± {std_error:.2f}")
    
    resultados_generales.append(f"K fold: {avg_accuracy:.2f}%")
                    
    #Bootstrap
    k = int(input("\nIngrese la cantidad de experimentos (K) para el bootstrap: "))
    muestras_entrenamiento = int(input("Ingrese la cantidad de muestras en el conjunto de entrenamiento: "))
    muestras_prueba = int(input("Ingrese la cantidad de muestras en el conjunto de prueba: "))

    accuracy_scores = []
    error_scores = []
    class_accuracy_scores = {}
    class_error_scores = {}
    
    print("\n////////////Precicion Bootstrap//////////////\n")

    for i in range(k):
        # Muestreo bootstrap para crear conjuntos de entrenamiento y prueba
        train_indices = np.random.choice(range(len(x)), size=muestras_entrenamiento, replace=True)
        test_indices = np.random.choice(range(len(x)), size=muestras_prueba, replace=True)
                
        x_train, x_test = x[train_indices], x[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        knn_classifier.fit(x_train, y_train)
        y_pred = knn_classifier.predict(x_test)
                
        # Calcular la precisión y el error para cada grupo
        accuracy = np.mean(y_pred == y_test) * 100
        error = np.mean(y_pred != y_test) * 100
                
        accuracy_scores.append(accuracy)
        error_scores.append(error)
                
        # Calcular precisión y error por clase
        unique_classes = np.unique(y_test)
        for cls in unique_classes:
            cls_indices = np.where(y_test == cls)[0]
            cls_pred = np.array(y_pred)[cls_indices]  # Filtrar predicciones para la clase actual
            cls_accuracy = np.mean(cls_pred == cls) * 100
            cls_error = 100 - cls_accuracy
                    
            if cls not in class_accuracy_scores:
                class_accuracy_scores[cls] = []
            if cls not in class_error_scores:
                class_error_scores[cls] = []
                    
            class_accuracy_scores[cls].append(cls_accuracy)
            class_error_scores[cls].append(cls_error)
                
        print(f"Porcentaje de precisión para el experimento {i+1}: {accuracy:.2f}%")
        print(f"Porcentaje de error para el experimento {i+1}: {error:.2f}%")
        
    avg_accuracy = np.mean(accuracy_scores)
    avg_error = np.mean(error_scores)
    std_accuracy = np.std(accuracy_scores)
    std_error = np.std(error_scores)

    print("\nResultados generales:")
    print(f"Porcentaje de precisión promedio: {avg_accuracy:.2f}% ± {std_accuracy:.2f}")
    print(f"Porcentaje de error promedio: {avg_error:.2f}% ± {std_error:.2f}")
    
    resultados_generales.append(f"Bootstrap: {avg_accuracy:.2f}%")
    resultados_generales.append("//////////////////////////////////")

    # Calcular el promedio y la desviación estándar de precisión y error por clase
    for cls, cls_acc_scores in class_accuracy_scores.items():
        cls_avg_acc = np.mean(cls_acc_scores)
        cls_std_acc = np.std(cls_acc_scores)
        cls_avg_err = np.mean(class_error_scores[cls])
        cls_std_err = np.std(class_error_scores[cls])
                
        print(f"\nResultados para la clase {cls}:")
        print(f"Porcentaje de precisión promedio: {cls_avg_acc:.2f}% ± {cls_std_acc:.2f}")
        print(f"Porcentaje de error promedio: {cls_avg_err:.2f}% ± {cls_std_err:.2f}")
    
    #Guardar en el txt
    with open("resultados_clasificacion.txt", "a") as file:
        file.write("\n".join(resultados_generales) + "\n")

    print("\nResultados generales guardados en 'resultados_clasificacion.txt'\n")
    
    
    
def class_min(x, y):
    class ClasificadorDistanciaMinima:
        def fit(self, X, y):
            self.X_train = X
            self.y_train = y
            self.class_means = {}

            # Calcular el promedio de cada clase
            unique_labels = set(y)
            for label in unique_labels:
                class_samples = X[y == label]
                class_mean = np.mean(class_samples, axis=0)
                self.class_means[label] = class_mean

        def predict(self, X):
            y_pred = []
            for sample in X:
                min_distance = float('inf')
                nearest_label = None
                for label, class_mean in self.class_means.items():
                    distance = distancia_euclidiana(sample, class_mean)

                    if distance < min_distance:
                        min_distance = distance
                        nearest_label = label
                y_pred.append(nearest_label)
            return y_pred

    min_distance = ClasificadorDistanciaMinima()
    
    #Arreglo para guardar los porcentajes generales de la validacion en el txt
    resultados_generales=[]

    #Train and test
    x_train, x_test, y_train, y_test = train_test(x, y)
    min_distance.fit(x_train, y_train)
    y_pred = min_distance.predict(x_test)
    accuracy = np.mean(y_pred == y_test) *100
    error = 100 - accuracy
    print("\n////////////Precicion Train and Test//////////////\n")
    print(f"Porcentaje de precisión de la clasificación de distancia mínima: {accuracy:.2f}%")
    print(f"Precisión de error en la clasificación de distancia mínima: {error:.2f}%")
    
    resultados_generales.append(f"Train and Test: {accuracy:.2f}%")
            
    #K fold
    k = int(input("\nIngrese la cantidad de grupos (K) para la validación cruzada: "))
    n_muestras = len(x)
    tamano_grupo = n_muestras // k

    accuracy_scores = []
    error_scores = []

    print("\n////////////Precision K fold//////////////\n")

    for i in range(k):
        inicio = i * tamano_grupo
        fin = (i + 1) * tamano_grupo if i < k - 1 else n_muestras

        x_test = x[inicio:fin]
        y_test = y[inicio:fin]

        x_train = np.concatenate([x[:inicio], x[fin:]])
        y_train = np.concatenate([y[:inicio], y[fin:]])

        min_distance.fit(x_train, y_train)
        y_pred = min_distance.predict(x_test)

        accuracy = np.mean(y_pred == y_test) * 100
        error = np.mean(y_pred != y_test) * 100

        accuracy_scores.append(accuracy)
        error_scores.append(error)

        print(f"Porcentaje de precisión para el experimento {i + 1}: {accuracy:.2f}%")
        print(f"Porcentaje de error para el experimento {i + 1}: {error:.2f}%")

    avg_accuracy = np.mean(accuracy_scores)
    avg_error = np.mean(error_scores)
    std_accuracy = np.std(accuracy_scores)
    std_error = np.std(error_scores)

    print("\nResultados generales:")
    print(f"Porcentaje de precisión promedio: {avg_accuracy:.2f}% ± {std_accuracy:.2f}")
    print(f"Porcentaje de error promedio: {avg_error:.2f}% ± {std_error:.2f}")
    
    resultados_generales.append(f"K fold: {avg_accuracy:.2f}%")
            
    #Bootstrap
    # Se solicita al usuario la cantidad de experimentos, la cantidad de muestras para entrenamiento y prueba
    k = int(input("\nIngrese la cantidad de experimentos (K) para el bootstrap: "))
    muestras_entrenamiento = int(input("Ingrese la cantidad de muestras en el conjunto de entrenamiento: "))
    muestras_prueba = int(input("Ingrese la cantidad de muestras en el conjunto de prueba: "))

    # Listas para almacenar resultados generales y por clase
    accuracy_scores = []
    error_scores = []
    class_accuracy_scores = {}
    class_error_scores = {}
    
    print("\n////////////Precicion Bootstrap//////////////\n")

    # Bucle para realizar K experimentos de bootstrap
    for i in range(k):
        # Muestreo bootstrap para crear conjuntos de entrenamiento y prueba
        train_indices = np.random.choice(range(len(x)), size=muestras_entrenamiento, replace=True)
        test_indices = np.random.choice(range(len(x)), size=muestras_prueba, replace=True)
                
        x_train, x_test = x[train_indices], x[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        # Entrenar el clasificador con el conjunto de entrenamiento y predecir el conjunto de prueba
        min_distance.fit(x_train, y_train)
        y_pred = min_distance.predict(x_test)
                
        # Calcular precisión y error para el experimento actual
        accuracy = np.mean(y_pred == y_test) * 100
        error = np.mean(y_pred != y_test) * 100
                
        # Almacenar precisión y error en las listas correspondientes
        accuracy_scores.append(accuracy)
        error_scores.append(error)
                
        # Calcular precisión y error por clase
        unique_classes = np.unique(y_test)
        for cls in unique_classes:
            cls_indices = np.where(y_test == cls)[0]
            cls_pred = np.array(y_pred)[cls_indices]  # Filtrar predicciones para la clase actual
            cls_accuracy = np.mean(cls_pred == cls) * 100
            cls_error = 100 - cls_accuracy
                    
            # Almacenar precisión y error por clase en diccionarios separados
            if cls not in class_accuracy_scores:
                class_accuracy_scores[cls] = []
            if cls not in class_error_scores:
                class_error_scores[cls] = []
                    
            class_accuracy_scores[cls].append(cls_accuracy)
            class_error_scores[cls].append(cls_error)
                
        # Imprimir resultados del experimento actual
        print(f"Porcentaje de precisión para el experimento {i+1}: {accuracy:.2f}%")
        print(f"Porcentaje de error para el experimento {i+1}: {error:.2f}%")

    # Calcular el promedio y la desviación estándar de precisión y error para todos los experimentos
    avg_accuracy = np.mean(accuracy_scores)
    avg_error = np.mean(error_scores)
    std_accuracy = np.std(accuracy_scores)
    std_error = np.std(error_scores)

    # Imprimir resultados generales
    print("\nResultados generales:")
    print(f"Porcentaje de precisión promedio: {avg_accuracy:.2f}% ± {std_accuracy:.2f}")
    print(f"Porcentaje de error promedio: {avg_error:.2f}% ± {std_error:.2f}")
    
    resultados_generales.append(f"Bootstrap: {avg_accuracy:.2f}%")
    resultados_generales.append("//////////////////////////////////")

    # Calcular el promedio y la desviación estándar de precisión y error por clase
    for cls, cls_acc_scores in class_accuracy_scores.items():
        cls_avg_acc = np.mean(cls_acc_scores)
        cls_std_acc = np.std(cls_acc_scores)
        cls_avg_err = np.mean(class_error_scores[cls])
        cls_std_err = np.std(class_error_scores[cls])
                
        # Imprimir resultados por clase
        print(f"\nResultados para la clase {cls}:")
        print(f"Porcentaje de precisión promedio: {cls_avg_acc:.2f}% ± {cls_std_acc:.2f}")
        print(f"Porcentaje de error promedio: {cls_avg_err:.2f}% ± {cls_std_err:.2f}")
        
    #Guardar en el txt
    with open("resultados_clasificacion.txt", "a") as file:
        file.write("\n".join(resultados_generales) + "\n")

    print("\nResultados generales guardados en 'resultados_clasificacion.txt'\n")
    
    
def train_test(x, y):
    porcentaje_entrenamiento = float(input("Ingrese el porcentaje de muestras para el conjunto de entrenamiento (0-1): "))
    #Obtener la cantidad de muestras para el conjunto de entrenamiento
    num_entrenamiento = int(len(x) * porcentaje_entrenamiento)
    #Dividir los datos en conjuntos de entrenamiento y prueba
    #Los primeros son entrenamiento, los demas son prueba 
    x_train, x_test = x[:num_entrenamiento], x[num_entrenamiento:]
    y_train, y_test = y[:num_entrenamiento], y[num_entrenamiento:]

    return x_train, x_test, y_train, y_test

    
main()