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


def main():
    archivo=input("Escriba el nombre del archivo de donde obtendremos la informacion: ")
    delimitador=input("Seleccione cual es el signo delimitador del archivo: ") 
    datos=cargardatos(archivo,delimitador)
    num_filas, num_columnas = datos.shape
    print(f"El DataFrame tiene {num_filas} patrones y {num_columnas} atributos.")
    tipos_de_datos = datos.dtypes
    print(tipos_de_datos)
    #Seleccionamos atributos para nuestro vector
    z=(str(input("Escriba el nombre de la columna que quiere predecir: ")))
    x = datos.drop(z, axis=1).values
    y = np.array(datos[z])
    limite_inferior_1 = int(input(f"Seleccione el limite inferior (Valores entre 0 y {num_columnas-2}) para generar el vector de attributos: "))
    limite_superior_1 = int(input(f"Ahora el limite superior (Valores entre {limite_inferior_1} y {num_columnas-2}): "))
    matriz_patrones = x[:, limite_inferior_1:limite_superior_1+1]
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
        def __init__(self, n_neighbors=1):
            self.n_neighbors = n_neighbors

        def fit(self, X, y):
            self.X_train = X
            self.y_train = y

        def predict(self, X):
            met_distancia=int(input("Ingrese el tipo de distancia que desee usar:\n 1.Euclidiana 2.Manhattan\n"))
            y_pred = []
            for sample in X:
                distances = []
                for i, train_sample in enumerate(self.X_train):
                    if(met_distancia==1):
                        distance = distancia_euclidiana(sample, train_sample)
                    else:
                        distance = distancia_manhattan(sample, train_sample)
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
    opc=0
    while(opc!=1 and opc!=2 and opc!=3):
        opc = int(input("Elija el tipo de validacion que desee usar: 1.Train and test 2.K-fold cross-validation 3.Bootstrap\n"))
        if opc == 1:
            x_train, x_test, y_train, y_test = train_test(x, y)
            knn_classifier.fit(x_train, y_train)
            y_pred = knn_classifier.predict(x_test)
            accuracy = np.mean(y_pred == y_test) *100
            error = 100 - accuracy
            print(f"Porcentaje de precisión de la clasificación de distancia mínima: {accuracy:.2f}%")
            print(f"Precisión de error en la clasificación de distancia mínima: {error:.2f}%")
            
        elif opc==2:
            #K fold
            k = int(input("\nIngrese la cantidad de grupos (K) para la validación cruzada: "))
            n_muestras = len(x)
            tamano_grupo = n_muestras // k

            accuracy_scores = []
            error_scores = []

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
            
        elif opc==3:
            k = int(input("Ingrese la cantidad de experimentos (K) para el bootstrap: "))
            muestras_entrenamiento = int(input("Ingrese la cantidad de muestras en el conjunto de entrenamiento: "))
            muestras_prueba = int(input("Ingrese la cantidad de muestras en el conjunto de prueba: "))

            accuracy_scores = []
            error_scores = []
            class_accuracy_scores = {}
            class_error_scores = {}

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

            # Calcular el promedio y la desviación estándar de precisión y error por clase
            for cls, cls_acc_scores in class_accuracy_scores.items():
                cls_avg_acc = np.mean(cls_acc_scores)
                cls_std_acc = np.std(cls_acc_scores)
                cls_avg_err = np.mean(class_error_scores[cls])
                cls_std_err = np.mean(class_error_scores[cls])
                
                print(f"\nResultados para la clase {cls}:")
                print(f"Porcentaje de precisión promedio: {cls_avg_acc:.2f}% ± {cls_std_acc:.2f}")
                print(f"Porcentaje de error promedio: {cls_avg_err:.2f}% ± {cls_std_err:.2f}")
        else:
                print("Seleccione una opcion correcta")
    
    
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

    opc=0
    while(opc!=1 and opc!=2 and opc!=3):
        opc = int(input("Elija el tipo de validacion que desee usar: 1.Train and test 2.K-fold cross-validation 3.Bootstrap\n"))
        if opc == 1:
            x_train, x_test, y_train, y_test = train_test(x, y)
            min_distance.fit(x_train, y_train)
            y_pred = min_distance.predict(x_test)
            accuracy = np.mean(y_pred == y_test) *100
            error = 100 - accuracy
            print(f"Porcentaje de precisión de la clasificación de distancia mínima: {accuracy:.2f}%")
            print(f"Precisión de error en la clasificación de distancia mínima: {error:.2f}%")
            
        elif opc==2:
            k = int(input("\nIngrese la cantidad de grupos (K) para la validación cruzada: "))
            n_muestras = len(x)
            tamano_grupo = n_muestras // k

            accuracy_scores = []
            error_scores = []

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
            
            
        elif opc==3:
            # Se solicita al usuario la cantidad de experimentos, la cantidad de muestras para entrenamiento y prueba
            k = int(input("Ingrese la cantidad de experimentos (K) para el bootstrap: "))
            muestras_entrenamiento = int(input("Ingrese la cantidad de muestras en el conjunto de entrenamiento: "))
            muestras_prueba = int(input("Ingrese la cantidad de muestras en el conjunto de prueba: "))

            # Listas para almacenar resultados generales y por clase
            accuracy_scores = []
            error_scores = []
            class_accuracy_scores = {}
            class_error_scores = {}

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
        else:
                print("Seleccione una opcion correcta")
    
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