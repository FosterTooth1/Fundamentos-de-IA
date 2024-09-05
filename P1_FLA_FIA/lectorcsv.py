import numpy as np
import pandas as pd

# Función para cargar el archivo de texto plano
def cargardatos(archivo, delimitador):
    data=pd.read_csv(archivo, delimiter=delimitador)
    return data

#Funcion para imprimir en el txt
def imprimirtxt(datos_array, tipos_de_datos, vector_atributos, matriz_patrones, matriz_patrones_reducida):
    with open('impresion.txt', 'w') as archivo:
        archivo.write('Informacion completa del dataset\n')
        archivo.write(str(datos_array))
        archivo.write("\n\nInformacion de el tipo de datos de los atributos del dataset\n")
        archivo.write(str(tipos_de_datos))
        archivo.write("\n\nAtributos selecionados del dataset\n")
        archivo.write(str(vector_atributos))
        archivo.write("\n\nMatriz que muestra los patrones seleccionados\n")
        archivo.write(str(matriz_patrones))
        archivo.write("\n\nMatriz que muestra los patrones seleccionados reducidos con base en los atributos seleccionados\n")
        archivo.write(str(matriz_patrones_reducida))
        archivo.write("\n")

def main():
    archivo=input("Escriba el nombre del archivo de donde obtendremos la informacion: ")
    delimitador=input("Seleccione cual es el signo delimitador del archivo: ") 
    datos=cargardatos(archivo,delimitador)
    num_filas, num_columnas = datos.shape
    print(f"El DataFrame tiene {num_filas} patrones y {num_columnas} atributos.")
    tipos_de_datos = datos.dtypes
    #Seleccionamos atributos al azar para generar el vector
    limite_inferior_1 = int(input(f"Seleccione el limite inferior (Valores entre 0 y {num_columnas}) para generar el vector de attributos: "))
    limite_superior_1 = int(input(f"Ahora el limite superior (Valores entre {limite_inferior_1} y {num_columnas}): "))
    vector_atributos= list(datos.columns[limite_inferior_1:limite_superior_1])
    #Seleccionamos patrones al azar para generar la matriz
    limite_inferior_2 = int(input(f"Seleccione el limite inferior (Valores entre 0 y {num_filas}) para generar la matriz de patrones: "))
    limite_superior_2 = int(input(f"Ahora el limite superior (Valores entre {limite_inferior_2} y {num_filas}): "))
    matriz_patrones= datos.iloc[limite_inferior_2:limite_superior_2]
    matriz_patrones=matriz_patrones.values
    matriz_patrones_reducida = matriz_patrones[:, limite_inferior_1:limite_superior_1]
    datos_array=np.array(datos)
    imprimirtxt(datos_array, tipos_de_datos,vector_atributos,matriz_patrones,matriz_patrones_reducida)
    print("Se ha impreso correctamente la informacion en el archivo de texto!")
main()