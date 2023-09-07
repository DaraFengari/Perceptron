import numpy as np
import csv

# Función para leer los patrones desde un archivo CSV
def leer_patrones(filename):
    entradas = []
    salidas = []
    with open(filename, mode='r', newline='') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            entradas.append([float(x) for x in row[:-1]])
            salidas.append(float(row[-1]))
    return np.array(entradas), np.array(salidas)

def activacion(pesos,x,b):
    z=pesos * x
    if z.sum() + b > 0:
        return 1
    else:
        return -1

def probando(pesos,x,b):
    z=pesos * x
    if z.sum() + b > 0 :
        return -1
    else:
        return 1

# Función para entrenar un perceptrón simple
def entrenar_perceptron(entradas, salidas, tasa_aprendizaje, max_epocas, criterio_error):
    num_entradas = entradas.shape[1]
    num_patrones = entradas.shape[0]

    # Inicialización de pesos y bias
    pesos = np.random.uniform(-1,1,size=2)
    print(f"{pesos}")
    bias = np.random.uniform(-1,1)
    epoca=0
    
    for epoca in range(max_epocas):
        error_epoca=0
        print(f"texto largoteeeeeeeeeeeeeeeee{epoca}")
        for i in range(num_patrones):
            prediccion=activacion(pesos,entradas[i],bias)
            error= salidas[i]-prediccion

            pesos[0] += tasa_aprendizaje * entradas[i][0] * error
            pesos[1] += tasa_aprendizaje * entradas[i][1] * error
            bias += tasa_aprendizaje * error
            error_epoca += error**2
            print(f"{error},{epoca},{bias},{pesos}")
        print(f"{error_epoca}")

    return pesos, bias

# Función para probar el perceptrón entrenado en datos reales
def probar_perceptron(entradas, pesos, bias):
    num_patrones = entradas.shape[0]
    resultados = []

    for i in range(num_patrones):
        prediccion=probando(pesos,entradas[i],bias)
        resultados.append(prediccion)

    return resultados

if __name__ == "__main__":
    # Lee los patrones de entrenamiento desde un archivo CSV
    archivo_entrenamiento = "patrones_entrenamiento.csv"
    entradas, salidas = leer_patrones(archivo_entrenamiento)

    # Configura los hiperparámetros
    tasa_aprendizaje = 0.01
    max_epocas = 1000
    criterio_error = 1

    # Entrena el perceptrón
    pesos, bias = entrenar_perceptron(entradas, salidas, tasa_aprendizaje, max_epocas, criterio_error)
    print(f"{pesos} y {bias}")

    # Lee los patrones de prueba desde otro archivo CSV
    archivo_prueba = "XOR_tst.csv"
    entradas_prueba, salidas_prueba = leer_patrones(archivo_prueba)

    # Prueba el perceptrón entrenado en datos de prueba
    resultados_prueba = probar_perceptron(entradas_prueba, pesos, bias)

    print("Resultados de prueba:")
    for i, resultado in enumerate(resultados_prueba):
        print(f"Entrada: {entradas_prueba[i]}, Salida: {resultado}")
