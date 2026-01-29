'''This file defines the mathematical activation functions.
- ReLU
- Softmax
'''
import numpy as np

def relu(Z):
    """
    Función ReLU (Rectified Linear Unit).
    Si Z > 0, devuelve Z. Si Z <= 0, devuelve 0.
    Esta funcion es la funcion ReLU que deja pasar la señal forward propagation.
    """
    return np.maximum(0, Z)

def relu_derivative(Z):
    """
    Derivada de ReLU. 
    Es 1 para valores positivos y 0 para negativos.
    Esta funcion de ReLU deja pasar la señal en el back propagation.
    """
    return Z > 0 # Devuelve un valor booleano True/False que en matematica de python funciona como 1 o 0
    # si Z > 0: la derivada es 1
    # si Z < o igual a 0: la derivada es 0

def softmax(Z):
    """
    Convierte las salidas de la última capa en probabilidades.
    Z es una matriz de (10, m).

    Softmax devuelve un porcentaje de probabilidad, por ejemplo 10%, 55%, 99.1%, etc.
    Por cada una de las predicciones se optiene su exponente natural con el numero de euler.
    Los valores resultantes lo suma, y luego didive cada valor resultante entre esa suma.

    La funcion softmax se utiliza en la capa de salida.
    """

    # Restamos el máximo de cada columna por estabilidad numérica 
    # (evita que np.exp explote con números grandes)
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

def categorical_cross_entropy(y_true, y_pred):
    """
    y_true: Etiquetas reales en formato One-Hot (10, m)
    y_pred: Predicciones de la red tras Softmax (10, m)

    El Trabajo de esta funcion es ponerle una nota (numero) a que tan bien o mal
    lo hizo la red en sus predicciones.Si la nota es alta, la red se equivoco mucho, 
    si es baja (cercana a cero), la red acerto.

    y_true, y_pred, estan organizadas como matrices, donde las filas son las categorias (0 al 9),
    y las columnas son los ejemplos individuales (imagenes)
    """
    
    m = y_true.shape[1] # m, es el numero total de imagenes que tenemos que procesar.

    # Usamos un pequeño epsilon (1e-15) para evitar el logaritmo de cero,
    # lo cual daría un error matemático.
    epsilon = 1e-15 # notacion cientifica (epsilon) es una notacion cientifica que representa
                    # un valor casi cercano a cero (0.000000000000001)
    ''' El valor de epsilon se crea para evitar que en la expresion mas abajo donde se utiliza 
        la funcion para hallar el logaritmo de un numero no sea infinito. Esto es porque el logaritmo
        de 0 no existe (tiende a infinito negativo). Esto evita que si la red neuronal asigna una 
        probabilidad de exactamente 0.00000, al intentar calcular log(0), python arrojara un error
        o un valor NaN (Not a Number). Esto romperia todo el entrenamiento.
    '''
    # linea de codigo que limita los valores minimos y maximos de y_pred
    # el valor minimo de y_pred =  epsilon (0.000000000000001)
    # el valor maximo de y_pred = 1 - epsilon (0.999999999999999)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Esta linea es la formula matematica final que calcula el error total.
    loss = -np.sum(y_true * np.log(y_pred)) / m
    
    return loss