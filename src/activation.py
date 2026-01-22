'''This file defines the mathematical activation functions.
- ReLU
- Softmax
'''
import numpy as np

def relu(Z):
    """
    Función ReLU (Rectified Linear Unit).
    Si Z > 0, devuelve Z. Si Z <= 0, devuelve 0.
    """
    return np.maximum(0, Z)

def relu_derivative(Z):
    """
    Derivada de ReLU. 
    Es 1 para valores positivos y 0 para negativos.
    """
    return Z > 0

def softmax(Z):
    """
    Convierte las salidas de la última capa en probabilidades.
    Z es una matriz de (10, m).
    """
    # Restamos el máximo de cada columna por estabilidad numérica 
    # (evita que np.exp explote con números grandes)
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)