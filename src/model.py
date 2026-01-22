'''This file defines function like:
- initialize_parameters(layers)
- forward_propagation(X, parameters, dropout_rate)
- backward_propagation(X, y, cache, parameters)
- update_parameters(parameters, gradients, learning_rate)
'''
import numpy as np
from src.activation import relu, softmax


class NeuralNetwork:

    '''
    Clase que representa una red neuronal simple para la clasificación de dígitos.
    Implementa la inicialización de pesos, propagación hacia adelante y entrenamiento.
    '''

    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        # Definicion de tamaño de matriz entre matriz de entrada con matriz oculta.
        self.w1 = np.random.randn( hidden_size, input_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((hidden_size, 1))

        # Definicion de tamaño de matriz entre matriz oculta con matriz de entrada.
        self.w2 = np.random.randn(output_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((output_size, 1))

    def show_parameters_info(self):
        """Muestra estadísticas clave de los pesos y sesgos."""
        params = [('w1', self.w1), ('b1', self.b1), 
                  ('w2', self.w2), ('b2', self.b2)]
         
        print("\n--- Diagnóstico de Parámetros ---")
        for name, p in params:
            print(f"{name:2} | Shape: {str(p.shape):10} | Media: {np.mean(p):.4f} | Desv. Est: {np.std(p):.4f}")

        print(f"\nPrimeros 5 valores de la primera fila de W1:\n{self.w1[0, :5]}")

    
