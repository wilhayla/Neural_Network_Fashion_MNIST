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
        # Definicion del peso de cada neurona de entrada con las neuronas ocultas
        # Resultado: una matriz W1 de tamaño 128 x 784
        # Cada neurona de entrada tiene 128 pesos W.
        # np.sqrt(2.0 / input_size): inicializacion He o (Kaiming Init). Formula que se utiliza para escalar los valores de W1 y evitar que sean muy grandes o muy pequeñas.
                  # Tambien esta formula esta diseñada para trabajar perfectamente con la funcion ReLU. Ayuda a que la mitad de las neuronas se activen y la otra mitad no, 
                  # manteniendo el aprendizaje dinamico.
        self.W1 = np.random.randn( hidden_size, input_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((hidden_size, 1)) # se inicializa el sesgo(bias) con un vector de ceros, de 128 filas.

        # Definicion de tamaño de matriz entre matriz oculta con matriz de entrada.
        self.W2 = np.random.randn(output_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((output_size, 1))

    def show_parameters_info(self):
        """Muestra estadísticas clave de los pesos y sesgos."""
        params = [('W1', self.W1), ('b1', self.b1), ('W2', self.W2), ('b2', self.b2)]

        print("\n--- Diagnóstico de Parámetros ---")
        for name, p in params:
            print(f"{name:2} | Shape: {str(p.shape):10} | Media: {np.mean(p):.4f} | Desv. Est: {np.std(p):.4f}")

        print(f"\nPrimeros 5 valores de la primera fila de W1:\n{self.W1[0, :5]}")

    def forward(self, X):
        """
        X: matriz de entrada (784, m) donde m es el número de ejemplos.
        Retorna la predicción final (A2).
        """
        # --- CAPA 1 (Oculta) ---
        # Combinación lineal: Z1 = W1 * X + b1
        self.Z1 = np.dot(self.W1, X) + self.b1
        # Activación: A1 = ReLU(Z1)
        self.A1 = relu(self.Z1)
        
        # --- CAPA 2 (Salida) ---
        # Combinación lineal: Z2 = W2 * A1 + b2
        self.Z2 = np.dot(self.W2, self.A1) + self.b2
        # Activación: A2 = Softmax(Z2)
        self.A2 = softmax(self.Z2)
        
        return self.A2
    
    def backward(self, X, Y):
        # Obtener la cantidad de imagenes en total
        m = X.shape[1]
        
        # --- Cálculo de Gradientes para la Capa de Salida ---
        dZ2 = self.A2 - Y  # Calculo de la direccion y magnitud de error entre la prediccion A2 y el valor de Y que es 1
        dW2 = (1 / m) * np.dot(dZ2, self.A1.T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
        
        # --- Cálculo de Gradientes para la Capa Oculta ---
        # Usamos la derivada de ReLU que programamos antes

        from src.activation import relu_derivative

        dZ1 = np.dot(self.W2.T, dZ2) * relu_derivative(self.Z1)
        dW1 = (1 / m) * np.dot(dZ1, X.T)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
        
        return dW1, db1, dW2, db2
    
    def update_parameters(self, dW1, db1, dW2, db2, learning_rate):
        """
        Aplica las correcciones a los pesos y sesgos.
        """
        self.W1 = self.W1 - learning_rate * dW1
        self.b1 = self.b1 - learning_rate * db1
        self.W2 = self.W2 - learning_rate * dW2
        self.b2 = self.b2 - learning_rate * db2

    def save_model(self, filename="model_weights.npz"):
        """Guarda los parámetros en un archivo comprimido de NumPy."""
        np.savez(filename, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)
        print(f"Modelo guardado en {filename}")

    def load_model(self, filename="model_weights.npz"):
        """Carga los parámetros desde un archivo."""
        data = np.load(filename)
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']
        print(f"Modelo cargado desde {filename}")
