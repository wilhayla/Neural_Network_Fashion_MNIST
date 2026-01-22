'''
It imports functions from the other files and contains the main loop.
'''
import numpy as np
from src.data_loader import load_mnist, preprocess_data
from src.model import NeuralNetwork
from src.activation import categorical_cross_entropy


def verify_data_loading(path):
    '''
    Funcion para validar que los datos se carguen, aplanan y normalizan correctamente
    '''
    print("--- iniciando verificacion de datos ---")

    try:
        # Carga
        images, labels = load_mnist(path, kind='train')
        # Preprocesamiento
        x_train, y_train = preprocess_data(images, labels)

        # Check de Dimensiones
        assert x_train.shape == (
            60000, 784), f"Error en shape de imágenes: {x_train.shape}"
        assert y_train.shape == (
            60000, 10), f"Error en shape de etiquetas: {y_train.shape}"

        # Check de Rango (Normalización)
        assert np.max(
            x_train) <= 1.0, "Los datos no están normalizados (máx > 1)"
        assert np.min(
            x_train) >= 0.0, "Los datos tienen valores negativos inesperados"

        # 3. Pruebas de verificación
        # Debería ser (60000, 784)
        print(f"Dimensiones de imágenes: {x_train.shape}")
        # Debería ser (60000, 10)
        print(f"Dimensiones de etiquetas: {y_train.shape}")
        print(f"Valor máximo en píxeles: {np.max(x_train)}")  # Debería ser 1.0
        # Debería ser un array con un solo '1'
        print(f"Ejemplo de etiqueta (One-Hot): {y_train[0]}")

        print("\n✅ ¡Datos cargados y preparados con éxito!")
        return x_train, y_train

    except AssertionError as e:
        print(f"❌ Falló la validación: {e}")
    except Exception as e:
        print(f"❌ Error crítico: {e}")

    return None, None

def get_accuracy(predictions, labels):
    """
    Compara el índice del valor máximo (la predicción) 
    con el índice del 1 en el One-Hot (la realidad).
    """
    pred_class = np.argmax(predictions, axis=0)
    true_class = np.argmax(labels, axis=0)
    return np.mean(pred_class == true_class)

def create_batches(X, Y, batch_size):
    """
    Divide los datos en grupos pequeños.
    X: (784, 60000), Y: (10, 60000)
    """
    m = X.shape[1]
    batches = []
    
    # Desordenar los datos para que la red no aprenda patrones por el orden
    permutation = np.random.permutation(m)
    X_shuffled = X[:, permutation]
    Y_shuffled = Y[:, permutation]
    
    # Crear los trozos
    for i in range(0, m, batch_size):
        X_batch = X_shuffled[:, i : i + batch_size]
        Y_batch = Y_shuffled[:, i : i + batch_size]
        batches.append((X_batch, Y_batch))
        
    return batches

def train():
    # --- 1. Carga y Preparación ---
    print("Cargando datos de Fashion MNIST...")
    raw_images, raw_labels = load_mnist('data', kind='train')
    X, Y = preprocess_data(raw_images, raw_labels)
    
    # Transponemos para que cada columna sea un ejemplo (784, 60000)
    X_input = X.T 
    Y_input = Y.T
    
    # --- 2. Inicialización del Modelo ---
    # Usamos 128 neuronas ocultas como planeamos
    nn = NeuralNetwork()
    
    # --- 3. Parámetros de Entrenamiento ---
    epochs = 100
    learning_rate = 0.1
    
    print(f"\nIniciando entrenamiento ({epochs} épocas)...")
    print("-" * 30)
    
    for i in range(epochs):
        # 1. Forward Pass
        A2 = nn.forward(X_input)
        
        # 2. Calcular el error (Loss)
        cost = categorical_cross_entropy(Y_input, A2)
        
        # 3. Backpropagation
        dW1, db1, dW2, db2 = nn.backward(X_input, Y_input)
        
        # 4. Actualización de parámetros
        nn.update_parameters(dW1, db1, dW2, db2, learning_rate)
        
        # 5. Monitoreo cada 10 épocas
        if i % 10 == 0:
            accuracy = get_accuracy(A2, Y_input)
            print(f"Epoch {i:3} | Costo: {cost:.4f} | Precisión: {accuracy:.2%}")

    print("-" * 30)
    final_acc = get_accuracy(nn.forward(X_input), Y_input)
    print(f"Entrenamiento finalizado. Precisión final: {final_acc:.2%}")

# --- Ejecución ---
if __name__ == "__main__":
    train()

    
