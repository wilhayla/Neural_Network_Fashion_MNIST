'''
It imports functions from the other files and contains the main loop.
'''
import numpy as np
from src.data_loader import load_mnist, preprocess_data
from src.model import NeuralNetwork


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

def run_test():
    # --- PASO 1: Cargar los parámetros (Datos y Etiquetas) ---
    # Asegúrate de que la carpeta 'data' contenga los archivos .ubyte
    print("Cargando datos...")
    raw_images, raw_labels = load_mnist('data', kind='train')
    
    # Preprocesamos: normalizamos píxeles y hacemos One-Hot a las etiquetas
    X, Y = preprocess_data(raw_images, raw_labels)
    
    # --- PASO 2: Preparar la entrada para la Red ---
    # Importante: Transponemos X para que sea (784, 60000)
    X_input = X.T 
    Y_input = Y.T # También transponemos las etiquetas para el futuro
    
    # --- PASO 3: Instanciar la Red ---
    # 784 entradas, 128 neuronas ocultas, 10 clases de salida
    nn = NeuralNetwork()
    
    # --- PASO 4: Probar el Forward Pass ---
    print("Ejecutando Forward Pass...")
    predicciones = nn.forward(X_input)
    
    # --- PASO 5: Verificaciones ---
    print(f"\nForma de X tras transponer: {X_input.shape}") # (784, 60000)
    print(f"Forma de las predicciones: {predicciones.shape}") # (10, 60000)
    
    # Verificamos si Softmax funcionó: la suma de la primera columna debe ser 1
    suma_ejemplo = np.sum(predicciones[:, 0])
    print(f"Suma de prob. del primer ejemplo: {suma_ejemplo:.2f}")

# --- Ejecución ---
if __name__ == "__main__":
    run_test()

    
