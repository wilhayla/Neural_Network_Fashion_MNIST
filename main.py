'''
It imports functions from the other files and contains the main loop.
'''
import numpy as np
from src.data_loader import load_mnist, preprocess_data


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


# --- Ejecución ---
if __name__ == "__main__":
    PATH_DATA = 'data'
    X, Y = verify_data_loading('data')
