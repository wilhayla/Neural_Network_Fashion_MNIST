'''
It imports functions from the other files and contains the main loop.
'''
import numpy as np
from src.data_loader import load_mnist, preprocess_data
from src.model import NeuralNetwork
from src.activation import categorical_cross_entropy

# --- 1. DEFINICIÓN DEL DICCIONARIO (Global) ---
label_names = {
    0: "Camiseta/Top",
    1: "Pantalón",
    2: "Pulóver",
    3: "Vestido",
    4: "Abrigo",
    5: "Sandalia",
    6: "Camisa",
    7: "Zapatilla",
    8: "Bolso",
    9: "Bota de tobillo"
}

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
    epochs = 60 # Con mini-batches, 50 épocas suelen ser suficientes
    learning_rate = 0.05
    batch_size = 64 # probar con trozos de 64 imagenes
    
    print(f"Entrenando con Mini-batches de {batch_size} ejemplos...")
    
    for epoch in range(epochs):
        # Creamos los batches desordenados para esta época
        batches = create_batches(X_input, Y_input, batch_size)
        
        for X_batch, Y_batch in batches:
            # A. Forward Pass
            nn.forward(X_batch)
            
            # B. Backpropagation (Gradientes)
            dW1, db1, dW2, db2 = nn.backward(X_batch, Y_batch)
            
            # C. Actualización de Pesos
            nn.update_parameters(dW1, db1, dW2, db2, learning_rate)
            
        # Monitoreo de progreso cada 5 épocas
        if epoch % 5 == 0:
            full_pred = nn.forward(X_input)
            loss = categorical_cross_entropy(Y_input, full_pred)
            acc = get_accuracy(full_pred, Y_input)
            print(f"Época {epoch:2} | Costo: {loss:.4f} | Precisión: {acc:.2%}")

    print("-" * 30)
    final_acc = get_accuracy(nn.forward(X_input), Y_input)
    print(f"¡Entrenamiento Terminado! Precisión Final: {final_acc:.2%}")
    return nn

def final_evaluation(nn):
    print("\n--- EVALUACIÓN FINAL CON DATOS DE TEST ---")
    
    # 1. Cargar datos de test (que la red NUNCA ha visto)
    test_images, test_labels = load_mnist('data', kind='t10k')
    X_test, Y_test = preprocess_data(test_images, test_labels)
    
    # 2. Preparar entrada (Transponer)
    X_test_input = X_test.T
    Y_test_input = Y_test.T
    
    # 3. Realizar predicción
    test_predictions = nn.forward(X_test_input)
    
    # 4. Calcular métricas
    test_loss = categorical_cross_entropy(Y_test_input, test_predictions)
    test_acc = get_accuracy(test_predictions, Y_test_input)
    
    print(f"Costo en Test: {test_loss:.4f}")
    print(f"Precisión en Test: {test_acc:.2%}")
    
    if test_acc >= 0.95:
        print("¡INCREÍBLE! Has superado el reto con creces. 🚀")
    elif test_acc >= 0.85:
        print("¡Excelente trabajo! Has superado el mínimo requerido. ✅")

def predict_random_image(nn):
    '''
    Funcion que elegira una imagen al azar del set de etiquetas, se le pasara a la red y dira que es
    '''
    # 1. Cargamos datos de test
    test_images, test_labels = load_mnist('data', kind='t10k')
    
    # 2. Elegimos un índice al azar
    idx = np.random.randint(0, test_images.shape[0])
    img = test_images[idx].reshape(1, 784) / 255.0 # Normalizamos
    
    # 3. La red hace su magia (predicción)
    prediction = nn.forward(img.T)
    predicted_class = np.argmax(prediction)
    actual_class = test_labels[idx]
    
    print("\n--- PREDICCIÓN INDIVIDUAL ---")
    print(f"La IA dice: {label_names[predicted_class]}")
    print(f"Realidad:   {label_names[actual_class]}")
    
    if predicted_class == actual_class:
        print("¡Resultado Correcto! ✅")
    else:
        print("La IA se confundió esta vez. ❌")

# --- Ejecución ---
if __name__ == "__main__":
    # 1. Ejecutar el entrenamiento y obtener el objeto nn entrenado
    # La función train ahora debe devolver el objeto nn

    # nn_entrenada = train()

    # nn_entrenada.save_model("modelo_fashion_94.npz")
    
    # 2. Llamar a la función de evaluación pasándole ese objeto
    # final_evaluation(nn_entrenada)

    '''una ves guardado el modelo cargarlo de este forma'''

    '''Creamos una red "vacía"'''
    nn = NeuralNetwork()
    
    ''' Cargamos el conocimiento guardado (tarda menos de 1 segundo)'''
    try:

        nn.load_model("modelo_fashion_94.npz")
    except FileNotFoundError:
        nn = train()
        nn.save_model("modelo_fashion_94.npz")
    
    ''' ¡Listo! Ya puedes evaluar o predecir sin entrenar de nuevo'''
    final_evaluation(nn)

    # Demostracion visual 
    predict_random_image(nn)

    
