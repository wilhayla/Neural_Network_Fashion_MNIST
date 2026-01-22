''' The purpose of this file is to deliver the data I will be using'''

import os
import numpy as np


def load_mnist(path, kind='train'):
    """Carga los datos de Fashion-MNIST desde el path especificado."""

    labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte')
    images_path = os.path.join(path, f'{kind}-images-idx3-ubyte')

    with open(labels_path, 'rb') as lbpath:
        # Leemos la cabecera (8 bytes) y luego el resto como etiquetas
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with open(images_path, 'rb') as imgpath:
        # Leemos la cabecera (16 bytes) y luego las imágenes
        # Reshapeamos a (número_imágenes, 784) para "aplanarlas"
        images = np.frombuffer(
            imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    return images, labels


def preprocess_data(images, labels):
    '''1. Normalización las imagenes y aplica one-hot-encoding a las etiquetas.'''

    images = images.astype('float32') / 255.0

    # 2. One-hot encoding para las etiquetas
    # Esto crea una matriz de (n_muestras, 10)
    n_classes = 10
    oh_labels = np.eye(n_classes)[labels]

    return images, oh_labels
