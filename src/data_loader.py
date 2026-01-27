''' The purpose of this file is to deliver the data I will be using'''

import os
import numpy as np


def load_mnist(path, kind='train'):
    """Carga los datos de Fashion-MNIST desde el path especificado."""

    # Carga las datos de etiquetas e imagenes de entrenamiento
    labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte')
    images_path = os.path.join(path, f'{kind}-images-idx3-ubyte')

    # Realizar la lectura fisica de los archivos binarios
    # Transformar "bytes" puros en arreglos numericos de MumPy para que la red neuronal pueda procesar.
    with open(labels_path, 'rb') as lbpath: # se encarga de abrir y cerrar automaticamente el archivo, con 'rb' asegurar que lean los datos binarios (read binary) y guardar en un alias "lbpath"
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

        # np.frombuffer: funcion que interpreta los datos directamente desde la memoria (buffer), sin necesidad de copiar los datos varias veces.
        # lbpath.read: lee todo el contenido del archivo de etiquetas.
        # dtype=np.uint8 : indica que cada etiqueta es un entero sin signo de 8 bits (valores de 0 a 255)
        # offset=8: ignora los primeros 8 bytes de informacion tecnica para empezar a leer los datos reales a partir de alli.

    # Trnasformar lo bytes crudos del archivo de imagenes en una matriz numerica que la red neuranal pueda entender
    # Proceso de conversion de binarios a pixeles
    with open(images_path, 'rb') as imgpath:
        
        images = np.frombuffer(
            imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)
        
        # reshape(len(labels), 784): convierte en una tabla matriz de len(labels)-60.000 filas y 784 columnas,
                                    # donde cada fila es una prenda, y cada columna representa la intensidad de brillo de un pixel espesifico (de 0 a 255)

    return images, labels


def preprocess_data(images, labels):
    '''1. Normalización las imagenes y aplica one-hot-encoding a las etiquetas.'''
    # Esclado de valores de 8 bits
    images = images.astype('float32') / 255.0  # divide cada bits de la imagen entre 255 para convertirlo en un numero de tipo decimal que varia entre 0 y 1

    # 2. One-hot encoding para las etiquetas
    # Esto crea una matriz de 10 columnas (etiquetas) y 60 mil imagenes(filas)
    # La matriz genera vectores con todo 0 escepto la que coincide con la etique, la cual sera 1.
    n_classes = 10
    oh_labels = np.eye(n_classes)[labels]

    # cuando ser realize el entrenamiento la capa de salida compara sus predicciones con esta matriz y verifica cual es la que mas se acerca al 1 haciendo una primera prediccion,
    # teniendo en cuesta la posicion en que esta el numero mas alto del vector de prediccion.

    return images, oh_labels
