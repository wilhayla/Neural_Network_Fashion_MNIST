# Neural_Network_Fashion_MNIST
A modular Deep Learning implementation from scratch to classify the Fashion MNIST dataset using Python and NumPy.

This project implements a deep neural network to classify clothing items using the Fashion MNIST dataset.

## 📂 Project Structure

- `data/`: Contains the binary dataset files (.ubyte).
- `src/`: Folder containing the modular source code.
  - `data_loader.py`: Logic for processing binary files.
  - `activation.py`: Implementation of ReLU and Softmax activation functions.
  - `model.py`: Network architecture (Forward/Backward propagation and Dropout).
- `main.py`: Main script to initiate the training process.