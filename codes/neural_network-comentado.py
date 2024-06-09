# -*- coding: utf-8 -*-
"""Neural_Network.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1hqYPWSvJLrAzcUGpdqoAWl9VsOdB1WBh
"""

import numpy as np
import matplotlib.pyplot as plt

def read_matrices_from_file(filename):
    """
    Função para ler as matrizes de um arquivo de texto.

    Args:
    filename (str): O caminho do arquivo de texto contendo as matrizes.

    Returns:
    list: Uma lista de matrizes lidas do arquivo.
    """
    with open(filename, 'r') as file:
        lines = file.readlines()
    matrices = [list(map(int, line.strip().split(','))) for line in lines]
    return matrices

def sigmoid(x):
    """
    Função de ativação sigmoid.

    Args:
    x (numpy.ndarray): O vetor de entrada.

    Returns:
    numpy.ndarray: O vetor resultante após a aplicação da função sigmoid.
    """
    return 1 / (1 + np.exp(-x))

def f_forward(x, w1, w2):
    """
    Função de feedforward.

    Args:
    x (numpy.ndarray): A entrada da rede.
    w1 (numpy.ndarray): Os pesos da camada de entrada para a camada oculta.
    w2 (numpy.ndarray): Os pesos da camada oculta para a camada de saída.

    Returns:
    numpy.ndarray: A saída da rede após o feedforward.
    """
    z1 = x.dot(w1)
    a1 = sigmoid(z1)
    z2 = a1.dot(w2)
    a2 = sigmoid(z2)
    return a2

def generate_wt(x, y):
    """
    Inicializa os pesos de uma camada da rede neural.

    Args:
    x (int): O número de neurônios na camada de entrada.
    y (int): O número de neurônios na camada de saída.

    Returns:
    numpy.ndarray: Os pesos inicializados.
    """
    return np.random.randn(x, y)

def loss(out, Y):
    """
    Calcula a função de perda (MSE).

    Args:
    out (numpy.ndarray): A saída prevista pela rede neural.
    Y (numpy.ndarray): O rótulo verdadeiro.

    Returns:
    float: O valor da função de perda.
    """
    s = np.square(out - Y)
    return np.sum(s) / len(Y)

def back_prop(x, y, w1, w2, alpha):
    """
    Realiza o backpropagation na rede neural para atualizar os pesos.

    Args:
    x (numpy.ndarray): A entrada da rede.
    y (numpy.ndarray): O rótulo verdadeiro.
    w1 (numpy.ndarray): Os pesos da camada de entrada para a camada oculta.
    w2 (numpy.ndarray): Os pesos da camada oculta para a camada de saída.
    alpha (float): A taxa de aprendizado.

    Returns:
    tuple: Os pesos atualizados w1 e w2.
    """
    z1 = x.dot(w1)
    a1 = sigmoid(z1)
    z2 = a1.dot(w2)
    a2 = sigmoid(z2)
    d2 = a2 - y
    d1 = np.multiply(w2.dot(d2.T).T, np.multiply(a1, 1 - a1))
    w1_adj = x.T.dot(d1)
    w2_adj = a1.T.dot(d2)
    w1 -= alpha * w1_adj
    w2 -= alpha * w2_adj
    return w1, w2

def train(x, Y, w1, w2, alpha=0.01, epoch=1000):
    """
    Treina a rede neural.

    Args:
    x (list): A lista de entradas da rede.
    Y (numpy.ndarray): Os rótulos verdadeiros.
    w1 (numpy.ndarray): Os pesos da camada de entrada para a camada oculta.
    w2 (numpy.ndarray): Os pesos da camada oculta para a camada de saída.
    alpha (float): A taxa de aprendizado (default: 0.01).
    epoch (int): O número de épocas de treinamento (default: 1000).

    Returns:
    tuple: As listas de acurácia e perda ao longo do treinamento, e os pesos atualizados w1 e w2.
    """
    acc = []
    losss = []
    for j in range(epoch):
        l = []
        for i in range(len(x)):
            out = f_forward(x[i], w1, w2)
            l.append(loss(out, Y[i]))
            w1, w2 = back_prop(x[i], Y[i], w1, w2, alpha)
        acc.append((1 - (sum(l) / len(x))) * 100)
        losss.append(sum(l) / len(x))
    return acc, losss, w1, w2

def predict(x, w1, w2):
    """
    Realiza a predição para uma entrada dada.

    Args:
    x (numpy.ndarray): A entrada da rede.
    w1 (numpy.ndarray): Os pesos da camada de entrada para a camada oculta.
    w2 (numpy.ndarray): Os pesos da camada oculta para a camada de saída.
    """
    Out = f_forward(x, w1, w2)
    k = np.argmax(Out)
    print(f"Image is of number {k}.")
    plt.imshow(x.reshape(5, 6))
    plt.show()

# Carregar as matrizes do arquivo
filename = '/content/arquivos_binarios.txt'
matrices = read_matrices_from_file(filename)

# Criando as entradas e rótulos
x = [np.array(matrix).reshape(1, 30) for matrix in matrices]
y = np.eye(10)

# Inicializando os pesos
w1 = generate_wt(30, 5)
w2 = generate_wt(5, 10)

# Treinando a rede
acc, losss, w1, w2 = train(x, y, w1, w2, alpha=0.1, epoch=1000)

"""
# Plotando a acurácia
plt.plot(acc)
plt.ylabel('Accuracy')
plt.xlabel("Epochs")
plt.show()

# Plotando a perda
plt.plot(losss)
plt.ylabel('Loss')
plt.xlabel("Epochs")
plt.show()
"""
# Testando a rede
predict(x[9], w1, w2)  # Predizendo para o número 2