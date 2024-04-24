import math
import numpy as np

# Função de Ativação Sigmóide
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# Classe Network
class Network(object):
    def __init__(self, sizes):
        """Inicialização da rede neural com pesos e vieses aleatórios."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Retorna a saída da rede se `a` for input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def predict(self, input_data):
        """Faz uma previsão com base nos dados de entrada."""
        return self.feedforward(np.array(input_data).reshape(self.sizes[0], 1))

    def load_weights(self, hidden_weights, output_weights, b1, b2):
        """Carrega os pesos e biases para a rede neural."""
        self.weights = [np.array(hidden_weights).reshape(self.sizes[1], self.sizes[0]),
                        np.array(output_weights).reshape(self.sizes[2], self.sizes[1])]
        self.biases = [np.array(b1).reshape(self.sizes[1], 1), np.array(b2).reshape(self.sizes[2], 1)]

# Função para ler um número binário de um arquivo de texto e converter para uma lista de inteiros
def read_binary_number(file_path):
    with open(file_path, 'r') as file:
        binary_number = file.readline().strip()
    return [int(bit) for bit in binary_number]

# Função para deduzir o número representado pelo número binário usando a rede neural treinada
def infer_number(network, binary_number):
    prediction = network.predict(binary_number)
    inferred_number = np.argmax(prediction)  # Assume que a saída com maior valor representa o número inferido
    return inferred_number

# Ler o número binário de um arquivo de texto
binary_number = read_binary_number("binary_number.txt")

# Tamanho da rede neural: N = número de bits do número binário, 4 camadas ocultas, 2 saídas
network_sizes = [len(binary_number), 4, 2]

# Criar uma instância da rede neural
network = Network(network_sizes)

# Carregar pesos e viés da rede neural
hidden_weights = np.random.randn(network_sizes[1], network_sizes[0])
output_weights = np.random.randn(network_sizes[2], network_sizes[1])
b1 = np.random.randn(network_sizes[1], 1)
b2 = np.random.randn(network_sizes[2], 1)

# Carregar os pesos e viés para a rede neural
network.load_weights(hidden_weights, output_weights, b1, b2)

# Deduzir o número representado pelo número binário usando a rede neural treinada
inferred_number = infer_number(network, binary_number)

print("Número binário:", binary_number)
print("Número inferido:", inferred_number)
