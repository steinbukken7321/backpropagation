import numpy as np

# Função de ativação sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivada da função de ativação sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Classe para a rede neural
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Inicialização dos pesos e viéses
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
        self.bias_hidden = np.random.rand(1, hidden_size)
        self.bias_output = np.random.rand(1, output_size)

    def feedforward(self, inputs):
        # Camada oculta
        hidden_sum = np.dot(
            inputs, self.weights_input_hidden) + self.bias_hidden
        hidden_output = sigmoid(hidden_sum)

        # Camada de saída
        output_sum = np.dot(
            hidden_output, self.weights_hidden_output) + self.bias_output
        final_output = sigmoid(output_sum)

        return final_output

    def train(self, inputs, targets, learning_rate, epochs):
        for epoch in range(epochs):
            # Feedforward
            hidden_sum = np.dot(
                inputs, self.weights_input_hidden) + self.bias_hidden
            hidden_output = sigmoid(hidden_sum)
            output_sum = np.dot(
                hidden_output, self.weights_hidden_output) + self.bias_output
            final_output = sigmoid(output_sum)

            # Backpropagation
            output_error = targets - final_output
            output_delta = output_error * sigmoid_derivative(final_output)
            # Transposta adicionada aqui
            hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
            hidden_delta = hidden_error * sigmoid_derivative(hidden_output)

            # Atualização dos pesos e viéses
            self.weights_hidden_output += np.dot(
                hidden_output.T, output_delta) * learning_rate
            self.weights_input_hidden += np.dot(inputs.T,
                                                hidden_delta) * learning_rate
            self.bias_output += np.sum(output_delta, axis=0) * learning_rate
            self.bias_hidden += np.sum(hidden_delta, axis=0) * learning_rate

            # Cálculo do erro
            error = np.mean(np.abs(output_error))
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Error: {error}")

# Função para ler números binários de um arquivo de texto
def read_binary_number(file_path):
    with open(file_path, 'r') as file:
        binary_number = file.read().strip()
    return binary_number

# Preparar os dados de entrada e saída


def prepare_data(binary_number):
    binary_array = [int(bit) for bit in binary_number]
    return np.array(binary_array).reshape(1, -1)


# Carregar os dados
binary_number = read_binary_number(
    "C:/Users/rafae/Desktop/backpropagation/codes/backpropandtxtfile/binary_number.txt")

# Preparar os dados
X = prepare_data(binary_number)

# Determinar o tamanho das entradas e saídas
input_size = X.shape[1]
output_size = 1

# Criar e treinar a rede neural
hidden_size = 4
learning_rate = 0.1
epochs = 10000

nn = NeuralNetwork(input_size, hidden_size, output_size)
# Aqui, estamos treinando a rede para prever as próprias entradas
nn.train(X, X, learning_rate, epochs)
