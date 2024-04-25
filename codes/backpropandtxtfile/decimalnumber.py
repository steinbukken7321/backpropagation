import numpy as np


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Inicialização dos pesos e viés
        self.weights_input_hidden = np.random.rand(
            self.input_size, self.hidden_size)
        self.bias_input_hidden = np.random.rand(1, self.hidden_size)
        self.weights_hidden_output = np.random.rand(
            self.hidden_size, self.output_size)
        self.bias_hidden_output = np.random.rand(1, self.output_size)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def feedforward(self, inputs):
        # Camada de entrada para camada oculta
        self.hidden_sum = np.dot(
            inputs, self.weights_input_hidden) + self.bias_input_hidden
        self.hidden_output = self.relu(self.hidden_sum)

        # Camada oculta para camada de saída
        self.output_sum = np.dot(
            self.hidden_output, self.weights_hidden_output) + self.bias_hidden_output
        self.output = self.relu(self.output_sum)

        return self.output

    def backward(self, inputs, targets, learning_rate):
        # Cálculo do erro
        self.error = targets - self.output

        # Cálculo dos gradientes para a camada de saída
        delta_output = self.error * self.relu_derivative(self.output)

        # Atualização dos pesos e viés da camada de saída
        self.weights_hidden_output += np.dot(
            self.hidden_output.T, delta_output) * learning_rate
        self.bias_hidden_output += np.sum(delta_output,
                                          axis=0, keepdims=True) * learning_rate

        # Cálculo dos gradientes para a camada oculta
        delta_hidden = np.dot(delta_output, self.weights_hidden_output.T) * \
            self.relu_derivative(self.hidden_output)

        # Atualização dos pesos e viés da camada oculta
        self.weights_input_hidden += np.dot(inputs.T,
                                            delta_hidden) * learning_rate
        self.bias_input_hidden += np.sum(delta_hidden,
                                         axis=0, keepdims=True) * learning_rate

    def train(self, inputs, targets, learning_rate, epochs):
        for epoch in range(epochs):
            output = self.feedforward(inputs)
            self.backward(inputs, targets, learning_rate)
            loss = np.mean(np.square(targets - output))
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Total Error: {loss}')

# Função para converter um número binário para um número decimal


def binary_to_decimal(binary_number):
    decimal_number = 0
    for i, bit in enumerate(reversed(binary_number)):
        decimal_number += int(bit) * (2 ** i)
    return decimal_number

# Função para converter um número decimal para sua representação binária


def decimal_to_binary(decimal_number, num_bits):
    binary_string = bin(decimal_number)[2:]
    binary_string = '0' * (num_bits - len(binary_string)) + binary_string
    return binary_string


# Exemplo de uso
input_size = 4
hidden_size = 10
output_size = 1

# Dados de entrada e saída
# Exemplo de números binários de comprimentos variáveis
binary_numbers = ['0101', '0011', '1010', '1111', '0000', '0001']
max_length = max(len(num) for num in binary_numbers)
inputs = np.array([[int(bit) for bit in num.zfill(max_length)]
                  for num in binary_numbers])
targets = np.array([[binary_to_decimal(num)] for num in binary_numbers])

# Criar e treinar a rede neural
nn = NeuralNetwork(input_size, hidden_size, output_size)
nn.train(inputs, targets, learning_rate=0.01, epochs=10000)

# Avaliação da rede neural
for i, binary_number in enumerate(binary_numbers):
    input_binary = np.array([int(bit) for bit in binary_number]).reshape(1, -1)
    output_decimal = nn.feedforward(input_binary)
    print(f'Entrada binária: {binary_number}, Saída decimal: {
          output_decimal[0, 0]}')
