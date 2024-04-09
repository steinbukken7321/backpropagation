import math

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def calculate_output(self, inputs):
        total_net_input = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        return 1 / (1 + math.exp(-total_net_input))


# Definição dos pesos e viés para a camada oculta e de saída
hidden_weights = [0.15, 0.20, 0.25, 0.30]
output_weights_o1 = [0.40, 0.45]
output_weights_o2 = [0.50, 0.55]
hidden_bias = 0.35
output_bias = 0.60

# criando neurônios para a camada oculta e de saída
hidden_neuron1 = Neuron(hidden_weights[:2], hidden_bias)
hidden_neuron2 = Neuron(hidden_weights[2:], hidden_bias)
output_neuron1 = Neuron(output_weights_o1, output_bias)
output_neuron2 = Neuron(output_weights_o2, output_bias)

# calculando as saídas
h1 = hidden_neuron1.calculate_output([0.05, 0.10])
h2 = hidden_neuron2.calculate_output([0.05, 0.10])
o1 = output_neuron1.calculate_output([h1, h2])
o2 = output_neuron2.calculate_output([h1, h2])

# saídas o1 e o2
print("Saídas da rede neural:")
print("O1:", o1)
print("O2:", o2)
