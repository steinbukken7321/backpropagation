import math

# Definição dos pesos e viés para a camada oculta e de saída
hidden_weights = [0.15, 0.20, 0.25, 0.30]
output_weights_o1 = [0.40, 0.45]
output_weights_o2 = [0.50, 0.55]
b1 = 0.35
b2 = 0.60

# saídas da camada oculta (hidden layer output)
neth1 = hidden_weights[0] * 0.05 + hidden_weights[1] * 0.10 + b1
outh1 = 1 / (1 + math.exp(-neth1))

neth2 = hidden_weights[2] * 0.05 + hidden_weights[3] * 0.10 + b1
outh2 = 1 / (1 + math.exp(-neth2))

# saídas da camada de saída (output layer output)
neto1 = output_weights_o1[0] * outh1 + output_weights_o1[1] * outh2 + b2
outo1 = 1 / (1 + math.exp(-neto1))

neto2 = output_weights_o2[0] * outh1 + output_weights_o2[1] * outh2 + b2
outo2 = 1 / (1 + math.exp(-neto2))

# resultados

"""
print("neth1:", neth1)
print("neth2:", neth2)

print("outh1:", outh1)
print("outh2:", outh2)

print("neto1:", neto1)
print("neto2:", neto2)

print("outo1:", outo1)
print("outo2:", outo2)
"""

# para calcular erro total (total error)

# definindo valores de alvo (targets)
target_o1 = 0.01
target_o2 = 0.99

# erros individuais para cada neurônio de saída
eo1 = 1/2 * (target_o1 - outo1) ** 2
eo2 = 1/2 * (target_o2 - outo2) ** 2

# erro total
etotal = eo1 + eo2

# resultados dos erros

"""
print("Erro para o neurônio o1:", eo1)
print("Erro para o neurônio o2:", eo2)
print("Erro total da rede neural:", etotal)
"""


