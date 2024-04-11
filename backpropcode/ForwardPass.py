import math

# Definição dos pesos e viés para a camada oculta e de saída
hidden_weights = [0.15, 0.20, 0.25, 0.30]
output_weights_o1 = [0.40, 0.45]
output_weights_o2 = [0.50, 0.55]
b1 = 0.35
b2 = 0.60
i1 = 0.05
i2 = 0.10

# saídas da camada oculta (hidden layer output)
neth1 = hidden_weights[0] * i1 + hidden_weights[1] * i2 + b1
outh1 = 1 / (1 + math.exp(-neth1))

neth2 = hidden_weights[2] * i1 + hidden_weights[3] * i2 + b1
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

# Derivadas parciais
# Calculando a derivada parcial de E_total em relação a w5 w6 w7 w8
partial_E_total_w5 = (outo1 - target_o1) * outo1 * (1 - outo1) * outh1
partial_E_total_w6 = (outo1 - target_o1) * outo1 * (1 - outo1) * outh2
partial_E_total_w7 = (outo2 - target_o2) * outo2 * (1 - outo2) * outh1
partial_E_total_w8 = (outo2 - target_o2) * outo2 * (1 - outo2) * outh2

# Resultado da derivada parcial w5 w6 w7 w8
"""
print("Derivada parcial de E_total em relação a w5:", partial_E_total_w5)
print("Derivada parcial de E_total em relação a w6:", partial_E_total_w6)
print("Derivada parcial de E_total em relação a w6:", partial_E_total_w7)
print("Derivada parcial de E_total em relação a w6:", partial_E_total_w8)
"""

# Taxa de aprendizado (eta)
eta = 0.5

output_weights_o1 = [0.40, 0.45]
output_weights_o2 = [0.50, 0.55]

# Novos pesos calculados usando as equações fornecidas
new_weight_w5 = output_weights_o1[0] - eta * partial_E_total_w5
new_weight_w6 = output_weights_o1[1] - eta * partial_E_total_w6
new_weight_w7 = output_weights_o2[0] - eta * partial_E_total_w7
new_weight_w8 = output_weights_o2[1] - eta * partial_E_total_w8

# Imprimindo os novos pesos
""""
print("Novo peso w5:", new_weight_w5)
print("Novo peso w6:", new_weight_w6)
print("Novo peso w7:", new_weight_w7)
print("Novo peso w8:", new_weight_w8)
"""
# Calculando a derivada parcial de E_total em relação a net_o1 e net_o1
partial_E_o1_net_o1 = (outo1 - target_o1) * outo1 * (1 - outo1)
partial_E_o2_net_o2 = (outo2 - target_o2) * outo2 * (1 - outo2)

# Calculando a derivada parcial de E_total em relação a out_h1 e out_h2
partial_E_total_out_h1 = partial_E_o1_net_o1 * output_weights_o1[0] + partial_E_o2_net_o2 * output_weights_o2[0]
partial_E_total_out_h2 = partial_E_o1_net_o1 * output_weights_o1[1] + partial_E_o2_net_o2 * output_weights_o2[1]

# Calculando a derivada parcial de out_h1 em relação a net_h1 e net_h2
partial_out_h1_net_h1 = outh1 * (1 - outh1)
partial_out_h2_net_h2 = outh2 * (1 - outh2)

# Calculando a derivada parcial de E_total em relação a w1 w2 w3 w4
partial_E_total_w1 = partial_E_total_out_h1 * partial_out_h1_net_h1 * i1
partial_E_total_w2 = partial_E_total_out_h2 * partial_out_h2_net_h2 * i1
partial_E_total_w3 = partial_E_total_out_h2 * partial_out_h2_net_h2 * i2
partial_E_total_w4 = partial_E_total_out_h2 * partial_out_h2_net_h2 * i2

# Atualizando o peso w1 w2 w3 w4
new_weight_w1 = hidden_weights[0] - eta * partial_E_total_w1
new_weight_w2 = hidden_weights[1] - eta * partial_E_total_w2
new_weight_w3 = hidden_weights[2] - eta * partial_E_total_w3
new_weight_w4 = hidden_weights[3] - eta * partial_E_total_w4

"""
print("Novo peso w1:", new_weight_w1)
print("Novo peso w2:", new_weight_w2)
print("Novo peso w3:", new_weight_w3)
print("Novo peso w4:", new_weight_w4)
"""