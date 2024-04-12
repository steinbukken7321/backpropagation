# bibliotecas
import math

# definição dos pesos e viés para a camada oculta e de saída
hidden_weights = [0.15, 0.20, 0.25, 0.30] # pesos w1 w2 w3 w4
output_weights = [0.40, 0.45, 0.50, 0.55] # pesos w5 w6 w7 w8
b1 = 0.35  # bias 1
b2 = 0.60  # bias 2
i1 = 0.05  # entrada 1
i2 = 0.10  # entrada 2

# -------------------------------------------------------------------------------
# saídas da camada oculta (hidden layer output)
neth1 = hidden_weights[0] * i1 + hidden_weights[1] * i2 + b1
outh1 = 1 / (1 + math.exp(-neth1))

neth2 = hidden_weights[2] * i1 + hidden_weights[3] * i2 + b1
outh2 = 1 / (1 + math.exp(-neth2))

# saídas da camada de saída (output layer output)
neto1 = output_weights[0] * outh1 + output_weights[1] * outh2 + b2
outo1 = 1 / (1 + math.exp(-neto1))

neto2 = output_weights[2] * outh1 + output_weights[3] * outh2 + b2
outo2 = 1 / (1 + math.exp(-neto2))
# -------------------------------------------------------------------------------

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

# ------------------------  para calcular erro total (total error)---------------------

# definindo valores de alvo (targets)
target_o1 = 0.01   # alvo
target_o2 = 0.99   # alvo prioritário

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

# --------------------------------- Derivadas parciais--------------------------------
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

# definindo taxa de aprendizado (eta)
eta = 0.5

# novos pesos calculados usando as equações fornecidas
new_weight_w5 = output_weights[0] - eta * partial_E_total_w5
new_weight_w6 = output_weights[1] - eta * partial_E_total_w6
new_weight_w7 = output_weights[2] - eta * partial_E_total_w7
new_weight_w8 = output_weights[3] - eta * partial_E_total_w8

# imprimindo novos pesos após os cálculos
""""
print("Novo peso w5:", new_weight_w5)
print("Novo peso w6:", new_weight_w6)
print("Novo peso w7:", new_weight_w7)
print("Novo peso w8:", new_weight_w8)
"""
# derivada parcial de E_total em relação a net_o1 e net_o1
partial_E_o1_net_o1 = (outo1 - target_o1) * outo1 * (1 - outo1)
partial_E_o2_net_o2 = (outo2 - target_o2) * outo2 * (1 - outo2)

# derivada parcial de E_total em relação a out_h1 e out_h2
partial_E_total_out_h1 = partial_E_o1_net_o1 * output_weights[0] + partial_E_o2_net_o2 * output_weights[2]
partial_E_total_out_h2 = partial_E_o1_net_o1 * output_weights[1] + partial_E_o2_net_o2 * output_weights[3]

# derivada parcial de out_h1 em relação a net_h1 e net_h2
partial_out_h1_net_h1 = outh1 * (1 - outh1)
partial_out_h2_net_h2 = outh2 * (1 - outh2)

# derivada parcial de E_total em relação a w1 w2 w3 w4
partial_E_total_w1 = partial_E_total_out_h1 * partial_out_h1_net_h1 * i1
partial_E_total_w2 = partial_E_total_out_h2 * partial_out_h2_net_h2 * i1
partial_E_total_w3 = partial_E_total_out_h2 * partial_out_h2_net_h2 * i2
partial_E_total_w4 = partial_E_total_out_h2 * partial_out_h2_net_h2 * i2

# atualizando o peso w1 w2 w3 w4
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