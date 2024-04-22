import random
import math

class neuronio:
    # Neurônio em uma rede neural
    def __init__(self, bias):
        # inicializa um neurônio com o viés (bias) e os pesos
        self.bias = bias
        self.pesos = []

    def calcular_saida(self, entradas):
        # calcula a saída do neurônio com base nas entradas fornecidas
        self.entradas = entradas
        self.saida = self.squash(self.calcular_total_net_input())
        return self.saida

    def calcular_total_net_input(self):
        # calcula a soma ponderada das entradas e pesos mais o viés
        total = 0
        for i in range(len(self.entradas)):
            total += self.entradas[i] * self.pesos[i]
        return total + self.bias

    def squash(self, total_net_input):
        # aplica a função de ativação (sigmóide) ao total da entrada
        return 1 / (1 + math.exp(-total_net_input))

    def calcular_pd_erro_wrt_total_net_input(self, target_output):
        # calcula a derivada parcial do erro em relação à soma ponderada das entradas
        return self.calcular_pd_erro_wrt_saida(target_output) * self.calcular_pd_total_net_input_wrt_entrada()

    def calcular_erro(self, target_output):
        # calcula o erro do neurônio com base na saída desejada
        return 0.5 * (target_output - self.saida) ** 2

    def calcular_pd_erro_wrt_saida(self, target_output):
        # calcula a derivada parcial do erro em relação à saída
        return -(target_output - self.saida)

    def calcular_pd_total_net_input_wrt_entrada(self):
        # calcula a derivada parcial da soma ponderada das entradas em relação à entrada
        return self.saida * (1 - self.saida)

    def calcular_pd_total_net_input_wrt_peso(self, index):
        # calcula a derivada parcial da soma ponderada das entradas em relação ao peso
        return self.entradas[index]

class Camada_Neuronios:
    # representa uma camada de neurônios em uma rede neural
    def __init__(self, num_neuronios, bias):
        # inicializa uma camada de neurônios com um número específico de neurônios e um viés (bias) comum
        self.bias = bias if bias else random.random()
        self.neuronios = [neuronio(self.bias) for _ in range(num_neuronios)]
    def feed_forward(self, entradas):
        # propaga as entradas através da camada e retorna as saídas
        return [neuronio.calcular_saida(entradas) for neuronio in self.neuronios]
    def obter_saidas(self):
        # retorna as saídas de todos os neurônios na camada
        return [neuronio.saida for neuronio in self.neuronios]

class RedeNeural:
    def __init__(self, num_inputs, num_ocultos, num_outputs, LR, pesos_camada_oculta=None,
                 bias_camada_oculta=None, pesos_camada_saida=None, bias_camada_saida=None):
        # inicializa a rede neural com o número de entradas, neurônios ocultos, neurônios de saída,
        # taxa de aprendizado e pesos opcionais para as camadas oculta e de saída
        self.num_inputs = num_inputs
        self.camada_oculta = Camada_Neuronios(num_ocultos, bias_camada_oculta)
        self.camada_saida = Camada_Neuronios(num_outputs, bias_camada_saida)
        self.LR = LR  # Taxa de aprendizado
        self.inicializar_pesos_entrada_oculta(pesos_camada_oculta)
        self.inicializar_pesos_oculta_saida(pesos_camada_saida)

    def inicializar_pesos_entrada_oculta(self, pesos_camada_oculta):
        # inicializa os pesos da camada oculta, se não forem fornecidos explicitamente
        num_peso = 0
        for h in range(len(self.camada_oculta.neuronios)):
            for i in range(self.num_inputs):
                if not pesos_camada_oculta:
                    self.camada_oculta.neuronios[h].pesos.append(
                        random.random())
                else:
                    self.camada_oculta.neuronios[h].pesos.append(
                        pesos_camada_oculta[num_peso])
                num_peso += 1

    def inicializar_pesos_oculta_saida(self, pesos_camada_saida):
        # inicializa os pesos da camada de saída, se não forem fornecidos explicitamente
        num_peso = 0
        for o in range(len(self.camada_saida.neuronios)):
            for h in range(len(self.camada_oculta.neuronios)):
                if not pesos_camada_saida:
                    self.camada_saida.neuronios[o].pesos.append(
                        random.random())
                else:
                    self.camada_saida.neuronios[o].pesos.append(
                        pesos_camada_saida[num_peso])
                num_peso += 1

    def feed_forward(self, entradas):
        # propaga as entradas pela rede neural e retorna as saídas da camada de saída
        saidas_camada_oculta = self.camada_oculta.feed_forward(entradas)
        return self.camada_saida.feed_forward(saidas_camada_oculta)

    def treinar(self, entradas_treinamento, saidas_treinamento):
        # executa o treinamento da rede neural usando o algoritmo de retropropagação
        self.feed_forward(entradas_treinamento)

        # calcula os deltas dos neurônios de saída
        pd_erros_wrt_saida_neuronio_total_net_input = [0] * len(self.camada_saida.neuronios)
        for o in range(len(self.camada_saida.neuronios)):
            pd_erros_wrt_saida_neuronio_total_net_input[o] = self.camada_saida.neuronios[o].calcular_pd_erro_wrt_total_net_input(saidas_treinamento[o])

        # calcula os deltas dos neurônios ocultos
        pd_erros_wrt_neuronio_oculto_total_net_input = [0] * len(self.camada_oculta.neuronios)
        for h in range(len(self.camada_oculta.neuronios)):
            d_erro_wrt_saida_neuronio_oculto = 0
            for o in range(len(self.camada_saida.neuronios)):
                d_erro_wrt_saida_neuronio_oculto += pd_erros_wrt_saida_neuronio_total_net_input[o] * self.camada_saida.neuronios[o].pesos[h]
            pd_erros_wrt_neuronio_oculto_total_net_input[h] = d_erro_wrt_saida_neuronio_oculto * self.camada_oculta.neuronios[h].calcular_pd_total_net_input_wrt_entrada()

        # atualiza os pesos dos neurônios de saída
        for o in range(len(self.camada_saida.neuronios)):
            for w_ho in range(len(self.camada_saida.neuronios[o].pesos)):
                pd_erro_wrt_peso = pd_erros_wrt_saida_neuronio_total_net_input[o] * self.camada_saida.neuronios[o].calcular_pd_total_net_input_wrt_peso(w_ho)
                self.camada_saida.neuronios[o].pesos[w_ho] -= self.LR * pd_erro_wrt_peso

        # atualiza os pesos dos neurônios ocultos
        for h in range(len(self.camada_oculta.neuronios)):
            for w_ih in range(len(self.camada_oculta.neuronios[h].pesos)):
                pd_erro_wrt_peso = pd_erros_wrt_neuronio_oculto_total_net_input[h] * self.camada_oculta.neuronios[h].calcular_pd_total_net_input_wrt_peso(w_ih)
                self.camada_oculta.neuronios[h].pesos[w_ih] -= self.LR * pd_erro_wrt_peso

    def calcular_erro_total(self, conjuntos_treinamento):
        # calcula o erro total da rede neural em um conjunto de treinamento
        erro_total = 0
        for t in range(len(conjuntos_treinamento)):
            entradas_treinamento, saidas_treinamento = conjuntos_treinamento[t]
            self.feed_forward(entradas_treinamento)
            for o in range(len(saidas_treinamento)):
                erro_total += self.camada_saida.neuronios[o].calcular_erro(saidas_treinamento[o])
        return erro_total

# definindo rede neural
rn = RedeNeural(num_inputs=2, num_ocultos=2, num_outputs=2, LR=0.5,
                pesos_camada_oculta=[0.15, 0.2, 0.25, 0.3],
                bias_camada_oculta=0.35,
                pesos_camada_saida=[0.4, 0.45, 0.5, 0.55],
                bias_camada_saida=0.6)

# saídas antes do treinamento
saida1, saida2 = rn.feed_forward([0.05, 0.1])
print("saída 1 antes do treinamento:", format(saida1, '.8f'))
print("saída 2 antes do treinamento:", format(saida2, '.8f'))

# erros de saídas 1 e 2 antes do treinamento
erro_saida1_antes = rn.camada_saida.neuronios[0].calcular_erro(0.01)
erro_saida2_antes = rn.camada_saida.neuronios[1].calcular_erro(0.99)
print("erro de saída 1 antes do treinamento:", format(erro_saida1_antes, '.9f'))
print("erro de saída 2 antes do treinamento:", format(erro_saida2_antes, '.9f'))

# erro total antes do treinamento
erro_total_antes = rn.calcular_erro_total([[[0.05, 0.1], [0.01, 0.99]]])
print("erro total antes do treinamento:", format(erro_total_antes, '.9f'), "\n")

# Treinamento da rede neural
for i in range(100):
    rn.treinar([0.05, 0.1], [0.01, 0.99])

# saídas após o treinamento
saida1_depois, saida2_depois = rn.feed_forward([0.05, 0.1])
print("saída 1 após o treinamento:", format(saida1_depois, '.9f'))
print("saída 2 após o treinamento:", format(saida2_depois, '.9f'))

# erro total após o treinamento
erro_total_depois = rn.calcular_erro_total([[[0.05, 0.1], [0.01, 0.99]]])
print("erro total após o treinamento:", format(erro_total_depois, '.9f'))

