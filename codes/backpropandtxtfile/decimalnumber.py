import random
import math

class RedeNeural:
    TAXA_APRENDIZAGEM = 0.5

    def __init__(self, num_inputs, num_ocultas, num_saidas, viés_camada_oculta=None, viés_camada_saida=None):
        self.num_inputs = num_inputs
        self.num_ocultas = num_ocultas
        self.num_saidas = num_saidas

        self.camada_oculta = CamadaNeuronios(num_ocultas, viés_camada_oculta)
        self.camada_saida = CamadaNeuronios(num_saidas, viés_camada_saida)

        self.inicializar_pesos_entrada_para_neuronios_camada_oculta()
        self.inicializar_pesos_neuronios_camada_oculta_para_neuronios_camada_saida()

    def inicializar_pesos_entrada_para_neuronios_camada_oculta(self):
        for h in range(len(self.camada_oculta.neuronios)):
            for i in range(self.num_inputs):
                self.camada_oculta.neuronios[h].pesos.append(random.random())

    def inicializar_pesos_neuronios_camada_oculta_para_neuronios_camada_saida(self):
        for o in range(len(self.camada_saida.neuronios)):
            for h in range(len(self.camada_oculta.neuronios)):
                self.camada_saida.neuronios[o].pesos.append(random.random())

    def feed_forward(self, entradas):
        saídas_camada_oculta = self.camada_oculta.feed_forward(entradas)
        return self.camada_saida.feed_forward(saídas_camada_oculta)

    def treinar(self, entradas_treino, saídas_treino):
        self.feed_forward(entradas_treino)

        pd_erros_em_relacao_ao_input_total_neuronio_saida = [0] * self.num_saidas
        for o in range(self.num_saidas):
            pd_erros_em_relacao_ao_input_total_neuronio_saida[o] = self.camada_saida.neuronios[o].calcular_pd_erro_em_relacao_ao_input_total(
                saídas_treino[o])

        pd_erros_em_relacao_ao_input_total_neuronio_oculto = [0] * self.num_ocultas
        for h in range(self.num_ocultas):
            d_erro_em_relacao_à_saída_neuronio_oculto = 0
            for o in range(self.num_saidas):
                d_erro_em_relacao_à_saída_neuronio_oculto += pd_erros_em_relacao_ao_input_total_neuronio_saida[o] * self.camada_saida.neuronios[o].pesos[h]

            pd_erros_em_relacao_ao_input_total_neuronio_oculto[h] = d_erro_em_relacao_à_saída_neuronio_oculto * self.camada_oculta.neuronios[h].calcular_pd_input_total_em_relacao_ao_input()

        for o in range(self.num_saidas):
            for w_ho in range(len(self.camada_saida.neuronios[o].pesos)):
                pd_erro_em_relacao_ao_peso = pd_erros_em_relacao_ao_input_total_neuronio_saida[o] * self.camada_saida.neuronios[o].calcular_pd_input_total_em_relacao_ao_peso(w_ho)
                self.camada_saida.neuronios[o].pesos[w_ho] -= self.TAXA_APRENDIZAGEM * pd_erro_em_relacao_ao_peso

        for h in range(self.num_ocultas):
            for w_ih in range(len(self.camada_oculta.neuronios[h].pesos)):
                pd_erro_em_relacao_ao_peso = pd_erros_em_relacao_ao_input_total_neuronio_oculto[h] * self.camada_oculta.neuronios[h].calcular_pd_input_total_em_relacao_ao_peso(w_ih)
                self.camada_oculta.neuronios[h].pesos[w_ih] -= self.TAXA_APRENDIZAGEM * pd_erro_em_relacao_ao_peso

    def calcular_erro_total(self, conjuntos_treino):
        erro_total = 0
        for t in range(len(conjuntos_treino)):
            entradas_treino, saídas_treino = conjuntos_treino[t]
            self.feed_forward(entradas_treino)
            for o in range(len(saídas_treino)):
                erro_total += self.camada_saida.neuronios[o].calcular_erro(saídas_treino[o])
        return erro_total

class CamadaNeuronios:
    def __init__(self, num_neuronios, viés):
        self.viés = viés if viés else random.random()
        self.neuronios = []
        for i in range(num_neuronios):
            self.neuronios.append(Neuronio(self.viés))

    def feed_forward(self, entradas):
        saídas = []
        for neuronio in self.neuronios:
            saídas.append(neuronio.calcular_saída(entradas))
        return saídas

class Neuronio:
    def __init__(self, viés):
        self.viés = viés
        self.pesos = []

    def calcular_saída(self, entradas):
        self.entradas = entradas + [0] * (9 - len(entradas))  # Adicionando zeros para completar 9 entradas
        self.saída = self.ativação(self.calcular_input_total())
        return self.saída

    def calcular_input_total(self):
        total = 0
        for i in range(len(self.entradas)):
            total += self.entradas[i] * self.pesos[i]
        return total + self.viés

    def ativação(self, input_total):
        return 1 / (1 + math.exp(-input_total))

    def calcular_pd_erro_em_relacao_ao_input_total(self, saída_desejada):
        return self.calcular_pd_erro_em_relacao_à_saída(saída_desejada) * self.calcular_pd_input_total_em_relacao_ao_input()

    def calcular_erro(self, saída_desejada):
        return 0.5 * (saída_desejada - self.saída) ** 2

    def calcular_pd_erro_em_relacao_à_saída(self, saída_desejada):
        return -(saída_desejada - self.saída)

    def calcular_pd_input_total_em_relacao_ao_input(self):
        return self.saída * (1 - self.saída)

    def calcular_pd_input_total_em_relacao_ao_peso(self, index):
        return self.entradas[index]

def ler_numero_binario_arquivo(caminho_arquivo):
    with open(caminho_arquivo, 'r') as arquivo:
        return [int(digito) for digito in arquivo.read().strip()]

# Ler o número binário do arquivo
numero_binario = ler_numero_binario_arquivo("C:\Users\rafae\Desktop\backpropagation\codes\backpropandtxtfile\binary_number.txt")
print("Número binário do arquivo:", numero_binario)

# Conjuntos de treinamento para números binários de 0 a 9
conjuntos_treino = [
    [[0, 0, 0], [0]],  # Para o número binário "000", a saída desejada é 0
    [[0, 0, 1], [1]],  # Para o número binário "001", a saída desejada é 1
    [[0, 1, 0], [2]],  # Para o número binário "010", a saída desejada é 2
    [[0, 1, 1], [3]],  # Para o número binário "011", a saída desejada é 3
    [[1, 0, 0], [4]],  # Para o número binário "100", a saída desejada é 4
    [[1, 0, 1], [5]],  # Para o número binário "101", a saída desejada é 5
    [[1, 1, 0], [6]],  # Para o número binário "110", a saída desejada é 6
    [[1, 1, 1], [7]],  # Para o número binário "111", a saída desejada é 7
    [[1, 0, 0, 0], [8]],  # Para o número binário "1000", a saída desejada é 8
    [[1, 0, 0, 1], [9]]   # Para o número binário "1001", a saída desejada é 9
]

# Criar e treinar a rede neural
rn = RedeNeural(9, 4, 2)
for i in range(10000):
    rn.treinar(numero_binario, numero_binario)

# Calcular e imprimir o erro total antes e após o treinamento
erro_total_antes = rn.calcular_erro_total(conjuntos_treino)
print("Erro total antes do treinamento:", erro_total_antes)

erro_total_após = rn.calcular_erro_total(conjuntos_treino)
print("Erro total após o treinamento:", erro_total_após)

# Deduzir o número do arquivo após o treinamento
numero_deduzido = rn.feed_forward(numero_binario)
print("Número deduzido do arquivo TXT após o treinamento:", numero_deduzido)
