import torch
import torch.nn as nn
import torch.optim as optim

# definição da classe da rede neural


class RedeNeural(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, pesos_oculta=None,
                 pesos_saida=None, bias_oculta=None, bias_saida=None):
        super(RedeNeural, self).__init__()
        # camada oculta
        self.camada_oculta = nn.Linear(input_size, hidden_size, bias=True)
        # camada de saída
        self.camada_saida = nn.Linear(hidden_size, output_size, bias=True)

        # inicialização dos pesos manualmente, se fornecidos
        if pesos_oculta is not None:
            self.camada_oculta.weight = nn.Parameter(
                torch.tensor(pesos_oculta))
        if pesos_saida is not None:
            # Transpondo os pesos da camada de saída
            self.camada_saida.weight = nn.Parameter(torch.tensor(pesos_saida).t())

        # inicialização dos bias manualmente, se fornecidos
        if bias_oculta is not None:
            self.camada_oculta.bias = nn.Parameter(torch.tensor(bias_oculta))
        if bias_saida is not None:
            self.camada_saida.bias = nn.Parameter(torch.tensor(bias_saida))
    # (forward pass)
    def forward(self, x):
        # Ativação da camada oculta (sigmoid)
        x = torch.sigmoid(self.camada_oculta(x))
        # Ativação da camada de saída (sigmoid)
        x = torch.sigmoid(self.camada_saida(x))
        return x

# função para treinar a rede neural


def treinar_rede(modelo, entradas, saidas, epochs=10000, lr=0.5):
    outputs_iniciais = modelo(entradas)
    # Definindo a função de perda

    def minha_funcao_perda(saidas, outputs_iniciais):
        loss_1 = 1/2 * (saidas[0, 0] - outputs_iniciais[0, 0])**2
        loss_2 = 1/2 * (saidas[0, 1] - outputs_iniciais[0, 1])**2
        perda_total = loss_1 + loss_2
        return perda_total, loss_1, loss_2

    # definição do otimizador (SGD - Gradiente Descendente Estocástico)
    optimizer = optim.SGD(modelo.parameters(), lr=lr)

    # antes do treinamento, calcula o erro total e imprime
    perda_total, loss_1, loss_2 = minha_funcao_perda(saidas, outputs_iniciais)
    print(f'erro 1 antes do treinamento: {loss_1:.8f}')
    print(f'erro 2 antes do treinamento: {loss_2:.8f}')
    print(f'erro total antes do treinamento: {perda_total:.8f}\n')

    criterion = nn.MSELoss()

    # Loop de treinamento
    for epoch in range(epochs):
        # zera os gradientes dos parâmetros do modelo
        optimizer.zero_grad()
        # realiza a passagem adiante (forward pass)
        outputs = modelo(entradas)
        # calcula o erro total
        loss = criterion(outputs, saidas)
        # (backward pass)
        loss.backward()
        # atualiza os parâmetros do modelo
        optimizer.step()

        # a cada x épocas, imprime informações sobre a perda e os valores de saída
        if (epoch+1) % 10000 == 0:
            perda_total = 1/2 * \
                ((saidas[0, 0] - outputs[0, 0])**2 +
                 (saidas[0, 1] - outputs[0, 1])**2)
            print(
                f'Época [{epoch+1}/{epochs}], erro total após treinamento: {perda_total:.9f}')
            # Imprimindo os valores de saída o1 e o2 após o treinamento
            print("Saída 1 após o treinamento:", format(outputs[0, 0], '.8f'))
            print("Saída 2 após o treinamento:", format(outputs[0, 1], '.8f'))


if __name__ == "__main__":
    # Definindo manualmente a entrada
    entradas = torch.tensor([[0.05, 0.1]])

    # Dados de exemplo para as saídas
    saidas = torch.tensor([[0.01, 0.99]])

    # Valores de pesos e bias manualmente definidos
    pesos_oculta = [[0.15, 0.25],
                    [0.2, 0.3],
                    [0.25, 0.35],
                    [0.3, 0.4]]

    pesos_saida = [[0.4, 0.45],
                   [0.5, 0.55],
                   [0.6, 0.65],
                   [0.7, 0.75]]

    bias_oculta = [0.35, 0.35, 0.35, 0.35]
    bias_saida = [0.6, 0.6]

    # modelo da rede neural
    modelo = RedeNeural(input_size=2, hidden_size=4, output_size=2,
                        pesos_oculta=pesos_oculta, pesos_saida=pesos_saida,
                        bias_oculta=bias_oculta, bias_saida=bias_saida)

    # imprimindo as saídas o1 e o2 antes do treinamento
    print("Saída 1 antes do treinamento:",
          format(modelo(entradas)[0, 0], '.8f'))
    print("Saída 2 antes do treinamento:",
          format(modelo(entradas)[0, 1], '.8f'))

    # treinando a rede neural
    treinar_rede(modelo, entradas, saidas)
