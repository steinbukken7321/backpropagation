import torch
import torch.nn as nn
import torch.optim as optim

# Resumo do Codigo:
# este código utiliza o PyTorch para definir o critério de perda, 
# criar um loop de treinamento para iterar sobre várias épocas e 
# realizar a otimização dos parâmetros da rede neural usando o 
# algoritmo SGD (gradiente descendente estocástico).


# Definição da classe da rede neural
class RedeNeural(nn.Module):
    # Inicializador da classe
    def __init__(self, input_size, hidden_size, output_size, pesos_oculta=None,
                 pesos_saida=None, bias_oculta=None, bias_saida=None):
        super(RedeNeural, self).__init__()
        # Camada oculta
        self.camada_oculta = nn.Linear(input_size, hidden_size, bias=True)
        # Camada de saída
        self.camada_saida = nn.Linear(hidden_size, output_size, bias=True)

        # Inicialização dos pesos manualmente, se fornecidos
        if pesos_oculta is not None:
            self.camada_oculta.weight = nn.Parameter(
                torch.tensor(pesos_oculta))
        if pesos_saida is not None:
            self.camada_saida.weight = nn.Parameter(torch.tensor(pesos_saida))

        # Inicialização dos bias manualmente, se fornecidos
        if bias_oculta is not None:
            self.camada_oculta.bias = nn.Parameter(torch.tensor(bias_oculta))
        if bias_saida is not None:
            self.camada_saida.bias = nn.Parameter(torch.tensor(bias_saida))

    # Método para a passagem adiante (forward pass) da rede neural
    def forward(self, x):
        # Ativação da camada oculta (sigmoid)
        x = torch.sigmoid(self.camada_oculta(x))
        # Ativação da camada de saída (sigmoid)
        x = torch.sigmoid(self.camada_saida(x))
        return x


# Função para treinar a rede neural
def treinar_rede(modelo, entradas, saidas, epochs=1, lr=0.5):
    outputs_iniciais = modelo(entradas)
    # Definindo a função de perda
    def minha_funcao_perda(saidas, outputs_iniciais):
        loss_1 = 1/2 * (saidas[0, 0] - outputs_iniciais[0, 0])**2
        loss_2 = 1/2 * (saidas[0, 1] - outputs_iniciais[0, 1])**2
        perda_total = loss_1 + loss_2
        return perda_total, loss_1, loss_2
    
    # Definição do otimizador (SGD - Gradiente Descendente Estocástico)
    optimizer = optim.SGD(modelo.parameters(), lr=lr)

    # Antes do treinamento: calcula a perda total
    perda_total, loss_1, loss_2 = minha_funcao_perda(saidas, outputs_iniciais)
    #for row in outputs_iniciais:
    #    print(f'[{row[0]:.8f}, {row[1]:.8f}]')
    print(f'Perda 1 antes do treinamento: {loss_1:.8f}')
    print(f'Perda 2 antes do treinamento: {loss_2:.8f}')
    print(f'Perda Total antes do treinamento: {perda_total:.8f}\n')
    
    # Calcula o erro quadrático médio (MSE) 
    criterion = nn.MSELoss()

    # Loop de treinamento
    for epoch in range(epochs):
        # Zera os gradientes dos parâmetros do modelo
        optimizer.zero_grad()
        # Realiza a passagem adiante (forward pass)
        outputs = modelo(entradas)
        # Calcula a perda total
        loss = criterion(outputs, saidas)
        # Realiza a retropropagação (backward pass)
        loss.backward()
        # Atualiza os parâmetros do modelo
        optimizer.step()

        # A cada 100 épocas, imprime informações sobre a perda e os valores de saída
        if (epoch+1) % 1 == 0:
            perda_total = 1/2 * ((saidas[0, 0] - outputs[0, 0])**2 + (saidas[0, 1] - outputs[0, 1])**2)
            print(f'Época [{epoch+1}/{epochs}], Perda Total: {perda_total:.9f}')
            print(f'Valores de Saída: {outputs.detach().numpy()}')

if __name__ == "__main__":
    # Definindo manualmente a entrada
    entradas = torch.tensor([[0.05, 0.1]])

    # Dados de exemplo para as saídas
    saidas = torch.tensor([[0.01, 0.99]])

    # Valores de pesos e bias manualmente definidos
    pesos_oculta = [[0.15, 0.2],
                    [0.25, 0.3]]

    pesos_saida = [[0.4, 0.45],
                   [0.5, 0.55]]

    bias_oculta = [0.35, 0.35]
    bias_saida = [0.6, 0.6]

    # Instanciando o modelo da rede neural
    modelo = RedeNeural(input_size=2, hidden_size=2, output_size=2,
                        pesos_oculta=pesos_oculta, pesos_saida=pesos_saida,
                        bias_oculta=bias_oculta, bias_saida=bias_saida)
    # Treinando a rede neural
    treinar_rede(modelo, entradas, saidas)

