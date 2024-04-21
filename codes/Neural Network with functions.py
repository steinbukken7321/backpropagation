import torch
import torch.nn as nn
import torch.optim as optim

# Definição da classe da rede neural
class MinhaRedeNeural(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, pesos_oculta=None, pesos_saida=None, bias_oculta=None, bias_saida=None):
        super(MinhaRedeNeural, self).__init__()
        # Camada oculta
        self.camada_oculta = nn.Linear(input_size, hidden_size, bias=True)
        # Camada de saída
        self.camada_saida = nn.Linear(hidden_size, output_size, bias=True)
        
        # Inicialização dos pesos manualmente, se fornecidos
        if pesos_oculta is not None:
            self.camada_oculta.weight = nn.Parameter(torch.tensor(pesos_oculta))
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
def treinar_rede(modelo, entradas, saidas, epochs=10000, lr=0.5):
    # Definição da função de perda (MSE Loss)
    criterion = nn.MSELoss()
    # Definição do otimizador (SGD - Gradiente Descendente Estocástico)
    optimizer = optim.SGD(modelo.parameters(), lr=lr)
    
    # Antes do treinamento: calcula a perda total
    outputs_iniciais = modelo(entradas)
    perda_inicial = criterion(outputs_iniciais, saidas)
    print(f'Valores de Saída antes do treinamento: {outputs_iniciais.detach().numpy()}')
    print(f'Perda Total antes do treinamento: {perda_inicial.item():.4f}\n')
    
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
        if (epoch+1) % 100 == 0:
            perda_saida1 = criterion(outputs[:, 0], saidas[:, 0])
            perda_saida2 = criterion(outputs[:, 1], saidas[:, 1])
            perda_total = loss.item()
            print(f'Época [{epoch+1}/{epochs}], Perda Total: {perda_total:.4f}, Perda Saída 1: {perda_saida1.item():.4f}, Perda Saída 2: {perda_saida2.item():.4f}')
            print(f'Valores de Saída: {outputs.detach().numpy()}')

if __name__ == "__main__":
    # Definindo manualmente a entrada
    entradas = torch.tensor([[0.05, 0.1]])
    
    # Dados de exemplo para as saídas
    saidas = torch.tensor([[0.99, 0.01]])
    
    # Valores de pesos e bias manualmente definidos
    pesos_oculta = [[0.15, 0.2],
                    [0.25, 0.3]]
    
    pesos_saida = [[0.4, 0.45],
                   [0.5, 0.55]]
    
    bias_oculta = [0.35, 0.35]
    bias_saida = [0.6, 0.6]
    
    # Instanciando o modelo da rede neural
    modelo = MinhaRedeNeural(input_size=2, hidden_size=2, output_size=2, pesos_oculta=pesos_oculta, pesos_saida=pesos_saida, bias_oculta=bias_oculta, bias_saida=bias_saida)
    # Treinando a rede neural
    treinar_rede(modelo, entradas, saidas)
