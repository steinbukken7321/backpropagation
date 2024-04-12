# Implementação de uma Rede Neural com Retropropagação do Zero

## Introdução as Redes Neurais e ao Backpropagation

### Redes Neurais
As redes neurais são modelos computacionais inspirados no funcionamento do cérebro humano. Elas consistem em neurônios interconectados organizados em camadas, onde cada neurônio recebe entradas, realiza um cálculo ponderado e produz uma saída.

Uma rede neural típica é composta por uma camada de entrada, uma ou mais camadas ocultas e uma camada de saída. As conexões entre os neurônios são representadas por pesos, que são ajustados durante o treinamento da rede para melhorar sua capacidade de fazer previsões ou classificações.

As redes neurais têm sido amplamente utilizadas em uma variedade de aplicações, incluindo reconhecimento de padrões, processamento de linguagem natural, visão computacional e muitas outras áreas de aprendizado de máquina e inteligência artificial.

![image](https://github.com/steinbukken7321/backpropagation/assets/83385968/f1414cdf-85a4-4891-b17a-45b03d70f50d)

### Backpropagation
Backpropagation é um algoritmo fundamental usado para treinar redes neurais artificiais. Ele permite que a rede neural aprenda a partir dos dados, ajustando os pesos das conexões entre os neurônios para minimizar o erro entre as previsões da rede e os valores reais dos dados de treinamento.

O processo de backpropagation envolve duas etapas principais: feedforward e retropropagação. Na etapa de feedforward, os dados são propagados pela rede neural, passando pelas diferentes camadas de neurônios, até que uma saída seja gerada. Em seguida, durante a etapa de retropropagação, o erro é calculado e propagado de volta pela rede, permitindo que os pesos das conexões sejam ajustados de acordo com a magnitude do erro.

![image](https://github.com/steinbukken7321/backpropagation/assets/83385968/8cb303b7-5a13-4176-aab0-ef35ad6f6518)



# Exemplo de Backpropagation em uma Rede Neural

Este é um exemplo simples de como o Backpropagation funciona em uma rede neural, no projeto, vamos usar uma rede neural com duas entradas, dois neurônios ocultos e dois neurônios de saída. Além disso, os neurônios ocultos e de saída incluirão um viés.

![image](https://github.com/steinbukken7321/backpropagation/assets/83385968/55024fac-4613-47d2-bba0-566b82797fbc)

## Sobre

Este projeto é baseado no tutorial de Matt Mazur sobre [Backpropagation](https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/). O método implementado aqui segue os passos detalhados no tutorial para treinar redes neurais usando Backpropagation.

## Como funciona?

1. **Feedforward**: A entrada é alimentada pela rede neural, passando pela camada oculta até a saída é calculada.

2. **Cálculo do Erro**: A diferença entre a saída prevista e a saída desejada é calculada para determinar o erro.

3. **Backpropagation**: Os gradientes são calculados para os pesos e os viéses usando a regra da cadeia.

4. **Atualização de Pesos e Viéses**: Os pesos e os viéses são ajustados usando os gradientes e uma taxa de aprendizado.


## languages ​​and tools used
![Python](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue)
![Vscode](https://img.shields.io/badge/VSCode-0078D4?style=for-the-badge&logo=visual%20studio%20code&logoColor=white)
![Github](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)
---

## Autores

- [@KarineFernandes](https://github.com/KaFernandes02)
- [@RafaelZiani](https://www.github.com/steinbukken7321)


## Contatos
- Karine Fernandes

[![LinkedIn](https://img.shields.io/badge/LinkedIn-7FFF00?style=for-the-badge&logo=linkedin&logoColor=000000)](https://www.linkedin.com/in/rafael-ziani-de-carvalho-a4546723a/)
[![Gmail](https://img.shields.io/badge/Gmail-7FFF00?style=for-the-badge&logo=gmail&logoColor=000000)](mailto:Rafael.ziani1@gmail.com)

- Rafael Ziani de Carvalho

[![LinkedIn](https://img.shields.io/badge/LinkedIn-000080?style=for-the-badge&logo=linkedin&logoColor=000000)](https://www.linkedin.com/in/rafael-ziani-de-carvalho-a4546723a/)
[![Gmail](https://img.shields.io/badge/Gmail-000080?style=for-the-badge&logo=gmail&logoColor=000000)](mailto:Rafael.ziani1@gmail.com)

## Status
- Karine Fernandes

![steinbukken7321 GitHub stats](https://github-readme-stats.vercel.app/api?username=KaFernandes02&theme=chartreuse-dark&show_icons=true)

- Rafael Ziani de Carvalho

![steinbukken7321 GitHub stats](https://github-readme-stats.vercel.app/api?username=steinbukken7321&theme=tokyonight&show_icons=true)


<code style="color : red">Created by Karine Fernandes and Rafael Ziani de Carvalho</code>
