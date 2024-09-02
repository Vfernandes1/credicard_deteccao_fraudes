# Otimização de Modelo Pré-Treinado para Detecção de Fraudes em Cartões de Crédito

## Aluno: Vinícius Oliveira Fernandes

A ponderada foca na aplicação de técnicas de ajuste fino de hiperparâmetros para melhorar o desempenho de um modelo de detecção de fraudes em transações com cartões de crédito. 

A intenção aqui era utilizar métodos como `RandomizedSearchCV`, utilizado aqui, para otimizar os parâmetros do modelo original, comparando o desempenho antes e depois da otimização.

*É IMPORTANTE CITAR QUE, EU PODERIA FAZER AJUSTES PARA TENTAR MELHORAR O RESULTADO, NO ENTANTO, PRIORIZEI O ENTENDIMENTO DO PROCESSO DE SELEÇÃO DE HIPERPARÂMETROS E DE ENTENDIMENTO DA UTILIZAÇÃO DO RandomizedSearchCV*

## Estrutura dos Modelos Utilizados

### Modelo Original

O modelo original é uma rede neural simples criada com a biblioteca Keras, a partir do que fizemos no projeto. Não utilizei nenhum tipo de dropout ou regularizador. Ele possui a seguinte arquitetura:

- **Camadas Ocultas**: Cinco camadas densas (arquitetura piramidal) com diferentes quantidades de neurônios:
  - 64, 32, 16, 8, e 4 neurônios, respectivamente.
  - Função de ativação ReLU (Rectified Linear Unit).
- **Output Layer**: Uma única camada densa com função de ativação sigmoid, adequada para a tarefa de classificação binária (fraude vs. não fraude).

#### Utilização do Modelo Original

Este modelo foi treinado com os seguintes parâmetros:
- **Função de perda**: `binary_crossentropy` — apropriada para tarefas de classificação binária.
- **Métricas**: `accuracy`, `precision`, `recall`, `mse` (mean squared error).
- **Épocas**: 10
- **Batch size**: 32

### Resultados do Modelo Original

Os resultados obtidos durante o treinamento do modelo original foram:

| Época | Accuracy | Loss    | MSE    | Precision | Recall | Val_Accuracy | Val_Loss | Val_MSE | Val_Precision | Val_Recall |
|-------|----------|---------|--------|-----------|--------|--------------|----------|---------|---------------|------------|
| 1     | 0.5195   | 100.0286| 0.4131 | 0.3054    | 0.2580 | 0.4932       | 0.6933   | 0.2501  | 0.4932        | 1.0000     |
| 2     | 0.5167   | 0.6929  | 0.2499 | 0.5167    | 1.0000 | 0.4932       | 0.6933   | 0.2501  | 0.4932        | 1.0000     |
| 3     | 0.4927   | 0.6933  | 0.2501 | 0.4927    | 1.0000 | 0.4932       | 0.6933   | 0.2501  | 0.4932        | 1.0000     |
| ...   | ...      | ...     | ...    | ...       | ...    | ...          | ...      | ...     | ...           | ...        |
| 10    | 0.5249   | 0.6928  | 0.2499 | 0.5249    | 1.0000 | 0.4932       | 0.6933   | 0.2501  | 0.4932        | 1.0000     |

O **AUC-ROC Score** do modelo original foi de 0.5.

### Modelo Otimizado

O modelo otimizado é semelhante ao original, mas aplicamos técnicas de ajuste fino de hiperparâmetros utilizando o `RandomizedSearchCV`. O objetivo foi explorar diferentes combinações de hiperparâmetros para encontrar a melhor configuração possível para maximizar o desempenho do modelo.

#### Faixa de Valores para os Hiperparâmetros

Optamos por otimizar os seguintes hiperparâmetros:

- **Número de neurônios nas camadas ocultas (`neurons`)**: [32, 64, 128]
- **Funções de ativação (`activation`)**: ['relu', 'tanh', 'sigmoid']
- **Taxa de aprendizado (`optimizer`)**: ['adam', 'rmsprop']
- **Batch size (`batch_size`)**: [16, 32, 64]
- **Número de épocas (`epochs`)**: [20, 30, 40]

A escolha dessas faixas foi baseada em práticas comuns de otimização de redes neurais, buscando um equilíbrio entre complexidade e desempenho, sem extrapolar os limites de tempo e recursos computacionais.

### Técnicas de Ajuste Fino de Hiperparâmetros

Utilizei o `RandomizedSearchCV` para explorar aleatoriamente várias combinações de hiperparâmetros. Este método é eficiente porque, em vez de testar exaustivamente todas as combinações possíveis (como no `GridSearchCV`), ele seleciona combinações aleatórias, permitindo encontrar boas configurações com menos tempo de processamento.

#### Métricas de Desempenho do Modelo Otimizado

| Métrica        | Valor    |
|----------------|----------|
| Precision (No Fraud) | 0.52    |
| Recall (No Fraud)    | 0.98    |
| Precision (Fraud)    | 0.79    |
| Recall (Fraud)       | 0.08    |
| Accuracy             | 0.53    |
| F1-Score (macro avg) | 0.41    |
| AUC-ROC Score        | 0.5277  |

### Comparação dos Resultados

| Métrica        | Modelo Original | Modelo Otimizado |
|----------------|-----------------|------------------|
| Accuracy       | 0.4932          | 0.53             |
| Precision      | 0.3054          | 0.52 (No Fraud) / 0.79 (Fraud) |
| Recall         | 1.0000          | 0.98 (No Fraud) / 0.08 (Fraud) |
| F1-Score       | Não disponível  | 0.41 (macro avg) |
| AUC-ROC Score  | 0.5             | 0.5277           |

### Análise dos Resultados

Após o ajuste fino de hiperparâmetros, houve uma leve melhoria no AUC-ROC Score (de 0.5 para 0.5277), indicando uma pequena melhoria na capacidade do modelo de distinguir entre fraudes e não fraudes. No entanto, a **precisão** e o **recall** para a classe de fraudes ainda são subótimos, sugerindo que o modelo otimizado tem uma taxa alta de falsos negativos (erros em detectar fraudes).

A técnica de ajuste fino utilizada permitiu explorar diferentes combinações de hiperparâmetros, mas os resultados indicam que, apesar da melhoria, o modelo ainda precisa de ajustes adicionais ou de uma abordagem diferente para atingir um desempenho aceitável.

## Etapas Realizadas

1. **Carregamento e Pré-processamento dos Dados**: Leitura do conjunto de dados de transações de cartões de crédito e normalização das features.
2. **Balanceamento dos Dados**: Aplicação do `RandomUnderSampler` para balancear o conjunto de dados e lidar com a alta desproporção de transações não fraudulentas.
3. **Treinamento do Modelo Original**: Treinamento de uma rede neural inicial e avaliação de seu desempenho.
4. **Definição de Hiperparâmetros**: Definição de uma faixa de valores para hiperparâmetros chave como `neurons`, `activation`, `optimizer`, `batch size` e `epochs`.
5. **Ajuste Fino com RandomizedSearchCV**: Aplicação de `RandomizedSearchCV` para explorar diferentes combinações de hiperparâmetros.
6. **Comparação de Resultados**: Avaliação e comparação dos desempenhos do modelo original e do modelo otimizado.
7. **Documentação e Análise**: Documentação das etapas realizadas, resultados observados e impacto das mudanças nos hiperparâmetros.

## Conclusões

Embora o ajuste NOS hiperparâmetros tenha trazido melhorias sutis ao modelo, as métricas de desempenho indicam que ele ainda não é suficientemente eficaz para identificar fraudes com alta precisão. A baixa melhora no **AUC-ROC Score** e o **Recall** para a classe de fraude apontam que o modelo precisa de uma arquitetura mais sofisticada ou de uma abordagem diferente para capturar melhor os padrões de fraude.

#### Referência

Chat GPT."Ajuda com documentação do processo de inserção de hiperparâmetros em modelos de detecção de fraudes". Disponível em: https://www.openai.com/chatgpt. Acesso em: 02 set. 2024.
