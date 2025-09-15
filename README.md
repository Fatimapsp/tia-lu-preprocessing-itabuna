# Desafio 2: Biblioteca de Pré-Processamento de Dados

## 📝 Visão Geral do Desafio

> Sejam bem-vindos à segunda etapa do nosso programa! Como vimos no desafio anterior, o principal objetivo de qualquer plataforma é auxiliar os consumidores na atividade mais complexa que todos eles possuem: escolher o que consumir. O nosso objetivo ao longo deste semestre é desenvolver o nosso modelo de recomendação para os usuários. Como trainees da área de dados, vocês estão no segundo desafio. Estamos numa etapa de pré-processamento dos nossos dados. Vocês vão perceber que quanto melhor o tratamento dos dados, mais preciso e eficiente será o nosso modelo.

Este repositório contém a solução para o desafio, que consiste na criação de uma biblioteca Python robusta para realizar as tarefas mais comuns de pré-processamento de dados, preparando um dataset para as futuras etapas de modelagem de machine learning.

## 📖 Tabela de Conteúdos

- [A Importância do Pré-Processamento](#-a-importância-do-pré-processamento)
- [Funcionalidades Implementadas](#-funcionalidades-implementadas)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Como Utilizar a Biblioteca](#-como-utilizar-a-biblioteca)
- [Como Executar os Testes](#-como-executar-os-testes)

## 🎯 A Importância do Pré-Processamento

Antes de construirmos qualquer modelo de aprendizado de máquina, precisamos garantir que os dados que o alimentarão sejam de alta qualidade. O pré-processamento é a etapa mais crucial e, muitas vezes, a que consome mais tempo em um projeto de dados. É aqui que transformamos dados brutos e "sujos" em um formato limpo, consistente e compreensível para os algoritmos.

Um tratamento cuidadoso e metódico assegura a integridade e a confiabilidade dos dados, permitindo que os algoritmos aprendam padrões verdadeiros de comportamento do consumidor, o que resulta em sugestões mais personalizadas e assertivas para nosso modelo de recomendação.

## ✨ Funcionalidades Implementadas

A biblioteca `preprocessing` foi construída de forma modular e encapsulada na classe principal `Preprocessing`. Ela oferece uma API simples e fluente (com encadeamento de métodos) para tratar os seguintes desafios:

### 1. Tratamento de Dados Ausentes (`MissingValueProcessor`)
A presença de dados ausentes ou nulos (`None`/`NaN`) é um desafio comum. A biblioteca oferece estratégias flexíveis para lidar com eles:
- **`isna()`**: Identifica e retorna todas as linhas que contêm valores nulos em colunas específicas.
- **`notna()`**: O inverso de `isna`, retorna todas as linhas que **não** contêm valores nulos.
- **`fillna()`**: Preenche valores ausentes utilizando métodos estatísticos (`mean`, `median`, `mode`) ou um valor padrão.
- **`dropna()`**: Remove completamente as linhas que contêm dados faltantes.

### 2. Escalonamento de Dados Numéricos (`Scaler`)
Algoritmos de machine learning podem ser sensíveis a features com escalas muito diferentes. Para evitar vieses, implementamos dois métodos de escalonamento:
- **`minMax_scaler()`**: Normaliza os dados para um intervalo fixo (geralmente [0, 1]).
- **`standard_scaler()`**: Padroniza os dados, resultando em uma distribuição com média 0 e desvio padrão 1 (Z-score).

### 3. Codificação de Dados Categóricos (`Encoder`)
Modelos de machine learning operam com números, não com texto. Nossos encoders traduzem variáveis categóricas para um formato numérico:
- **`label_encode()`**: Atribui um número inteiro único para cada categoria em uma coluna.
- **`oneHot_encode()`**: Cria novas colunas binárias (0 ou 1) para cada categoria, evitando a criação de uma relação de ordem artificial.

## 📂 Estrutura do Projeto

O repositório está organizado da seguinte forma:

```
.
├── preprocessing.py    # O módulo principal contendo todas as classes da biblioteca.
├── test_preprocessing.py   # O arquivo com os testes unitários para a biblioteca.
├── food_statistics.py   # O arquivo contendo a classe de Statistics implementada no primeiro desafio
└── README.md               # Este arquivo.
```

## 🚀 Como Utilizar a Biblioteca

A classe principal `Preprocessing` serve como uma fachada, simplificando o uso de todas as funcionalidades. Abaixo está um exemplo prático de uso:

```python
# Importe a classe principal
from preprocessing_lib import Preprocessing

# 1. Carregue seu dataset no formato de dicionário
dados = {
    'idade': [25, 30, None, 45],
    'salario': [50000, 60000, 45000, None],
    'cidade': ['Recife', 'Salvador', 'Recife', 'São Paulo']
}

# 2. Instancie a classe de pré-processamento
preprocessador = Preprocessing(dados)

# 3. Aplique as transformações de forma encadeada
preprocessador.fillna(columns={'idade'}, method='mean') \
              .fillna(columns={'salario'}, method='median') \
              .scale(columns={'idade', 'salario'}, method='standard') \
              .encode(columns={'cidade'}, method='oneHot')

# 4. O dataset original foi modificado e está pronto para uso
print(preprocessador.dataset)
```

## ✅ Como Executar os Testes

Para garantir a confiabilidade e o correto funcionamento da biblioteca, foi criada uma suíte de testes unitários utilizando o módulo `unittest` do Python.

Para executar os testes, abra seu terminal na raiz do projeto e execute o seguinte comando:

```bash
python -m unittest test_preprocessing.py
```

A saída esperada é uma mensagem indicando que todos os testes passaram (`OK`). Isso confirma que todas as funcionalidades da biblioteca estão operando conforme o esperado.
