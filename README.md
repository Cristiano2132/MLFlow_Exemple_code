
# Projeto MLflow - Detecção de Diabetes

Este projeto configura um ambiente Docker para executar o MLflow, focado na criação de um modelo de Machine Learning para prever fraude.

## Visão Geral
O objetivo deste projeto é construir um modelo preditivo de machine learning para determinar se um paciente apresenta sinais de diabetes com base em medições diagnósticas. O conjunto de dados utilizado neste projeto é originário do Instituto Nacional de Diabetes e Doenças Digestivas e Renais (National Institute of Diabetes and Digestive and Kidney Diseases). Todos os pacientes do conjunto de dados são mulheres com pelo menos 21 anos de idade e de herança Pima.
Iremos registrar todos os esperimentos e modelos no MLflow. Este projeto abrange:

1. Coleta e preparação de dados.
2. Treinamento de diferentes modelos de Machine Learning.
3. Avaliação e comparação de modelos.
4. Registro e deploy de modelos usando MLflow.

## Setup

### Pré-requisitos

- Docker e Docker Compose instalados
- Python e `virtualenv` instalados

### Passos para configuração

1. **Clone o repositório:**
   ```sh
   git clone <URL-do-repositório>
   cd <nome-do-diretório>
   ```

2. **Criar um ambiente virtual Python:**
   ```sh
   python -m venv venv
   source venv/bin/activate  # Para Linux/Mac
   venv\Scripts\activate  # Para Windows
   ```

3. **Instalar dependências:**
   ```sh
   pip install -r requirements.txt
   ```

4. **Subir os containers Docker:**


4.1. Construindo a Imagem Docker

Execute o comando abaixo no diretório onde o Dockerfile está localizado para criar a imagem:
```bash
docker build -t mlflow_image . 
```


•	-t mlflow_image: Define o nome da imagem como mlflow_image.
•	.: Indica o contexto de construção no diretório atual.

4.2. Iniciando o Contêiner

Use o comando abaixo para iniciar o contêiner:

```bash
docker run -it -p 8000:5000 mlflow_image   
```


5. **Acesse o MLflow:**
   - Abra seu navegador e vá para: `http://localhost:8000`

## Estrutura do projeto
Ao trabalhar com MLflow, é importante organizar o projeto de forma clara e escalável, permitindo o rastreamento fácil de experimentos, reprodutibilidade e colaboração. Abaixo está uma estrutura de pastas geralmente recomendada:

Estrutura Geral de Pastas


```bash
my_mlflow_project/
├── data/                    # Dados brutos e processados
│   ├── raw/                 # Dados brutos
│   └── processed/           # Dados processados
├── notebooks/               # Notebooks para exploração inicial e análise
├── src/                     # Código fonte principal
│   ├── data/                # Scripts de manipulação de dados
│   ├── models/              # Modelos criados ou treinados
│   ├── features/            # Scripts para engenharia de features
│   ├── training/            # Scripts para treinamento do modelo
│   └── evaluation/          # Scripts para avaliação e métricas
├── mlruns/                  # Diretório onde MLflow salva os experimentos (gerado automaticamente)
├── config/                  # Arquivos de configuração
│   ├── params.yaml          # Hiperparâmetros e configurações do modelo
│   └── settings.yaml        # Configurações gerais do projeto
├── tests/                   # Testes automatizados para o código
├── scripts/                 # Scripts utilitários ou automações
├── Dockerfile               # Arquivo Docker para criar ambientes reprodutíveis
├── requirements.txt         # Dependências do projeto
├── environment.yml          # Alternativa ao `requirements.txt`
├── README.md                # Documentação do projeto
└── .gitignore               # Arquivo para ignorar arquivos no Git
```

Descrição das Pastas

1.	data/:
  •	Armazena os dados brutos (raw/) e processados (processed/).
  •	A separação ajuda na rastreabilidade e na limpeza de dados.
2.	notebooks/:
  •	Contém notebooks para experimentação inicial e prototipagem.
3.	src/:
  •	Estruturada para organizar os componentes do pipeline:
  •	data/: Scripts de carregamento e pré-processamento de dados.
  •	features/: Scripts para criar e transformar features.
  •	models/: Modelos finais e serializados (por exemplo, arquivos .pkl ou .h5).
  •	training/: Scripts que definem o pipeline de treinamento e integração com MLflow.
  •	evaluation/: Scripts para avaliação, incluindo cálculo de métricas.
4.	mlruns/:
  •	Diretório gerado automaticamente pelo MLflow para armazenar logs e metadados dos experimentos.
5.	config/:
  •	Arquivos YAML para armazenar configurações como hiperparâmetros, caminhos de dados, e definições de ambiente.
6.	tests/:
  •	Scripts de teste para validar o código e pipelines (usando, por exemplo, pytest).
7.	scripts/:
  •	Scripts adicionais para automação, como inicializar MLflow, criar registros ou realizar deploy.
8.	Arquivos principais:
  •	Dockerfile: Para criar ambientes reprodutíveis.
  •	requirements.txt / environment.yml: Listas de dependências do projeto.
  •	README.md: Documentação que explica o objetivo do projeto e como executá-lo.


Dicas para Integração com MLflow

1.	Logs Automatizados:
  •	Configure os scripts em src/training/ e src/evaluation/ para registrar experimentos, parâmetros e métricas no MLflow usando o mlflow.start_run().
2.	Versionamento de Dados e Código:
  •	Use ferramentas como DVC ou git-lfs para versionar dados grandes, enquanto o MLflow cuida do rastreamento dos modelos.
3.	Model Registry:
  •	Configure o MLflow Model Registry para gerenciar modelos versionados e promover para produção.
4.	Envio para Deploy:
  •	Organize scripts no diretório scripts/ para fazer deploy dos modelos usando MLflow Serving, Docker, ou serviços como Azure.

Se a pasta do projeto já está criada e o terminal está aberto nela, os comandos para criar a estrutura podem ser simplificados. Aqui está o Markdown atualizado:


## Comandos para Unix (Mac/Linux)
```bash
# Cria os subdiretórios a partir da pasta atual
mkdir -p data/{raw,processed} notebooks src/{data,models,features,training,evaluation} config tests scripts

# Cria os arquivos principais
touch Dockerfile requirements.txt environment.yml README.md .gitignore
touch config/{params.yaml,settings.yaml}
```
Comandos para Windows (cmd)

```bash
REM Cria os subdiretórios a partir da pasta atual
mkdir data\raw data\processed
mkdir notebooks
mkdir src\data src\models src\features src\training src\evaluation
mkdir config
mkdir tests
mkdir scripts
```
REM Cria os arquivos principais

```bash
type nul > Dockerfile
type nul > requirements.txt
type nul > environment.yml
type nul > README.md
type nul > .gitignore
type nul > config\params.yaml
type nul > config\settings.yaml
```
Notas
```
1.	Posicionamento no Terminal:
  •	Certifique-se de que o terminal está aberto na raiz do projeto (onde será criada a estrutura).
2.	Flexibilidade:
  •	Altere ou remova diretórios conforme necessário se a estrutura completa não for necessária.

```