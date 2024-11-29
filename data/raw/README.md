# Predição de Diabetes: Projeto de Machine Learning

## Descrição do Conjunto de Dados

O conjunto de dados contém medições diagnósticas e uma variável-alvo binária (`Outcome`), que indica se o paciente possui diabetes:

- **Pregnancies**: Número de vezes que a paciente esteve grávida.
- **Glucose**: Concentração de glicose plasmática após 2 horas em um teste oral de tolerância à glicose.
- **BloodPressure**: Pressão arterial diastólica (mm Hg).
- **SkinThickness**: Espessura da dobra cutânea do tríceps (mm).
- **Insulin**: Insulina sérica em 2 horas (mu U/ml).
- **BMI**: Índice de massa corporal (peso em kg/(altura em m)^2).
- **DiabetesPedigreeFunction**: Função que mede a probabilidade de diabetes com base no histórico familiar.
- **Age**: Idade da paciente (em anos).
- **Outcome**: Variável-alvo (0: Sem diabetes, 1: Com diabetes).

### Informações Principais do Conjunto de Dados

- **Instâncias**: 768
- **Atributos**: 8 características preditivas e 1 variável-alvo (`Outcome`).
- **Distribuição das Classes**:
  - `Outcome = 1` (Positivo para diabetes).
  - `Outcome = 0` (Negativo para diabetes).
- **Valores Faltantes**: Sim, algumas variáveis possuem valores ausentes.

## Pipeline do Projeto

O projeto está dividido nas seguintes etapas:

1. **Exploração e Limpeza dos Dados**:
   - Compreender a distribuição de cada característica.
   - Tratar valores ausentes.
   - Analisar correlações entre as variáveis e o alvo.

2. **Engenharia de Features**:
   - Escalonar e normalizar variáveis.
   - Criar novas variáveis, se necessário, para melhorar o desempenho do modelo.

3. **Desenvolvimento do Modelo**:
   - Treinar diferentes modelos de machine learning, como Regressão Logística, Random Forest e XGBoost.
   - Avaliar o desempenho com validação cruzada.
   - Selecionar o melhor modelo com base em métricas como AUC-ROC, precisão e recall.

4. **Implantação do Modelo**:
   - Rastrear os experimentos utilizando o MLflow.
   - Criar APIs ou scripts para disponibilizar as previsões.

## Início Rápido

### Pré-requisitos

- Python 3.10 ou superior.
- Bibliotecas:
  - `pandas`, `numpy`, `matplotlib`, `seaborn` para análise e visualização de dados.
  - `scikit-learn` para machine learning.
  - `MLflow` para rastreamento de experimentos.
  - `pytest` para testes (opcional).

### Estrutura do Projeto

```bash
.
├── Dockerfile
├── README-CONTAINERS.md
├── README.md
├── data
│   ├── processed               # Dados processados
│   └── raw                     # Dados brutos
│       ├── README.md
│       └── diabetes.csv
├── docker-compose.yml
├── experiments                 # Scripts para criação e execução de experimentos
│   ├── create_or_set_mlflow_experiment.py
│   ├── exp_model_logistica.py
│   └── exp_model_logistica_autolog.py
├── main.py
├── mlruns                      # Diretório gerado automaticamente pelo MLflow para armazenar logs e metadados dos experimentos
├── notebooks                   # Notebooks para exploração inicial e análise
├── requirements-dev.txt
├── requirements.in
├── requirements.txt
├── src                         # Código fonte principal
│   ├── __init__.py
│   ├── data                    # Scripts de manipulação de dados
│   │   ├── __init__.py
│   │   ├── download_data.py
│   │   └── load_data.py
│   ├── evaluation              # Scripts para avaliação e métricas
│   │   ├── __init__.py
│   │   └── calcule_ks.py
│   ├── features                # Scripts para engenharia de features
│   │   ├── __init__.py
│   │   └── feature_engineering.py
│   ├── models                  # Modelos criados ou treinados
│   │   ├── __init__.py
│   │   └── reg_logistica.py
│   ├── training                # Scripts para treinamento do modelo
│   │   └── __init__.py
│   └── utils.py
└── tmp                         # Diretório temporário para arquivos e logs intermediários
```



### Comandos para Unix (Mac/Linux)
>**Atenção**: (**Posicionamento no Terminal**) Certifique-se de que o terminal está aberto na raiz do projeto (onde será criada a estrutura).


```bash
# Cria os subdiretórios a partir da pasta atual
mkdir -p data/{raw,processed} notebooks src/{data,models,features,training,evaluation} config tests scripts

# Cria os arquivos principais
touch Dockerfile docker-compose.yml requirements.txt environment.yml README.md .gitignore .env
touch config/{params.yaml,settings.yaml}
```

### Comandos para Windows (cmd)
>**Atenção**: (**Posicionamento no Terminal**) Certifique-se de que o terminal está aberto na raiz do projeto (onde será criada a estrutura).


```bash
REM Cria os subdiretórios a partir da pasta atual
mkdir data\raw data\processed
mkdir notebooks
mkdir src\data src\models src\features src\training src\evaluation
mkdir config
mkdir tests
mkdir scripts

REM Cria os arquivos principais
type nul > Dockerfile
type nul > docker-compose.yml
type nul > requirements.txt
type nul > environment.yml
type nul > README.md
type nul > .gitignore
type nul > .env
type nul > config\params.yaml
type nul > config\settings.yaml
```


### Configuração e Uso do Docker


### Variáveis de Ambiente

Crie um arquivo `.env` na raiz do projeto com as seguintes variáveis:

```plaintext
MLFLOW_LOCAL_VOLUME=/Users/...diretorio/do/projeto/mlflow
MLFLOW_PORT=5002
```



### Executando o Projeto>**Atenção**: (**Posicionamento no Terminal**) Certifique-se de que o terminal está aberto na raiz do projeto (onde será criada a estrutura).

>**Atenção**: (**Posicionamento no Terminal**) Certifique-se de que o terminal está aberto na raiz do projeto (onde será criada a estrutura).



1. **Construa e inicie os serviços** usando Docker Compose:

```bash
docker-compose up --build
```

2. **Acesse o servidor MLflow** no navegador, usando o endereço `http://localhost:${MLFLOW_PORT}`.

3. **Parar o container**:

```bash
docker-compose down
```

>Nota: Altere ou remova diretórios conforme necessário se a estrutura completa não for necessária.

