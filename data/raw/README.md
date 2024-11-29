# Predição de Diabetes: Projeto de Machine Learning


## **Descrição do Conjunto de Dados**
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

### **Informações Principais do Conjunto de Dados**
- **Instâncias**: 768
- **Atributos**: 8 características preditivas e 1 variável-alvo (`Outcome`).
- **Distribuição das Classes**:
  - `Outcome = 1` (Positivo para diabetes).
  - `Outcome = 0` (Negativo para diabetes).
- **Valores Faltantes**: Sim, algumas variáveis possuem valores ausentes.

## **Pipeline do Projeto**
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

## **Início Rápido**
### **Pré-requisitos**
- Python 3.10 ou superior.
- Bibliotecas:
  - `pandas`, `numpy`, `matplotlib`, `seaborn` para análise e visualização de dados.
  - `scikit-learn` para machine learning.
  - `MLflow` para rastreamento de experimentos.
  - `pytest` para testes (opcional).

