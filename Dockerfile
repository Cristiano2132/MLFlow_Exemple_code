# Baseado na imagem oficial do Python
FROM python:3.10-slim

# Instalar dependências necessárias para o MLflow e PostgreSQL
RUN pip install mlflow psycopg2-binary

# Expor a porta padrão usada pelo MLflow
EXPOSE 5000

# Definir diretório de trabalho para MLflow 
WORKDIR /mlflow

# Comando para iniciar o servidor MLflow automaticamente
CMD ["mlflow", "server", "--backend-store-uri", "postgresql://mlflow_user:mlflow_password@db:5432/mlflow_db", "--host", "0.0.0.0", "--port", "5000"]
