import mlflow
from mlflow.tracking import MlflowClient

# Conectar ao cliente MLflow
client = MlflowClient()

# Identificar o melhor modelo (pode ser com base nas métricas de interesse)
best_run_id = "6cb1a0711fdf43858f380affc676dc38"  # Substitua com o ID da melhor run

# Registrar o modelo da melhor run
model_uri = f"runs:/{best_run_id}/model"
mlflow.register_model(model_uri, "Diabetes_Detection")

# Obter a versão mais recente do modelo registrado
model_version = client.get_latest_versions("Diabetes_Detection")[0].version

# Promover o modelo para produção
client.transition_model_version_stage(
    name="Diabetes_Detection",
    version=model_version,
    stage="Production"
)