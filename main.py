import sys
import pickle
from pathlib import Path
import mlflow
import pandas as pd
import json
from mlflow.client import MlflowClient

# Configurar paths
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR / "src"))
sys.path.append(str(BASE_DIR / "data"))

from data.load_data import load_data
from features.feature_engineering import custom_cut
from evaluation.metrics import get_ks
from utils import get_summary
from mlflow.tracking import MlflowClient


def list_registered_models():
    """
    Lista todos os nomes de modelos registrados no MLflow Model Registry.
    
    Returns:
        list: Lista de nomes dos modelos registrados.
    """
    client = MlflowClient()
    registered_models = client.search_registered_models()
    model_names = [model.name for model in registered_models]
    return model_names


def get_run_ids_by_experiment_name(experiment_name):
    """Retorna uma lista de run_ids dado o nome do experimento."""
    try:
        client = MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        
        if experiment is None:
            print(f"Experimento '{experiment_name}' não encontrado.")
            return []
        
        runs = client.search_runs(experiment_ids=[experiment.experiment_id])
        run_ids = [run.info.run_id for run in runs]
        
        return run_ids
    
    except Exception as e:
        print(f"Erro ao buscar runs do experimento '{experiment_name}': {e}")
        return []

def load_artifact_from_mlflow(run_id, artifact_path):
    """Carrega um artefato salvo no MLflow dado o run_id e o artifact_path."""
    try:
        client = MlflowClient()
        # Download artifact
        artifact_uri = client.download_artifacts(run_id, artifact_path, dst_path='/tmp')
        with open(artifact_uri, 'rb') as f:
            artifact = json.load(f)
        return artifact
    
    except Exception as e:
        print(f"Erro ao baixar o artefato '{artifact_path}': {e}")
        sys.exit(1)
    
    
        
if __name__ == "__main__":
    import mlflow



    model_name = "diabetes_detection"
    tags = {"ml": "modelagem", "app": "diabetes"}
    desc = "Modelo para detecção de diabetes"

    client = mlflow.MlflowClient()

    try:
        print("Verificando se o modelo já está registrado...")
        registered_model = client.get_registered_model(model_name)
        print(f"Modelo '{model_name}' encontrado.")
        print(f"name: {registered_model.name}")
        print(f"tags: {registered_model.tags}")
        print(f"description: {registered_model.description}")
    except Exception as e:
        print(f"Modelo '{model_name}' não encontrado.")
        print(f"Registrando modelo")
        client.create_registered_model(model_name, tags, desc)
        registered_model = client.get_registered_model(model_name)
        print(f"name: {registered_model.name}")
        print(f"tags: {registered_model.tags}")
        print(f"description: {registered_model.description}")



    # # Configuração inicial
    # data_path = BASE_DIR / "data" / "raw" / "diabetes.csv"
    # df = load_data(data_path)
    
    # features = df.columns[:-1]
    # label = df.columns[-1]
    # train_index = df.sample(frac=0.8, random_state=42).index
    # test_index = df.drop(train_index).index
    
    # # Configurar o MLflow
    # mlflow.set_tracking_uri("http://0.0.0.0:5002/")
    # experiment_name = "diabetes_modeling"
    
    # # Carregar artefatos
    # print("Carregando artefatos do MLflow...")
    # runid = '62b821f9c3654f32a17e129abdea6d72'
    # bins_dict = load_artifact_from_mlflow(run_id=runid, artifact_path="feature_engineering/bins.json")
    # woe_dict = load_artifact_from_mlflow(run_id=runid, artifact_path="feature_engineering/woe.json")

    # print("Artefatos carregados com sucesso!")
    
    # print("Carregando modelo...")
    # # para carregar diretamente e fazer predict proba como a seguir utilize o log_personalizado ao invés do autolog
    # model_name = "models:/diabetes_detection"
    # model_version = 2
    # model = mlflow.xgboost.load_model(model_uri=f"{model_name}/{model_version}")
    
    # # print("Modelo carregado com sucesso!")
    
    # # # Transformar os dados
    # # for feat in features:
    # #     df[feat] = custom_cut(df, feat, bins_dict[feat])
    # #     df[feat] = df[feat].map(woe_dict[feat])
    
    # # print("Dados transformados com sucesso!")
    # # print(get_summary(df))
    
    # # # Previsão com o modelo carregado
    # # y_pred = model.predict_proba(df[features])[:, 1]
    # # print(y_pred)
    
    # # # Avaliação do modelo
    # # df_result = pd.DataFrame({'y': df[label], 'y_pred': y_pred})
    # # ks_train = get_ks(df=df_result.loc[train_index], proba_col='y_pred', true_value_col='y')
    # # ks_test = get_ks(df=df_result.loc[test_index], proba_col='y_pred', true_value_col='y')
    
    # # print(f"KS no conjunto de treino: {ks_train}")
    # # print(f"KS no conjunto de teste: {ks_test}")
    # # print(model)

