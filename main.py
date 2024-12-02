import sys
import pickle
from pathlib import Path
import mlflow
import pandas as pd

# Configurar paths
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR / "src"))
sys.path.append(str(BASE_DIR / "data"))

from data.load_data import load_data
from features.feature_engineering import custom_cut
from evaluation.metrics import get_ks
from utils import get_summary
from mlflow.client import MlflowClient
import json

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


def main():
    # Configuração inicial
    data_path = BASE_DIR / "data" / "raw" / "diabetes.csv"
    df = load_data(data_path)
    
    features = df.columns[:-1]
    label = df.columns[-1]
    train_index = df.sample(frac=0.8, random_state=42).index
    test_index = df.drop(train_index).index
    
    # Configurar o MLflow
    mlflow.set_tracking_uri("http://0.0.0.0:5002/")
    experiment_name = "diabetes_modeling"
    
    # Carregar artefatos
    print("Carregando artefatos do MLflow...")
    runid = '62b821f9c3654f32a17e129abdea6d72'
    artifact_path = "feature_engineering/bins.json"
    bins_dict = load_artifact_from_mlflow(run_id=runid, artifact_path=artifact_path)
    
    artifact_path = "feature_engineering/woe.json"
    woe_dict = load_artifact_from_mlflow(run_id=runid, artifact_path=artifact_path)

    print("Artefatos carregados com sucesso!")
    
    print("Carregando modelo...")
    model_name = "diabetes_detection"
    model_version = 1
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
    
    print("Modelo carregado com sucesso!")
    
    # Transformar os dados
    for feat in features:
        df[feat] = custom_cut(df, feat, bins_dict[feat])
        df[feat] = df[feat].map(woe_dict[feat])
    
    print("Dados transformados com sucesso!")
    print(get_summary(df))
    
    # Previsão com o modelo carregado
    y_pred = model.predict(df[features])
    
    # Avaliação do modelo
    df_result = pd.DataFrame({'y': df[label], 'y_pred': y_pred})
    ks_train = get_ks(df=df_result.loc[train_index], proba_col='y_pred', true_value_col='y')
    ks_test = get_ks(df=df_result.loc[test_index], proba_col='y_pred', true_value_col='y')
    
    print(f"KS no conjunto de treino: {ks_train}")
    print(f"KS no conjunto de teste: {ks_test}")


if __name__ == "__main__":
    main()

