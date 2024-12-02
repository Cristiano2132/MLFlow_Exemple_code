import os
import sys
from pathlib import Path
import pandas as pd
import json
import pickle
import mlflow
import mlflow.sklearn

# Configurar paths
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "src"))
sys.path.append(str(BASE_DIR / "data"))

from data.load_data import load_data
from features.feature_engineering import TreeCategizer, WOEEncoder, custom_cut
from models.reg_logistica import build_logistic_model
from evaluation.metrics import get_report_metrics
from utils import get_summary



def set_experiment(exp_name: str):
    # Verifica se o experimento já existe
    experiment = mlflow.get_experiment_by_name(exp_name)
    if experiment is None:
        # Se não existir, cria um novo
        mlflow.create_experiment(exp_name)
        print(f"Experimento '{exp_name}' criado.")
    else:
        print(f"Usando experimento existente: '{exp_name}'")
    
    # Define o experimento como o ativo
    mlflow.set_experiment(exp_name)


if __name__ == "__main__":
    
    # Configurar URI do MLflow
    mlflow.set_tracking_uri("http://0.0.0.0:5002/")
    exp_name = "diabetes_reg_logistica"
    set_experiment(exp_name)
    # Habilitar autolog
    mlflow.sklearn.autolog()  # Não registra o modelo automaticamente no registry
    # if there is active run stop it
    if mlflow.active_run():
        mlflow.end_run()
    with mlflow.start_run(run_name="novo_modelo_lgbm"):
        # Configuração inicial
        data_path = BASE_DIR / "data" / "raw" / "diabetes.csv"
        df = load_data(data_path)
        
        # Pré-processamento
        features = df.columns[:-1]
        label = df.columns[-1]
        train_index = df.sample(frac=0.8, random_state=42).index
        test_index = df.drop(train_index).index
        
        print("Resumo inicial dos dados:")
        print(get_summary(df))
        
        
        # Feature engineering
        bins_dict = {}
        tc = TreeCategizer(df=df.loc[train_index].dropna())
        for feat in features:
            bins_dict[feat] = tc.get_splits(target_column=label, feature_column=feat)
        
        for feat in features:
            df[feat] = custom_cut(df, feat, bins_dict[feat])
        
        print("Resumo após TreeCategizer e custom_cut:")
        print(get_summary(df))
        
        encoder = WOEEncoder(df.loc[train_index], label)
        for feat in features:
            encoder.fit(feat)
            df[feat] = df[feat].map(encoder.woe_dict.get(feat))
        
        print("Resumo após WOE encoding:")
        print(get_summary(df))
        
        # Logando os artefatos
        mlflow.log_dict(bins_dict, "feature_engineering/bins.json")
        mlflow.log_dict(encoder.woe_dict, "feature_engineering/woe.json")
        
        # Treinamento do modelo
        model = build_logistic_model(df.loc[train_index][features], df.loc[train_index][label])
        print("Modelo treinado com sucesso!")
        
        # Salvando o modelo como artefato
        model_path = BASE_DIR / "logistic_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        mlflow.log_artifact(str(model_path), artifact_path="models")
        os.remove(model_path)
        
        y_pred = model.predict(df[features])
        
        df_result = pd.DataFrame({'y': df[label], 'y_pred': y_pred})
        report_train = get_report_metrics(df=df_result, proba_col='y_pred', true_value_col='y', base="train")

        y_pred = model.predict(df.loc[test_index][features])
        df_result = pd.DataFrame({'y': df.loc[test_index][label], 'y_pred': y_pred})
        report_test = get_report_metrics(df=df_result, proba_col='y_pred', true_value_col='y', base="test")
        report = {}
        report.update(report_train)
        report.update(report_test)
        mlflow.log_metrics(report)
    print("Run finalizado com sucesso!")

