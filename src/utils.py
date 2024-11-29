import pandas as pd
import mlflow

def get_summary(df: pd.DataFrame):
    df_summary = pd.DataFrame(columns = ['clumn_dtype', 'na', 'na_pct', 'top_class', 'top_class_pct', 'nunique', 'unique_values'])
    for col in df.columns:
        df_summary.at[col, 'clumn_dtype'] = df[col].dtype
        df_summary.at[col, 'na'] = df[col].isna().sum()
        df_summary.at[col, 'na_pct'] = df[col].isna().sum() / len(df)*100
        df_summary.at[col, 'top_class'] = df[col].value_counts().index[0]
        df_summary.at[col, 'top_class_pct'] = df[col].value_counts().values[0] / len(df)*100
        df_summary.at[col, 'nunique'] = df[col].nunique()
        if df[col].nunique() < 10:
            df_summary.at[col, 'unique_values'] = df[col].unique().tolist()
        else:
            df_summary.at[col, 'unique_values'] = '...'
    return df_summary



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
