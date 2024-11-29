import mlflow

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
