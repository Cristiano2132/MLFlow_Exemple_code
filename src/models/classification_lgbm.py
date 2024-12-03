from lightgbm import LGBMClassifier
from models.bayesian_opt import HyperparameterOptimizer


def build_lgbm_model(X, y):
    
    param_space = {
        'max_depth': {'type': 'int', 'low': 2, 'high': 10},
        'num_leaves': {'type': 'int', 'low': 10, 'high': 50},
        'learning_rate': {'type': 'float', 'low': 1e-3, 'high': 1e-1, 'log': True},
        'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
        'min_child_samples': {'type': 'int', 'low': 10, 'high': 50},
        'subsample': {'type': 'float', 'low': 0.2, 'high': 1.0},
        'colsample_bytree': {'type': 'float', 'low': 0.5, 'high': 1.0},
    }

    # optimizer = HyperparameterOptimizer(LGBMClassifier, param_space)
    # study = optimizer.optimize(X, y)
    # best_params = study.best_params
    model = LGBMClassifier()
    model.fit(X, y)
    return model
