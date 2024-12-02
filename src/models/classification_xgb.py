from xgboost import XGBClassifier
from models.bayesian_opt import HyperparameterOptimizer

def build_xgb_model(X, y):
    
    param_space = {
        'max_depth': {'type': 'int', 'low': 3, 'high': 10},
        'learning_rate': {'type': 'float', 'low': 1e-3, 'high': 1e-1, 'log': True},
        'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
        'min_child_weight': {'type': 'int', 'low': 1, 'high': 10},
        'subsample': {'type': 'float', 'low': 0.5, 'high': 1.0},
        'colsample_bytree': {'type': 'float', 'low': 0.5, 'high': 1.0},
        'gamma': {'type': 'float', 'low': 0, 'high': 5},
        'reg_alpha': {'type': 'float', 'low': 0, 'high': 1},
        'reg_lambda': {'type': 'float', 'low': 0, 'high': 1},
    }

    # optimizer = HyperparameterOptimizer(XGBClassifier, param_space)
    # study = optimizer.optimize(X, y)
    # best_params = study.best_params
    model = XGBClassifier()
    model.fit(X, y)
    return model
