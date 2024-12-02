import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from scipy.stats import ks_2samp
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.model_selection import StratifiedKFold
from typing import Dict, Any


def get_ks(df: pd.DataFrame, proba_col: str, true_value_col: str) -> float:
    """
    Calculate the Kolmogorov-Smirnov (KS) statistic for the given DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the true labels and predicted probabilities.
        proba_col (str): Column name for predicted probabilities.
        true_value_col (str): Column name for true labels.

    Returns:
        float: KS statistic.
    """
    class0 = df[df[true_value_col] == 0]
    class1 = df[df[true_value_col] == 1]
    ks = ks_2samp(class0[proba_col], class1[proba_col])
    return ks.statistic

class HyperparameterOptimizer:
    def __init__(self, model_class: Any, param_space: Dict[str, Any], n_folds: int = 5, max_evals: int = 50, random_state: int = 42):
        """
        Initialize the hyperparameter optimizer.

        Args:
            model_class (Any): The model class to be optimized.
            param_space (Dict[str, Any]): The hyperparameter space for optimization.
            n_folds (int): Number of cross-validation folds. Default is 5.
            max_evals (int): Number of optimization evaluations. Default is 50.
            random_state (int): Random state for reproducibility. Default is 42.
        """
        self.model_class = model_class
        self.param_space = param_space
        self.n_folds = n_folds
        self.max_evals = max_evals
        self.random_state = random_state

    def objective(self, params: Dict[str, Any], X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Objective function for hyperparameter optimization.

        Args:
            params (Dict[str, Any]): Hyperparameters to be evaluated.
            X (pd.DataFrame): Training feature data.
            y (pd.Series): Training target data.

        Returns:
            Dict[str, Any]: Dictionary containing the loss and status.
        """
        valid_ks = []
        strat = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        for train_index, valid_index in strat.split(X, y):
            x_train, x_valid = X.iloc[train_index], X.iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

            clf = self.model_class(**params)
            clf.fit(x_train, y_train)

            preds_valid = clf.predict_proba(x_valid)[:, 1]
            df_classifier = pd.DataFrame({'y_true': y_valid, 'score': preds_valid})
            ks_ = get_ks(df=df_classifier, proba_col='score', true_value_col='y_true')
            valid_ks.append(ks_)

        mean_ks = np.mean(valid_ks)
        return {'loss': -mean_ks, 'status': STATUS_OK}

    def optimize(self, X: pd.DataFrame, y: pd.Series) -> Trials:
        """
        Run the hyperparameter optimization process.

        Args:
            X (pd.DataFrame): Training feature data.
            y (pd.Series): Training target data.

        Returns:
            Trials: The Trials object containing the optimization results.
        """
        trials = Trials()
        best = fmin(fn=lambda params: self.objective(params, X, y),
                    space=self.param_space,
                    algo=tpe.suggest,
                    max_evals=self.max_evals,
                    trials=trials,
                    rstate=np.random.RandomState(self.random_state))
        print("Melhores hiperparâmetros:", best)
        return trials

    def _suggest_value(self, param_info: Dict[str, Any]) -> Any:
        """
        Suggest a value for a given hyperparameter.

        Args:
            param_info (Dict[str, Any]): Information about the hyperparameter space.

        Returns:
            Any: Suggested value for the hyperparameter.
        """
        if param_info['type'] == 'int':
            return hp.quniform(param_info['name'], param_info['low'], param_info['high'], 1)
        elif param_info['type'] == 'float':
            return hp.loguniform(param_info['name'], np.log(param_info['low']), np.log(param_info['high']))
        else:
            raise ValueError(f"Unsupported parameter type for {param_info['name']}")

