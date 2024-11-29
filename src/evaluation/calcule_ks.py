import pandas as pd
from scipy.stats import ks_2samp


def get_ks(df: pd.DataFrame, proba_col: str, true_value_col: str):
    
    # Recover each class
    class0 = df[df[true_value_col] == 0]
    class1 = df[df[true_value_col] == 1]
    ks = ks_2samp(class0[proba_col], class1[proba_col])
    return ks.statistic
