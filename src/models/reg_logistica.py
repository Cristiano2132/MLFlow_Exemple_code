from sklearn.linear_model import LogisticRegression

def build_logistic_model(X, y):
    """Construir e treinar o modelo de regressão logística"""
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)
    return model