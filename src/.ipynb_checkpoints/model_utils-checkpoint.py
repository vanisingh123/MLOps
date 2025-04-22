from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn

def train_model(X, y, params):
    model = RandomForestClassifier(**params)
    model.fit(X, y)
    return model

def log_model(model, params, model_name="best_model"):
    mlflow.set_tracking_uri("http://localhost:5000")
    with mlflow.start_run():
        mlflow.sklearn.log_model(model, model_name)
        mlflow.log_params(params)
