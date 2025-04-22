from metaflow import FlowSpec, step, Parameter
import mlflow
import os
import dataprocessing
import model

class TrainingFlow(FlowSpec):

    seed = Parameter("seed", default=42)

    @step
    def start(self):
        self.df = dataprocessing.load_data("data/adult.data")
        self.next(self.preprocess)

    @step
    def preprocess(self):
        df = dataprocessing.preprocess_data(self.df)
        self.X = df.drop("income_>50K", axis=1)
        self.y = df["income_>50K"]
        self.next(self.train)

    @step
    def train(self):
        self.clf, self.X_test, self.y_test = model.train_model(self.X, self.y, seed=self.seed)
        self.score = model.evaluate_model(self.clf, self.X_test, self.y_test)
        self.next(self.register)

    @step
    def register(self):
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment("IncomePrediction")

        with mlflow.start_run():
            mlflow.log_param("seed", self.seed)
            mlflow.log_metric("accuracy", self.score)
            mlflow.sklearn.log_model(self.clf, artifact_path="model", registered_model_name="BestIncomeModel")

        self.next(self.end)

    @step
    def end(self):
        print(f"Model registered with accuracy: {self.score}")

if __name__ == "__main__":
    TrainingFlow()
