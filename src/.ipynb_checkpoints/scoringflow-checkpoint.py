from metaflow import FlowSpec, step
import mlflow
import pandas as pd
import dataprocessing

class ScoringFlow(FlowSpec):

    @step
    def start(self):
        self.df = dataprocessing.load_data("data/adult.data")
        self.next(self.preprocess)

    @step
    def preprocess(self):
        df = dataprocessing.preprocess_data(self.df)
        self.X = df.drop("income_>50K", axis=1)
        self.y = df["income_>50K"]
        self.next(self.predict)

    @step
    def predict(self):
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        model_uri = "models:/BestIncomeModel/Production"
        self.model = mlflow.sklearn.load_model(model_uri)
        self.predictions = self.model.predict(self.X)
        self.next(self.end)

    @step
    def end(self):
        print("Sample predictions:", self.predictions[:10])

if __name__ == "__main__":
    ScoringFlow()
