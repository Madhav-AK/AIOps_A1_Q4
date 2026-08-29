import subprocess
import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


SEED = 0

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("q4-reproducibility")

data = pd.read_csv("data/iris.csv")
X = data.drop(columns=["target"])
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)


def train_and_evaluate(n_estimators=100, max_depth=5, min_samples_split=2):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=SEED,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")
    return model, acc, f1


def train_and_log(n_estimators=100, max_depth=5, min_samples_split=2, run_name=None):
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)
        mlflow.log_param("seed", SEED)

        model, acc, f1 = train_and_evaluate(n_estimators, max_depth, min_samples_split)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1)
        mlflow.set_tag("git_commit", subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())

        mlflow.sklearn.log_model(model, artifact_path="model")
        run_id = mlflow.active_run().info.run_id
        print(f"Logged run {run_id} | acc={acc:.4f} f1={f1:.4f}")
        return run_id


baseline_run_id = train_and_log(100, 5, 2, run_name="rf-baseline")

registered_model = mlflow.register_model(
    model_uri=f"runs:/{baseline_run_id}/model",
    name="q4-iris-random-forest",
)

client = MlflowClient()
client.transition_model_version_stage(
    name="q4-iris-random-forest",
    version=registered_model.version,
    stage="Staging",
)

print(f"Registered model version: {registered_model.version}")
print("Model stage: Staging")