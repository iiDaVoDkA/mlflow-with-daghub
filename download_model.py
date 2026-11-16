import os
import shutil
import tempfile
from pathlib import Path

import mlflow
from mlflow import MlflowClient

os.environ["MLFLOW_TRACKING_USERNAME"] = "iiDaVoDkA"
os.environ["MLFLOW_TRACKING_PASSWORD"] = Path("/run/secrets/dagshub_token").read_text().strip()
mlflow.set_tracking_uri("https://dagshub.com/iiDaVoDkA/mlflow-with-daghub.mlflow")
client = MlflowClient()
models = sorted(client.search_logged_models(experiment_ids=["0"]),
                key=lambda model: model.creation_timestamp, reverse=True)
model = next(model for model in models if model.status == "READY"
             and model.source_run_id
             and client.get_run(model.source_run_id).info.status == "FINISHED")

with tempfile.TemporaryDirectory() as folder:
    source = mlflow.artifacts.download_artifacts(artifact_uri=model.model_uri, dst_path=folder)
    shutil.copytree(source, "model")
print("Downloaded latest successful model:", model.name)
