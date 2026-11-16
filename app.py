import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, FiniteFloat

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
model = mlflow.sklearn.load_model("model")


class Input(BaseModel):
    instances: list[list[FiniteFloat]] = Field(min_length=1)


@app.post("/predict")
def predict(data: Input):
    if any(len(row) != 64 for row in data.instances):
        raise HTTPException(422, "Each sample needs 64 pixel values.")
    return {"predictions": model.predict(data.instances).tolist()}
