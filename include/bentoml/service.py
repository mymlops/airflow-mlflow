import bentoml

runner = bentoml.mlflow.get("toxic-comment-classifier:latest").to_runner()

svc = bentoml.Service('toxic-comment-classifier', runners=[runner])


@svc.api(input=bentoml.io.Text(), output=bentoml.io.NumpyNdarray())
def predict(input_text: str):
    return runner.predict.run([input_text])[0]
