from evaluation_metrics import PrometheusScore


model_name = "meta-llama/Llama-3.1-8B-Instruct"

predictions = [
        "El sol brilla debido a reacciones de fusión nuclear en su núcleo.",
        "La capital de Francia es París."
    ]

references = [
    "El sol produce luz y calor mediante reacciones de fusión nuclear en su núcleo.",
    "París es la capital de Francia."
    ]

contexts = [
    "El Sol emite energía en forma de luz y calor, generada a través de reacciones de fusión nuclear en su núcleo.",
    "Francia es un país europeo cuya capital es París."
    ]

metric = PrometheusScore()
score = metric.compute(model_name=model_name, predictions=predictions, references=references, contexts=contexts)

assert isinstance(score, float)
assert 1 <= score <= 10