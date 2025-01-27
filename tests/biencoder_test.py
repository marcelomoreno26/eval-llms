from evaluation_metrics import BiEncoderScore


references = [
    "El sol brilla en el cielo.",
    "Las bicicletas son ecológicas.",
    "El café es una bebida popular."
]

predictions = [
    "El sol está en el cielo.",
    "Las bicicletas son buenas para el ambiente.",
    "El café es adictivo."
]

metric = BiEncoderScore()
score = metric.compute(predictions=predictions, references=references, batch_size=4)
assert isinstance(score, float)
assert -1 <= score <= 1
