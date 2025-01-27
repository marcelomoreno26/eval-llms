from evaluation_metrics import SASEnsemble

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

models = ["sentence-transformers/paraphrase-multilingual-mpnet-base-v2", "cross-encoder/stsb-roberta-large"]

metric = SASEnsemble()
results = metric.compute(models=models, predictions=predictions, references=references, batch_size=4)

assert isinstance(results, float)
assert -1 <= results <= 1
