import torch
import datasets
import evaluate
import numpy as np
from transformers import AutoConfig
from .sas import SemanticAnswerSimilarity
from .biencoder import BiEncoderScore



_CITATION = """\
@misc{risch2021semantic,
      title={Semantic Answer Similarity for Evaluating Question Answering Models},
      author={Julian Risch and Timo Möller and Julian Gutsch and Malte Pietsch},
      year={2021},
      eprint={2108.06130},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""



_DESCRIPTION = """\
Ensemble of Semantic Answer Similarity and BiEncoder metrics.
Improves assessment by applying cross-encoder and bi-encoder models and averages their results.
"""



_KWARGS_DESCRIPTION = """
Compute Ensemble of Semantic Answer Similarity and BiEncoder metric.
Args:
    predictions: list of predictions.
    references: list of references.
    batch_size: An integer specifying the batch size for embedding computation.
    models : List of cross-encoder/bi-encoder model names or paths to be used for ensemble evaluation.

Returns: float
       The average score of  SAS/BiEncoder metrics.
Examples:

        >>> references = ["El sol brilla en el cielo.", "Las bicicletas son ecológicas.", "El café es una bebida popular."]
        >>> predictions = ["El sol está en el cielo.", "Las bicicletas son buenas para el ambiente.", "El café es adictivo."]
        >>> models = ["sentence-transformers/paraphrase-multilingual-mpnet-base-v2", "cross-encoder/stsb-roberta-large"]
        >>> metric = SASEnsemble()
        >>> score = metric.compute(models=models, predictions=predictions, references=references, batch_size=4)
        >>> print(score)
            0.8148
"""



@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class SASEnsemble(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value(dtype="string"),
                    "references": datasets.Value(dtype="string"),
                },
            ),
            reference_urls=["https://arxiv.org/abs/2108.06130"],
        )


    def _compute(self, models: list[str], predictions: list[str], references: list[str], batch_size: int) -> float:
        scores = []

        for model in models:
            model_config = AutoConfig.from_pretrained(model)

            if model_config.architectures is not None:
                is_cross_encoder = any(architectures.endswith("ForSequenceClassification") for architectures in model_config.architectures)

            if is_cross_encoder:
                metric = SemanticAnswerSimilarity(model_name=model)
            else:
                metric = BiEncoderScore(model_name=model)

            score = metric.compute(predictions=predictions, references=references, batch_size=batch_size)
            scores.append(score)
        
        return float(np.mean(scores))
