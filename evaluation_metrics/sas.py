import torch
import datasets
import evaluate
import numpy as np
from sentence_transformers import CrossEncoder



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
SAS utilizes a cross-encoder model where prediction and reference are joined together with a separator token.
The model then generates a similarity score ranging from 0 to 1.
"""



_KWARGS_DESCRIPTION = """
Compute Semantic Answer Similarity with a Cross Encoder.
Args:
    predictions: list of predictions.
    references: list of references.
    batch_size: An integer specifying the batch size for embedding computation.
    model_name: A string with the name of the Cross Encoder to be used.
Returns: float
        SAS score.
Examples:


        >>> references = ["El sol brilla en el cielo.", "Las bicicletas son ecológicas.", "El café es una bebida popular."]
        >>> predictions = ["El sol está en el cielo.", "Las bicicletas son buenas para el ambiente.", "El café es adictivo."]
        >>> metric = SemanticAnswerSimilarity()
        >>> score = metric.compute(predictions=predictions, references=references, batch_size=4)
        >>> print(score)
            0.7047
"""



@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class SemanticAnswerSimilarity(evaluate.Metric):
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


    def _compute(self, predictions: list[str], references: list[str], batch_size: int, model_name: str = "cross-encoder/stsb-roberta-large") -> float:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = CrossEncoder(model_name, device=device)
        pairs = []
        
        for prediction, reference in zip(predictions, references):
            pairs.append([prediction, reference])
        
        scores = model.predict(pairs, batch_size=batch_size)
        
        return float(scores.mean())
