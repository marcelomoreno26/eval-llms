import torch
import datasets
import evaluate
from torch.nn import CosineSimilarity
from sentence_transformers import SentenceTransformer



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
The BiEncoder score computes the cosine similarity (ranges from -1  to 1) between prediction and reference
after encoding them with a sentence transformer model. 
"""



_KWARGS_DESCRIPTION = """
Computes BiEncoder Score.
Arguments:
    predictions: A list of predicted text sequences.
    references: A list of reference text sequences.
    batch_size: An integer specifying the batch size for embedding computation.
    model_name: A string with the name of the Sentence Transformer to be used.


Returns: A float value representing the mean similarity score across all text pairs (BiEncoder Score).

Example:
    >>> references = ["El sol brilla en el cielo.", "Las bicicletas son ecológicas.", "El café es una bebida popular."]
    >>> predictions = ["El sol está en el cielo.", "Las bicicletas son buenas para el ambiente.", "El café es adictivo."]
    >>> metric = BiEncoderScore()
    >>> score = metric.compute(predictions=predictions, references=references, batch_size=4)
    >>> print(score)
        0.9248
"""



@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class BiEncoderScore(evaluate.Metric):
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


    def _compute(self, predictions: list[str], references: list[str], batch_size: int, model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2") -> float:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        metric = CosineSimilarity(dim=1)
        model = SentenceTransformer(model_name, device=device)

        predictions_embeddings = model.encode(predictions, batch_size=batch_size, convert_to_tensor=True)
        references_embeddings = model.encode(references, batch_size=batch_size, convert_to_tensor=True)
        
        similarities = metric(predictions_embeddings, references_embeddings)
        
        return similarities.mean().item()

