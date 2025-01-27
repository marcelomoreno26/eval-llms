import datasets
import evaluate



_CITATION = ""



_DESCRIPTION = """
This metric computes accuracy for multiple-choice or test-type tasks by comparing the predicted answers to the reference answers.
It checks for exact equality between the predictions and references after converting both to lowercase and stripping extra spaces.
"""



_KWARGS_DESCRIPTION = """
Compute the accuracy for multiple-choice or test-type tasks.
Args:
    predictions: list of predictions.
    references: list of references.
Returns: float
        Accuracy score.
Examples:

        >>> references = ["A", "B", "C"]
        >>> predictions = ["A", "D", "C"]
        >>> metric = AccuracyMetric()
        >>> score = metric.compute(predictions=predictions, references=references)
        >>> print(score)
            0.6667
"""



class Accuracy(evaluate.Metric):
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
        )

    def _compute(self, predictions: list[str], references: list[str]) -> float:
        correct_predictions = sum([prediction.lower().strip() == reference.lower().strip() for prediction, reference in zip(predictions, references)])
        accuracy = correct_predictions / len(predictions)
        
        return float(accuracy)