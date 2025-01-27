from evaluation_metrics import Accuracy

references = ["A", "B", "C"]
predictions = ["A", "D", "C"]
metric = Accuracy()
score = metric.compute(predictions=predictions, references=references)

assert isinstance(score, float)
assert 0 <= score <= 1