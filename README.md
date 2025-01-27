# eval-llms
`eval-llms` is a library that provides various useful metrics to evaluate LLMs and has been created to facilitate the creation of benchmarks 
with old, recent and upcoming models.

The metrics follow the same interface as those of Hugging Face's [Evaluate](https://huggingface.co/docs/evaluate/index) library.

## Instructions
1. Follow [uv](https://github.com/astral-sh/uv) library installation instructions.
2. Clone this repo.
3. Run command on terminal to setup venv
```bash
uv sync
```
4. Evaluate you models with any of the metrics available.


## Example Usage

```python
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
print(score)
```

## Available Metrics
### [Biencoder Similarity Score](https://arxiv.org/abs/2108.06130)
Computes the cosine similarity (ranges from -1  to 1) between prediction and reference
after encoding them with a sentence transformer model. 

### [Semantic Answer Similarity (SAS)](https://arxiv.org/abs/2108.06130)
SAS utilizes a cross-encoder model where prediction and reference are joined together with a separator token.
The model then generates a similarity score ranging from 0 to 1.

### [SAS Ensemble](https://arxiv.org/abs/2108.06130)
Ensemble of Semantic Answer Similarity and BiEncoder metrics.
Improves assessment by applying cross-encoder and bi-encoder models and averages their results.

### [Prometheus Score](https://arxiv.org/abs/2405.01535)
The Prometheus Score is generated by one of the Prometheus Feedback collection models described in the
Prometheus 2 paper using the direct assesment task. In direct assessment, the model assigns a score to each response based on predefined criteria. 
In the case of this metric the Prometheus evaluates models responses based on five criteria: Veracity, Relevance, 
Completeness, Fidelity to Documentation, and Clarity and Coherence. Each response is scored from 1 to 10, 
reflecting its alignment with these criteria. The average score for all responses is calculated.

### Accuracy
Accuracy for multiple-choice or test-type tasks by comparing the predicted answers to the reference answers.
It checks for exact equality between the predictions and references.



