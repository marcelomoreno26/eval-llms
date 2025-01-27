# llm-eval
`eval-llms` is library that provides various useful metrics to evaluate LLMs and has been created to facilitate the creation of benchmarks 
with old, recent and upcoming models.

The metrics follow the same interface as those of Hugging Face's [Evaluate](https://huggingface.co/docs/evaluate/index) library.

## Installation


## How to Use


## Available Metrics
### Biencoder Similarity Score
Computes the cosine similarity (ranges from -1  to 1) between prediction and reference
after encoding them with a sentence transformer model. 

### Semantic Answer Similarity (SAS)
SAS utilizes a cross-encoder model where prediction and reference are joined together with a separator token.
The model then generates a similarity score ranging from 0 to 1.

### SAS Ensemble
Ensemble of Semantic Answer Similarity and BiEncoder metrics.
Improves assessment by applying cross-encoder and bi-encoder models and averages their results.

### Accuracy
Accuracy for multiple-choice or test-type tasks by comparing the predicted answers to the reference answers.
It checks for exact equality between the predictions and references.


