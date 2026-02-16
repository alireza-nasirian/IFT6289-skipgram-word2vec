# Skip-gram Word2Vec

**IFT6289-A-H26 Assignment 1** — Universit&eacute; de Montr&eacute;al

Implementation of the skip-gram word2vec model trained on the Penn Treebank dataset using stochastic gradient descent with negative sampling.

## Project Structure

```
.
├── word2vec.py          # Skip-gram model: naive softmax, negative sampling, skipgram
├── sgd.py               # Stochastic gradient descent optimizer
├── data_process.py      # Penn Treebank dataset loader and preprocessing
├── run.py               # Training script (35K iterations)
├── utils/
│   └── utils.py         # Sigmoid, softmax, negative sampling, visualization
├── data/
│   └── penntreebank/
│       └── sentences.txt
├── requirements.txt
└── IFT6289_A1_report.pdf
```

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.5+ with `numpy` and `matplotlib`.

## Training

```bash
python run.py
```

This trains 10-dimensional word vectors for 35,000 iterations using:
- **Skip-gram** model with context window size C=5
- **Negative sampling** loss (K=10 negative samples)
- **SGD** with learning rate 0.2 (annealed by 0.5x every 20K iterations)

Training takes approximately 30--60 minutes on CPU. Checkpoints and word vector visualizations are saved every 5,000 iterations.

The final loss should converge to around or below **9**.

## Implemented Functions

| Function | File | Description |
|----------|------|-------------|
| `sigmoid()` | `utils/utils.py` | Numerically stable sigmoid activation |
| `naive_softmax_loss_and_gradient()` | `word2vec.py` | Full-vocabulary softmax loss and gradients |
| `neg_sampling_loss_and_gradient()` | `word2vec.py` | Negative sampling loss and gradients |
| `skipgram()` | `word2vec.py` | Skip-gram model aggregating loss over context window |
| `sgd()` | `sgd.py` | Vanilla SGD parameter update |

## Results

Training converges to a loss of approximately **7.89** after 35,000 iterations. The learned word vectors capture semantic relationships visible in 2D PCA projections:

- Sentiment words cluster together (e.g., *great*, *good*, *well*)
- Gender-related structure is captured (*queen*/*female* vs. *king*/*man*)
- Progressive refinement of clusters from early to late iterations

## References

Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013). *Distributed representations of words and phrases and their compositionality*. Advances in Neural Information Processing Systems, 26.
