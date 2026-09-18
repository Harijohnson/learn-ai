# Step 3 · Text classification (spam vs. ham)

The first real-world task: decide whether an SMS message is **spam** or **ham**
(not spam). This is where we learn to turn **text into numbers** a model can use —
first with a classic scikit-learn model, then with a PyTorch neural network.

## Files (read in this order)

| File | What it shows |
| ---- | ------------- |
| `classifier.py` | The simplest starter: a bag-of-words `CountVectorizer` + `LogisticRegression` on a handful of sentences. |
| `train.py` | Trains logistic regression on the full `data/spam.csv`, prints what it learned, and **saves** the model + vectorizer into `data/`. |
| `predict.py` | Loads the saved model and classifies a brand-new message. |
| `text_nn.py` | The same spam problem, solved with a **PyTorch neural network** on TF-IDF features (mini-batches, training loop, evaluation). |

## Run

```powershell
python 03_text_classification/classifier.py
python 03_text_classification/train.py      # creates data/spam_model.pkl + data/vectorizer.pkl
python 03_text_classification/predict.py     # uses the saved model
python 03_text_classification/text_nn.py     # creates data/spam_nn_model.pth
```

## What to look for

- The **vocabulary** and per-word **weights** the model learns (spammy words get positive weights).
- The **accuracy** and **confusion matrix** on the held-out test messages.
- `predict.py` labelling a new message as `spam`/`ham` with a probability.

## Data used / produced (in `data/`)

- **Input:** `spam.csv`
- **Produced:** `spam_model.pkl`, `vectorizer.pkl` (from `train.py`);
  `tfidf_vectorizer.pkl`, `spam_nn_model.pth` (from `text_nn.py`)

⬅️ Prev: [Step 2 · Neural networks](../02_neural_networks/)
➡️ Next: [Step 4 · Embeddings & tokens](../04_embeddings_and_tokens/)
