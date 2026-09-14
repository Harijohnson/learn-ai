# ai-learn

My playground for learning AI and machine learning with Python. 🧠

This repo holds small experiments as I work through the basics — starting with a
simple text sentiment classifier built using [scikit-learn](https://scikit-learn.org/).

## What's inside

| File            | What it does                                                        |
| --------------- | ------------------------------------------------------------------- |
| `classifier.py` | A tiny sentiment classifier that labels text as positive/negative.  |
| `script.py`     | Scratch file for quick experiments.                                 |

## How `classifier.py` works

It's a first look at how a machine learning model learns from examples:

1. **Training data** — a few example sentences, each tagged `positive` or `negative`.
2. **CountVectorizer** — turns the words into numbers the model can understand
   (a "bag of words").
3. **LogisticRegression** — the model learns which words point to which label.
4. **Predict** — feed it a new sentence and it guesses the sentiment.

## Getting started

### 1. Set up the virtual environment

```powershell
# Create it (only needed once)
python -m venv venv

# Activate it (Windows PowerShell)
venv\Scripts\Activate.ps1
```

### 2. Install the dependencies

```powershell
pip install scikit-learn
```

### 3. Run the classifier

```powershell
python classifier.py
```

You'll see the vocabulary it learned, the word/number matrix, the model's
learned weights, and finally the prediction for the test sentence
`"I love this product"`.

## Ideas to try next

- Add more training sentences and see if predictions improve.
- Try a `"neutral"` label alongside positive and negative.
- Print the model's confidence with `model.predict_proba()`.
- Swap in a different model (like `MultinomialNB`) and compare results.

## Notes

- Built with Python and scikit-learn.
- `venv/` is git-ignored — each machine creates its own.
