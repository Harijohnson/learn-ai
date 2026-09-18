# 🧠 AI Learning Journey (`ai-learn`)

A hands-on, **from-scratch** walk through modern AI — starting with a single
neuron whose maths you can do by hand, and ending with a working **Transformer
language model** that writes little stories.

Every numbered folder is **one step of the journey**. Read and run them in order
(`01` → `07`) and you'll see how the ideas stack on top of each other:

> a neuron → a network → text classification → embeddings → attention →
> the transformer block → a language model.

This repo is meant to be **read**. Each script is small, self-contained, and
heavily commented so that anyone — including future-me or someone I share it
with — can follow exactly what is happening and *why*.

---

## 📚 The journey at a glance

| Step | Folder | What you learn | Key idea |
| ---- | ------ | -------------- | -------- |
| 1 | [`01_neuron_basics/`](01_neuron_basics/) | A single neuron & gradient descent | "Learning" = nudging weights to reduce error |
| 2 | [`02_neural_networks/`](02_neural_networks/) | Stacking neurons into a network | Layers + ReLU can learn non-linear patterns |
| 3 | [`03_text_classification/`](03_text_classification/) | Turning **text** into numbers & classifying it | Bag-of-words / TF-IDF + a classifier (spam vs. ham) |
| 4 | [`04_embeddings_and_tokens/`](04_embeddings_and_tokens/) | Tokens & embeddings | Words become **learnable vectors** |
| 5 | [`05_attention/`](05_attention/) | Attention & positional info | Tokens learn **which other tokens to look at** |
| 6 | [`06_transformer/`](06_transformer/) | The Transformer block | Attention + feed-forward + residual + LayerNorm |
| 7 | [`07_language_model/`](07_language_model/) | A real language model | Predict the next token → **generate text** |

---

## 🗂️ Repository layout

```text
ai-learn/
├── 01_neuron_basics/          # the single neuron, by hand and in PyTorch
├── 02_neural_networks/        # a small classifier network
├── 03_text_classification/    # spam vs. ham (scikit-learn + a PyTorch net)
├── 04_embeddings_and_tokens/  # tokenizing text, learnable embeddings
├── 05_attention/              # Q/K/V attention, self-attention, positions
├── 06_transformer/            # the reusable Transformer block
├── 07_language_model/         # toy LM + the full TinyStories pipeline
├── data/                      # datasets + trained models (all committed)
├── _scratch/                  # old scratch files, kept for history
├── requirements.txt           # Python dependencies
├── .gitignore
└── README.md                  # you are here
```

Each stage folder also has its **own `README.md`** with the details for that step.

---

## ⚙️ Setup (one time)

This project uses a Python virtual environment. On **Windows PowerShell**:

```powershell
# 1. Create the virtual environment (only needed once)
python -m venv venv

# 2. Activate it (do this every new terminal)
venv\Scripts\Activate.ps1

# 3. Install the dependencies
pip install -r requirements.txt
```

> The `venv/` folder is git-ignored — every machine builds its own.

---

## ▶️ How to run a script

All scripts figure out where the `data/` folder is on their own, so you can run
them from **anywhere**. The simplest habit is to run from the repo root:

```powershell
python 01_neuron_basics/neuron.py
python 03_text_classification/predict.py
python 07_language_model/tinystories_6_generate.py
```

---

## 🧭 Stage-by-stage guide

### 1 · Neuron basics — [`01_neuron_basics/`](01_neuron_basics/)

The atom of a neural network. `neuron.py` does gradient descent **by hand in
pure Python** (no libraries) so you can see the maths; `neuron_torch.py` does the
same thing with PyTorch's `nn.Linear` + an optimizer.

### 2 · Neural networks — [`02_neural_networks/`](02_neural_networks/)

`nn_classifier.py` stacks neurons (`Linear → ReLU → Linear`) into a small network
that separates two groups of points, with a train/test split and an accuracy check.

### 3 · Text classification — [`03_text_classification/`](03_text_classification/)

The first "real" task: **is this SMS spam or ham?**

- `classifier.py` — a tiny bag-of-words + logistic-regression starter.
- `train.py` — trains on the full `data/spam.csv`, saves the model to `data/`.
- `predict.py` — loads the saved model and classifies a new message.
- `text_nn.py` — the same problem solved with a **PyTorch neural network** on
  TF-IDF features.

### 4 · Embeddings & tokens — [`04_embeddings_and_tokens/`](04_embeddings_and_tokens/)

How text becomes vectors a model can learn. `tokenize_demo.py` maps words → IDs →
embedding vectors. `embedding_train.py` shows embeddings being trained and a first
hand-rolled **Query/Key/Value** attention calculation (the bridge into step 5).

### 5 · Attention — [`05_attention/`](05_attention/)

The core idea behind transformers.

- `learn_attention.py` — trainable Q/K where a token *learns* what to focus on.
- `self_attention.py` — PyTorch's `nn.MultiheadAttention`.
- `positional.py` — adding position information so order matters.

### 6 · The Transformer block — [`06_transformer/`](06_transformer/)

`transformer_block.py` shows the residual-connection + LayerNorm pattern;
`transformer_block_example.py` assembles the full reusable block
(attention → add & norm → feed-forward → add & norm).

### 7 · Language model — [`07_language_model/`](07_language_model/)

The payoff.

- `tiny_lm.py` — a complete tiny language model on a **toy 8-word vocabulary**,
  including next-token prediction and sampling with temperature/top-k.
- The `tinystories_*` scripts are a **full pipeline** on real data — see below.

---

## 🔁 The TinyStories pipeline (run in order)

These live in [`07_language_model/`](07_language_model/) and must be run **in
sequence**, because each step produces a file the next one needs. The numbers in
the filenames are the run order:

| # | Script | Produces (in `data/`) |
| - | ------ | --------------------- |
| 1 | `tinystories_1_download_dataset.py` | *(inspects the HuggingFace dataset)* |
| 2 | `tinystories_2_prepare_data.py`     | `tinystories.txt` |
| 3 | `tinystories_3_train_tokenizer.py`  | `tinystories_tokenizer.json` |
| 4 | `tinystories_4_prepare_tokens.py`   | `X.pt`, `Y.pt` |
| 5 | `tinystories_5_train_lm.py`         | `tiny_transformer.pth` |
| 6 | `tinystories_6_generate.py`         | *(prints generated stories)* |

```powershell
python 07_language_model/tinystories_2_prepare_data.py
python 07_language_model/tinystories_3_train_tokenizer.py
python 07_language_model/tinystories_4_prepare_tokens.py
python 07_language_model/tinystories_5_train_lm.py
python 07_language_model/tinystories_6_generate.py
```

The trained model is already committed, so you can jump straight to step 6 to see
it generate text.

---

## 💾 The `data/` folder

Datasets and trained models are **committed on purpose** so this repo works right
after a clone. What's in there and which script creates it:

| File | Created by | What it is |
| ---- | ---------- | ---------- |
| `spam.csv` | *(source data)* | The SMS spam/ham dataset |
| `spam_model.pkl`, `vectorizer.pkl` | `03_.../train.py` | scikit-learn spam model + TF-IDF vectorizer |
| `tfidf_vectorizer.pkl`, `spam_nn_model.pth` | `03_.../text_nn.py` | Vectorizer + PyTorch spam network |
| `tinystories.txt` | `tinystories_2_...` | Raw story text |
| `tinystories_tokenizer.json` | `tinystories_3_...` | Trained BPE tokenizer |
| `X.pt`, `Y.pt` | `tinystories_4_...` | Tokenized training tensors |
| `tiny_transformer.pth` | `tinystories_5_...` | The trained language model |

---

## 📦 Requirements

Core libraries used across the project:

- **PyTorch** (`torch`) — neural networks, tensors, autograd
- **scikit-learn** — classic ML (logistic regression, TF-IDF, metrics)
- **pandas** — reading `spam.csv`
- **tokenizers** — the BPE tokenizer for the language model
- **datasets** (HuggingFace) — downloading TinyStories
- **joblib** — saving/loading scikit-learn models

Full pinned versions are in [`requirements.txt`](requirements.txt).

---

## 🗃️ `_scratch/`

Old experiment files kept for history, not part of the learning path:

- `script.py` — an empty scratch file.
- `spam_ham.py` — an early hard-coded copy of the spam dataset (now replaced by
  `data/spam.csv`).

---

## 🚀 Where this is going next

Ideas to keep building on this foundation:

- Train the language model for more epochs / on more stories and compare quality.
- Add more Transformer blocks or attention heads and watch the loss.
- Try different sampling settings (temperature, top-k) in generation.
- Package the shared `TransformerBlock` / `TinyLanguageModel` into a reusable module.
