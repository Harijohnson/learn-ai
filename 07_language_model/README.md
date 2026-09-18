# Step 7 · Language model

Everything comes together here. A language model **predicts the next token**, and
by doing that over and over it can **generate text**.

There are two parts to this step:

1. **`tiny_lm.py`** — a complete, tiny language model on a **toy 8-word
   vocabulary**. Small enough to read end-to-end: token + position embeddings,
   a causal (masked) attention layer, a feed-forward layer, next-token
   prediction, and sampling with **temperature** and **top-k**.

2. **The TinyStories pipeline** (`tinystories_*`) — the same ideas scaled up to a
   real dataset of children's stories, split into ordered steps.

## Run the toy model

```powershell
python 07_language_model/tiny_lm.py
```

## Run the TinyStories pipeline (in order)

Each step writes a file the next step needs. The number in each filename **is**
the run order:

| # | Script | Produces (in `data/`) |
| - | ------ | --------------------- |
| 1 | `tinystories_1_download_dataset.py` | *(inspects the HuggingFace dataset)* |
| 2 | `tinystories_2_prepare_data.py` | `tinystories.txt` |
| 3 | `tinystories_3_train_tokenizer.py` | `tinystories_tokenizer.json` |
| 4 | `tinystories_4_prepare_tokens.py` | `X.pt`, `Y.pt` |
| 5 | `tinystories_5_train_lm.py` | `tiny_transformer.pth` |
| 6 | `tinystories_6_generate.py` | *(prints generated stories)* |

```powershell
python 07_language_model/tinystories_2_prepare_data.py
python 07_language_model/tinystories_3_train_tokenizer.py
python 07_language_model/tinystories_4_prepare_tokens.py
python 07_language_model/tinystories_5_train_lm.py
python 07_language_model/tinystories_6_generate.py
```

> The trained model (`data/tiny_transformer.pth`) is already committed, so you can
> jump straight to step 6 and watch it generate stories from a prompt like
> "Once upon a time".

## What to look for

- In `tiny_lm.py`: how **temperature** changes how "adventurous" the output is,
  and how a **prompt** steers generation.
- In the pipeline: **train vs. validation loss** dropping over epochs, then the
  final model producing story-like text.

⬅️ Prev: [Step 6 · The Transformer block](../06_transformer/)
🏠 Back to the [project overview](../README.md)
