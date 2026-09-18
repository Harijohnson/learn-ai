# Step 4 · Embeddings & tokens

TF-IDF treats every word as an isolated column. Modern models instead give each
token a **learnable vector** (an *embedding*) so that meaning can be captured and
similar words can end up close together. This step introduces tokens and embeddings.

## Files

| File | What it shows |
| ---- | ------------- |
| `tokenize_demo.py` | Turning words → **token IDs** → **embedding vectors** with `nn.Embedding`. |
| `embedding_train.py` | Embeddings being *trained* (commented walkthrough) **plus** a first hand-rolled **Query/Key/Value** attention calculation — the bridge into Step 5. |

## Run

```powershell
python 04_embeddings_and_tokens/tokenize_demo.py
python 04_embeddings_and_tokens/embedding_train.py
```

## What to look for

- How a list of words becomes a list of integer IDs, then a matrix of vectors.
- In `embedding_train.py`: the **attention scores** and **attention weights**
  computed from Q, K, and V — your first look at attention.

⬅️ Prev: [Step 3 · Text classification](../03_text_classification/)
➡️ Next: [Step 5 · Attention](../05_attention/)
