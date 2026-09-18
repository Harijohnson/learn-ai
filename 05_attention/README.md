# Step 5 · Attention

**Attention** is the heart of the Transformer. It lets each token decide **which
other tokens to pay attention to** when building its own representation.

## Files

| File | What it shows |
| ---- | ------------- |
| `learn_attention.py` | Trainable Query/Key layers where a token **learns** to focus its attention on a specific other token. |
| `self_attention.py` | The same idea using PyTorch's built-in `nn.MultiheadAttention`, and inspecting the attention weights. |
| `positional.py` | **Positional embeddings** — because attention alone has no sense of word order, we add position information. |

## Run

```powershell
python 05_attention/learn_attention.py
python 05_attention/self_attention.py
python 05_attention/positional.py
```

## What to look for

- In `learn_attention.py`: the **attention distribution shifting** over training
  until it focuses where we asked it to.
- The **attention weights** matrix (each row sums to 1).
- How token embeddings change once **position** is added.

⬅️ Prev: [Step 4 · Embeddings & tokens](../04_embeddings_and_tokens/)
➡️ Next: [Step 6 · The Transformer block](../06_transformer/)
