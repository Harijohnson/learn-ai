# Step 6 · The Transformer block

A Transformer is just this block repeated. Each block is:

> self-attention → **add & normalize** → feed-forward network → **add & normalize**

The "add" is a **residual connection** (add the input back to the output) and the
"normalize" is **LayerNorm** — together they make deep networks train stably.

## Files

| File | What it shows |
| ---- | ------------- |
| `transformer_block.py` | The **residual + LayerNorm** pattern on its own, so you can see what it does. |
| `transformer_block_example.py` | The full reusable `TransformerBlock` class: attention + feed-forward + both add-&-norm steps. |

## Run

```powershell
python 06_transformer/transformer_block.py
python 06_transformer/transformer_block_example.py
```

## What to look for

- How the output changes after the residual add, and again after LayerNorm.
- The `TransformerBlock` class — this is the exact building block reused in Step 7.

⬅️ Prev: [Step 5 · Attention](../05_attention/)
➡️ Next: [Step 7 · Language model](../07_language_model/)
