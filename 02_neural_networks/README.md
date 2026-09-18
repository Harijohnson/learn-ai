# Step 2 · Neural networks

One neuron can only learn straight-line relationships. **Stacking** neurons into
layers (with a ReLU in between) lets a network learn more interesting patterns.

## Files

| File | What it shows |
| ---- | ------------- |
| `nn_classifier.py` | A small network (`Linear → ReLU → Linear`) that classifies 2-D points into two groups, with a manual train/test split and an accuracy check. |

## Run

```powershell
python 02_neural_networks/nn_classifier.py
```

## What to look for

- The **loss dropping** over the training epochs.
- The final **predictions vs. actual** labels and the printed accuracy.

⬅️ Prev: [Step 1 · Neuron basics](../01_neuron_basics/)
➡️ Next: [Step 3 · Text classification](../03_text_classification/)
