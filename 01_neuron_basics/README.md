# Step 1 · Neuron basics

The starting point: a **single neuron** and how it "learns" using gradient
descent — nudging its weights a little at a time to reduce its error.

## Files

| File | What it shows |
| ---- | ------------- |
| `neuron.py` | Gradient descent done **by hand in pure Python** — no libraries. You can read every line of the maths. |
| `neuron_torch.py` | The same one-neuron idea using PyTorch's `nn.Linear`, an MSE loss, and an SGD optimizer. |

## Run

```powershell
python 01_neuron_basics/neuron.py
python 01_neuron_basics/neuron_torch.py
```

## What to look for

- The **error shrinking** each epoch as the weights update.
- `neuron_torch.py` should learn a weight ≈ 2 and bias ≈ 0 (it's learning `y = 2x`).

➡️ Next: [Step 2 · Neural networks](../02_neural_networks/)
