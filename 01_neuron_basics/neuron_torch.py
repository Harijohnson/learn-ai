import torch
import torch.nn as nn

# Training data
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

# One neuron
model = nn.Linear(1, 1)

# Loss
loss_function = nn.MSELoss()

# Optimizer
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)

for epoch in range(1000):

    prediction = model(x)

    loss = loss_function(prediction, y)

    loss.backward()

    optimizer.step()

    optimizer.zero_grad()

    if epoch % 10 == 0:
        print(
            f"Epoch {epoch}: "
            f"loss={loss.item():.4f}"
        )

print("Weight:", model.weight.item())
print("Bias:", model.bias.item())