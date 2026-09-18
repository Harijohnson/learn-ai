# import torch
# import torch.nn as nn

# # Training data
# X = torch.tensor([
#     [1.0, 1.0],
#     [1.0, 2.0],
#     [4.0, 4.0],
#     [5.0, 4.0]
# ])

# # Class labels
# y = torch.tensor([
#     0,
#     0,
#     1,
#     1
# ])

# # Neural network
# model = nn.Sequential(
#     nn.Linear(2, 4),
#     nn.ReLU(),
#     nn.Linear(4, 2)
# )

# # Loss for classification
# loss_function = nn.CrossEntropyLoss()

# # Optimizer
# optimizer = torch.optim.SGD(
#     model.parameters(),
#     lr=0.1
# )

# for epoch in range(1000):

#     # Forward pass
#     output = model(X)

#     # Calculate loss
#     loss = loss_function(output, y)

#     # Backpropagation
#     loss.backward()

#     # Update weights
#     optimizer.step()

#     # Clear gradients
#     optimizer.zero_grad()

#     if epoch % 100 == 0:
#         print(
#             f"Epoch {epoch}: "
#             f"loss={loss.item():.4f}"
#         )

# print("Training output:")
# print(model(X))

# print("Training predictions:")
# print(torch.argmax(model(X), dim=1))

# print("Actual labels:")
# print(y)



# # Test
# test = torch.tensor([
#     [1.5, 1.5],
#     [4.5, 4.5]
# ])

# output = model(test)

# prediction = torch.argmax(output, dim=1)

# print("Predictions:", prediction)


import torch
import torch.nn as nn

# All data
X = torch.tensor([
    [1.0, 1.0],
    [1.0, 2.0],
    [2.0, 1.0],
    [2.0, 2.0],
    [4.0, 4.0],
    [4.0, 5.0],
    [5.0, 4.0],
    [5.0, 5.0]
])

# Labels
y = torch.tensor([
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1
])

# Split manually
X_train = X[:6]
y_train = y[:6]

X_test = X[6:]
y_test = y[6:]

# Neural network
model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 2)
)

loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1
)

# Training
for epoch in range(1000):

    output = model(X_train)

    loss = loss_function(output, y_train)

    loss.backward()

    optimizer.step()

    optimizer.zero_grad()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: loss={loss.item():.4f}")

# Test
test_output = model(X_test)

predictions = torch.argmax(test_output, dim=1)

print("Predictions:", predictions)
print("Actual:", y_test)

# Accuracy
accuracy = (predictions == y_test).float().mean()

print("Accuracy:", accuracy.item())