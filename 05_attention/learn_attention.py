import torch
import torch.nn as nn

# 3 token representations
tokens = torch.tensor([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
])

# We will use token 0 as the query
query_token = tokens[0:1]

# All 3 tokens are possible keys
key_tokens = tokens

# Trainable Q and K projections
query_layer = nn.Linear(4, 4)
key_layer = nn.Linear(4, 4)

optimizer = torch.optim.Adam(
    list(query_layer.parameters()) +
    list(key_layer.parameters()),
    lr=0.05
)

loss_function = nn.CrossEntropyLoss()

# We want query token 0 to focus on token 2
target = torch.tensor([2])

for epoch in range(500):

    Q = query_layer(query_token)
    K = key_layer(key_tokens)

    # Q: [1, 4]
    # K: [3, 4]
    # K.T: [4, 3]
    scores = Q @ K.T

    # scores shape = [1, 3]
    loss = loss_function(scores, target)

    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    if epoch % 100 == 0:
        attention = torch.softmax(scores, dim=1)

        print(
            f"Epoch {epoch}: "
            f"loss={loss.item():.4f}, "
            f"attention={attention.detach().numpy()}"
        )

# Final attention
Q = query_layer(query_token)
K = key_layer(key_tokens)

scores = Q @ K.T
attention = torch.softmax(scores, dim=1)

print("\nFinal scores:")
print(scores)

print("\nFinal attention:")
print(attention)