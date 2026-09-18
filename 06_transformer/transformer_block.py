# import torch
# import torch.nn as nn

# # 3 tokens, embedding size = 4
# tokens = torch.tensor([
#     [1.0, 0.0, 0.0, 0.0],
#     [0.0, 1.0, 0.0, 0.0],
#     [0.0, 0.0, 1.0, 0.0],
# ])

# # Add batch dimension
# tokens = tokens.unsqueeze(0)


# # -------------------------
# # Self-attention
# # -------------------------

# attention = nn.MultiheadAttention(
#     embed_dim=4,
#     num_heads=1,
#     batch_first=True
# )

# attention_output, attention_weights = attention(
#     tokens,
#     tokens,
#     tokens
# )


# # -------------------------
# # Feed-forward network
# # -------------------------

# feed_forward = nn.Sequential(
#     nn.Linear(4, 8),
#     nn.ReLU(),
#     nn.Linear(8, 4)
# )

# output = feed_forward(attention_output)


# print("Input shape:")
# print(tokens.shape)

# print("\nAttention output:")
# print(attention_output)

# print("\nFinal output after feed-forward:")
# print(output)

import torch
import torch.nn as nn

tokens = torch.tensor([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
])

tokens = tokens.unsqueeze(0)

attention = nn.MultiheadAttention(
    embed_dim=4,
    num_heads=1,
    batch_first=True
)

attention_output, _ = attention(
    tokens,
    tokens,
    tokens
)

# Residual connection
residual_output = tokens + attention_output

# Layer normalization
layer_norm = nn.LayerNorm(4)

normalized_output = layer_norm(residual_output)

print("Attention output:")
print(attention_output)

print("\nAfter residual connection:")
print(residual_output)

print("\nAfter LayerNorm:")
print(normalized_output)