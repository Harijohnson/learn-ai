import torch
import torch.nn as nn

# --------------------------------------------------
# 1. Three tokens
# --------------------------------------------------

tokens = torch.tensor([
    [1.0, 0.0, 0.0, 0.0],  # token 0
    [0.0, 1.0, 0.0, 0.0],  # token 1
    [0.0, 0.0, 1.0, 0.0],  # token 2
])

# Add batch dimension
tokens = tokens.unsqueeze(0)

# Shape:
# [batch, sequence_length, embedding_size]
print("Input shape:", tokens.shape)


# --------------------------------------------------
# 2. Self-attention
# --------------------------------------------------

attention = nn.MultiheadAttention(
    embed_dim=4,
    num_heads=1,
    batch_first=True
)


# --------------------------------------------------
# 3. Run self-attention
# --------------------------------------------------

output, attention_weights = attention(
    tokens,
    tokens,
    tokens,
    need_weights=True
)


# --------------------------------------------------
# 4. Inspect
# --------------------------------------------------

print("\nAttention weights:")
print(attention_weights)

print("\nAttention output:")
print(output)