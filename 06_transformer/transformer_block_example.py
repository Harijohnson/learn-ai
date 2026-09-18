import torch
import torch.nn as nn


class TransformerBlock(nn.Module):

    def __init__(self, embedding_size=4, num_heads=1):
        super().__init__()

        # Self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_size,
            num_heads=num_heads,
            batch_first=True
        )

        # Normalize after attention
        self.norm1 = nn.LayerNorm(embedding_size)

        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_size, embedding_size * 2),
            nn.ReLU(),
            nn.Linear(embedding_size * 2, embedding_size)
        )

        # Normalize after feed-forward
        self.norm2 = nn.LayerNorm(embedding_size)

    def forward(self, x):

        # -------------------------
        # Self-attention
        # -------------------------

        attention_output, attention_weights = self.attention(
            x,
            x,
            x
        )

        # Residual connection
        x = x + attention_output

        # LayerNorm
        x = self.norm1(x)

        # -------------------------
        # Feed-forward
        # -------------------------

        feed_forward_output = self.feed_forward(x)

        # Residual connection
        x = x + feed_forward_output

        # LayerNorm
        x = self.norm2(x)

        return x, attention_weights


# --------------------------------------------------
# Test
# --------------------------------------------------

tokens = torch.tensor([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
])

# Add batch dimension
tokens = tokens.unsqueeze(0)

block = TransformerBlock()

output, attention_weights = block(tokens)

print("Input:")
print(tokens)

print("\nAttention weights:")
print(attention_weights)

print("\nOutput:")
print(output)