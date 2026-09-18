import torch
import torch.nn as nn

# 3 token embeddings
tokens = torch.tensor([
    [1.0, 0.0, 0.0, 0.0],  # position 0
    [0.0, 1.0, 0.0, 0.0],  # position 1
    [0.0, 0.0, 1.0, 0.0],  # position 2
])

# 3 positions, embedding size = 4
position_embedding = nn.Embedding(
    num_embeddings=3,
    embedding_dim=4
)

# Position IDs
positions = torch.tensor([0, 1, 2])

# Get position vectors
position_vectors = position_embedding(positions)

print("Token embeddings:")
print(tokens)

print("\nPosition embeddings:")
print(position_vectors)

# Add token + position
final_embeddings = tokens + position_vectors

print("\nToken + Position:")
print(final_embeddings)