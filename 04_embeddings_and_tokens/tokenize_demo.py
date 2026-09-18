text = "I love free money"

words = text.lower().split()

vocab = {
    "i": 1,
    "love": 2,
    "free": 3,
    "money": 4
}

tokens = [vocab[word] for word in words]

print("Words:", words)
print("Token IDs:", tokens)



import torch
import torch.nn as nn

# Token IDs
tokens = torch.tensor([1, 2, 3, 4])

# 5 possible tokens, each represented by a vector of size 3
embedding = nn.Embedding(
    num_embeddings=5,
    embedding_dim=3
)

vectors = embedding(tokens)

print("Token IDs:")
print(tokens)

print("\nEmbedding vectors:")
print(vectors)