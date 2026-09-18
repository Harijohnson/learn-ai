# import torch
# import torch.nn as nn

# # -------------------------
# # 1. Vocabulary
# # -------------------------

# vocab = {
#     "win": 1,
#     "free": 2,
#     "money": 3,
#     "claim": 4,
#     "prize": 5,
#     "meeting": 6,
#     "office": 7,
#     "call": 8,
#     "me": 9,
#     "tomorrow": 10
# }

# # -------------------------
# # 2. Training data
# # -------------------------
# # 1 = spam
# # 0 = ham

# X = torch.tensor([
#     [1, 2, 3],      # win free money
#     [4, 2, 5],      # claim free prize
#     [6, 7, 8],      # meeting office call
#     [8, 9, 10]      # call me tomorrow
# ])

# y = torch.tensor([
#     1,
#     1,
#     0,
#     0
# ])

# # -------------------------
# # 3. Model
# # -------------------------

# embedding = nn.Embedding(
#     num_embeddings=len(vocab) + 1,
#     embedding_dim=4
# )

# classifier = nn.Linear(4, 2)

# # -------------------------
# # 4. Print "free" before training
# # -------------------------

# print("Before training:")
# print(embedding.weight[2])

# # -------------------------
# # 5. Optimizer + loss
# # -------------------------

# loss_function = nn.CrossEntropyLoss()

# optimizer = torch.optim.Adam(
#     list(embedding.parameters()) +
#     list(classifier.parameters()),
#     lr=0.01
# )

# # -------------------------
# # 6. Training
# # -------------------------

# for epoch in range(500):

#     # Convert token IDs → vectors
#     vectors = embedding(X)

#     # Combine the 3 token vectors
#     sentence_vector = vectors.mean(dim=1)

#     # Classification
#     output = classifier(sentence_vector)

#     # Loss
#     loss = loss_function(output, y)

#     # Backpropagation
#     loss.backward()

#     # Update embedding + classifier
#     optimizer.step()

#     optimizer.zero_grad()

#     if epoch % 100 == 0:
#         print(
#             f"Epoch {epoch}: "
#             f"loss={loss.item():.4f}"
#         )

# # -------------------------
# # 7. Print "free" after training
# # -------------------------

# print("\nAfter training:")
# print(embedding.weight[2])

# # -------------------------
# # 8. Test
# # -------------------------

# test = torch.tensor([
#     [1, 2, 3]   # win free money
# ])

# with torch.no_grad():

#     vectors = embedding(test)

#     sentence_vector = vectors.mean(dim=1)

#     output = classifier(sentence_vector)

#     prediction = torch.argmax(output, dim=1)

# print("\nPrediction:", prediction.item())





import torch
import torch.nn as nn

# 3 tokens, each represented by 4 numbers
tokens = torch.tensor([
    [1.0, 0.5, 0.2, 0.1],
    [0.3, 0.8, 0.4, 0.2],
    [0.1, 0.2, 0.9, 0.7]
])

# Create Q, K, V transformations
query_layer = nn.Linear(4, 4)
key_layer = nn.Linear(4, 4)
value_layer = nn.Linear(4, 4)

# Create Query, Key, Value
Q = query_layer(tokens)
K = key_layer(tokens)
V = value_layer(tokens)

print("Q:")
print(Q)

print("\nK:")
print(K)

print("\nV:")
print(V)

# Attention scores
scores = Q @ K.T

print("\nScores:")
print(scores)

# Convert scores into attention weights
attention_weights = torch.softmax(scores, dim=1)

print("\nAttention weights:")
print(attention_weights)

# Combine Values using attention weights
output = attention_weights @ V

print("\nAttention output:")
print(output)