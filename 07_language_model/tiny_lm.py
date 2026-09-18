import torch
import torch.nn as nn


# ----------------------------------
# 1. Vocabulary
# ----------------------------------




# vocab = {
#     "i": 0,
#     "love": 1,
#     "free": 2,
#     "money": 3
# }

# id_to_word = {
#     0: "i",
#     1: "love",
#     2: "free",
#     3: "money"
# }



sentences = [
    ["i", "love", "free", "money"],
    ["you", "love", "free", "gifts"],
    ["i", "like", "good", "food"],
    ["you", "like", "good", "music"],
]

# Create vocabulary automatically
all_words = set()

for sentence in sentences:
    for word in sentence:
        all_words.add(word)

vocab = {
    word: index
    for index, word in enumerate(sorted(all_words))
}

id_to_word = {
    index: word
    for word, index in vocab.items()
}

print("Vocabulary:")
print(vocab)


# ----------------------------------
# 2. Training sequence
# ----------------------------------

# "i love free money"
# sequence = torch.tensor([0, 1, 2, 3])

# Input:
# i love free
# x = sequence[:-1]

# # Target:
# # love free money
# y = sequence[1:]

# print("Input IDs:")
# print(x)

# print("Target IDs:")
# print(y)

training_inputs = []
training_targets = []

for sentence in sentences:
    ids = [vocab[word] for word in sentence]

    training_inputs.append(ids[:-1])
    training_targets.append(ids[1:])

x = torch.tensor(training_inputs)
y = torch.tensor(training_targets)

print("Input:")
print(x)

print("\nTarget:")
print(y)
# ----------------------------------
# 3. Model
# ----------------------------------

class TinyLanguageModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_size=16,
        max_length=3
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_size
        )

        self.position_embedding = nn.Embedding(
            max_length,
            embedding_size
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_size,
            num_heads=1,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(embedding_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_size, 32),
            nn.ReLU(),
            nn.Linear(32, embedding_size)
        )

        self.norm2 = nn.LayerNorm(embedding_size)

        # Convert final representation → vocabulary scores
        self.output_layer = nn.Linear(
            embedding_size,
            vocab_size
        )

    def forward(self, x):

        batch_size, sequence_length = x.shape

        # -----------------------------
        # Token + position embeddings
        # -----------------------------

        positions = torch.arange(
            sequence_length,
            device=x.device
        )

        token_vectors = self.token_embedding(x)

        position_vectors = self.position_embedding(
            positions
        )

        x = token_vectors + position_vectors

        # -----------------------------
        # Causal attention mask
        # -----------------------------

        mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                device=x.device
            ),
            diagonal=1
        ).bool()

        # -----------------------------
        # Self-attention
        # -----------------------------

        attention_output, _ = self.attention(
            x,
            x,
            x,
            attn_mask=mask
        )

        x = self.norm1(
            x + attention_output
        )

        # -----------------------------
        # Feed-forward
        # -----------------------------

        feed_forward_output = self.feed_forward(x)

        x = self.norm2(
            x + feed_forward_output
        )

        # -----------------------------
        # Vocabulary scores
        # -----------------------------

        logits = self.output_layer(x)

        return logits


# ----------------------------------
# 4. Create model
# ----------------------------------

model = TinyLanguageModel(
    vocab_size=len(vocab)
)


# ----------------------------------
# 5. Loss + optimizer
# ----------------------------------

loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)


# ----------------------------------
# 6. Training
# ----------------------------------

# x = x.unsqueeze(0)
# y = y.unsqueeze(0)

for epoch in range(1000):

    logits = model(x)

    # logits:
    # [batch, sequence, vocabulary]

    loss = loss_function(
        logits.reshape(-1, len(vocab)),
        y.reshape(-1)
    )

    loss.backward()

    optimizer.step()

    optimizer.zero_grad()

    if epoch % 100 == 0:
        print(
            f"Epoch {epoch}: "
            f"loss={loss.item():.4f}"
        )

# Generate text one token at a time

# def generate(start_word, max_tokens=4):

#     model.eval()

#     generated = [vocab[start_word]]

#     for _ in range(max_tokens - 1):

#         input_ids = torch.tensor(
#             [generated]
#         )

#         with torch.no_grad():
#             logits = model(input_ids)

#         # Get prediction for the last token
#         next_token_logits = logits[0, -1]

#         # Convert logits to probabilities
#         probabilities = torch.softmax(
#             next_token_logits,
#             dim=0
#         )

#         # Print probabilities
#         print("\nNext-token probabilities:")

#         for token_id, probability in enumerate(probabilities):
#             if probability.item() > 0.001:
#                 print(
#                     id_to_word[token_id],
#                     probability.item()
#                 )

#         # Pick highest probability token
#         next_token = torch.argmax(
#             probabilities
#         ).item()

#         generated.append(next_token)

#     return [
#         id_to_word[token_id]
#         for token_id in generated
#     ]


def generate(start_word, max_tokens=4, temperature=1.0):

	model.eval()

	generated = [vocab[start_word]]

	for _ in range(max_tokens - 1):

		input_ids = torch.tensor([generated])

		with torch.no_grad():
			logits = model(input_ids)

		# Get logits for the last token
		next_token_logits = logits[0, -1]

		# Temperature
		next_token_logits = next_token_logits / temperature

		# Convert logits to probabilities
		probabilities = torch.softmax(
			next_token_logits,
			dim=0
		)


		top_k = 2
		values, indices = torch.topk(
			probabilities,
			top_k
		)

		next_token_index = torch.multinomial(
			values,
			num_samples=1
		).item()

		next_token = indices[next_token_index].item()

		generated.append(next_token)

	return [
		id_to_word[token_id]
		for token_id in generated
	]
print("\nGenerated:")
print(generate("i"))



# ----------------------------------
# 7. Predict next tokens
# ----------------------------------

model.eval()

with torch.no_grad():

    logits = model(x)

    predictions = torch.argmax(
        logits,
        dim=-1
    )

print("\nPredicted IDs:")
print(predictions)

print("\nPredicted words:")

for token_id in predictions[0]:
    print(id_to_word[token_id.item()])

for i in range(10):
    print(generate("i", temperature=1.0))
print("\nTemperature 0.5")

for i in range(5):
    print(generate("i", temperature=0.5))


print("\nTemperature 2.0")

for i in range(5):
    print(generate("i", temperature=2.0))


def generate_from_prompt(prompt, max_tokens=4):

    model.eval()

    generated = [
        vocab[word]
        for word in prompt
    ]

    for _ in range(max_tokens - len(generated)):

        input_ids = torch.tensor([generated])

        with torch.no_grad():
            logits = model(input_ids)

        next_token_logits = logits[0, -1]

        probabilities = torch.softmax(
            next_token_logits,
            dim=0
        )

        next_token = torch.multinomial(
            probabilities,
            num_samples=1
        ).item()

        generated.append(next_token)

    return [
        id_to_word[token_id]
        for token_id in generated
    ]


print(generate_from_prompt(["i", "love"]))
print(generate_from_prompt(["i", "like"]))
print(generate_from_prompt(["you", "love"]))
print(generate_from_prompt(["you", "like"]))