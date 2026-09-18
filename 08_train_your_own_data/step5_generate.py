import torch
import torch.nn as nn
from tokenizers import Tokenizer
from pathlib import Path

# Everything for this step lives inside THIS folder (not the shared data/).
HERE = Path(__file__).resolve().parent


# =========================================================
# 1. Transformer block
# =========================================================

class TransformerBlock(nn.Module):

    def __init__(self, embedding_size=64, num_heads=4):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_size,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(embedding_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_size, embedding_size * 4),
            nn.ReLU(),
            nn.Linear(embedding_size * 4, embedding_size)
        )

        self.norm2 = nn.LayerNorm(embedding_size)

    def forward(self, x):

        sequence_length = x.size(1)

        mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length
            ),
            diagonal=1
        ).bool()

        attention_output, _ = self.attention(
            x,
            x,
            x,
            attn_mask=mask
        )

        x = self.norm1(
            x + attention_output
        )

        feed_forward_output = self.feed_forward(x)

        x = self.norm2(
            x + feed_forward_output
        )

        return x


# =========================================================
# 2. Language model
# =========================================================

class TinyLanguageModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_size=64,
        max_length=32
    ):
        super().__init__()

        self.max_length = max_length

        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_size
        )

        self.position_embedding = nn.Embedding(
            max_length,
            embedding_size
        )

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embedding_size=embedding_size,
                num_heads=4
            ),
            TransformerBlock(
                embedding_size=embedding_size,
                num_heads=4
            )
        ])

        self.output_layer = nn.Linear(
            embedding_size,
            vocab_size
        )

    def forward(self, x):

        batch_size, sequence_length = x.shape

        positions = torch.arange(
            sequence_length
        )

        token_vectors = self.token_embedding(x)

        position_vectors = self.position_embedding(
            positions
        )

        x = token_vectors + position_vectors

        for block in self.transformer_blocks:
            x = block(x)

        logits = self.output_layer(x)

        return logits


# =========================================================
# 3. Load tokenizer
# =========================================================

tokenizer = Tokenizer.from_file(
    str(HERE / "creative_tokenizer.json")
)

vocab_size = tokenizer.get_vocab_size()

print("Vocabulary size:", vocab_size)


# =========================================================
# 4. Load model
# =========================================================

model = TinyLanguageModel(
    vocab_size=vocab_size
)

model.load_state_dict(
    torch.load(
        HERE / "creative_transformer.pth",
        weights_only=True
    )
)

model.eval()


# =========================================================
# 5. Generate
# =========================================================

def generate(
    prompt,
    max_new_tokens=40,
    temperature=1.0
):

    encoded = tokenizer.encode(prompt)

    generated = encoded.ids

    for _ in range(max_new_tokens):

        # Keep context within model limit
        input_ids = generated[-32:]

        input_tensor = torch.tensor(
            [input_ids],
            dtype=torch.long
        )

        with torch.no_grad():

            logits = model(input_tensor)

        # Last position
        next_token_logits = logits[0, -1]

        # Temperature
        next_token_logits = (
            next_token_logits / temperature
        )

        # Probabilities
        probabilities = torch.softmax(
            next_token_logits,
            dim=0
        )

        # Sampling
        next_token = torch.multinomial(
            probabilities,
            num_samples=1
        ).item()

        generated.append(next_token)

    return tokenizer.decode(generated)


# =========================================================
# 6. Test
# =========================================================

prompts = [
    "In the quaint town",
    "One crisp autumn morning",
    "The young artist",
    "Once upon a time"
]

for prompt in prompts:
    print("\nPrompt:", prompt)
    print(generate(
        prompt,
        max_new_tokens=40,
        temperature=0.8
    ))
