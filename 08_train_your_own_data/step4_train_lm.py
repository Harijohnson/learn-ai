import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tokenizers import Tokenizer
from pathlib import Path

# Everything for this step lives inside THIS folder (not the shared data/).
HERE = Path(__file__).resolve().parent


# =========================================================
# 1. Load data
# =========================================================

X = torch.load(HERE / "creative_X.pt")
Y = torch.load(HERE / "creative_Y.pt")

print("X shape:", X.shape)
print("Y shape:", Y.shape)


# =========================================================
# 2. Train / validation split
# =========================================================

split = int(len(X) * 0.9)

X_train = X[:split]
Y_train = Y[:split]

X_val = X[split:]
Y_val = Y[split:]

print("Training:", X_train.shape)
print("Validation:", X_val.shape)


# =========================================================
# 3. DataLoader
# =========================================================

train_dataset = TensorDataset(
    X_train,
    Y_train
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)


# =========================================================
# 4. Transformer Block
# =========================================================

class TransformerBlock(nn.Module):

    def __init__(
        self,
        embedding_size=64,
        num_heads=4
    ):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_size,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(
            embedding_size
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(
                embedding_size,
                embedding_size * 4
            ),
            nn.ReLU(),
            nn.Linear(
                embedding_size * 4,
                embedding_size
            )
        )

        self.norm2 = nn.LayerNorm(
            embedding_size
        )

    def forward(self, x):

        sequence_length = x.size(1)

        # Causal mask
        mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                device=x.device
            ),
            diagonal=1
        ).bool()

        # Self-attention
        attention_output, _ = self.attention(
            x,
            x,
            x,
            attn_mask=mask
        )

        # Residual + LayerNorm
        x = self.norm1(
            x + attention_output
        )

        # Feed-forward
        feed_forward_output = self.feed_forward(x)

        # Residual + LayerNorm
        x = self.norm2(
            x + feed_forward_output
        )

        return x


# =========================================================
# 5. Language Model
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

        # Token embedding
        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_size
        )

        # Position embedding
        self.position_embedding = nn.Embedding(
            max_length,
            embedding_size
        )

        # Transformer blocks
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

        # Vocabulary prediction
        self.output_layer = nn.Linear(
            embedding_size,
            vocab_size
        )

    def forward(self, x):

        batch_size, sequence_length = x.shape

        positions = torch.arange(
            sequence_length,
            device=x.device
        )

        # Token + position embeddings
        token_vectors = self.token_embedding(x)

        position_vectors = self.position_embedding(
            positions
        )

        x = token_vectors + position_vectors

        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Predict vocabulary token
        logits = self.output_layer(x)

        return logits


# =========================================================
# 6. Create model
# =========================================================

tokenizer = Tokenizer.from_file(
    str(HERE / "creative_tokenizer.json")
)

vocab_size = tokenizer.get_vocab_size()

print("Vocabulary size:", vocab_size)

model = TinyLanguageModel(
    vocab_size=vocab_size,
    embedding_size=64,
    max_length=32
)


# =========================================================
# 7. Loss + optimizer
# =========================================================

loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001
)


# =========================================================
# 8. Training
# =========================================================

epochs = 30

for epoch in range(epochs):

    model.train()

    total_loss = 0

    for batch_X, batch_Y in train_loader:

        # Forward
        logits = model(batch_X)

        # [batch, sequence, vocab] -> [batch * sequence, vocab]
        logits = logits.reshape(
            -1,
            vocab_size
        )

        targets = batch_Y.reshape(-1)

        # Loss
        loss = loss_function(
            logits,
            targets
        )

        # Backpropagation
        loss.backward()

        # Update
        optimizer.step()

        # Clear gradients
        optimizer.zero_grad()

        total_loss += loss.item()

    average_loss = (
        total_loss /
        len(train_loader)
    )

    # -----------------------------
    # Validation
    # -----------------------------

    model.eval()

    with torch.no_grad():

        val_logits = model(X_val)

        val_loss = loss_function(
            val_logits.reshape(-1, vocab_size),
            Y_val.reshape(-1)
        )

    print(
        f"Epoch {epoch + 1}/{epochs} "
        f"| train loss: {average_loss:.4f} "
        f"| val loss: {val_loss.item():.4f}"
    )


# =========================================================
# 9. Save model
# =========================================================

torch.save(
    model.state_dict(),
    HERE / "creative_transformer.pth"
)

print("\nModel saved successfully.")
