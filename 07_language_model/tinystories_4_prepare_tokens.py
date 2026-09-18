# from tokenizers import Tokenizer
# import torch


# # ----------------------------------
# # 1. Load tokenizer
# # ----------------------------------

# tokenizer = Tokenizer.from_file(
#     "data/tinystories_tokenizer.json"
# )

# eos_id = tokenizer.token_to_id("<eos>")

# print("EOS token ID:", eos_id)


# # ----------------------------------
# # 2. Read stories
# # ----------------------------------

# stories = []

# with open(
#     "data/tinystories.txt",
#     "r",
#     encoding="utf-8"
# ) as file:

#     for line in file:
#         line = line.strip()

#         if line:
#             stories.append(line)


# print("Stories:", len(stories))


# # ----------------------------------
# # 3. Convert text → token IDs
# # ----------------------------------

# all_tokens = []

# for story in stories:

#     encoded = tokenizer.encode(story)

#     all_tokens.extend(encoded.ids)

#     # Separate stories
#     all_tokens.append(eos_id)


# print("Total tokens:", len(all_tokens))


# # ----------------------------------
# # 4. Limit size for learning
# # ----------------------------------

# max_tokens = 100_000

# all_tokens = all_tokens[:max_tokens]

# print("Using tokens:", len(all_tokens))


# # ----------------------------------
# # 5. Create fixed-size sequences
# # ----------------------------------

# block_size = 32

# inputs = []
# targets = []

# for i in range(
#     0,
#     len(all_tokens) - block_size,
#     block_size
# ):

#     chunk = all_tokens[
#         i:i + block_size + 1
#     ]

#     inputs.append(
#         chunk[:-1]
#     )

#     targets.append(
#         chunk[1:]
#     )


# # ----------------------------------
# # 6. Convert to tensors
# # ----------------------------------

# X = torch.tensor(
#     inputs,
#     dtype=torch.long
# )

# Y = torch.tensor(
#     targets,
#     dtype=torch.long
# )


# print("Input shape:", X.shape)
# print("Target shape:", Y.shape)


# # ----------------------------------
# # 7. Save
# # ----------------------------------

# torch.save(
#     X,
#     "data/X.pt"
# )

# torch.save(
#     Y,
#     "data/Y.pt"
# )

# print("Saved:")
# print("data/X.pt")
# print("data/Y.pt")

from datasets import load_dataset
from tokenizers import Tokenizer
import torch
from pathlib import Path

# All data files live in the shared data/ folder at the repo root.
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


# ----------------------------------
# 1. Load dataset
# ----------------------------------

dataset = load_dataset(
    "roneneldan/TinyStories",
    split="train[:5000]"
)

print("Stories:", len(dataset))


# ----------------------------------
# 2. Load tokenizer
# ----------------------------------

tokenizer = Tokenizer.from_file(
    str(DATA_DIR / "tinystories_tokenizer.json")
)

eos_id = tokenizer.token_to_id("<eos>")

print("EOS token ID:", eos_id)


# ----------------------------------
# 3. Tokenize each complete story
# ----------------------------------

all_tokens = []

for item in dataset:

    story = item["text"]

    encoded = tokenizer.encode(story)

    all_tokens.extend(encoded.ids)

    # End of story
    all_tokens.append(eos_id)


print("Total tokens:", len(all_tokens))


# ----------------------------------
# 4. Limit tokens
# ----------------------------------

max_tokens = 1163218

all_tokens = all_tokens[:max_tokens]

print("Using tokens:", len(all_tokens))


# ----------------------------------
# 5. Create sequences
# ----------------------------------

block_size = 32

inputs = []
targets = []

for i in range(
    0,
    len(all_tokens) - block_size,
    block_size
):

    chunk = all_tokens[
        i:i + block_size + 1
    ]

    inputs.append(
        chunk[:-1]
    )

    targets.append(
        chunk[1:]
    )


# ----------------------------------
# 6. Tensor conversion
# ----------------------------------

X = torch.tensor(
    inputs,
    dtype=torch.long
)

Y = torch.tensor(
    targets,
    dtype=torch.long
)


print("Input shape:", X.shape)
print("Target shape:", Y.shape)


# ----------------------------------
# 7. Save
# ----------------------------------

torch.save(X, DATA_DIR / "X.pt")
torch.save(Y, DATA_DIR / "Y.pt")

print("Saved successfully.")