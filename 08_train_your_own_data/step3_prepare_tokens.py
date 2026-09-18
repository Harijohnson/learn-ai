from tokenizers import Tokenizer
import torch
from pathlib import Path

# Everything for this step lives inside THIS folder (not the shared data/).
HERE = Path(__file__).resolve().parent


# ----------------------------------
# 1. Load our tokenizer
# ----------------------------------

tokenizer = Tokenizer.from_file(
    str(HERE / "creative_tokenizer.json")
)

eos_id = tokenizer.token_to_id("<eos>")

print("EOS token ID:", eos_id)


# ----------------------------------
# 2. Read the cleaned stories
# ----------------------------------
# step1_prepare_data.py wrote one story per line.

stories = []

with open(
    HERE / "creative_clean.txt",
    "r",
    encoding="utf-8"
) as file:

    for line in file:
        line = line.strip()

        if line:
            stories.append(line)


print("Stories:", len(stories))


# ----------------------------------
# 3. Convert text -> token IDs
# ----------------------------------

all_tokens = []

for story in stories:

    encoded = tokenizer.encode(story)

    all_tokens.extend(encoded.ids)

    # Mark the end of each story
    all_tokens.append(eos_id)


print("Total tokens:", len(all_tokens))


# ----------------------------------
# 4. Create fixed-size sequences
# ----------------------------------
# block_size must match max_length in the model (see step4).

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
# 5. Convert to tensors
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
# 6. Save
# ----------------------------------

torch.save(X, HERE / "creative_X.pt")
torch.save(Y, HERE / "creative_Y.pt")

print("Saved:")
print(HERE / "creative_X.pt")
print(HERE / "creative_Y.pt")
