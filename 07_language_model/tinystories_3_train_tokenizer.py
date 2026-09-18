from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer
from pathlib import Path

# All data files live in the shared data/ folder at the repo root.
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


# ----------------------------------
# 1. Create empty BPE tokenizer
# ----------------------------------

tokenizer = Tokenizer(
    BPE(unk_token="<unk>")
)

tokenizer.pre_tokenizer = ByteLevel()
tokenizer.decoder = ByteLevelDecoder()


# ----------------------------------
# 2. Configure BPE trainer
# ----------------------------------

trainer = BpeTrainer(
    vocab_size=2000,
    min_frequency=2,
    special_tokens=[
        "<unk>",
        "<pad>",
        "<bos>",
        "<eos>"
    ]
)


# ----------------------------------
# 3. Train tokenizer
# ----------------------------------

tokenizer.train(
    [str(DATA_DIR / "tinystories.txt")],
    trainer
)


# ----------------------------------
# 4. Save tokenizer
# ----------------------------------

tokenizer.save(
    str(DATA_DIR / "tinystories_tokenizer.json")
)


# ----------------------------------
# 5. Test tokenizer
# ----------------------------------

text = "Lily went to her mom and fixed her shirt."

encoded = tokenizer.encode(text)

print("Original:")
print(text)

print("\nTokens:")
print(encoded.tokens)

print("\nToken IDs:")
print(encoded.ids)

print("\nNumber of tokens:")
print(len(encoded.ids))