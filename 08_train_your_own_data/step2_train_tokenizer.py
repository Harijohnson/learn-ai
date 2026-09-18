from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer
from pathlib import Path

# Everything for this step lives inside THIS folder (not the shared data/).
HERE = Path(__file__).resolve().parent


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
# Your stories use richer vocabulary than TinyStories, so a slightly
# bigger vocab (4000) captures more whole words. Lower it to 2000 if
# you want a smaller model.

trainer = BpeTrainer(
    vocab_size=4000,
    min_frequency=2,
    special_tokens=[
        "<unk>",
        "<pad>",
        "<bos>",
        "<eos>"
    ]
)


# ----------------------------------
# 3. Train tokenizer on OUR data
# ----------------------------------

tokenizer.train(
    [str(HERE / "creative_clean.txt")],
    trainer
)


# ----------------------------------
# 4. Save tokenizer
# ----------------------------------

tokenizer.save(
    str(HERE / "creative_tokenizer.json")
)


# ----------------------------------
# 5. Test tokenizer
# ----------------------------------

text = "In the quaint town of Willowbrook lived a young artist."

encoded = tokenizer.encode(text)

print("Original:")
print(text)

print("\nTokens:")
print(encoded.tokens)

print("\nToken IDs:")
print(encoded.ids)

print("\nNumber of tokens:")
print(len(encoded.ids))
