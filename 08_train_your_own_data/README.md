# Step 8 · Train a language model on YOUR own data

This folder is **completely self-contained**. Unlike Step 7 (which reads/writes the
shared `data/` folder at the repo root), *everything* here — the raw text, the
cleaned text, the tokenizer, the token tensors, and the trained model — lives
**inside this folder**. Nothing leaks into `data/`.

It takes the file `creative_stories.txt` (2,503 short stories) and turns it into a
tiny transformer language model you can generate new stories from.

---

## Folder contents

**You start with just two things:**

```
08_train_your_own_data/
├── README.md              ← this guide
├── creative_stories.txt   ← the raw data (your input)
├── step1_prepare_data.py
├── step2_train_tokenizer.py
├── step3_prepare_tokens.py
├── step4_train_lm.py
└── step5_generate.py
```

**After you run the pipeline, these files appear here too** (all generated, all
local to this folder):

```
├── creative_clean.txt        ← step 1 output (one story per line, cleaned)
├── creative_tokenizer.json   ← step 2 output (the tokenizer)
├── creative_X.pt             ← step 3 output (model inputs)
├── creative_Y.pt             ← step 3 output (model targets)
└── creative_transformer.pth  ← step 4 output (the trained model)
```

---

## How every script finds its files

Each script starts with the same line:

```python
HERE = Path(__file__).resolve().parent
```

`__file__` is the script's own path, so `HERE` is **this folder**. Every read and
write is `HERE / "something"`. That is the whole trick that keeps the data inside
this folder instead of under `data/` — and it means you can run the scripts from
*any* working directory.

---

## Run it (in order)

Each step writes a file the next step needs, so run them top to bottom:

```powershell
python 08_train_your_own_data/step1_prepare_data.py
python 08_train_your_own_data/step2_train_tokenizer.py
python 08_train_your_own_data/step3_prepare_tokens.py
python 08_train_your_own_data/step4_train_lm.py
python 08_train_your_own_data/step5_generate.py
```

> Using the repo's virtual env? Swap `python` for `venv\Scripts\python.exe`.

---

## What each step does (in detail)

### Step 1 — Clean the raw file → `creative_clean.txt`

**Why:** `creative_stories.txt` mixes real story text with decoration:

```
===== STORY 1 =====
In the quaint town of Willowbrook, ...
(several paragraphs)
====================================================================================================

===== STORY 2 =====
...
```

Those `===== STORY N =====` headers and long `====` separator lines are **not part
of your stories**. If we leave them in, the model wastes capacity learning to
produce `===== STORY` noise.

**How:** the script

1. reads the whole file,
2. splits it on every `===== STORY <number> =====` header (each piece = one story),
3. deletes the `====...====` separator lines,
4. collapses each story's paragraphs into a **single line** (newlines → spaces),
   because the later steps read the file one story per line.

**Expected output:**

```
Stories found: 2503
Saved to: ...\08_train_your_own_data\creative_clean.txt
```

If "Stories found" isn't ~2503, your header format changed — make sure each story
still begins with `===== STORY <number> =====`.

### Step 2 — Train a tokenizer on YOUR words → `creative_tokenizer.json`

**Why:** a model can't read letters, only numbers. A *tokenizer* chops text into
sub-word pieces and gives each a number. Because your stories use richer words than
the toy TinyStories set ("Willowbrook", "enchanting"), we train a **fresh** one.

**How:** a Byte-Pair-Encoding (BPE) tokenizer is trained directly on
`creative_clean.txt` with:

- `vocab_size = 4000` — how many distinct pieces to learn (bigger = more whole
  words survive as single tokens).
- `min_frequency = 2` — ignore pieces that appear only once.
- special tokens `<unk> <pad> <bos> <eos>` — reserved slots. `<eos>`
  ("end of story") is used in step 3 to mark where one story ends.

It prints a sample tokenization at the end so you can eyeball that it works.

### Step 3 — Turn stories into training pairs → `creative_X.pt`, `creative_Y.pt`

**Why:** a language model learns by **predicting the next token**. So training data
is pairs of (what it sees, what should come next).

**How:**

1. Encode every story to token IDs, appending `<eos>` after each one.
2. Glue them into one long stream of tokens.
3. Slide a window of `block_size = 32` across the stream. For each window:
   - `X` = the 32 tokens (the input)
   - `Y` = those same tokens shifted right by one (the "next token" at every
     position)

`block_size` here **must equal** `max_length` in step 4 (both are 32).

**Expected output:**

```
Stories: 2503
Total tokens: <a few hundred thousand>
Input shape: torch.Size([N, 32])
Target shape: torch.Size([N, 32])
```

### Step 4 — Train the model → `creative_transformer.pth`

**Why:** this is where learning actually happens.

**The model** (a small GPT-style transformer):

- token embedding + position embedding (turn IDs into vectors that carry meaning
  and order),
- **2** stacked transformer blocks — each does masked/causal self-attention
  (every token may look only at earlier tokens) plus a feed-forward layer,
- a final linear layer that outputs a score for **every** vocabulary token — the
  model's guess for what comes next.

**Training loop:** 30 epochs, AdamW optimizer (`lr = 0.001`), 90% of the data for
training and 10% held out for validation. Each epoch prints:

```
Epoch 5/30 | train loss: 4.1234 | val loss: 4.3456
```

**What to watch:** both losses should fall. If `val loss` climbs while `train loss`
keeps dropping, the model is **overfitting** (memorizing) — common with only ~2500
stories. Fixes: fewer epochs, or more data.

> This is the slow step. On CPU expect **minutes**, not seconds.

### Step 5 — Generate text → prints to the screen

Loads the tokenizer + trained model and continues a prompt one token at a time,
using **temperature** to control randomness (lower = safer/repetitive, higher =
more surprising). It tries a few prompts tuned to your data, e.g.
`"In the quaint town"`.

Expect **story-like, on-topic** text — not polished prose. It's a tiny model on a
small dataset.

---

## Knobs you can turn

| Want... | Change | In file |
| ------- | ------ | ------- |
| Bigger vocabulary | `vocab_size` (4000 → 6000) | `step2` |
| Longer context | `block_size` **and** `max_length` (keep equal) | `step3` + `step4` |
| A stronger model | `embedding_size` (64 → 128) or add more blocks | `step4` |
| Train longer/shorter | `epochs` | `step4` |
| More/less random output | `temperature` | `step5` |

> If you change `vocab_size`, `block_size`, `embedding_size`, or the number of
> blocks, re-run **from the step that produces the affected file** onward. Changing
> the tokenizer means re-running steps 2 → 5.

---

## Use your OWN stories instead

Want to train on something else entirely? Two options:

- **Same format:** replace `creative_stories.txt` with your file using the same
  `===== STORY N =====` layout, then re-run from step 1.
- **Different format:** put your text in `creative_clean.txt` yourself — just make
  it **one story (or document) per line** — and skip straight to step 2.

---

## Troubleshooting

- **`FileNotFoundError`** for a `.txt`/`.json`/`.pt`/`.pth` — you skipped an earlier
  step. Run them in order.
- **Output full of `<unk>`** — the tokenizer never saw those words. Retrain the
  tokenizer (step 2) on the same data you generate from.
- **Loss won't drop** — dataset is small; try more `epochs`, a larger
  `embedding_size`, or more data.
- **It's slow** — training on CPU is expected to take a few minutes; that's normal.

---

⬅️ Prev: [Step 7 · Language model](../07_language_model/)
🏠 Back to the [project overview](../README.md)
