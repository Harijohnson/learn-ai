import re
from pathlib import Path

# Everything for this step lives inside THIS folder (not the shared data/).
HERE = Path(__file__).resolve().parent

RAW_FILE = HERE / "creative_stories.txt"
CLEAN_FILE = HERE / "creative_clean.txt"


# ----------------------------------
# 1. Read the raw file
# ----------------------------------
# creative_stories.txt looks like:
#
#   ===== STORY 1 =====
#   <the story, over several paragraphs>
#   ====================================
#
#   ===== STORY 2 =====
#   ...
#
# We must throw away the "STORY N" headers and the "====" separator
# lines, keeping only the actual story text.

raw_text = RAW_FILE.read_text(encoding="utf-8")


# ----------------------------------
# 2. Split into individual stories
# ----------------------------------
# Split on every "===== STORY <number> =====" header. Whatever sits
# between two headers is one story.

chunks = re.split(r"=====\s*STORY\s+\d+\s*=====", raw_text)


# ----------------------------------
# 3. Clean each story
# ----------------------------------

stories = []

for chunk in chunks:

    # Remove the long "====..====" separator lines.
    text = re.sub(r"^=+$", "", chunk, flags=re.MULTILINE)

    # Collapse newlines + extra spaces into single spaces so each
    # story becomes ONE line. The next steps read line by line.
    text = " ".join(text.split())

    if text:
        stories.append(text)


# ----------------------------------
# 4. Save one story per line
# ----------------------------------

with open(CLEAN_FILE, "w", encoding="utf-8") as file:
    for story in stories:
        file.write(story)
        file.write("\n")


print("Stories found:", len(stories))
print("Saved to:", CLEAN_FILE)
