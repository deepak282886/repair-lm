"""
Vocabulary per curriculum stage.
Stages 1-3: controlled gym vocabulary.
Stages 4-7: open vocabulary from corpus.
"""

# ─────────────────────────────────────────────
# Stage 1 — atoms only, most common words
# ─────────────────────────────────────────────

STAGE1_WORDS = [
    # articles / determiners
    "the", "a", "an", "this", "that", "my", "your",
    # pronouns
    "i", "you", "he", "she", "it", "we", "they",
    # verbs
    "is", "are", "was", "run", "sit", "eat", "see",
    "go", "come", "get", "do", "say", "know", "think",
    # nouns
    "cat", "dog", "bird", "fish", "tree", "sun", "moon",
    "house", "door", "book", "food", "water", "ball", "sky",
    # adjectives
    "big", "small", "red", "blue", "green", "fast", "slow",
    "hot", "cold", "old", "new", "good", "bad",
]

# ─────────────────────────────────────────────
# Stage 2 — bigrams, add connectors
# ─────────────────────────────────────────────

STAGE2_EXTRA = [
    "on", "in", "at", "by", "to", "of", "with",
    "and", "but", "or",
    "mat", "hat", "box", "hill", "road", "wall",
    "black", "white", "bright", "dark",
    "play", "look", "walk", "jump", "fly", "swim",
]

# ─────────────────────────────────────────────
# Stage 3 — simple sentences
# ─────────────────────────────────────────────

STAGE3_EXTRA = [
    "can", "will", "not", "no", "yes",
    "very", "more", "most", "so", "too",
    "man", "woman", "child", "boy", "girl",
    "up", "down", "out", "back", "here", "there",
    "make", "take", "give", "find", "keep", "put",
]

# ─────────────────────────────────────────────
# Stage sentence templates (for gym)
# ─────────────────────────────────────────────

# (template, slots)  — slots filled from vocab lists above
STAGE1_TEMPLATES = [
    ["the", "{noun}"],
    ["a", "{noun}"],
    ["the", "{adj}", "{noun}"],
    ["{noun}", "is", "{adj}"],
]

STAGE2_TEMPLATES = [
    ["the", "{noun}", "{verb}"],
    ["a", "{adj}", "{noun}", "{verb}"],
    ["the", "{noun}", "is", "on", "the", "{noun2}"],
    ["the", "{noun}", "{verb}", "and", "{verb2}"],
    ["{pronoun}", "{verb}", "the", "{noun}"],
]

STAGE3_TEMPLATES = [
    ["the", "{noun}", "can", "{verb}"],
    ["{pronoun}", "{verb}", "the", "{adj}", "{noun}"],
    ["the", "{noun}", "{verb}", "{prep}", "the", "{noun2}"],
    ["{noun}", "is", "very", "{adj}"],
    ["the", "{adj}", "{noun}", "{verb}", "{prep}", "the", "{noun2}"],
]

# Slot fill lists
NOUNS   = ["cat","dog","bird","fish","tree","sun","moon",
            "house","book","ball","child","man","woman","box"]
VERBS   = ["run","sit","eat","see","go","walk","jump",
            "fly","swim","play","look","make","find"]
ADJS    = ["big","small","red","blue","green","fast","slow",
            "hot","cold","old","new","good","black","white","bright"]
PRNS    = ["i","you","he","she","it","we","they"]
PREPS   = ["on","in","at","by","to","with"]


def fill_template(template: list) -> list:
    """Fill slot markers with random vocab."""
    import random
    result = []
    used_nouns = []
    for tok in template:
        if tok == "{noun}":
            n = random.choice(NOUNS)
            used_nouns.append(n)
            result.append(n)
        elif tok == "{noun2}":
            # pick different noun if possible
            choices = [n for n in NOUNS if n not in used_nouns]
            result.append(random.choice(choices or NOUNS))
        elif tok == "{verb}" or tok == "{verb2}":
            result.append(random.choice(VERBS))
        elif tok == "{adj}":
            result.append(random.choice(ADJS))
        elif tok == "{pronoun}":
            result.append(random.choice(PRNS))
        elif tok == "{prep}":
            result.append(random.choice(PREPS))
        else:
            result.append(tok)
    return result