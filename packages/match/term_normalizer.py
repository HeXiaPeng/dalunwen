import json
import os
import re
import unicodedata

def load_synonyms(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # build synonyms -> canonical map
    syn2canon = {}
    for canon, syns in data.get("canonical_to_synonyms", {}).items():
        canon_l = canon.lower()
        syn2canon[canon_l] = canon_l
        for s in syns:
            syn2canon[s.lower()] = canon_l
    return syn2canon

def normalize_text(text, syn2canon):
    t = unicodedata.normalize("NFKC", (text or ""))
    t = t.lower()
    t = t.replace("/", " ").replace("-", " ").replace("_", " ")
    t = re.sub(r"\s+", " ", t).strip()
    # replace synonyms on word boundaries
    words = t.split(" ")
    norm_words = []
    for w in words:
        # simple punctuation strip
        w_clean = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", w)
        canon = syn2canon.get(w_clean, w_clean)
        # keep original punctuation around the word
        prefix = w[:len(w)-len(w.lstrip(".,;:()[]{}"))]
        suffix = w[len(w.rstrip(".,;:()[]{}")):]
        norm_words.append(canon)
    return " ".join(norm_words)

def truncate_text(text, max_chars=450):
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    # prefer cutting at whitespace
    cut = t[:max_chars]
    last_space = cut.rfind(" ")
    if last_space > max_chars * 0.7:
        cut = cut[:last_space]
    return cut

