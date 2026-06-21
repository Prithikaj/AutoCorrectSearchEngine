"""
corrector.py
------------
Context-aware spelling corrector using:
  - Norvig edit-distance candidates (edit1 + edit2)
  - Unigram language model (P(word))
  - Bigram language model (P(word | prev_word)) for context scoring
  - Confidence scoring to distinguish real words from OOV terms
"""

import re
import string
import difflib
from collections import Counter, defaultdict


# ── Corpus loading ────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def load_corpus(filename: str):
    """
    Returns
    -------
    corpus       : list[str]      all tokens in order
    vocab        : set[str]       unique tokens
    word_probs   : dict[str,float] unigram probabilities
    bigram_probs : dict[str, dict[str,float]]  P(w2 | w1)
    """
    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()

    corpus = _tokenize(text)
    vocab = set(corpus)

    # Unigram model
    counts = Counter(corpus)
    total = float(sum(counts.values()))
    word_probs = {w: counts[w] / total for w in counts}

    # Bigram model  P(w2 | w1)  with Laplace smoothing (k=1)
    bigram_counts: dict[str, Counter] = defaultdict(Counter)
    for w1, w2 in zip(corpus, corpus[1:]):
        bigram_counts[w1][w2] += 1

    V = len(vocab)
    bigram_probs: dict[str, dict[str, float]] = {}
    for w1, followers in bigram_counts.items():
        denom = counts[w1] + V  # Laplace denominator
        bigram_probs[w1] = {w2: (cnt + 1) / denom for w2, cnt in followers.items()}

    return corpus, vocab, word_probs, bigram_probs


# ── Edit-distance candidate generation ───────────────────────────────────────

def _splits(word: str):
    return [(word[:i], word[i:]) for i in range(len(word) + 1)]


def _deletes(word: str):
    return [L + R[1:] for L, R in _splits(word) if R]


def _transposes(word: str):
    return [L + R[1] + R[0] + R[2:] for L, R in _splits(word) if len(R) > 1]


def _replaces(word: str):
    return [L + c + R[1:] for L, R in _splits(word) if R for c in string.ascii_lowercase]


def _inserts(word: str):
    return [L + c + R for L, R in _splits(word) for c in string.ascii_lowercase]


def edits1(word: str) -> set[str]:
    return set(_deletes(word) + _transposes(word) + _replaces(word) + _inserts(word))


def edits2(word: str) -> set[str]:
    return {e2 for e1 in edits1(word) for e2 in edits1(e1)}


# ── Core correction logic ─────────────────────────────────────────────────────

def _known(words, vocab):
    return {w for w in words if w in vocab}


def _score_candidate(candidate: str, word_probs: dict, bigram_probs: dict,
                      prev_word: str | None, alpha: float = 0.85) -> float:
    """
    Blend unigram and bigram scores.
      score = alpha * P(candidate) + (1-alpha) * P(candidate | prev_word)
    Falls back to unigram when no bigram context is available.
    A high alpha (0.85) keeps unigram frequency dominant so the bigram
    context nudges rather than overrides the frequency-based choice.
    """
    unigram = word_probs.get(candidate, 0.0)
    if prev_word and prev_word in bigram_probs:
        bigram = bigram_probs[prev_word].get(candidate, 1e-9)
        return alpha * unigram + (1 - alpha) * bigram
    return unigram


def _is_proper_noun(original_token: str) -> bool:
    """
    Return True if the token looks like a proper noun (name, place, brand).
    Heuristics:
      - Original casing starts with uppercase AND length >= 3
      - Contains digits (e.g. "B2B", "MP3")
      - All uppercase (acronym like "AI", "NLP")
    """
    if not original_token:
        return False
    stripped = original_token.strip("'\".,!?;:()")
    if len(stripped) < 2:
        return False
    if stripped[0].isupper() and len(stripped) >= 3:
        return True
    if stripped.isupper() and len(stripped) >= 2:
        return True
    if any(ch.isdigit() for ch in stripped):
        return True
    return False


def correct_spelling(word: str, vocab: set, word_probs: dict,
                     bigram_probs: dict | None = None,
                     prev_word: str | None = None,
                     original_token: str | None = None,
                     protected_words: set | None = None) -> dict:
    """
    Returns a structured dict:
      {
        "status"     : "correct" | "corrected" | "unknown" | "proper_noun",
        "original"   : str,
        "suggestion" : str,
        "confidence" : float,
        "candidates" : list[tuple[str, float]],
      }

    Skips correction for:
      - Words in protected_words set (user-defined)
      - Proper nouns detected by casing heuristic (original_token starts with capital)
      - Words where no candidates exist at all (truly OOV)
    """
    original_token = original_token or word
    word_lower = word.lower().strip()

    if not word_lower:
        return {"status": "unknown", "original": original_token,
                "suggestion": original_token, "confidence": 0.0, "candidates": []}

    # ── Protected / user-defined words ─────────────────────────
    if protected_words and word_lower in {p.lower() for p in protected_words}:
        return {"status": "correct", "original": original_token,
                "suggestion": original_token, "confidence": 1.0, "candidates": []}

    # ── Proper noun detection (capital letter, acronym, digit) ──
    if _is_proper_noun(original_token):
        return {"status": "proper_noun", "original": original_token,
                "suggestion": original_token, "confidence": 1.0, "candidates": []}

    # ── Already in vocabulary ───────────────────────────────────
    if word_lower in vocab:
        return {"status": "correct", "original": original_token,
                "suggestion": word_lower, "confidence": word_probs.get(word_lower, 0.0),
                "candidates": []}

    bp = bigram_probs or {}

    # ── Build candidate pool ─────────────────────────────────────
    e1 = _known(edits1(word_lower), vocab)
    e2 = _known(edits2(word_lower), vocab) if not e1 else set()
    dl = set(difflib.get_close_matches(word_lower, list(vocab), n=5, cutoff=0.6)) if not e1 and not e2 else set()
    candidates = e1 or e2 or dl

    if not candidates:
        # Absolutely no candidates anywhere — truly OOV, keep as-is (likely a name)
        return {"status": "proper_noun", "original": original_token,
                "suggestion": original_token, "confidence": 0.0, "candidates": []}

    # ── Score and rank ───────────────────────────────────────────
    scored = sorted(
        [(c, _score_candidate(c, word_probs, bp, prev_word)) for c in candidates],
        key=lambda x: x[1],
        reverse=True,
    )

    best, best_score = scored[0]
    max_prob = max(word_probs.values()) if word_probs else 1.0
    confidence = min(best_score / (max_prob + 1e-12), 1.0)

    # ── Guard: suppress only genuinely long OOV words ───────────
    # Words ≥ 8 chars with NO edit-1 candidates AND very low confidence
    # are almost certainly proper nouns / foreign words typed lowercase
    # (e.g. "coimbatore"=10, "prithika"=8).
    # "stduing"=7 is a real 7-char typo so it still gets corrected.
    if not e1 and len(word_lower) >= 8 and confidence < 0.002:
        return {"status": "proper_noun", "original": original_token,
                "suggestion": original_token, "confidence": 0.0, "candidates": []}

    return {
        "status": "corrected",
        "original": original_token,
        "suggestion": best,
        "confidence": round(confidence, 4),
        "candidates": [(c, round(s, 6)) for c, s in scored[:5]],
    }


def correct_query(query: str, vocab: set, word_probs: dict,
                  bigram_probs: dict | None = None,
                  protected_words: set | None = None) -> dict:
    """
    Correct a multi-word query token by token using bigram context.
    Preserves original casing for proper noun detection.

    Returns
    -------
    {
      "original"    : str,
      "corrected"   : str,
      "tokens"      : list[dict],
      "was_changed" : bool,
    }
    """
    # Split preserving original casing for proper noun detection
    original_tokens = query.split()
    results = []
    prev = None

    for orig_tok in original_tokens:
        clean = re.sub(r"[^a-zA-Z0-9]", "", orig_tok)   # strip punctuation for lookup
        res = correct_spelling(
            clean.lower(), vocab, word_probs, bigram_probs,
            prev_word=prev,
            original_token=orig_tok,
            protected_words=protected_words,
        )
        results.append(res)
        prev = res["suggestion"].lower()

    corrected_tokens = [r["suggestion"] for r in results]
    corrected_query = " ".join(corrected_tokens)

    return {
        "original": query,
        "corrected": corrected_query,
        "tokens": results,
        "was_changed": corrected_query.lower() != query.lower(),
    }
