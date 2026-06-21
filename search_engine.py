"""
search_engine.py
----------------
NLP pipeline for the AutoCorrect Search Engine:
  - Corpus loading with unigram statistics
  - Exact keyword search with context windows
  - N-gram frequency analysis
  - Word similarity scoring (edit-distance + frequency blend)
  - Keyboard adjacency model for typo analysis
  - PDF exact-match sentence search
  - Autocomplete suggestions by frequency
  - Keyword highlighting in snippets
  - Result deduplication
"""

import re
import string
from collections import Counter
from itertools import islice

import nltk
import numpy as np

# Download NLTK stopwords once
for _pkg in ("stopwords",):
    try:
        nltk.data.find(f"corpora/{_pkg}")
    except LookupError:
        nltk.download(_pkg, quiet=True)

from nltk.corpus import stopwords as _sw
STOP_WORDS = set(_sw.words("english"))

# ── Keyboard adjacency map ────────────────────────────────────────────────────
# Each key → set of physically adjacent keys (QWERTY layout)
KEYBOARD_ADJACENCY: dict[str, set[str]] = {
    "q": {"w", "a", "s"},          "w": {"q", "e", "a", "s", "d"},
    "e": {"w", "r", "s", "d", "f"},"r": {"e", "t", "d", "f", "g"},
    "t": {"r", "y", "f", "g", "h"},"y": {"t", "u", "g", "h", "j"},
    "u": {"y", "i", "h", "j", "k"},"i": {"u", "o", "j", "k", "l"},
    "o": {"i", "p", "k", "l"},     "p": {"o", "l"},
    "a": {"q", "w", "s", "z"},     "s": {"a", "w", "e", "d", "z", "x"},
    "d": {"s", "e", "r", "f", "x", "c"},"f": {"d", "r", "t", "g", "c", "v"},
    "g": {"f", "t", "y", "h", "v", "b"},"h": {"g", "y", "u", "j", "b", "n"},
    "j": {"h", "u", "i", "k", "n", "m"},"k": {"j", "i", "o", "l", "m"},
    "l": {"k", "o", "p"},
    "z": {"a", "s", "x"},          "x": {"z", "s", "d", "c"},
    "c": {"x", "d", "f", "v"},     "v": {"c", "f", "g", "b"},
    "b": {"v", "g", "h", "n"},     "n": {"b", "h", "j", "m"},
    "m": {"n", "j", "k"},
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, remove stopwords."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if t and t not in STOP_WORDS]


def _highlight(snippet: str, query_tokens: list[str]) -> str:
    """Wrap query terms in **bold** markdown."""
    for tok in query_tokens:
        if not tok:
            continue
        pattern = re.compile(r"\b(" + re.escape(tok) + r")\b", re.IGNORECASE)
        snippet = pattern.sub(r"**\1**", snippet)
    return snippet


def _deduplicate(results: list[dict], threshold: float = 0.85) -> list[dict]:
    """Remove near-duplicate snippets using Jaccard similarity on tokens."""
    kept: list[dict] = []
    for r in results:
        tokens_r = set(r["snippet"].lower().split())
        is_dup = any(
            len(tokens_r & set(k["snippet"].lower().split())) /
            max(len(tokens_r | set(k["snippet"].lower().split())), 1) >= threshold
            for k in kept
        )
        if not is_dup:
            kept.append(r)
    return kept


# ── Corpus loading ────────────────────────────────────────────────────────────

def load_corpus(filename: str):
    """
    Load corpus and build vocabulary + unigram statistics.

    Returns
    -------
    corpus       : list[str]         all tokens in order
    vocab        : set[str]          unique tokens
    word_probs   : dict[str,float]   unigram probabilities
    text_blocks  : list[str]         sentence-level chunks (≥5 words)
    word_counts  : Counter           raw word frequency counts
    """
    with open(filename, "r", encoding="utf-8") as f:
        raw_text = f.read()

    corpus = re.findall(r"[a-z]+", raw_text.lower())
    vocab = set(corpus)
    word_counts = Counter(corpus)
    total = float(sum(word_counts.values()))
    word_probs = {w: word_counts[w] / total for w in word_counts}

    # Build sentence blocks
    raw_blocks = re.split(r"(?<=[.!?])\s+", raw_text)
    text_blocks = []
    buffer = ""
    for blk in raw_blocks:
        blk = blk.strip()
        if not blk:
            continue
        if buffer:
            combined = buffer + " " + blk
            if len(combined) <= 500:
                buffer = combined
                continue
            else:
                text_blocks.append(buffer)
                buffer = blk
        else:
            buffer = blk
    if buffer:
        text_blocks.append(buffer)
    text_blocks = [b for b in text_blocks if len(b.split()) >= 5]

    return corpus, vocab, word_probs, text_blocks, word_counts


# ── Exact keyword search ──────────────────────────────────────────────────────

def search_corpus(word: str, corpus: list[str], text_blocks: list[str],
                  context_size: int = 8) -> list[dict]:
    """
    Find exact token matches and return surrounding context windows.
    Falls back to sentence-level search if no token-level hits.
    """
    word = word.lower()
    pattern = re.compile(r"\b" + re.escape(word) + r"\b")
    results = []

    for i, w in enumerate(corpus):
        if w == word:
            start = max(i - context_size, 0)
            end = min(i + context_size + 1, len(corpus))
            snippet = " ".join(corpus[start:end])
            results.append({
                "snippet": snippet,
                "score": 1.0,
                "source": "Exact",
                "highlighted_snippet": _highlight(snippet, [word]),
            })
            if len(results) >= 6:
                break

    if not results:
        for blk in text_blocks:
            if pattern.search(blk):
                results.append({
                    "snippet": blk,
                    "score": 1.0,
                    "source": "Exact",
                    "highlighted_snippet": _highlight(blk, [word]),
                })
                if len(results) >= 6:
                    break

    return _deduplicate(results)


# ── N-gram frequency analysis ─────────────────────────────────────────────────

def ngram_frequency(text: str, word_counts: Counter, total_words: int,
                    n: int = 2, top_k: int = 15) -> list[tuple[str, int]]:
    """
    Build n-gram counts from a text string and return the top-k by frequency.
    Used to show which phrases are most common in the corpus.

    Parameters
    ----------
    text        : the raw corpus text (lowercased tokens)
    word_counts : pre-computed unigram Counter
    total_words : total token count
    n           : gram size (1=unigram, 2=bigram, 3=trigram)
    top_k       : how many top results to return

    Returns list of (ngram_string, count) tuples.
    """
    tokens = re.findall(r"[a-z]+", text.lower())
    tokens = [t for t in tokens if t not in STOP_WORDS]
    ngrams = zip(*[tokens[i:] for i in range(n)])
    ngram_counts = Counter(" ".join(g) for g in ngrams)
    return ngram_counts.most_common(top_k)


# ── Word similarity explorer ──────────────────────────────────────────────────

def _edit_distance(a: str, b: str) -> int:
    """Standard dynamic-programming edit distance (Levenshtein)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev = dp[:]
        dp[0] = i
        for j, cb in enumerate(b, 1):
            dp[j] = min(prev[j] + 1, dp[j - 1] + 1,
                        prev[j - 1] + (0 if ca == cb else 1))
    return dp[len(b)]


def word_similarity(query_word: str, vocab: set, word_probs: dict,
                    top_n: int = 10, max_edit: int = 2) -> list[dict]:
    """
    Find the top-N most similar words to query_word using a blend of:
      - Inverse edit distance  (closer = more similar)
      - Unigram frequency      (common words score higher)

    Returns list of dicts: {word, edit_dist, frequency, similarity_score}
    """
    query_word = query_word.lower().strip()
    if not query_word:
        return []

    results = []
    for w in vocab:
        if w == query_word:
            continue
        # Only consider words within max_edit distance (for performance)
        # Quick length filter before computing full edit distance
        if abs(len(w) - len(query_word)) > max_edit:
            continue
        dist = _edit_distance(query_word, w)
        if dist > max_edit:
            continue
        freq = word_probs.get(w, 0.0)
        # Blend: edit similarity (0-1) weighted with log frequency
        edit_sim = 1.0 / (1.0 + dist)
        score = 0.6 * edit_sim + 0.4 * (freq / (max(word_probs.values()) + 1e-12))
        results.append({
            "word": w,
            "edit_dist": dist,
            "frequency": word_probs.get(w, 0.0),
            "similarity_score": round(score, 5),
        })

    results.sort(key=lambda x: x["similarity_score"], reverse=True)
    return results[:top_n]


# ── Keyboard typo analysis ────────────────────────────────────────────────────

def keyboard_typo_pairs(vocab: set, word_probs: dict,
                        top_n: int = 20) -> list[dict]:
    """
    Find the most frequent word pairs that differ by exactly one
    adjacent-key substitution (common keyboard typo pattern).

    Returns list of dicts: {original, typo, key_from, key_to, freq_original}
    sorted by frequency of the correct word.
    """
    pairs = []
    seen = set()

    for word in vocab:
        if len(word) < 3:
            continue
        for i, ch in enumerate(word):
            if ch not in KEYBOARD_ADJACENCY:
                continue
            for adj_key in KEYBOARD_ADJACENCY[ch]:
                typo = word[:i] + adj_key + word[i + 1:]
                if typo in vocab and typo != word:
                    key = tuple(sorted([word, typo]))
                    if key not in seen:
                        seen.add(key)
                        pairs.append({
                            "original": word,
                            "typo": typo,
                            "key_from": ch,
                            "key_to": adj_key,
                            "freq_original": round(word_probs.get(word, 0.0), 7),
                        })

    pairs.sort(key=lambda x: x["freq_original"], reverse=True)
    return pairs[:top_n]


# ── PDF search ────────────────────────────────────────────────────────────────

def search_pdf(query: str, pdf_text: str, top_n: int = 6) -> list[dict]:
    """
    Search inside extracted PDF text in two passes:
      1. PDF-Exact  – sentences that literally contain all query tokens.
      2. PDF-Fuzzy  – sentences containing at least one query token
                      (partial match fallback).
    Both passes highlight matched terms.
    """
    if not query or not pdf_text:
        return []

    sentences = [
        s.strip() for s in re.split(r"(?<=[.!?])\s+", pdf_text)
        if len(s.strip()) > 20
    ]
    if not sentences:
        return []

    query_tokens = [t.lower() for t in query.split() if t]

    # Pass 1 – all tokens present
    exact_results = []
    for sent in sentences:
        sent_lower = sent.lower()
        if all(re.search(r"\b" + re.escape(tok) + r"\b", sent_lower)
               for tok in query_tokens):
            exact_results.append({
                "snippet": sent,
                "score": 1.0,
                "source": "PDF-Exact",
                "highlighted_snippet": _highlight(sent, query_tokens),
            })

    # Pass 2 – at least one token present (fuzzy fallback)
    fuzzy_results = []
    for sent in sentences:
        sent_lower = sent.lower()
        matched = [tok for tok in query_tokens
                   if re.search(r"\b" + re.escape(tok) + r"\b", sent_lower)]
        if matched and sent not in {r["snippet"] for r in exact_results}:
            score = round(len(matched) / len(query_tokens), 4)
            fuzzy_results.append({
                "snippet": sent,
                "score": score,
                "source": "PDF-Fuzzy",
                "highlighted_snippet": _highlight(sent, matched),
            })

    fuzzy_results.sort(key=lambda x: x["score"], reverse=True)

    combined = _deduplicate(exact_results)[:top_n] + \
               _deduplicate(fuzzy_results)[:top_n]
    return _deduplicate(combined)[:top_n * 2]


# ── Autocomplete suggestions ──────────────────────────────────────────────────

def extract_suggestions(prefix: str, vocab: set, word_probs: dict,
                        top_n: int = 8) -> list[str]:
    """Return vocabulary words starting with prefix, ranked by frequency."""
    if not prefix or len(prefix) < 2:
        return []
    prefix = prefix.lower().strip()
    matches = [w for w in vocab if w.startswith(prefix) and w != prefix]
    matches.sort(key=lambda w: word_probs.get(w, 0), reverse=True)
    return matches[:top_n]
