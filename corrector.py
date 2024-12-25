import re
import string
from collections import Counter

# Function to read the corpus from a text file and extract words
def read_corpus(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        words = []
        for line in lines:
            words += re.findall(r'\w+', line.lower())
    return words

# Load the corpus and compute probabilities
def load_corpus(filename):
    corpus = read_corpus(filename)
    vocab = set(corpus)
    words_count = Counter(corpus)
    total_words_count = float(sum(words_count.values()))
    word_probabs = {word: words_count[word] / total_words_count for word in words_count.keys()}
    return corpus, vocab, word_probabs

# Helper functions for generating word edits
def split(word):
    return [(word[:i], word[i:]) for i in range(len(word) + 1)]

def delete(word):
    return [left + right[1:] for left, right in split(word) if right]

def swap(word):
    return [left + right[1] + right[0] + right[2:] for left, right in split(word) if len(right) > 1]

def replace(word):
    return [left + char + right[1:] for left, right in split(word) if right for char in string.ascii_lowercase]

def insert(word):
    return [left + char + right for left, right in split(word) for char in string.ascii_lowercase]

# Function to generate level one edits
def level_one_edits(word):
    return set(delete(word) + swap(word) + replace(word) + insert(word))

# Function to generate level two edits
def level_two_edits(word):
    return set(e2 for e1 in level_one_edits(word) for e2 in level_one_edits(e1))

# Spell corrector function
def correct_spelling(word, vocab, word_probabs):
    if word in vocab:
        return f"'{word}' is already correct."

    # Generate suggestions
    suggestions = level_one_edits(word) or level_two_edits(word) or [word]
    best_guesses = [w for w in suggestions if w in vocab]

    if not best_guesses:
        return f"Sorry, no suggestions found for '{word}'."

    # Rank suggestions based on probabilities
    suggestions_with_probabs = [(w, word_probabs[w]) for w in best_guesses]
    suggestions_with_probabs.sort(key=lambda x: x[1], reverse=True)

    return f"Suggestions for '{word}':\n" + "\n".join([f"{w}: {p:.4f}" for w, p in suggestions_with_probabs])