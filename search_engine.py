# Basic search function that searches for a word in the corpus and returns surrounding words (context)
def search_corpus(word, corpus, context_size=5):
    results = []
    for i, w in enumerate(corpus):
        if w == word:
            # Extract context around the found word
            start = max(i - context_size, 0)
            end = min(i + context_size + 1, len(corpus))
            context = " ".join(corpus[start:end])
            results.append(context)

    return results if results else None