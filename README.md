# AutoCorrectSearchEngine

## Overview

AutoCorrectSearchEngine is an AI-powered search dashboard built with Streamlit. It combines spelling correction, semantic search, PDF search, contextual AI summaries, and search analytics into one structured user experience.

## Key Features

- **AI Summary 🤖**: Generates a short explanation after search results are returned.
- **Smart Suggestions 🔍**: Provides type-ahead suggestions while the user types a query.
- **Semantic Search 🧠**: Finds meaning-based results, not just exact keyword matches.
- **Chat with Search Results 💬**: Lets users ask questions like "explain this result" about a chosen result.
- **PDF Search 📄**: Upload PDF files and search inside them.
- **Trending Searches 📈**: Displays popular search queries.
- **Search Analytics Dashboard 📊**: Shows total searches, failed searches, top search terms, and failed search rate.

## Files

- `app.py` — Streamlit application UI and search orchestration.
- `search_engine.py` — Corpus loading, exact/semantic search, and suggestion generation.
- `corrector.py` — Spell correction logic and corpus probability model.
- `big.txt` — Corpus data used for search and spelling suggestions.
- `requirements.txt` — Python dependencies.

## Installation

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## Usage

1. Enter a search query or use voice input in the sidebar.
2. Review spelling correction and smart suggestions.
3. View exact search results, semantic results, and PDF search matches.
4. Read the AI-generated summary.
5. Ask questions about any selected result using the chat section.
6. Monitor trending searches and analytics in the dashboard.

## Notes

- The app uses a local TF-IDF semantic search model and fallback AI summarization.
- A Groq API key can be configured via Streamlit secrets if external search capabilities are added later.
- PDF search works for uploaded PDF files using `PyPDF2` text extraction.

## License

This project is available for experimentation and improvement.
