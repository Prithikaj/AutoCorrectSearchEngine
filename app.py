import streamlit as st
from autocorrect.corrector import correct_spelling, load_corpus
from search.search_engine import search_corpus
from transformers import pipeline
import speech_recognition as sr

# Load corpus data
corpus, vocab, word_probabs = load_corpus('data/big.txt')

# Load Hugging Face POS tagger model
pos_tagger = pipeline("token-classification", model="vblagoje/bert-english-uncased-finetuned-pos", aggregation_strategy="simple")

# Set page config for better appearance
st.set_page_config(page_title="AutoCorrect Dashboard", layout="wide")

# Function to transcribe audio input
def transcribe_audio():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.sidebar.text("Listening...")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        st.sidebar.text(f"Transcribed text: {text}")
        return text
    except sr.UnknownValueError:
        st.sidebar.error("Could not understand audio.")
        return ""
    except sr.RequestError:
        st.sidebar.error("Could not request results from Google Speech Recognition service.")
        return ""

# Sidebar for user input
st.sidebar.header("AutoCorrect Settings")
input_method = st.sidebar.selectbox("Choose Input Method:", ["Text Input", "Voice Input"])

if input_method == "Voice Input":
    word = transcribe_audio()
else:
    word = st.sidebar.text_input("Enter a word to check spelling:", "")

# Main title
st.title("🔍 AutoCorrect & POS Categorized Search System")

# If the user has entered a word, show suggestions and search results
if word:
    # Correct spelling
    result = correct_spelling(word.lower(), vocab, word_probabs)

    # Display the original and corrected word
    st.subheader("Spelling Correction")
    if ":" in result:
        corrected_word = result.split(":")[1].strip()
        st.markdown(f"Original Word: {word}")
        st.markdown(f"Corrected Suggestion: {corrected_word}")
    else:
        st.markdown(f"Original Word: {word}")
        st.markdown("No corrections found.")
        corrected_word = word  # Use the original word if no correction is found

    # Perform search using corrected spelling (first suggestion)
    search_results = search_corpus(corrected_word, corpus)

    # Display search results categorized by POS
    st.subheader("Search Results Categorized by POS")

    # Define the POS categories you want to include
    pos_categories = ["NOUN", "VERB", "ADJ", "ADV", "PROPN", "PRON", "NUM", "INTJ"]

    # Create a dictionary to hold results for each POS category
    pos_results = {pos: [] for pos in pos_categories}

    if search_results:
        for res in search_results:
            # Apply POS tagging to each result
            pos_tags = pos_tagger(res)

            # Check which POS the corrected word appears as
            for tag in pos_tags:
                entity = tag['entity_group']
                if entity in pos_results:  # Check if entity is in pos_results
                    if tag['word'].lower() == corrected_word:
                        pos_results[entity].append(res)
                        break  # Exit the loop after finding the first match

        # Display results for each POS category
        for pos in pos_categories:
            st.subheader(f"Results where '{corrected_word}' is used as a {pos}:")
            if pos_results[pos]:
                for i, res in enumerate(pos_results[pos], start=1):
                    st.markdown(f"Result {i}: {res}")
            else:
                st.markdown(f"No results found where '{corrected_word}' is used as a {pos}.")
    else:
        st.warning(f"No search results found for {corrected_word}.")

# Footer for additional info
st.markdown("---")
st.markdown("### About This Tool")
st.write(
    "This AutoCorrect & POS Categorized Search System helps you find corrections for your misspelled words and search relevant information from a corpus, while categorizing the results based on the grammatical usage of the word."
)