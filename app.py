"""
app.py  –  WordRight: Smart AutoCorrect & Search
=================================================
A real-world NLP application for everyday users.
Pages:
  🏠 Home          – landing page with feature overview
  ✏️  AutoCorrect   – type any text, get instant corrections
  🔍 Search        – search the knowledge corpus
  📄 PDF Analyzer  – upload & search any PDF document
  📊 Analytics     – your personal usage dashboard
"""

import re
import io
import time
from collections import Counter

import pandas as pd
import plotly.express as px
import streamlit as st
from PyPDF2 import PdfReader

try:
    import pypdf as _pypdf
    _HAS_PYPDF = True
except ImportError:
    _HAS_PYPDF = False
try:
    import pdfplumber as _pdfplumber
    _HAS_PDFPLUMBER = True
except ImportError:
    _HAS_PDFPLUMBER = False

from corrector import load_corpus as load_corrector_corpus, correct_query, correct_spelling
from search_engine import (
    load_corpus, search_corpus, search_pdf,
    extract_suggestions, ngram_frequency,
    word_similarity, keyboard_typo_pairs, KEYBOARD_ADJACENCY,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WordRight – Smart AutoCorrect & Search",
    page_icon="✏️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
[data-testid="stAppViewContainer"] { background: #0f1117; }
[data-testid="stSidebar"] { display: none; }
[data-testid="collapsedControl"] { display: none; }

/* ── Top nav bar ── */
.navbar {
  display: flex; align-items: center; gap: 10px;
  background: #161b2e; border-bottom: 1px solid #2d3651;
  padding: 10px 24px; margin: -1rem -1rem 1.5rem -1rem;
  position: sticky; top: 0; z-index: 999;
}
.nav-logo { font-size: 1.4rem; font-weight: 800; color: #4f8bf9; letter-spacing: -0.5px; }
.nav-logo span { color: #e2e8f0; }
.nav-links { display: flex; gap: 6px; margin-left: auto; }
.nav-btn {
  background: transparent; border: none; color: #a0aec0;
  padding: 6px 14px; border-radius: 8px; font-size: 0.88rem;
  cursor: pointer; transition: all .15s;
}
.nav-btn:hover, .nav-btn.active {
  background: #2d3651; color: #e2e8f0;
}

/* ── Hero ── */
.hero {
  background: linear-gradient(135deg, #161b2e 0%, #1a2744 100%);
  border: 1px solid #2d3651; border-radius: 16px;
  padding: 48px 40px; text-align: center; margin-bottom: 2rem;
}
.hero h1 { font-size: 2.8rem; font-weight: 800; color: #e2e8f0; margin: 0 0 12px; }
.hero h1 span { color: #4f8bf9; }
.hero p  { font-size: 1.1rem; color: #a0aec0; max-width: 560px; margin: 0 auto 24px; }

/* ── Feature cards ── */
.feat-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 2rem; }
.feat-card {
  background: #161b2e; border: 1px solid #2d3651;
  border-radius: 12px; padding: 24px; text-align: center;
}
.feat-card .icon { font-size: 2rem; margin-bottom: 10px; }
.feat-card h3 { color: #e2e8f0; font-size: 1rem; margin: 0 0 6px; }
.feat-card p  { color: #718096; font-size: 0.83rem; margin: 0; line-height: 1.5; }

/* ── Correction UI ── */
.correction-output {
  background: #161b2e; border: 1px solid #2d3651;
  border-radius: 12px; padding: 20px; margin-top: 16px;
}
.token-row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.tok {
  display: inline-flex; align-items: center; gap: 6px;
  border-radius: 8px; padding: 6px 12px; font-size: 0.92rem;
  font-weight: 500;
}
.tok-ok         { background: #1a3d2b; color: #68d391; }
.tok-fixed      { background: #3d2b1a; color: #f6ad55; }
.tok-unknown    { background: #3d1a1a; color: #fc8181; }
.tok-propernoun { background: #1a2744; color: #90cdf4; }
.tok-arrow      { color: #718096; font-size: 0.75rem; }

/* ── Search bar ── */
.search-wrap {
  background: #161b2e; border: 2px solid #2d3651;
  border-radius: 50px; padding: 4px 6px 4px 20px;
  display: flex; align-items: center; gap: 8px;
  transition: border-color .2s; margin-bottom: 1.5rem;
}
.search-wrap:focus-within { border-color: #4f8bf9; }

/* ── Result card ── */
.result-card {
  background: #161b2e; border: 1px solid #2d3651;
  border-left: 4px solid #4f8bf9;
  border-radius: 10px; padding: 16px 20px; margin-bottom: 12px;
}
.result-card.exact { border-left-color: #68d391; }
.result-meta { display: flex; gap: 8px; margin-bottom: 8px; align-items: center; }
.badge {
  display: inline-block; border-radius: 20px;
  padding: 2px 10px; font-size: 0.74rem; font-weight: 600;
}
.badge-rank   { background:#4f8bf9; color:#fff; }
.badge-source { background:#2d3651; color:#a0aec0; }
.badge-score  { background:#1a3d2b; color:#68d391; }
.result-text  { color: #cbd5e0; font-size: 0.91rem; line-height: 1.65; }

/* ── Stats card ── */
.stat-card {
  background: #161b2e; border: 1px solid #2d3651;
  border-radius: 12px; padding: 20px; text-align: center;
}
.stat-card .val { font-size: 2rem; font-weight: 800; color: #4f8bf9; }
.stat-card .lbl { font-size: 0.82rem; color: #718096; margin-top: 2px; }

/* ── Misc ── */
.section-title {
  font-size: 1.25rem; font-weight: 700; color: #e2e8f0;
  margin: 0 0 12px; padding-bottom: 8px;
  border-bottom: 1px solid #2d3651;
}
.tip-box {
  background: #1a2744; border: 1px solid #4f8bf9;
  border-radius: 8px; padding: 10px 14px;
  font-size: 0.84rem; color: #a0aec0; margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────
CORPUS_FILE = "big.txt"

@st.cache_resource(show_spinner="Loading language model…")
def get_all_data():
    corpus, vocab, word_probs, text_blocks, word_counts = load_corpus(CORPUS_FILE)
    _, _, word_probs_lm, bigram_probs = load_corrector_corpus(CORPUS_FILE)
    raw_text = open(CORPUS_FILE, encoding="utf-8").read()
    return corpus, vocab, word_probs, text_blocks, word_counts, word_probs_lm, bigram_probs, raw_text

corpus, vocab, word_probs, text_blocks, word_counts, word_probs_lm, bigram_probs, raw_text = get_all_data()
total_words = sum(word_counts.values())

# ── Session state ─────────────────────────────────────────────────────────────
_defaults = {
    "page": "Home",
    "search_history": [],
    "search_counts": Counter(),
    "autocorrect_history": [],
    "failed_searches": 0,
    "total_searches": 0,
    "pdf_text": "",
    "pdf_name": "",
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Navigation ────────────────────────────────────────────────────────────────
PAGES = ["Home", "AutoCorrect", "Search", "PDF Analyzer", "Analytics"]
PAGE_ICONS = {"Home": "🏠", "AutoCorrect": "✏️", "Search": "🔍",
              "PDF Analyzer": "📄", "Analytics": "📊"}

cols = st.columns([2, 1, 1, 1, 1, 1])
with cols[0]:
    st.markdown('<div style="font-size:1.3rem;font-weight:800;color:#4f8bf9;padding-top:4px;">✏️ Word<span style="color:#e2e8f0;">Right</span></div>', unsafe_allow_html=True)
for i, page in enumerate(PAGES):
    with cols[i + 1]:
        active_style = "background:#2d3651;color:#e2e8f0;" if st.session_state.page == page else "background:transparent;color:#a0aec0;"
        if st.button(f"{PAGE_ICONS[page]} {page}", key=f"nav_{page}",
                     use_container_width=True):
            st.session_state.page = page
            st.rerun()

st.markdown("<hr style='border-color:#2d3651;margin:0 0 1.5rem;'>", unsafe_allow_html=True)
page = st.session_state.page

# ── Shared helpers ────────────────────────────────────────────────────────────
def load_pdf_file(uploaded) -> tuple[str, str | None]:
    try:
        raw_bytes = uploaded.read()
    except Exception as e:
        return "", str(e)
    if _HAS_PDFPLUMBER:
        try:
            with _pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
                t = "\n".join(p.extract_text() or "" for p in pdf.pages).strip()
            if t:
                return t, None
        except Exception:
            pass
    if _HAS_PYPDF:
        try:
            r = _pypdf.PdfReader(io.BytesIO(raw_bytes))
            t = "\n".join(p.extract_text() or "" for p in r.pages).strip()
            if t:
                return t, None
        except Exception:
            pass
    try:
        r = PdfReader(io.BytesIO(raw_bytes))
        t = "\n".join(p.extract_text() or "" for p in r.pages).strip()
        if t:
            return t, None
    except Exception as e:
        return "", str(e)
    return "", "No text could be extracted (scanned/image PDF)."


def render_result(res: dict, rank: int):
    src = res.get("source", "")
    card_cls = "result-card exact" if "Exact" in src else "result-card"
    score = res.get("score", 0)
    snippet = res.get("highlighted_snippet", res.get("snippet", ""))
    st.markdown(f"""
    <div class="{card_cls}">
      <div class="result-meta">
        <span class="badge badge-rank">#{rank}</span>
        <span class="badge badge-source">{src}</span>
        <span class="badge badge-score">score {score:.3f}</span>
      </div>
      <div class="result-text">{snippet}</div>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════
if page == "Home":
    st.markdown("""
    <div class="hero">
      <h1>Meet <span>WordRight</span></h1>
      <p>An intelligent writing assistant that fixes your spelling, searches your documents,
         and understands what you mean — even when you make mistakes.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="feat-grid">
      <div class="feat-card">
        <div class="icon">✏️</div>
        <h3>Smart AutoCorrect</h3>
        <p>Paste any text and get instant corrections. Understands context using a bigram language model.</p>
      </div>
      <div class="feat-card">
        <div class="icon">🔍</div>
        <h3>Intelligent Search</h3>
        <p>Search a large knowledge corpus with typo tolerance. Even misspelled queries return great results.</p>
      </div>
      <div class="feat-card">
        <div class="icon">📄</div>
        <h3>PDF Analyzer</h3>
        <p>Upload any PDF and instantly search its contents sentence by sentence with keyword highlighting.</p>
      </div>
      <div class="feat-card">
        <div class="icon">💡</div>
        <h3>Autocomplete</h3>
        <p>Real-time word suggestions based on corpus frequency as you type.</p>
      </div>
      <div class="feat-card">
        <div class="icon">🔗</div>
        <h3>Word Explorer</h3>
        <p>Explore similar words, see edit distances and how the corrector ranks its candidates.</p>
      </div>
      <div class="feat-card">
        <div class="icon">📊</div>
        <h3>Usage Analytics</h3>
        <p>Track your search history, success rates, and most-searched terms over time.</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Quick Start</div>', unsafe_allow_html=True)
    q1, q2, q3 = st.columns(3)
    with q1:
        st.markdown("""
        <div class="tip-box">
        <b>1. Fix your writing</b><br>
        Go to <b>✏️ AutoCorrect</b> → paste or type any text → see instant corrections word by word.
        </div>""", unsafe_allow_html=True)
    with q2:
        st.markdown("""
        <div class="tip-box">
        <b>2. Search anything</b><br>
        Go to <b>🔍 Search</b> → type your query (typos are fine!) → get highlighted results from the corpus.
        </div>""", unsafe_allow_html=True)
    with q3:
        st.markdown("""
        <div class="tip-box">
        <b>3. Search your PDF</b><br>
        Go to <b>📄 PDF Analyzer</b> → upload your document → search its contents instantly.
        </div>""", unsafe_allow_html=True)

    # Corpus stats
    st.markdown('<div class="section-title" style="margin-top:1.5rem;">Corpus Statistics</div>', unsafe_allow_html=True)
    cs1, cs2, cs3, cs4 = st.columns(4)
    cs1.markdown(f'<div class="stat-card"><div class="val">{len(vocab):,}</div><div class="lbl">Unique Words</div></div>', unsafe_allow_html=True)
    cs2.markdown(f'<div class="stat-card"><div class="val">{total_words:,}</div><div class="lbl">Total Tokens</div></div>', unsafe_allow_html=True)
    cs3.markdown(f'<div class="stat-card"><div class="val">{len(text_blocks):,}</div><div class="lbl">Text Blocks</div></div>', unsafe_allow_html=True)
    cs4.markdown(f'<div class="stat-card"><div class="val">{len(bigram_probs):,}</div><div class="lbl">Bigram Pairs</div></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# PAGE: AUTOCORRECT
# ══════════════════════════════════════════════════════════════════
elif page == "AutoCorrect":
    st.markdown('<h2 style="color:#e2e8f0;margin-bottom:4px;">✏️ AutoCorrect</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color:#718096;margin-bottom:1.5rem;">Type or paste any text. WordRight checks every word and suggests corrections instantly.</p>', unsafe_allow_html=True)

    input_col, opts_col = st.columns([3, 1])

    with opts_col:
        st.markdown('<div class="section-title">Options</div>', unsafe_allow_html=True)
        show_correct = st.checkbox("Show correct words", value=True)
        show_unknown = st.checkbox("Show unknown words", value=True)
        show_alts    = st.checkbox("Show alternatives", value=False)

        st.markdown('<div class="section-title" style="margin-top:1rem;">Protected Words</div>', unsafe_allow_html=True)
        st.caption("Names, places, brands — one per line. These are never changed.")
        protected_raw = st.text_area(
            "protected_words", height=90,
            placeholder="Coimbatore\nDelhi\nGoogle\nPython",
            label_visibility="collapsed", key="protected_raw",
        )
        protected_words = {w.strip() for w in protected_raw.splitlines() if w.strip()}

        st.markdown('<div class="section-title" style="margin-top:1rem;">Voice Input</div>', unsafe_allow_html=True)
        st.caption("Record your voice — works in browser, no microphone setup needed.")
        try:
            from audiorecorder import audiorecorder
            _HAS_AUDIO = True
        except ImportError:
            _HAS_AUDIO = False

        if _HAS_AUDIO:
            audio = audiorecorder("🎙️ Click to record", "⏹️ Stop recording", key="voice_rec")
            if len(audio) > 0:
                # Save to a BytesIO buffer and transcribe
                import io as _io
                import speech_recognition as _sr
                buf = _io.BytesIO()
                audio.export(buf, format="wav")
                buf.seek(0)
                recognizer = _sr.Recognizer()
                try:
                    with _sr.AudioFile(buf) as source:
                        audio_data = recognizer.record(source)
                    st.session_state["voice_text"] = recognizer.recognize_google(audio_data)
                    st.success(f"Heard: **{st.session_state['voice_text']}**")
                    st.rerun()
                except _sr.UnknownValueError:
                    st.warning("Could not understand the audio. Please speak clearly.")
                except Exception as e:
                    st.error(f"Transcription error: {e}")
        else:
            st.info("Install `streamlit-audiorecorder` for voice input.")

    with input_col:
        voice_default = st.session_state.get("voice_text", "")
        user_text = st.text_area(
            "Your text",
            value=voice_default,
            placeholder="e.g.  I wnt to lern machne lerning and artficial inteligence",
            height=140,
            label_visibility="collapsed",
        )
        check_btn = st.button("✅ Check & Correct", type="primary", use_container_width=True)

    if check_btn and user_text.strip():
        correction = correct_query(user_text.strip(), vocab=vocab,
                                   word_probs=word_probs_lm, bigram_probs=bigram_probs,
                                   protected_words=protected_words)
        tokens = correction["tokens"]

        # ── Corrected text output ──────────────────────────────
        st.markdown('<div class="section-title" style="margin-top:1.5rem;">Corrected Text</div>', unsafe_allow_html=True)
        corrected_text = correction["corrected"]
        st.markdown(f"""
        <div class="correction-output">
          <p style="color:#e2e8f0;font-size:1.05rem;line-height:1.7;margin:0;">{corrected_text}</p>
        </div>""", unsafe_allow_html=True)
        st.code(corrected_text, language=None)   # copyable version

        # ── Word-by-word breakdown ─────────────────────────────
        st.markdown('<div class="section-title" style="margin-top:1.5rem;">Word-by-Word Breakdown</div>', unsafe_allow_html=True)
        chips_html = '<div class="token-row">'
        for t in tokens:
            if t["status"] == "correct":
                if show_correct:
                    chips_html += f'<span class="tok tok-ok">✓ {t["suggestion"]}</span>'
            elif t["status"] == "corrected":
                chips_html += (f'<span class="tok tok-fixed">'
                               f'{t["original"]} <span class="tok-arrow">→</span> {t["suggestion"]}'
                               f'</span>')
            elif t["status"] == "proper_noun":
                chips_html += f'<span class="tok tok-propernoun">📍 {t["original"]}</span>'
            else:
                if show_unknown:
                    chips_html += f'<span class="tok tok-unknown">? {t["original"]}</span>'
        chips_html += '</div>'
        st.markdown(chips_html, unsafe_allow_html=True)

        # Legend
        st.markdown("""
        <div style="display:flex;gap:14px;margin-top:10px;font-size:0.78rem;color:#718096;">
          <span><span style="color:#68d391;">✓ Green</span> — already correct</span>
          <span><span style="color:#f6ad55;">→ Orange</span> — corrected</span>
          <span><span style="color:#90cdf4;">📍 Blue</span> — proper noun / name (kept as-is)</span>
          <span><span style="color:#fc8181;">? Red</span> — unknown word</span>
        </div>""", unsafe_allow_html=True)

        # ── Alternatives table ─────────────────────────────────
        if show_alts:
            corrected_tokens = [t for t in tokens if t["status"] == "corrected"]
            if corrected_tokens:
                st.markdown('<div class="section-title" style="margin-top:1.5rem;">Top Alternatives</div>', unsafe_allow_html=True)
                rows = []
                for t in corrected_tokens:
                    for cand, score in (t.get("candidates") or [])[:4]:
                        rows.append({"Original": t["original"], "Candidate": cand,
                                     "Score": score, "Chosen": cand == t["suggestion"]})
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # ── Summary stats ──────────────────────────────────────
        n_corrected = sum(1 for t in tokens if t["status"] == "corrected")
        n_unknown   = sum(1 for t in tokens if t["status"] == "unknown")
        n_correct   = sum(1 for t in tokens if t["status"] in ("correct", "proper_noun"))
        n_proper    = sum(1 for t in tokens if t["status"] == "proper_noun")
        st.markdown(f"""
        <div style="display:flex;gap:12px;margin-top:1.2rem;">
          <div class="stat-card" style="flex:1;padding:14px;">
            <div class="val" style="font-size:1.5rem;">{n_correct}</div>
            <div class="lbl">Already correct</div>
          </div>
          <div class="stat-card" style="flex:1;padding:14px;">
            <div class="val" style="font-size:1.5rem;color:#f6ad55;">{n_corrected}</div>
            <div class="lbl">Corrections made</div>
          </div>
          <div class="stat-card" style="flex:1;padding:14px;">
            <div class="val" style="font-size:1.5rem;color:#90cdf4;">{n_proper}</div>
            <div class="lbl">Names kept</div>
          </div>
          <div class="stat-card" style="flex:1;padding:14px;">
            <div class="val" style="font-size:1.5rem;color:#fc8181;">{n_unknown}</div>
            <div class="lbl">Unknown words</div>
          </div>
        </div>""", unsafe_allow_html=True)

        # Save to history
        st.session_state.autocorrect_history.append({
            "original": user_text.strip()[:60],
            "corrected": corrected_text[:60],
            "corrections": n_corrected,
            "time": time.strftime("%H:%M:%S"),
        })

    elif not user_text.strip():
        st.markdown("""
        <div class="tip-box">
          💡 <b>Try it:</b> paste a sentence with typos like
          <em>"I wnt to lern machne lerning"</em> and click Check &amp; Correct.
        </div>""", unsafe_allow_html=True)

    # ── Autocomplete widget ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-title">💡 Autocomplete</div>', unsafe_allow_html=True)
    ac_prefix = st.text_input("Start typing a word…", placeholder="e.g.  spel", key="ac_prefix")
    if ac_prefix.strip():
        suggs = extract_suggestions(ac_prefix.strip(), vocab, word_probs, top_n=10)
        if suggs:
            cols_ac = st.columns(5)
            for i, s in enumerate(suggs):
                with cols_ac[i % 5]:
                    st.markdown(f"""
                    <div style="background:#1e2130;border:1px solid #2d3651;border-radius:8px;
                                padding:6px 10px;text-align:center;font-size:0.88rem;color:#e2e8f0;
                                margin-bottom:6px;">{s}</div>""", unsafe_allow_html=True)
        else:
            st.info("No suggestions — try a different prefix.")

# ══════════════════════════════════════════════════════════════════
# PAGE: SEARCH
# ══════════════════════════════════════════════════════════════════
elif page == "Search":
    st.markdown('<h2 style="color:#e2e8f0;margin-bottom:4px;">🔍 Search</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color:#718096;margin-bottom:1rem;">Search the knowledge corpus. Typos are automatically corrected before searching.</p>', unsafe_allow_html=True)

    # ── Big search bar ─────────────────────────────────────────────────────
    search_col, btn_col = st.columns([5, 1])
    with search_col:
        query_input = st.text_input("search_input", placeholder="🔍  Search anything…",
                                    label_visibility="collapsed", key="search_input")
    with btn_col:
        search_btn = st.button("Search", type="primary", use_container_width=True)

    # Autocomplete suggestions
    last_word = query_input.split()[-1] if query_input.strip() else ""
    suggs = extract_suggestions(last_word, vocab, word_probs, top_n=5)
    if suggs and query_input.strip():
        sugg_cols = st.columns(len(suggs))
        for i, s in enumerate(suggs):
            with sugg_cols[i]:
                if st.button(s, key=f"sq_{s}"):
                    words = query_input.split(); words[-1] = s
                    st.session_state["search_input"] = " ".join(words)
                    st.rerun()

    if search_btn and query_input.strip():
        correction = correct_query(query_input.strip(), vocab=vocab,
                                   word_probs=word_probs_lm, bigram_probs=bigram_probs)
        corrected_q = correction["corrected"]

        if correction["was_changed"]:
            st.info(f"🔧 Searching for **{corrected_q}** (auto-corrected from *{query_input}*)")

        with st.spinner("Searching…"):
            first_word = corrected_q.split()[0] if corrected_q else ""
            results = search_corpus(first_word, corpus, text_blocks)

        # Register analytics
        st.session_state.total_searches += 1
        st.session_state.search_counts[query_input.strip()] += 1
        st.session_state.search_history.append({
            "query": query_input.strip(),
            "corrected": corrected_q,
            "found": bool(results),
            "timestamp": time.strftime("%H:%M:%S"),
        })
        if not results:
            st.session_state.failed_searches += 1

        if results:
            st.markdown(f'<div class="section-title">Found {len(results)} result(s) for "{corrected_q}"</div>', unsafe_allow_html=True)
            for rank, res in enumerate(results, 1):
                render_result(res, rank)

            # Frequency bar for query terms
            q_tokens = [t for t in corrected_q.split() if t in word_counts]
            if q_tokens:
                st.markdown("---")
                st.markdown('<div class="section-title">Query term frequency in corpus</div>', unsafe_allow_html=True)
                df_freq = pd.DataFrame({"Word": q_tokens, "Count": [word_counts[t] for t in q_tokens]})
                fig = px.bar(df_freq, x="Word", y="Count", color="Count",
                             color_continuous_scale="Blues", height=220)
                fig.update_layout(margin=dict(t=10, b=10), paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="rgba(0,0,0,0)", font_color="#a0aec0")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No results found. Try different keywords or check spelling.")

    elif not query_input.strip():
        # Show popular searches or suggestions
        st.markdown('<div class="section-title">💡 Try searching for…</div>', unsafe_allow_html=True)
        examples = ["justice", "adventure", "discovery", "knowledge", "wonder", "truth"]
        ex_cols = st.columns(3)
        for i, ex in enumerate(examples):
            with ex_cols[i % 3]:
                if st.button(f"🔍 {ex}", key=f"ex_{ex}", use_container_width=True):
                    st.session_state["search_input"] = ex
                    st.rerun()

    # ── Word Similarity Explorer (bonus feature on search page) ─────────────
    st.markdown("---")
    with st.expander("🔗 Word Similarity Explorer — see how the corrector ranks candidates"):
        sim_w = st.text_input("Enter any word (try a misspelling)", placeholder="speling", key="sim_w")
        max_ed = st.radio("Max edit distance", [1, 2], horizontal=True, key="sim_ed")
        if sim_w.strip():
            with st.spinner("Computing…"):
                sim_res = word_similarity(sim_w.strip(), vocab, word_probs, top_n=8, max_edit=max_ed)
            if sim_res:
                df_s = pd.DataFrame(sim_res)
                df_s["freq_%"] = (df_s["frequency"] * 100).round(5)
                c1, c2 = st.columns(2)
                with c1:
                    fig_s = px.bar(df_s, x="word", y="similarity_score",
                                   color="similarity_score", color_continuous_scale="Viridis",
                                   title="Candidate ranking", height=280)
                    fig_s.update_layout(margin=dict(t=30,b=10), paper_bgcolor="rgba(0,0,0,0)",
                                        plot_bgcolor="rgba(0,0,0,0)", font_color="#a0aec0")
                    st.plotly_chart(fig_s, use_container_width=True)
                with c2:
                    st.dataframe(
                        df_s[["word","edit_dist","freq_%","similarity_score"]].rename(columns={
                            "word":"Word","edit_dist":"Edit Dist",
                            "freq_%":"Freq %","similarity_score":"Score"}),
                        use_container_width=True, hide_index=True)
            else:
                st.info(f"No similar words within edit distance {max_ed}.")

# ══════════════════════════════════════════════════════════════════
# PAGE: PDF ANALYZER
# ══════════════════════════════════════════════════════════════════
elif page == "PDF Analyzer":
    st.markdown('<h2 style="color:#e2e8f0;margin-bottom:4px;">📄 PDF Analyzer</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color:#718096;margin-bottom:1.5rem;">Upload any PDF and search its contents. WordRight finds exact sentences containing your keywords.</p>', unsafe_allow_html=True)

    upload_col, info_col = st.columns([2, 1])
    with upload_col:
        uploaded = st.file_uploader("Upload a PDF", type=["pdf"], label_visibility="collapsed")
        if uploaded:
            with st.spinner("Reading PDF…"):
                pdf_text, pdf_err = load_pdf_file(uploaded)
            if pdf_err:
                st.error(f"Could not extract text: {pdf_err}")
            elif pdf_text:
                st.session_state.pdf_text = pdf_text
                st.session_state.pdf_name = uploaded.name
                st.success(f"✅ **{uploaded.name}** loaded — {len(pdf_text):,} characters")
            else:
                st.warning("PDF read but no text found (may be a scanned image).")

    with info_col:
        if st.session_state.pdf_text:
            words_in_pdf = len(st.session_state.pdf_text.split())
            sents_in_pdf = len(re.split(r"(?<=[.!?])\s+", st.session_state.pdf_text))
            st.markdown(f"""
            <div class="stat-card" style="margin-bottom:10px;">
              <div class="val" style="font-size:1.4rem;">{words_in_pdf:,}</div>
              <div class="lbl">Words in PDF</div>
            </div>
            <div class="stat-card">
              <div class="val" style="font-size:1.4rem;">{sents_in_pdf:,}</div>
              <div class="lbl">Sentences found</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="tip-box">
              Upload a searchable PDF (not a scanned image). <br>
              Works great with reports, research papers, notes, and books.
            </div>""", unsafe_allow_html=True)

    if st.session_state.pdf_text:
        st.markdown("---")
        pdf_q_col, pdf_btn_col = st.columns([5, 1])
        with pdf_q_col:
            pdf_query = st.text_input("Search inside PDF", placeholder="🔍  Type a word or phrase…",
                                      label_visibility="collapsed", key="pdf_query")
        with pdf_btn_col:
            pdf_search_btn = st.button("Search PDF", type="primary", use_container_width=True)

        if pdf_search_btn and pdf_query.strip():
            # Correct the query first
            corr = correct_query(pdf_query.strip(), vocab=vocab,
                                 word_probs=word_probs_lm, bigram_probs=bigram_probs)
            pdf_cq = corr["corrected"]
            if corr["was_changed"]:
                st.info(f"🔧 Searching for **{pdf_cq}** (auto-corrected from *{pdf_query}*)")

            with st.spinner("Searching PDF…"):
                pdf_results = search_pdf(pdf_cq, st.session_state.pdf_text, top_n=10)

            if pdf_results:
                by_src: dict[str, list] = {}
                for r in pdf_results:
                    by_src.setdefault(r["source"], []).append(r)

                if "PDF-Exact" in by_src:
                    n = len(by_src["PDF-Exact"])
                    st.markdown(f'<div class="section-title">📌 {n} exact match(es) — sentences containing "{pdf_cq}"</div>', unsafe_allow_html=True)
                    for rank, res in enumerate(by_src["PDF-Exact"], 1):
                        render_result(res, rank)

                if "PDF-Fuzzy" in by_src:
                    with st.expander(f"📑 {len(by_src['PDF-Fuzzy'])} partial match(es) — sentences with some keywords"):
                        for rank, res in enumerate(by_src["PDF-Fuzzy"], 1):
                            render_result(res, rank)
            else:
                st.warning(f"The word **{pdf_cq}** was not found in this PDF.")

        # Full text preview
        with st.expander("👁️ Preview extracted text"):
            st.text_area("Extracted text", st.session_state.pdf_text[:3000] + "…"
                         if len(st.session_state.pdf_text) > 3000 else st.session_state.pdf_text,
                         height=200, label_visibility="collapsed")

# ══════════════════════════════════════════════════════════════════
# PAGE: ANALYTICS
# ══════════════════════════════════════════════════════════════════
elif page == "Analytics":
    st.markdown('<h2 style="color:#e2e8f0;margin-bottom:4px;">📊 Analytics</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color:#718096;margin-bottom:1.5rem;">Your personal usage dashboard — searches, corrections, and trends.</p>', unsafe_allow_html=True)

    sh = st.session_state.search_history
    ah = st.session_state.autocorrect_history
    sc = st.session_state.search_counts

    # Top stats row
    s1, s2, s3, s4 = st.columns(4)
    s1.markdown(f'<div class="stat-card"><div class="val">{st.session_state.total_searches}</div><div class="lbl">Total Searches</div></div>', unsafe_allow_html=True)
    s2.markdown(f'<div class="stat-card"><div class="val">{len(ah)}</div><div class="lbl">Texts Corrected</div></div>', unsafe_allow_html=True)
    found_c = sum(1 for h in sh if h.get("found")) if sh else 0
    rate = round(found_c / max(len(sh), 1) * 100)
    s3.markdown(f'<div class="stat-card"><div class="val">{rate}%</div><div class="lbl">Search Success Rate</div></div>', unsafe_allow_html=True)
    top_q = sc.most_common(1)
    s4.markdown(f'<div class="stat-card"><div class="val" style="font-size:1.1rem;">{top_q[0][0] if top_q else "—"}</div><div class="lbl">Top Query</div></div>', unsafe_allow_html=True)

    if sh:
        st.markdown("---")
        ch1, ch2 = st.columns(2)
        with ch1:
            top10 = sc.most_common(10)
            df_top = pd.DataFrame(top10, columns=["Query", "Count"])
            fig_bar = px.bar(df_top, x="Count", y="Query", orientation="h",
                             title="Top Queries", color="Count",
                             color_continuous_scale="Blues", height=320)
            fig_bar.update_layout(yaxis=dict(autorange="reversed"),
                                  paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="rgba(0,0,0,0)", font_color="#a0aec0",
                                  margin=dict(t=40, b=10))
            st.plotly_chart(fig_bar, use_container_width=True)

        with ch2:
            fig_pie = px.pie(
                values=[found_c, len(sh) - found_c],
                names=["Results Found", "No Results"],
                title="Search Success",
                color_discrete_sequence=["#4f8bf9", "#fc8181"], height=320,
            )
            fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#a0aec0",
                                  margin=dict(t=40, b=10))
            st.plotly_chart(fig_pie, use_container_width=True)

        # Search history table
        st.markdown('<div class="section-title">Search History</div>', unsafe_allow_html=True)
        df_sh = pd.DataFrame(sh)
        df_sh.insert(0, "#", range(1, len(df_sh) + 1))
        show_cols = [c for c in ["#", "query", "corrected", "found", "timestamp"] if c in df_sh.columns]
        st.dataframe(df_sh[show_cols].rename(columns={
            "query": "Original Query", "corrected": "Corrected",
            "found": "Found", "timestamp": "Time"
        }), use_container_width=True, hide_index=True)
        st.download_button("⬇️ Export search history",
                           df_sh.to_csv(index=False).encode("utf-8"),
                           "search_history.csv", "text/csv")
    else:
        st.info("No search data yet. Head to the Search page and run some queries!")

    if ah:
        st.markdown("---")
        st.markdown('<div class="section-title">AutoCorrect History</div>', unsafe_allow_html=True)
        df_ah = pd.DataFrame(ah)
        df_ah.insert(0, "#", range(1, len(df_ah) + 1))
        st.dataframe(df_ah.rename(columns={
            "original": "Original Text", "corrected": "Corrected",
            "corrections": "# Fixes", "time": "Time"
        }), use_container_width=True, hide_index=True)

    # N-gram explorer (power feature tucked away here)
    st.markdown("---")
    with st.expander("📊 N-gram Frequency Explorer — analyse corpus language patterns"):
        ng_n = st.radio("Gram size", [1, 2, 3],
                        format_func=lambda x: {1:"Unigram",2:"Bigram",3:"Trigram"}[x],
                        horizontal=True, key="ng_n")
        ng_k = st.slider("Top N", 10, 30, 15, key="ng_k")
        with st.spinner("Computing…"):
            ngs = ngram_frequency(raw_text, word_counts, total_words, n=ng_n, top_k=ng_k)
        if ngs:
            df_ng = pd.DataFrame(ngs, columns=["N-gram", "Count"])
            fig_ng = px.bar(df_ng, x="Count", y="N-gram", orientation="h",
                            color="Count", color_continuous_scale="Teal", height=420)
            fig_ng.update_layout(yaxis=dict(autorange="reversed"),
                                 paper_bgcolor="rgba(0,0,0,0)",
                                 plot_bgcolor="rgba(0,0,0,0)", font_color="#a0aec0",
                                 margin=dict(t=10, b=10))
            st.plotly_chart(fig_ng, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style="border-color:#2d3651;margin-top:3rem;">
<div style="text-align:center;color:#4a5568;font-size:0.78rem;padding:10px 0 20px;">
  ✏️ <b>WordRight</b> — Built with Python · Streamlit · NLTK · Plotly
</div>""", unsafe_allow_html=True)
