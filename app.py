import streamlit as st
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
import uuid
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import openai
import os

# UI Config
st.set_page_config(page_title="AI-Powered Research Dashboard", layout="wide")
st.title("🤖 Daily Research Summary (Simulated + GPT Ready)")

st.markdown("""
Upload Excel sheets from analysts. The app will:
- Deduplicate and group similar news
- Generate summaries using GPT or simulated output
- Tag each item with sectors/types
- Present a clean, categorized dashboard
""")

# Optional OpenAI Key
api_key = st.sidebar.text_input("🔐 Enter your OpenAI API Key (Optional)", type="password")
use_gpt = False
if api_key:
    openai.api_key = api_key
    use_gpt = True

# Load Embedding Model
model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')

# File Upload
uploaded_files = st.file_uploader("📥 Upload Analyst Research Files (Excel)", type=["xlsx"], accept_multiple_files=True)

@st.cache_data
def compute_embeddings(texts):
    return model.encode(texts, convert_to_tensor=False)

# Simulated GPT Output
SIMULATED_SUMMARIES = [
    ("India and UK advance FTA talks, easing trade restrictions", ["Macro", "Policy"]),
    ("US pushes domestic pharma growth amid tightening FDA approvals", ["Pharma", "Policy"]),
    ("Sustainable transport added to India's climate plan", ["Energy", "Transport"]),
    ("Private equity investments in India see strong momentum", ["Markets", "Finance"]),
    ("Bank of Baroda posts strong quarterly results", ["Banking"])
]

# GPT Summary + Tagging or fallback
@st.cache_data
def gpt_summarize_and_tag(texts, fallback_id):
    if not use_gpt:
        return SIMULATED_SUMMARIES[fallback_id % len(SIMULATED_SUMMARIES)]

    prompt = f"""
You are an assistant that condenses financial news. Summarize the following entries into one line and suggest a few relevant sectors (e.g., Banking, IT, Pharma, Macro, Policy, Markets, Energy).

Entries:
{texts[:1000]}

Return as:
Summary: <one-liner summary>
Tags: <comma-separated sector/type tags>
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You summarize and tag financial news."},
                {"role": "user", "content": prompt}
            ]
        )
        reply = response.choices[0].message.content
        summary_line = reply.split("Summary:")[1].split("Tags:")[0].strip()
        tag_line = reply.split("Tags:")[1].strip()
        return summary_line, [t.strip() for t in tag_line.split(",")]
    except Exception as e:
        st.warning(f"Falling back to simulated GPT due to: {e}")
        return SIMULATED_SUMMARIES[fallback_id % len(SIMULATED_SUMMARIES)]

if uploaded_files:
    raw_entries = []

    for uploaded_file in uploaded_files:
        analyst_name = Path(uploaded_file.name).stem
        df = pd.read_excel(uploaded_file)
        for col in df.columns:
            for item in df[col].dropna():
                if isinstance(item, str) and item.strip():
                    entry = {
                        "text": item.strip(),
                        "analyst": analyst_name,
                        "tags": [],
                        "sector": "Unknown",
                        "id": str(uuid.uuid4())
                    }
                    raw_entries.append(entry)

    st.success(f"✅ Parsed {len(raw_entries)} entries from uploaded files.")

    # Deduplication
    texts = [entry["text"] for entry in raw_entries]
    embeddings = compute_embeddings(texts)
    similarity_matrix = cosine_similarity(embeddings)
    threshold = 0.8
    grouped = []
    used = set()

    for i in range(len(texts)):
        if i in used:
            continue
        group = [raw_entries[i]]
        used.add(i)
        for j in range(i + 1, len(texts)):
            if j in used:
                continue
            if similarity_matrix[i][j] > threshold:
                group.append(raw_entries[j])
                used.add(j)
        grouped.append(group)

    # Enhance each group with summary & tags
    enhanced = []
    for idx, group in enumerate(grouped):
        combined_text = "\n".join([item["text"] for item in group[:5]])  # limit size
        summary, tags = gpt_summarize_and_tag(combined_text, idx)
        analysts = list({item['analyst'] for item in group})
        enhanced.append({
            "summary": summary,
            "tags": tags,
            "analysts": analysts,
            "entries": group
        })

    # Sidebar filters
    all_tags = sorted(set(tag for item in enhanced for tag in item["tags"]))
    all_analysts = sorted(set(a for item in enhanced for a in item["analysts"]))
    selected_tags = st.sidebar.multiselect("Filter by Sector/Tag", all_tags, default=all_tags)
    selected_analysts = st.sidebar.multiselect("Filter by Analyst", all_analysts, default=all_analysts)

    # Display
    st.subheader("📚 Categorized News Summaries")
    for item in enhanced:
        if not any(tag in selected_tags for tag in item["tags"]):
            continue
        if not any(a in selected_analysts for a in item["analysts"]):
            continue

        st.markdown(f"### 📝 {item['summary']}")
        st.caption("Tags: " + ", ".join(item["tags"]))
        st.caption("Analyst(s): " + ", ".join(item["analysts"]))

        with st.expander("💬 Analyst Inputs"):
            for entry in item["entries"]:
                st.write(f"- {entry['text']} ({entry['analyst']})")

else:
    st.info("Upload Excel files from analysts to get started.")
