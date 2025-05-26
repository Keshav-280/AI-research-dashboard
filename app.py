import streamlit as st
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
import uuid
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Initialize the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

st.set_page_config(page_title="Daily Research Consolidator", layout="wide")
st.title("📊 Daily Research Summary Dashboard")

st.markdown("""
Upload structured Excel sheets from analysts. The system will:
1. Remove duplicate/similar news using embeddings
2. Tag news items (e.g., sector, type)
3. Retain analyst-specific insights
4. Present a clean dashboard view with filters
""")

uploaded_files = st.file_uploader("Upload Research Files (Excel)", type=["xlsx"], accept_multiple_files=True)

@st.cache_data

def compute_embeddings(texts):
    return model.encode(texts, convert_to_tensor=False)

if uploaded_files:
    raw_entries = []
    all_tags = set()
    all_sectors = set()

    for uploaded_file in uploaded_files:
        analyst_name = Path(uploaded_file.name).stem
        df = pd.read_excel(uploaded_file)
        for col in df.columns:
            for item in df[col].dropna():
                if isinstance(item, str) and item.strip():
                    entry = {
                        "text": item.strip(),
                        "analyst": analyst_name,
                        "tags": [],  # Placeholder for tagging logic
                        "sector": "Unknown",  # Placeholder
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

    st.info(f"🧹 Reduced from {len(raw_entries)} to {len(grouped)} unique items.")

    # Sidebar filters
    all_analysts = sorted(set(e['analyst'] for g in grouped for e in g))
    selected_analysts = st.sidebar.multiselect("Filter by Analyst", all_analysts, default=all_analysts)

    st.subheader("📰 Consolidated News Feed")
    for idx, group in enumerate(grouped):
        analysts = list({e['analyst'] for e in group})
        if not any(a in selected_analysts for a in analysts):
            continue

        st.markdown(f"### {idx+1}. {group[0]['text']}")
        st.caption("Analyst(s): " + ", ".join(analysts))
        with st.expander("See individual entries"):
            for entry in group:
                st.write(f"- {entry['text']} ({entry['analyst']})")
else:
    st.info("Upload at least one Excel file to begin.")

