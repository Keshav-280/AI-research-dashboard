import streamlit as st
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
import uuid

# Initialize the embedding model (small one for performance)
model = SentenceTransformer("all-MiniLM-L6-v2")

st.set_page_config(page_title="Daily Research Consolidator", layout="wide")
st.title("📊 Daily Research Summary Dashboard")

st.markdown("""
Upload research summaries from multiple analysts. The app will:
- Parse the files
- Remove duplicate or similar items
- Group insights
- Preserve analyst-specific insights
""")

uploaded_files = st.file_uploader("Upload Research Files (Excel)", type=["xlsx"], accept_multiple_files=True)

if uploaded_files:
    raw_entries = []

    for uploaded_file in uploaded_files:
        analyst_name = Path(uploaded_file.name).stem
        df = pd.read_excel(uploaded_file)
        for _, row in df.iterrows():
            for cell in row:
                if pd.notna(cell) and isinstance(cell, str):
                    raw_entries.append({
                        "text": cell.strip(),
                        "analyst": analyst_name,
                        "id": str(uuid.uuid4())
                    })

    st.subheader("🧾 Raw Extracted Items")
    st.write(f"Total extracted entries: {len(raw_entries)}")
    if st.checkbox("Show raw items"):
        st.json(raw_entries)

    # Deduplication using semantic similarity
    st.subheader("🔍 Deduplicating Similar News Items")
    texts = [entry['text'] for entry in raw_entries]
    embeddings = model.encode(texts, convert_to_tensor=True)
    threshold = 0.75

    # Group similar entries
    groups = []
    used = set()

    for i in range(len(texts)):
        if i in used:
            continue
        group = [raw_entries[i]]
        used.add(i)
        for j in range(i+1, len(texts)):
            if j in used:
                continue
            if util.pytorch_cos_sim(embeddings[i], embeddings[j]) > threshold:
                group.append(raw_entries[j])
                used.add(j)
        groups.append(group)

    st.success(f"Grouped into {len(groups)} distinct items")

    # Display grouped items with analyst tags
    for idx, group in enumerate(groups):
        st.markdown(f"### {idx+1}. {group[0]['text']}")
        analysts = list({entry['analyst'] for entry in group})
        st.caption("Analyst(s): " + ", ".join(analysts))

else:
    st.info("Please upload at least one Excel file to begin.")
