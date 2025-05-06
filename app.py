import streamlit as st
import json
import openai
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Чат с ИИ по Jira", page_icon="🤖")
st.title("🤖 Чат с ИИ по данным из Jira")

query = st.text_input("Введите вопрос к задачам из Jira")

if query:
    with open("jira_data.json", "r", encoding="utf-8") as f:
        docs = json.load(f)

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(docs)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec), k=3)
    relevant_docs = [docs[i] for i in I[0]]

    prompt = f"Ответь на вопрос: '{query}' на основе следующих задач Jira:\n\n" + "\n".join(relevant_docs)

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )

    st.write("**Ответ:**")
    st.write(response["choices"][0]["message"]["content"])
