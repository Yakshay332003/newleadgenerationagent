import streamlit as st
from astrapy import DataAPIClient
import pandas as pd
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import requests
import numpy as np

ASTRA_DB_APPLICATION_TOKEN = st.secrets["ASTRA_DB_APPLICATION_TOKEN"]
ASTRA_DB_API_ENDPOINT = st.secrets["ASTRA_DB_API_ENDPOINT"]
hf_token = st.secrets["HF_TOKEN"]
gemini_api_key = st.secrets["GEMINI_API_KEY"]
client = DataAPIClient(ASTRA_DB_APPLICATION_TOKEN)
db = client.get_database(ASTRA_DB_API_ENDPOINT)
collection = db.get_collection("news_headlines")


genai.configure(api_key=gemini_api_key)


@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2", use_auth_token=hf_token)

model = load_model()

def embed(text):
    return model.encode([text])[0]
def load_news_with_embeddings():
    docs = list(collection.find({}))  # Get all docs
    df = pd.DataFrame(docs)

    # Check if embeddings are stored; if not, generate and save
    if "embedding" not in df.columns or df["embedding"].isnull().any():
        df["embedding"] = df["Headline"].apply(lambda text: embed(text).tolist())
        # Save embeddings back to DB here if you want

    # Convert embeddings back to numpy arrays for similarity calculation
    df["embedding"] = df["embedding"].apply(np.array)
    return df

df_news = load_news_with_embeddings()
def semantic_search(query, top_k=8):
    query_emb = embed(query).reshape(1, -1)  # embed query, shape (1, dim)
    embeddings = np.vstack(df_news["embedding"].values)  # (N, dim)
    
    # Compute cosine similarity
    similarities = cosine_similarity(query_emb, embeddings)[0]
    
    # Get indices of top_k highest similarities
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    # Retrieve matching documents
    results = df_news.iloc[top_indices]
    return results.to_dict(orient="records")







def llm_agent(messages, context):
    context_text = "\n".join(
        [f"- {c['Headline']} (Source: {c['Source']}, Link: {c['URL']})" for c in context]
    )

    system_prompt = """You are a business development news analyst agent. You help users to find partnership with our CDMO.
You answer using recent pharma/biotech/CDMO/AI headlines.
Rules:
1. Use ONLY the provided headlines as factual basis.
2. If context is missing, rely on past conversation memory.
3. If the user asks for leads/partnerships, suggest and EXPLAIN reasoning.
4. Always answer in a conversational style with clear reasoning steps."""

    
    history = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages])

    full_prompt = f"""{system_prompt}

Conversation so far:
{history}

Relevant headlines:
{context_text}

Now answer the last USER query in the conversation with reasoning and suggestions.
"""

    model = genai.GenerativeModel("gemini-2.5-pro")
    response = model.generate_content(full_prompt)

    return response.text


st.set_page_config(page_title="BD News Agent", page_icon="🤖", layout="wide")
st.title("🤖 Pharma & Biotech Business Development Agent")

if "messages" not in st.session_state:
    st.session_state.messages = [] 


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask a question about pharma, biotech, CDMO news..."):
    
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)


    context = semantic_search(user_input, top_k=8)


    answer = llm_agent(st.session_state.messages, context)

    
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
