import os
import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load precomputed document embeddings and documents
EMBEDDINGS_PATH = "embeddings.npy"
DOCUMENTS_PATH = "documents.txt"

if not os.path.exists(EMBEDDINGS_PATH):
	st.error(f"Missing embeddings file: {EMBEDDINGS_PATH}")
	st.stop()

if not os.path.exists(DOCUMENTS_PATH):
	st.error(f"Missing documents file: {DOCUMENTS_PATH}")
	st.stop()

embeddings = np.load(EMBEDDINGS_PATH)
with open(DOCUMENTS_PATH, "r", encoding="utf-8") as f:
	documents = [line.strip() for line in f.readlines()]


def retrieve_top_k(query_embedding, embeddings, documents, k=10):
	"""Retrieve top-k most similar documents using cosine similarity."""
	similarities = cosine_similarity(query_embedding.reshape(1, -1), embeddings)[0]
	top_k_indices = similarities.argsort()[-k:][::-1]
	return [(documents[i], float(similarities[i])) for i in top_k_indices]


# Streamlit UI
st.title("Information Retrieval using Document Embeddings")
query = st.text_input("Enter your query:")


def get_query_embedding(query):
	"""Placeholder for query embedding. Replace with real model call."""
	if not query:
		return np.zeros(embeddings.shape[1])
	return np.random.rand(embeddings.shape[1])


if st.button("Search"):
	if not query:
		st.warning("Please enter a query.")
	else:
		query_embedding = get_query_embedding(query)
		results = retrieve_top_k(query_embedding, embeddings, documents, k=10)

		st.write("### Top 10 Relevant Documents:")
		for doc, score in results:
			st.write(f"- **{doc}** (Score: {score:.4f})")