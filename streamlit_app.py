import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import numpy as np
import logging

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="Document Genie", layout="wide")

st.markdown("""
## Next-Gen-PDF-Exploration: Get instant insights from your Documents

This chatbot is built using the Retrieval-Augmented Generation (RAG) framework. It processes uploaded PDF documents by breaking them down into manageable chunks, creates a searchable vector store, and generates accurate answers to user queries. This approach ensures high-quality, contextually relevant responses.

### How It Works

1. Upload Your Documents: The system accepts multiple PDF files and creates a searchable vector index.  
2. Ask a Question: Ask any question related to the uploaded documents.  
3. Cluster Questions: Use KMeans, Agglomerative, or Hierarchical clustering to group similar questions.
""")

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

# Function to build a vector store
def get_vector_store(text_chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(text_chunks)
    vector_store = FAISS.from_embeddings(embeddings)
    vector_store.save_local("faiss_index")

# Function to load the vector store and retrieve answers
def user_input(user_question):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode([user_question])
    vector_store = FAISS.load_local("faiss_index", model)
    docs = vector_store.similarity_search(user_question)
    st.write("Documents matching your question:")
    for doc in docs:
        st.write(doc)

# Clustering Function
def cluster_questions(questions, method, n_clusters):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(questions)

    # Perform clustering
    if method == "KMeans":
        cluster_model = KMeans(n_clusters=n_clusters, random_state=42)
        labels = cluster_model.fit_predict(embeddings)
        inertia = cluster_model.inertia_  # Only for KMeans
        st.write(f"Inertia: {inertia:.2f}")

    elif method == "Agglomerative":
        cluster_model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = cluster_model.fit_predict(embeddings)

    elif method == "Hierarchical":
        Z = linkage(embeddings, 'ward')
        plt.figure(figsize=(10, 5))
        dendrogram(Z, labels=questions, leaf_rotation=90)
        st.pyplot(plt)
        return {}  # No further clustering for Hierarchical

    # Evaluation Metrics
    silhouette = silhouette_score(embeddings, labels)
    db_index = davies_bouldin_score(embeddings, labels)

    st.write(f"Silhouette Score: {silhouette:.2f}")
    st.write(f"Davies-Bouldin Index: {db_index:.2f}")

    # Grouping questions by clusters
    clustered_questions = {i: [] for i in range(n_clusters)}
    for idx, label in enumerate(labels):
        clustered_questions[label].append(questions[idx])

    return clustered_questions

# Main Function
def main():
    st.header("AI Clone Chatbot 💁")

    user_question = st.text_input("Ask a Question from the PDF Files", key="user_question")

    if user_question:
        try:
            user_input(user_question)
        except Exception as e:
            st.error(f"An error occurred while processing your question: {e}")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", 
                                    accept_multiple_files=True, key="pdf_uploader")

        if st.button("Submit & Process", key="process_button"):
            try:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Documents processed successfully!")
            except Exception as e:
                st.error(f"An error occurred while processing the PDFs: {e}")

        # Question Clustering Section
        st.subheader("Question Clustering")
        question_input = st.text_area("Enter your questions (one per line):", key="question_input")

        clustering_method = st.selectbox("Select Clustering Method", 
                                         options=["KMeans", "Agglomerative", "Hierarchical"], key="cluster_method")

        n_clusters = st.slider("Select number of clusters", min_value=2, max_value=10, value=3)

        if st.button("Cluster Questions", key="cluster_button"):
            questions = question_input.splitlines()
            if len(questions) < 2:
                st.warning("Please enter at least two questions to cluster.")
            else:
                clusters = cluster_questions(questions, clustering_method, n_clusters)
                if clusters:
                    st.write("Clustered Questions:")
                    for cluster, qs in clusters.items():
                        st.write(f"Cluster {cluster}:")
                        for q in qs:
                            st.write(f"- {q}")

if __name__ == "__main__":
    main()
