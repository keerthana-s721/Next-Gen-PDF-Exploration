import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import numpy as np
import os
import logging

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="Document Genie", layout="wide")

st.markdown("""
## Next-Gen-PDF-Exploration: Get instant insights from your Documents

This chatbot is built using the Retrieval-Augmented Generation (RAG) framework, leveraging Google's Generative AI model Gemini-PRO. It processes uploaded PDF documents by breaking them down into manageable chunks, creates a searchable vector store, and generates accurate answers to user queries. This advanced approach ensures high-quality, contextually relevant responses for an efficient and effective user experience.

### How It Works

1. *Enter Your API Key:* You'll need a Google API key for access to Google's Generative AI models.  
2. *Upload Your Documents:* The system accepts multiple PDF files and creates a searchable vector index.  
3. *Ask a Question:* Ask any question related to the uploaded documents.  
4. *Cluster Questions:* Use KMeans, Agglomerative, or Hierarchical clustering to group similar questions.
""")

# API key input
api_key = st.text_input("Enter your Google API Key:", type="password", key="api_key_input")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain(api_key):
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not available, say "answer is not available in the context".\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain(api_key)
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

def cluster_questions(questions, method, n_clusters):
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Load the Sentence Transformer model
    embeddings = model.encode(questions)

    if method == "KMeans":
        cluster_model = KMeans(n_clusters=n_clusters, random_state=42)
    elif method == "Agglomerative":
        cluster_model = AgglomerativeClustering(n_clusters=n_clusters)
    elif method == "Hierarchical":
        # Use linkage matrix for dendrogram visualization
        Z = linkage(embeddings, 'ward')
        plt.figure(figsize=(10, 5))
        dendrogram(Z, labels=questions, leaf_rotation=90)
        st.pyplot(plt)
        return {}  # No further clustering needed for dendrogram

    labels = cluster_model.fit_predict(embeddings)
    clustered_questions = {i: [] for i in range(n_clusters)}
    for idx, label in enumerate(labels):
        clustered_questions[label].append(questions[idx])
    
    return clustered_questions

def main():
    st.header("AI Clone Chatbot 💁")

    user_question = st.text_input("Ask a Question from the PDF Files", key="user_question")

    if user_question and api_key:
        try:
            user_input(user_question, api_key)
        except Exception as e:
            st.error(f"An error occurred while processing your question: {e}")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True, key="pdf_uploader")
        
        if st.button("Submit & Process", key="process_button") and api_key:
            try:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks, api_key)
                    st.success("Documents processed successfully!")
            except Exception as e:
                st.error(f"An error occurred while processing the PDFs: {e}")

        # Question clustering section
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
                        st.write(f"*Cluster {cluster}:*")
                        for q in qs:
                            st.write(f"- {q}")

if __name__ == "__main__":
    main()
