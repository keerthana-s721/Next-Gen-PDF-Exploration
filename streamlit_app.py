import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import wordnet
import nltk
import logging

# Download WordNet if not already downloaded
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)

# Streamlit app configuration
st.set_page_config(page_title="Document Genie", layout="wide")

st.markdown("""
## Next-Gen-PDF-Exploration: Get instant insights from your Documents

This chatbot is built using the Retrieval-Augmented Generation (RAG) framework, leveraging Google's Generative AI model Gemini-PRO. It processes uploaded PDF documents by breaking them down into manageable chunks, creates a searchable vector store, and generates accurate answers to user queries.

### How It Works

1. **Enter Your API Key**: Obtain your API key [here](https://makersuite.google.com/app/apikey).
2. **Upload Your Documents**: Upload PDF files for analysis.
3. **Ask a Question**: After processing the documents, ask questions related to the uploaded content.
""")

# API key input
api_key = st.text_input("Enter your Google API Key:", type="password", key="api_key_input")

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to expand user query with synonyms
def expand_query(query):
    synonyms = set()
    for word in query.split():
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
    return ' '.join(list(synonyms))

# BM25 retrieval method
def compute_bm25_scores(text_chunks):
    tokenized_chunks = [chunk.split(" ") for chunk in text_chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    return bm25

# Vector store retrieval method
def get_vector_store(text_chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Tfidf retrieval method
def tfidf_retrieval(query, text_chunks):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(text_chunks)
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, X).flatten()
    return similarities

# KMeans clustering method
def kmeans_clustering(text_chunks, n_clusters=5):
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(text_chunks)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(X)
    labels = kmeans.labels_
    clusters = {i: [] for i in range(n_clusters)}
    for i, label in enumerate(labels):
        clusters[label].append(text_chunks[i])
    return clusters

# DBSCAN clustering method
def dbscan_clustering(text_chunks):
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(text_chunks)
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    labels = dbscan.fit_predict(X)
    unique_labels = set(labels)
    clusters = {label: [] for label in unique_labels if label != -1}
    for i, label in enumerate(labels):
        if label != -1:
            clusters[label].append(text_chunks[i])
    return clusters

# Agglomerative Clustering method
def agglomerative_clustering(text_chunks, n_clusters=5):
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(text_chunks)
    agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
    labels = agglomerative.fit_predict(X.toarray())
    clusters = {i: [] for i in range(n_clusters)}
    for i, label in enumerate(labels):
        clusters[label].append(text_chunks[i])
    return clusters

# Conversational chain loading function
def get_conversational_chain(api_key):
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, just say, "answer is not available in the context."
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to process user input and generate response
def user_input(user_question, api_key, bm25, text_chunks, clusters, retrieval_method):
    expanded_query = expand_query(user_question)

    # Check the retrieval method and perform the appropriate action
    if retrieval_method == "BM25" and bm25:
        tokenized_query = expanded_query.split(" ")
        bm25_scores = bm25.get_scores(tokenized_query)

        # Select relevant documents based on BM25 scores
        relevant_docs = [text_chunks[i] for i in range(len(bm25_scores)) if bm25_scores[i] > 0]
        top_chunks = sorted(relevant_docs, key=lambda x: bm25_scores[text_chunks.index(x)], reverse=True)[:5]

        chain = get_conversational_chain(api_key)
        response = chain({"input_documents": top_chunks, "question": user_question}, return_only_outputs=True)
        st.write("Reply: ", response["output_text"])

    elif retrieval_method == "Vector Store":
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(expanded_query)

        chain = get_conversational_chain(api_key)
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        st.write("Reply: ", response["output_text"])

    elif retrieval_method == "Tfidf":
        similarities = tfidf_retrieval(expanded_query, text_chunks)
        top_indices = similarities.argsort()[-5:][::-1]  # Get the indices of the top 5 documents
        top_chunks = [text_chunks[i] for i in top_indices]

        chain = get_conversational_chain(api_key)
        response = chain({"input_documents": top_chunks, "question": user_question}, return_only_outputs=True)
        st.write("Reply: ", response["output_text"])
    else:
        st.warning("BM25 not initialized or retrieval method not set.")

# Main Streamlit function
def main():
    st.header("AI Clone Chatbot 💁")

    if not api_key:
        st.warning("Please enter your API key to proceed.")
        return

    # Dropdown to select retrieval and clustering algorithms
    retrieval_method = st.sidebar.selectbox("Select Retrieval Algorithm:", ("BM25", "Vector Store", "Tfidf"))
    clustering_method = st.sidebar.selectbox("Select Clustering Algorithm:", ("KMeans", "DBSCAN", "Agglomerative Clustering"))

    user_question = st.text_input("Ask a Question from the PDF Files", key="user_question")

    # Initialize variables
    bm25 = None
    text_chunks = []

    if user_question:
        try:
            user_input(user_question, api_key, bm25, text_chunks, {}, retrieval_method)
        except Exception as e:
            st.error(f"An error occurred while processing your question: {e}")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True, key="pdf_uploader")
        
        if st.button("Submit & Process", key="process_button"):
            if pdf_docs:
                try:
                    with st.spinner("Processing..."):
                        raw_text = get_pdf_text(pdf_docs)
                        text_chunks = get_text_chunks(raw_text)
                        
                        # Perform clustering based on selected method
                        if clustering_method == "KMeans":
                            clusters = kmeans_clustering(text_chunks)
                        elif clustering_method == "DBSCAN":
                            clusters = dbscan_clustering(text_chunks)
                        elif clustering_method == "Agglomerative Clustering":
                            clusters = agglomerative_clustering(text_chunks)

                        # Initialize BM25 for retrieval
                        bm25 = compute_bm25_scores(text_chunks)

                        st.success("Documents processed successfully!")
                except Exception as e:
                    st.error(f"An error occurred while processing your PDFs: {e}")
            else:
                st.warning("Please upload PDF files before processing.")

if __name__ == "__main__":
    main()
