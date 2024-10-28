import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import wordnet
import logging

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)

# Streamlit app configuration
st.set_page_config(page_title="Document Genie", layout="wide")

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

# Three clustering options: KMeans, Agglomerative Clustering, and DBSCAN
def cluster_text_chunks(text_chunks, algorithm="kmeans", n_clusters=5):
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(text_chunks)

    if algorithm == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=0)
    elif algorithm == "agglomerative":
        model = AgglomerativeClustering(n_clusters=n_clusters)
    elif algorithm == "dbscan":
        model = DBSCAN(eps=0.5, min_samples=5)

    model.fit(X.toarray())
    labels = model.labels_

    clusters = {i: [] for i in range(n_clusters) if i != -1}
    for i, label in enumerate(labels):
        if label != -1:  # Ignore noise points in DBSCAN (-1 label)
            clusters[label].append(text_chunks[i])
    return clusters

# Function to compute BM25 scores for ranking
def compute_bm25_scores(text_chunks):
    tokenized_chunks = [chunk.split(" ") for chunk in text_chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    return bm25

# Function to create a vector store from text chunks
def get_vector_store(text_chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to load conversational chain
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

# Three retrieval methods: BM25, Cosine Similarity, and FAISS
def retrieve_chunks(method, expanded_query, bm25, text_chunks, vector_store):
    if method == "bm25":
        tokenized_query = expanded_query.split(" ")
        bm25_scores = bm25.get_scores(tokenized_query)
        sorted_chunks = [text_chunks[i] for i in bm25_scores.argsort()[::-1]]
        return sorted_chunks[:5]

    elif method == "cosine":
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(text_chunks)
        query_vector = vectorizer.transform([expanded_query])
        cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        sorted_chunks = [text_chunks[i] for i in cosine_similarities.argsort()[::-1]]
        return sorted_chunks[:5]

    elif method == "faiss":
        docs = vector_store.similarity_search(expanded_query)
        return [doc['text'] for doc in docs[:5]]

# Function to process user input and generate response
def user_input(user_question, api_key, retrieval_method, clustering_method, bm25, text_chunks, clusters, vector_store):
    expanded_query = expand_query(user_question)

    # Filter chunks based on cluster relevance using cosine similarity
    relevant_cluster = max(clusters.keys(), key=lambda k: cosine_similarity(
        TfidfVectorizer().fit_transform([" ".join(clusters[k]) + " " + expanded_query])))

    top_chunks = [chunk for chunk in text_chunks if chunk in clusters[relevant_cluster]]

    # Retrieve chunks based on selected method
    relevant_chunks = retrieve_chunks(retrieval_method, expanded_query, bm25, top_chunks, vector_store)

    # Generate answer
    chain = get_conversational_chain(api_key)
    response = chain({"input_documents": relevant_chunks, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

# Main Streamlit function
def main():
    st.header("AI Clone Chatbot 💁")
    user_question = st.text_input("Ask a Question from the PDF Files", key="user_question")
    retrieval_method = st.selectbox("Select Retrieval Method", ["bm25", "cosine", "faiss"])
    clustering_method = st.selectbox("Select Clustering Method", ["kmeans", "agglomerative", "dbscan"])

    if user_question and api_key:
        try:
            user_input(user_question, api_key, retrieval_method, clustering_method, bm25, text_chunks, clusters, vector_store)
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
                    
                    # Perform clustering on text chunks
                    clusters = cluster_text_chunks(text_chunks, algorithm=clustering_method)
                    st.write(f"Document content has been clustered using {clustering_method}.")
                    
                    bm25 = compute_bm25_scores(text_chunks)
                    vector_store = get_vector_store(text_chunks, api_key)
                    st.success("Documents processed successfully!")
            except Exception as e:
                st.error(f"An error occurred while processing the PDFs: {e}")

if __name__ == "__main__":
    main()
