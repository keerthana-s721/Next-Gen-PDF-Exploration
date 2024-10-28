import subprocess
import sys

# Install necessary packages if they are not already installed
required_packages = ["rank_bm25", "PyPDF2", "langchain", "langchain_google_genai", "google-generativeai", "scikit-learn", "nltk"]
for package in required_packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import logging
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import wordnet
import os

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="Document Genie", layout="wide")

st.markdown("""
## Next-Gen-PDF-Exploration: Get instant insights from your Documents

This chatbot is built using the Retrieval-Augmented Generation (RAG) framework, leveraging Google's Generative AI model Gemini-PRO. It processes uploaded PDF documents by breaking them down into manageable chunks, creates a searchable vector store, and generates accurate answers to user queries. This advanced approach ensures high-quality, contextually relevant responses for an efficient and effective user experience.

### How It Works

Follow these simple steps to interact with the chatbot:

1. **Enter Your API Key**: You'll need a Google API key for the chatbot to access Google's Generative AI models. Obtain your API key [here](https://makersuite.google.com/app/apikey).
2. **Upload Your Documents**: The system accepts multiple PDF files at once, analyzing the content to provide comprehensive insights.
3. **Ask a Question**: After processing the documents, ask any question related to the content of your uploaded documents for a precise answer.
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

def extract_keywords_tfidf(text_chunks):
    vectorizer = TfidfVectorizer(max_df=0.8, max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(text_chunks)
    keywords = vectorizer.get_feature_names_out()
    return keywords

def expand_query(query):
    synonyms = set()
    for word in query.split():
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
    return ' '.join(list(synonyms))

def compute_bm25_scores(text_chunks):
    tokenized_chunks = [chunk.split(" ") for chunk in text_chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    return bm25

def get_vector_store(text_chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain(api_key):
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, api_key, bm25, text_chunks):
    expanded_query = expand_query(user_question)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(expanded_query)

    top_chunks = [doc['text'] for doc in docs]
    tokenized_query = expanded_query.split(" ")
    bm25_scores = bm25.get_scores(tokenized_query)

    # Sort chunks by BM25 scores in descending order
    top_chunks_sorted = [top_chunks[i] for i in bm25_scores.argsort()[::-1]]
    
    chain = get_conversational_chain(api_key)
    response = chain({"input_documents": top_chunks_sorted[:5], "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

def main():
    st.header("AI Clone Chatbot 💁")

    user_question = st.text_input("Ask a Question from the PDF Files", key="user_question")

    if user_question and api_key:
        try:
            # Process the question with query expansion and BM25 re-ranking
            user_input(user_question, api_key, bm25, text_chunks)
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
                    keywords = extract_keywords_tfidf(text_chunks)
                    st.write("Extracted Keywords (TF-IDF):", ", ".join(keywords))

                    bm25 = compute_bm25_scores(text_chunks)
                    get_vector_store(text_chunks, api_key)
                    st.success("Documents processed successfully!")
            except Exception as e:
                st.error(f"An error occurred while processing the PDFs: {e}")

if __name__ == "__main__":
    main()
