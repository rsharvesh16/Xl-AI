import os
import boto3
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

load_dotenv()

# Initialize Bedrock client
bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

def data_ingestion():
    csv_files = [f for f in os.listdir("data") if f.endswith('.csv')]
    all_data = []
    for file in csv_files:
        df = pd.read_csv(os.path.join("data", file))
        # Include column names in the text
        text = f"File: {file}\n"
        text += f"Columns: {', '.join(df.columns)}\n"
        text += df.to_string(index=False)
        all_data.append(text)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.create_documents(all_data)
    return docs

def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")

def get_mistral_llm():
    llm = Bedrock(model_id="mistral.mistral-7b-instruct-v0:2", client=bedrock,
                  model_kwargs={'max_tokens': 8000, 'temperature': 0.2})
    return llm

prompt_template = """
Human: You are an AI assistant tasked with answering questions based on the information provided in CSV files. Use the following context, which contains relevant information from the CSV files, to answer the question. Follow these guidelines strictly:

1. Provide comprehensive answers based solely on the information present in the given context.
2. If the context doesn't contain enough information to answer the question fully, state what information is missing and answer with the parts you can address accurately.
3. If you're unsure about any part of the answer, explicitly state your uncertainty and provide the information you are confident about.
4. If the question cannot be answered at all based on the given context, clearly state that you don't have enough information to provide an accurate answer.
5. Do not make assumptions or include information that isn't explicitly stated in the context.
6. Always mention the specific columns, data points, or file names from the CSV that you're using to form your answer.
7. Provide detailed, thorough responses that fully address all aspects of the question.

Context: {context}

Question: {question}
Assistant: I'll provide a comprehensive answer based on the information available in the CSV files. Here's what I found:

"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity", search_kwargs={"k": 5}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answer['result']


def main():
    st.set_page_config(page_title="CSV Chatbot", layout="wide")

    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
            body {
                font-family: 'Roboto', sans-serif;
            }
            .container {
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                max-width: 800px;
                margin: 0 auto;
            }
            .title {
                text-align: center;
                color: #007bff;
                font-weight: 700;
            }
            .upload-container {
                padding: 15px;
                border: 2px dashed #007bff;
                border-radius: 10px;
                text-align: center;
                background-color: #e9ecef;
                margin-bottom: 20px;
            }
            .sidebar-title {
                font-size: 18px;
                font-weight: bold;
                color: #28a745;
                margin-bottom: 10px;
            }
            .footer {
                text-align: center;
                margin-top: 30px;
                font-size: 14px;
                color: #6c757d;
            }
        </style>
        <div class="container">
            <h1 class="title">CSV Chatbot</h1>
            <p>Upload your CSV files and ask questions about your data!</p>
        </div>
    """, unsafe_allow_html=True)

    st.write("<br>", unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown('<div class="sidebar-title">Data Management</div>', unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)
        if uploaded_files:
            for uploaded_file in uploaded_files:
                with open(os.path.join("data", uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
            st.success("CSV files uploaded successfully!")
        
        if st.button("Update Knowledge Base"):
            with st.spinner("Processing CSV files..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Knowledge base updated successfully!")

    user_question = st.text_input("What would you like to know about your data?")

    if st.button("Get Answer"):
        if user_question:
            with st.spinner("Analyzing your data..."):
                try:
                    faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
                    llm = get_mistral_llm()
                    response = get_response_llm(llm, faiss_index, user_question)
                    st.write("### Answer")
                    st.write(response)
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.info("Make sure you've uploaded CSV files and updated the knowledge base before asking questions.")
        else:
            st.warning("Please enter a question.")

    st.markdown('<div class="footer">Powered by Mistral LLM and Streamlit</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()