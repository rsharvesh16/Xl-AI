from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env.
import streamlit as st
import os
import pandas as pd
import requests
import json

# Configure Groq API
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("Groq API key not found in environment variables.")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Custom HTML for the header
html_code = """
<div style="background-color: #f0f0f0; padding: 20px; border-radius: 10px;">
    <h1 style="color: #333;">CSV Chatbot</h1>
    <p style="color: #555;">Upload your CSV and ask questions about your data.</p>
</div>
"""

def get_groq_response(input_prompt, csv_content):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that analyzes CSV data and answers questions about it only on text, dont provide any wrong answers. Give only related to The document present."
            },
            {
                "role": "user",
                "content": f"Based on the following CSV data:\n\n{csv_content}\n\nAnswer the following question:\n\n{input_prompt}"
            }
        ],
        "temperature": 0.5,
        "max_tokens": 50000
    }
    
    response = requests.post(GROQ_API_URL, headers=headers, data=json.dumps(data))
    
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return f"Error: {response.status_code} - {response.text}"

st.markdown(html_code, unsafe_allow_html=True)

st.header("CSV Data Analysis Chatbot")

uploaded_file = st.file_uploader("Choose a CSV file...", type="csv")
csv_content = ""

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    csv_content = df.to_string()
    st.write("CSV file uploaded successfully!")
    st.dataframe(df.head())  # Display the first few rows of the CSV

input_prompt = st.text_input("Ask a question about your CSV data:", key="input")

submit = st.button("Get Answer")

if submit and csv_content:
    if input_prompt:
        with st.spinner("Generating response..."):
            response = get_groq_response(input_prompt, csv_content)
        st.subheader("Chatbot Response")
        st.write(response)
    else:
        st.warning("Please enter a question about your data.")
elif submit:
    st.warning("Please upload a CSV file first.")

st.markdown("---")
st.markdown("Created with Streamlit and LLaMA-3.1 model")