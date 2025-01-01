# Xl-AI

## Overview
The **Xl-AI** is a Streamlit application designed to interactively analyze CSV files and provide answers to user queries based on the data in those files. Powered by Amazon Bedrock's Mistral LLM and FAISS (Facebook AI Similarity Search), this tool leverages advanced AI models to enable natural language understanding and retrieval-augmented generation for data-driven insights.

## Features
- **Upload CSV Files:** Users can upload multiple CSV files, which are then processed to extract and index the data.
- **Data Ingestion:** The data from the uploaded CSV files is read, and the column names and rows are combined into a searchable context.
- **Vectorization:** Data is converted into embeddings using Amazon Titan Embeddings for efficient similarity search.
- **Question Answering:** Users can input natural language queries, and the chatbot will provide answers by retrieving the most relevant data from the CSV files.
- **Streamlit Interface:** A user-friendly web interface for seamless interaction.

## Technologies Used
1. **Streamlit:** For creating the interactive web interface.
2. **Amazon Bedrock:** To integrate the Mistral LLM and Titan Embeddings.
3. **FAISS:** For building and querying vector stores.
4. **Python:** Core programming language for the implementation.
5. **Pandas:** To handle and manipulate CSV data.
6. **dotenv:** For managing environment variables.

## How It Works
### 1. **Data Ingestion**
Uploaded CSV files are processed by reading their contents into Pandas DataFrames. The column names and data rows are structured into a text format to create a searchable context.

### 2. **Text Splitting**
The context is divided into manageable chunks using the Recursive Character Text Splitter to ensure efficient processing and retrieval.

### 3. **Embedding Creation**
The text chunks are embedded into vectors using Amazon Titan Embeddings, which are then stored in a FAISS index for similarity search.

### 4. **Question Answering**
When a user submits a query, the FAISS index retrieves the most relevant text chunks. The Mistral LLM processes these chunks along with the user query to generate a detailed answer following strict guidelines.

### 5. **Dynamic Updates**
The application allows users to update the knowledge base with new CSV files at any time. The FAISS index is recreated with the latest data to ensure up-to-date responses.

## Usage
1. **Upload CSV Files:** Drag and drop one or more CSV files into the upload area.
2. **Update Knowledge Base:** Click the "Update Knowledge Base" button to process the uploaded files.
3. **Ask Questions:** Enter a question related to your data in the input box and click "Get Answer" to receive a response.

## Prompt Template
The prompt used by the LLM ensures:
- Answers are derived strictly from the provided context.
- Uncertainty and missing information are clearly stated.
- References to specific files, columns, and data points are mentioned.

## Error Handling
- If no files are uploaded, the app prompts users to upload CSV files.
- Detailed error messages guide users in case of issues like missing files or invalid questions.


## Future Enhancements
- Add support for other file types (e.g., Excel, JSON).
- Enable advanced filtering and visualization of data within the app.
- Implement caching for faster retrieval of frequently queried data.

## License
This project is licensed under the MIT License. Feel free to use and modify it as per your requirements.

## Contact
For any queries or feedback, please contact:
- **Developer:** Sharvesh R
- **Email:** [rsharvesh16@gmail.com]

