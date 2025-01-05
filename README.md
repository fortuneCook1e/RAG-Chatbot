# 🤖 RAG Chatbot: Domain-Specific Q&A Bot

A **Retrieval-Augmented Generation (RAG)** chatbot designed to answer domain-specific questions accurately by combining **retrieval-based systems** with **large language models (LLMs)**. This project preprocesses domain-related documents, retrieves relevant context dynamically, and generates precise answers using **Llama 3** hosted locally via **Ollama**.

---

## 🛠️ Features

- **Domain-Specific Question Answering**:
  - Accurately answers user queries based on a predefined set of documents.
  
- **Retrieval-Augmented Generation**:
  - Combines document retrieval from **ChromaDB** with LLM-powered generation for contextually accurate responses.

- **Local LLM Deployment**:
  - Uses **Ollama** to host the **Llama 3 model** locally, ensuring privacy and security.

- **Context-Aware Responses**:
  - Designed with a strict **prompt template** to enforce reliable and domain-specific answers, preventing out-of-scope responses.

- **API Integration**:
  - Built with **FastAPI**, exposing a RESTful endpoint for seamless interaction between the frontend and the backend.

---

## 📚 Technologies Used

### Backend:
- **FastAPI**: For defining the chatbot's RESTful API.
- **ChromaDB**: A vector database for storing and retrieving document embeddings.
- **Ollama**: To host the **Llama 3 model** locally.

### Artificial Intelligence:
- **Llama 3**: Large language model for generating context-aware answers.
- **Retrieval-Augmented Generation (RAG)**: To enhance LLM responses with domain-specific context.

### Data Processing:
- **Document Preprocessing**:
  - Chunking and embedding documents for efficient retrieval.

---

## 📦 Installation

### 1. Create and activate your virtual environment
	## For Conda:
	conda create -n myenv python=3.10
	conda activate myenv

### 2. Run the following command to install all libraries listed in requirements.txt
	pip install -r requirements.txt

### 3. Navigate to the app directory and run the following to start the server:
	uvicorn src.main:app

### 4. The server is now started and user can send API request to it.
