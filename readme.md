# 🚀 PDF RAG Assistant: Your Intelligent Document Companion

Welcome to the **PDF RAG Assistant**, a powerful Retrieval-Augmented Generation (RAG) system designed to unlock the knowledge within your PDF documents. Powered by a sleek **FastAPI** backend and an intuitive **Streamlit** frontend, this tool brings your documents to life with cutting-edge AI, supporting multiple LLM providers for ultimate flexibility.

---

## ✨ Why You'll Love It

- 📑 **Multi-PDF Mastery**: Seamlessly process multiple PDFs at once
- 🧠 **Smart Semantic Search**: Leverages HuggingFace embeddings for precise document retrieval
- 🤖 **Choose Your AI**: Pick from OpenAI, Groq, or Gemini models to power your answers
- 🌐 **Interactive Web UI**: A polished Streamlit interface with chat history and LLM selection
- ⚡ **Robust API**: FastAPI backend with clean endpoints and error handling
- 🕒 **Live Status Updates**: Real-time system monitoring for a smooth experience

---

## 🏛️ System Architecture

```
┌────────────────────────┐    🚀 HTTP Requests    ┌────────────────────────┐
│                        │ ─────────────────────► │                        │
│  Streamlit Web UI      │                       │  FastAPI Backend       │
│  (Interactive Frontend) │ ◄──────────────────── │  (Powerful Server)     │
│                        │    📩 JSON Responses   │                        │
└────────────────────────┘                       └────────────────────────┘
                                                        │
                                                        ▼
                                                ┌────────────────────────┐
                                                │     RAG Pipeline       │
                                                │  • PDF Extraction      │
                                                │  • Text Chunking       │
                                                │  • Vector Embeddings   │
                                                │  • FAISS Vector Store  │
                                                │  • LLM Integration     │
                                                └────────────────────────┘
```

---

## 🌈 Get Started in Minutes

### 📋 Prerequisites

- Python 3.8 or higher
- API keys for your chosen LLM providers (OpenAI, Groq, Google)
- PDF documents ready for analysis

### 1. 📥 Clone or Set Up the Project

Create a project directory and include these essential files:
- `main.py`: The FastAPI backend powerhouse
- `streamlit_app.py`: The Streamlit frontend magic
- `requirements.txt`: All the dependencies you’ll need

### 2. 🛠️ Install Dependencies

**Option A: One-Click Install (Recommended)**
```bash
pip install fastapi uvicorn streamlit requests langchain langchain-openai langchain-community langchain-huggingface langchain-groq langchain-google-genai faiss-cpu PyPDF2 sentence-transformers tiktoken python-dotenv pydantic
```

**Option B: Step-by-Step Install**
```bash
pip install fastapi uvicorn streamlit requests
pip install langchain langchain-openai langchain-community langchain-huggingface langchain-groq langchain-google-genai
pip install faiss-cpu PyPDF2 sentence-transformers tiktoken python-dotenv pydantic
```

### 3. 🔑 Configure Your Environment

1. **Set Up API Keys** in a `.env` file:
   ```plaintext
   OPENAI_API_KEY=your-openai-api-key
   GROQ_API_KEY=your-groq-api-key
   GOOGLE_API_KEY=your-google-api-key
   ```

2. **Create a PDF Folder** and add your documents:
   ```bash
   mkdir pdfs
   # Copy your PDF files into the pdfs/ folder
   ```

### 4. 🚀 Launch the Application

**Terminal 1 - Fire Up the FastAPI Backend:**
```bash
python main.py
```
Access the server at: `http://localhost:8000`

**Terminal 2 - Launch the Streamlit Frontend:**
```bash
streamlit run streamlit_app.py
```
Open the web interface at: `http://localhost:8501`

---

## 📂 Project Structure

```
pdf-rag-assistant/
├── main.py                 # FastAPI backend server
├── streamlit_app.py        # Streamlit frontend application
├── requirements.txt        # Python dependencies
├── pdfs/                   # Folder for your PDF documents
│   ├── document1.pdf
│   ├── document2.pdf
│   └── ...
├── README.md               # This file
└── .env                    # Environment variables for API keys
```

---

## 🛠️ API Endpoints

### 🩺 Health Check
```http
GET /
```
Checks if the API is up and running.

### 📊 System Status
```http
GET /status
```
View the RAG system’s status and current LLM configuration.

### 🤖 LLM Options
```http
GET /llm-options
```
Lists available LLM providers and their models.

### ⚙️ Configure Default LLM
```http
POST /configure-llm
Content-Type: application/json

{
    "llm_provider": "openai",
    "model_name": "gpt-4o-mini"
}
```
Sets the default LLM for the system.

### ❓ Ask a Question
```http
POST /ask
Content-Type: application/json

{
    "question": "What is the main topic of the documents?",
    "llm_provider": "openai",
    "model_name": "gpt-4o-mini"
}
```

**Response Example:**
```json
{
    "answer": "The documents discuss...",
    "status": "success",
    "llm_used": "OpenAI (gpt-4o-mini)"
}
```

---

## 🔍 How It Works

1. **PDF Processing**: Automatically extracts text from all PDFs in the `pdfs/` folder on startup
2. **Text Chunking**: Splits documents into manageable chunks (1000 chars, 200-char overlap)
3. **Embedding Magic**: Converts chunks into vector embeddings using HuggingFace’s `all-MiniLM-L6-v2`
4. **Vector Storage**: Stores embeddings in FAISS for lightning-fast similarity searches
5. **Question Handling**: Embeds user questions and retrieves relevant document chunks
6. **AI-Powered Answers**: Uses your chosen LLM (OpenAI, Groq, or Gemini) to generate precise, context-based responses

---

## 🎯 Example Questions to Ask

- "What’s the main theme of these documents?"
- "Can you summarize the key points?"
- "What recommendations are provided?"
- "Explain [specific topic] based on the PDFs."

---

## 🌟 Features in Depth

### 🌍 Streamlit Frontend
- **Chat Interface**: Ask questions and get instant, conversational responses
- **Chat History**: Review all past questions and answers
- **LLM Flexibility**: Switch between providers and models on the fly
- **Real-Time Status**: Stay informed with live backend connection updates
- **Error Handling**: Clear, user-friendly error messages
- **Sleek Design**: A modern, responsive interface for a seamless experience

### ⚙️ FastAPI Backend
- **Automated PDF Processing**: Scans and processes PDFs at startup
- **RESTful API**: Clean, well-documented endpoints with proper status codes
- **Multi-LLM Integration**: Supports OpenAI, Groq, and Gemini for versatile answer generation
- **Robust Validation**: Uses Pydantic for secure input handling
- **High Performance**: Optimized vector search and caching for speed

### 🧠 RAG Pipeline
- **Multi-Document Support**: Combines insights from multiple PDFs
- **Semantic Precision**: Finds contextually relevant information with embeddings
- **Context-Driven Answers**: Ensures responses are grounded in your documents
- **No Hallucinations**: Answers strictly based on provided content

---

## 🛠️ Troubleshooting Tips

### 🔴 Common Issues

**"Cannot connect to API"**
- Verify the FastAPI server is running on `http://localhost:8000`
- Check for firewall restrictions
- Ensure no other application is using port 8000

**"System Not Ready"**
- Confirm PDFs are in the `pdfs/` folder
- Check FastAPI logs for initialization errors
- Ensure PDFs contain extractable text (not scanned images)

**"Poor or Empty Responses"**
- Validate all API keys and ensure they have sufficient credits
- Review startup logs to confirm PDFs were processed
- Try more specific or rephrased questions
- Check if the selected LLM model suits your needs

**Installation Hiccups**
- Confirm Python 3.8+ is installed
- Install dependencies individually if the batch command fails
- Use a virtual environment for a clean setup

### 🕵️‍♂️ Debug Steps

1. **Inspect FastAPI Logs**: Check the terminal running `main.py` for errors
2. **Test the API**: Visit `http://localhost:8000` in your browser
3. **Verify PDFs**: Ensure PDFs are readable and text-based
4. **Test API Keys**: Validate keys with a simple API call
5. **Check LLM Settings**: Confirm the selected provider and model are supported

---

## 🔒 Security Best Practices

- **Protect API Keys**: Store them in `.env` and exclude from version control
- **Local Processing**: All data stays on your machine for maximum privacy
- **Secure Setup**: No external data sharing or cloud uploads

---

## ⚡ Performance Optimization

- **PDF Size**: Large PDFs may slow initial processing
- **Chunk Tuning**: Adjust `chunk_size` (1000) and `overlap` (200) for better results
- **LLM Selection**: Experiment with models for optimal speed and accuracy
- **Caching**: Vector store is reused for faster queries after initial setup

---

## 🤝 Contribute & Customize

This is an open learning project! Here’s how you can enhance it:
- Add file upload functionality to the UI
- Support additional document formats (e.g., DOCX, TXT)
- Integrate more LLM providers
- Improve the UI with custom themes or layouts
- Optimize the RAG pipeline for larger datasets

---

## 📜 License & Compliance

This project is for **educational purposes**. Ensure compliance with:
- Terms of service for OpenAI, Groq, and Google APIs
- Copyright restrictions for your PDF content
- Relevant data privacy regulations

---

## 🆘 Need Help?

If you hit a snag:
1. Review the troubleshooting section
2. Check FastAPI logs for detailed errors
3. Verify dependency installation
4. Confirm API key validity
5. Ensure your chosen LLM provider is operational

---

**Unleash the Power of Your Documents! 🚀**

*Crafted with ❤️ using FastAPI, Streamlit, LangChain, and multiple LLM providers*