# 📚 PDF RAG Assistant

A Retrieval-Augmented Generation (RAG) system that allows you to ask questions about your PDF documents using FastAPI backend and Streamlit frontend.

## 🌟 Features

- **Multi-PDF Support**: Process multiple PDF documents simultaneously
- **Semantic Search**: Uses HuggingFace embeddings for intelligent document retrieval
- **OpenAI Integration**: Powered by GPT-4o-mini for accurate answers
- **Web Interface**: Clean Streamlit frontend with chat history
- **REST API**: FastAPI backend with proper error handling
- **Real-time Status**: Live system status monitoring

## 🏗️ Architecture

```
┌─────────────────┐    HTTP Requests    ┌─────────────────┐
│                 │ ──────────────────► │                 │
│  Streamlit UI   │                     │  FastAPI Server │
│  (Frontend)     │ ◄────────────────── │  (Backend)      │
│                 │    JSON Responses   │                 │
└─────────────────┘                     └─────────────────┘
                                                │
                                                ▼
                                        ┌─────────────────┐
                                        │   RAG Pipeline  │
                                        │  • PDF Parser   │
                                        │  • Text Chunker │
                                        │  • Embeddings   │
                                        │  • Vector Store │
                                        │  • LLM Chain    │
                                        └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API Key
- PDF documents to analyze

### 1. Clone/Download the Project

Create a new directory and add the following files:
- `main.py` (FastAPI backend)
- `streamlit_app.py` (Streamlit frontend)
- `requirements.txt` (dependencies)

### 2. Install Dependencies

**Option A: Single command (recommended)**
```bash
pip install fastapi uvicorn streamlit requests langchain langchain-openai langchain-community langchain-huggingface faiss-cpu PyPDF2 sentence-transformers tiktoken python-dotenv pydantic
```

**Option B: Step by step**
```bash
pip install fastapi uvicorn streamlit requests
pip install langchain langchain-openai langchain-community langchain-huggingface
pip install faiss-cpu PyPDF2 sentence-transformers tiktoken python-dotenv pydantic
```

### 3. Setup Your Environment

1. **Add your OpenAI API Key** in `main.py`:
   ```python
   os.environ["OPENAI_API_KEY"] = "your-actual-openai-api-key-here"
   ```

2. **Create PDF folder** and add your documents:
   ```bash
   mkdir pdfs
   # Copy your PDF files to the pdfs/ folder
   ```

### 4. Run the Application

**Terminal 1 - Start FastAPI Backend:**
```bash
python main.py
```
Server will start at: `http://localhost:8000`

**Terminal 2 - Start Streamlit Frontend:**
```bash
streamlit run streamlit_app.py
```
Web interface will open at: `http://localhost:8501`

## 📁 Project Structure

```
pdf-rag-assistant/
├── main.py                 # FastAPI backend server
├── streamlit_app.py        # Streamlit frontend application
├── requirements.txt        # Python dependencies
├── pdfs/                   # Your PDF documents folder
│   ├── document1.pdf
│   ├── document2.pdf
│   └── ...
├── README.md              # This file
└── .env                   # Environment variables (optional)
```

## 🔧 API Endpoints

### Health Check
```http
GET /
```
Returns API status and health information.

### System Status
```http
GET /status
```
Returns RAG system initialization status.

### Ask Question
```http
POST /ask
Content-Type: application/json

{
    "question": "What is the main topic of the documents?"
}
```

**Response:**
```json
{
    "answer": "Based on the documents provided...",
    "status": "success"
}
```

## 🎯 How It Works

1. **Document Processing**: PDFs are automatically processed on startup
2. **Text Chunking**: Documents are split into overlapping chunks (1000 chars with 200 overlap)
3. **Embedding Generation**: Each chunk is converted to vector embeddings using HuggingFace
4. **Vector Storage**: Embeddings are stored in FAISS for fast similarity search
5. **Question Processing**: User questions are embedded and matched with relevant document chunks
6. **Answer Generation**: GPT-4o-mini generates answers using only the retrieved context

## 💡 Usage Examples

**Sample Questions:**
- "What is the main topic discussed in these documents?"
- "Can you summarize the key findings?"
- "What are the recommendations mentioned?"
- "How does [specific concept] work according to the documents?"

## 🔍 Features in Detail

### Frontend (Streamlit)
- **Interactive Chat**: Ask questions and get immediate responses
- **Chat History**: View all previous questions and answers
- **System Status**: Real-time backend connection monitoring
- **Error Handling**: User-friendly error messages
- **Responsive Design**: Clean, professional interface

### Backend (FastAPI)
- **Automatic PDF Processing**: Scans and processes all PDFs on startup
- **RESTful API**: Clean HTTP endpoints with proper status codes
- **Error Handling**: Comprehensive error catching and reporting
- **Validation**: Input validation using Pydantic models
- **Performance**: Efficient vector search and caching

### RAG Pipeline
- **Multi-PDF Support**: Combines knowledge from multiple documents
- **Semantic Search**: Finds contextually relevant information
- **Context-Aware**: Answers are grounded in document content
- **No Hallucination**: System only answers from provided documents

## 🛠️ Troubleshooting

### Common Issues

**"Cannot connect to API"**
- Ensure FastAPI server is running on port 8000
- Check if `http://localhost:8000` is accessible
- Verify no firewall blocking the connection

**"System Not Ready"**
- Confirm PDF files are in the `pdfs/` folder
- Check FastAPI terminal for initialization errors
- Ensure PDFs contain readable text (not just images)

**"Empty or poor responses"**
- Verify OpenAI API key is correct and has credits
- Check if PDFs were processed correctly (view startup logs)
- Try more specific questions

**Installation Issues**
- Ensure Python 3.8+ is installed
- Try installing packages individually if batch install fails
- Consider using virtual environment for clean installation

### Debug Steps

1. **Check FastAPI logs**: Look for errors in the terminal running `main.py`
2. **Test API directly**: Visit `http://localhost:8000` in browser
3. **Verify PDFs**: Ensure PDFs are readable and contain text
4. **Check API key**: Test OpenAI API key with a simple request

## 🔒 Security Notes

- **API Key**: Never commit your OpenAI API key to version control
- **Local Only**: This setup runs locally - PDFs never leave your machine
- **Data Privacy**: All processing happens on your local system

## 📈 Performance Tips

- **PDF Size**: Larger PDFs take longer to process initially
- **Chunk Size**: Adjust chunk_size (1000) and overlap (200) for your use case
- **Model Choice**: Switch to different embedding models if needed
- **Caching**: Vector store is created once and reused for all queries

## 🤝 Contributing

This is a learning project! Feel free to:
- Add new features (file upload via UI, different document types)
- Improve error handling
- Enhance the user interface
- Optimize performance

## 📄 License

This project is for educational purposes. Please ensure you comply with:
- OpenAI API terms of service
- PDF content copyright restrictions
- Any applicable data privacy regulations

## 🆘 Support

For issues:
1. Check the troubleshooting section above
2. Review FastAPI logs for error details
3. Ensure all dependencies are properly installed
4. Verify OpenAI API key and credits

---

**Happy Learning! 🎓**

*Built with FastAPI, Streamlit, LangChain, and OpenAI*