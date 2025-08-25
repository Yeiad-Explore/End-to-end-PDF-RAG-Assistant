from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Literal
import os
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

# Set API keys - use environment variables for security
os.environ["OPENAI_API_KEY"] = os.getenv("you_api_key_here")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "you_api_key_here")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "you_api_key_here")

app = FastAPI(title="PDF RAG API", description="RAG system for PDF documents with multiple LLM support")

# Pydantic models
class QuestionRequest(BaseModel):
    question: str
    llm_provider: Literal["openai", "groq", "gemini"] = "openai"
    model_name: Optional[str] = None

class QuestionResponse(BaseModel):
    answer: str
    status: str
    llm_used: str

class LLMConfigRequest(BaseModel):
    llm_provider: Literal["openai", "groq", "gemini"]
    model_name: Optional[str] = None

# Global variables to store the RAG components
vector_store = None
retriever = None
current_llm_config = {"provider": "openai", "model": "gpt-4o-mini"}

def get_llm(provider: str, model_name: Optional[str] = None):
    """Get the appropriate LLM based on provider and model"""
    
    if provider == "openai":
        model = model_name or "gpt-4o-mini"
        return ChatOpenAI(model=model, temperature=0.2), f"OpenAI ({model})"
    
    elif provider == "groq":
        model = model_name or "deepseek-r1-distill-llama-70b"
        return ChatGroq(model=model, temperature=0.2), f"Groq ({model})"
    
    elif provider == "gemini":
        model = model_name or "gemini-pro"
        return ChatGoogleGenerativeAI(model=model, temperature=0.2), f"Gemini ({model})"
    
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

def extract_text_from_pdf(pdf_path):
    """Extract text from a single PDF file"""
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + " "
    return text.strip()

def initialize_rag_system(pdf_folder_path: str):
    """Initialize the RAG system with PDFs from a folder"""
    global vector_store, retriever
    
    # Get all PDF files
    pdf_files = []
    for filename in os.listdir(pdf_folder_path):
        if filename.lower().endswith('.pdf'):
            pdf_files.append(os.path.join(pdf_folder_path, filename))
    
    if not pdf_files:
        raise Exception("No PDF files found in the specified folder")
    
    # Extract text from all PDFs
    all_pdf_text = ""
    for pdf_path in pdf_files:
        pdf_text = extract_text_from_pdf(pdf_path)
        all_pdf_text += pdf_text + "\n\n"
    
    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([all_pdf_text])
    
    # Create embeddings and vector store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # Set up retriever
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    
    return len(pdf_files), len(chunks)

def create_rag_chain(llm_provider: str, model_name: Optional[str] = None):
    """Create RAG chain with specified LLM"""
    if retriever is None:
        raise Exception("RAG system not initialized")
    
    # Get LLM
    llm, llm_description = get_llm(llm_provider, model_name)
    
    # Set up prompt
    prompt = PromptTemplate(
        template="""
      You are a helpful assistant.
      Answer ONLY from the provided document context.
      If the context is insufficient, just say you don't know.
      You are a company assistant and will assist the fellow employee. Be polite. Write in bangla if user writes in bangla, otherwise keep it english.
      Answer descriptive, answer broad and better. Use points, sub points, guide to where he/she can find more information in the pdf.
      
      {context}
      Question: {question}
    """,
        input_variables=['context', 'question']
    )
    
    # Create the chain
    def format_docs(retrieved_docs):
        context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
        return context_text

    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })
    
    parser = StrOutputParser()
    chain = parallel_chain | prompt | llm | parser
    
    return chain, llm_description

@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on startup"""
    try:
        pdf_folder = "pdfs"  
        
        if not os.path.exists(pdf_folder):
            os.makedirs(pdf_folder)
            print(f"Created {pdf_folder} folder. Please add your PDF files there.")
            return
            
        num_pdfs, num_chunks = initialize_rag_system(pdf_folder)
        print(f"RAG system initialized with {num_pdfs} PDFs and {num_chunks} chunks")
        
    except Exception as e:
        print(f"Error initializing RAG system: {e}")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "PDF RAG API with multiple LLM support is running", "status": "healthy"}

@app.get("/status")
async def get_status():
    """Get system status"""
    if retriever is None:
        return {"status": "not_initialized", "message": "RAG system not initialized"}
    return {
        "status": "ready", 
        "message": "RAG system ready to answer questions",
        "current_llm": current_llm_config
    }

@app.get("/llm-options")
async def get_llm_options():
    """Get available LLM options"""
    return {
        "providers": {
            "openai": {
                "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
                "default": "gpt-4o-mini"
            },
            "groq": {
                "models": ["deepseek-r1-distill-llama-70b", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"],
                "default": "deepseek-r1-distill-llama-70b"
            },
            "gemini": {
                "models": ["gemini-pro", "gemini-1.5-pro", "gemini-1.5-flash"],
                "default": "gemini-1.5-flash"
            }
        }
    }

@app.post("/configure-llm")
async def configure_llm(config: LLMConfigRequest):
    """Configure default LLM settings"""
    global current_llm_config
    
    try:
        # Test the LLM configuration
        _, llm_description = get_llm(config.llm_provider, config.model_name)
        
        current_llm_config = {
            "provider": config.llm_provider,
            "model": config.model_name or {
                "openai": "gpt-4o-mini",
                "groq": "deepseek-r1-distill-llama-70b",
                "gemini": "gemini-pro"
            }[config.llm_provider]
        }
        
        return {
            "status": "success",
            "message": f"LLM configured to {llm_description}",
            "config": current_llm_config
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error configuring LLM: {str(e)}")

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question to the RAG system with specified LLM"""
    if retriever is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        # Use specified LLM or fall back to current config
        llm_provider = request.llm_provider or current_llm_config["provider"]
        model_name = request.model_name or current_llm_config["model"]
        
        # Create RAG chain with specified LLM
        chain, llm_description = create_rag_chain(llm_provider, model_name)
        
        # Get answer from RAG system
        answer = chain.invoke(request.question)
        
        return QuestionResponse(
            answer=answer,
            status="success",
            llm_used=llm_description
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)