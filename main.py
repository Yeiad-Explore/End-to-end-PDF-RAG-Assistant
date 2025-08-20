
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings

# Set OpenAI API key
# os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"

app = FastAPI(title="PDF RAG API", description="RAG system for PDF documents")

# Pydantic models
class QuestionRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    answer: str
    status: str

# Global variables to store the RAG components
vector_store = None
main_chain = None

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
    global vector_store, main_chain
    
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
    
    # Set up LLM and prompt
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    prompt = PromptTemplate(
        template="""
      You are a helpful assistant.
      Answer ONLY from the provided document context.
      If the context is insufficient, just say you don't know.
      You are a copany assistant and will assist the fellow employee. be polite. Write in bangla if user writes in bangla, otherwise keep it english
        answer descriptive,  answer brod and better. USe points, sub points, guide to where he/she can find more information in the pdf.
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
    main_chain = parallel_chain | prompt | llm | parser
    
    return len(pdf_files), len(chunks)

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
    return {"message": "PDF RAG API is running", "status": "healthy"}

@app.get("/status")
async def get_status():
    """Get system status"""
    if main_chain is None:
        return {"status": "not_initialized", "message": "RAG system not initialized"}
    return {"status": "ready", "message": "RAG system ready to answer questions"}

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question to the RAG system"""
    if main_chain is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        # Get answer from RAG system
        answer = main_chain.invoke(request.question)
        
        return QuestionResponse(
            answer=answer,
            status="success"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)