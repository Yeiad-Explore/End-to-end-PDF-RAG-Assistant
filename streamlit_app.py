# streamlit_app.py - Streamlit Frontend with LLM Selection
import streamlit as st
import requests
import json
import time

# FastAPI backend URL
API_URL = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide"
)

# Title and description
st.title("📚 PDF RAG Assistant")
st.markdown("Ask questions about your PDF documents with multiple LLM options!")

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'llm_options' not in st.session_state:
    st.session_state.llm_options = {}

# Load LLM options
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_llm_options():
    try:
        response = requests.get(f"{API_URL}/llm-options", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return {
        "providers": {
            "openai": {"models": ["gpt-4o-mini"], "default": "gpt-4o-mini"},
            "groq": {"models": ["deepseek-r1-distill-llama-70b"], "default": "deepseek-r1-distill-llama-70b"},
            "gemini": {"models": ["gemini-pro"], "default": "gemini-pro"}
        }
    }

# Sidebar for system status and LLM configuration
with st.sidebar:
    st.header("System Status")
    
    # Check API status
    try:
        response = requests.get(f"{API_URL}/status", timeout=5)
        if response.status_code == 200:
            status_data = response.json()
            if status_data["status"] == "ready":
                st.success("✅ System Ready")
                if "current_llm" in status_data:
                    current_llm = status_data["current_llm"]
                    st.info(f"Current LLM: {current_llm['provider']} ({current_llm['model']})")
            else:
                st.warning("⚠️ System Not Ready")
                st.warning(status_data["message"])
        else:
            st.error("❌ API Error")
            st.error(f"Status code: {response.status_code}")
    except requests.exceptions.RequestException:
        st.error("❌ Cannot connect to API")
        st.error("Make sure FastAPI server is running on localhost:8000")
    
    st.markdown("---")
    
    # LLM Configuration Section
    st.header("🤖 LLM Configuration")
    
    # Load LLM options
    llm_options = load_llm_options()
    st.session_state.llm_options = llm_options
    
    # Provider selection
    providers = list(llm_options["providers"].keys())
    selected_provider = st.selectbox(
        "Choose LLM Provider:",
        providers,
        index=0,
        key="provider_select"
    )
    
    # Model selection based on provider
    if selected_provider in llm_options["providers"]:
        models = llm_options["providers"][selected_provider]["models"]
        default_model = llm_options["providers"][selected_provider]["default"]
        
        selected_model = st.selectbox(
            "Choose Model:",
            models,
            index=models.index(default_model) if default_model in models else 0,
            key="model_select"
        )
    else:
        selected_model = None
    
    # Configure default LLM button
    if st.button("Set as Default LLM"):
        try:
            response = requests.post(
                f"{API_URL}/configure-llm",
                json={
                    "llm_provider": selected_provider,
                    "model_name": selected_model
                },
                timeout=10
            )
            if response.status_code == 200:
                st.success("✅ Default LLM updated!")
                time.sleep(1)
                st.rerun()
            else:
                st.error(f"Error: {response.text}")
        except Exception as e:
            st.error(f"Error setting default LLM: {str(e)}")
    
    st.markdown("---")
    st.markdown("**Instructions:**")
    st.markdown("1. Make sure FastAPI server is running")
    st.markdown("2. Add PDF files to the 'pdfs' folder")
    st.markdown("3. Configure your preferred LLM")
    st.markdown("4. Ask questions about your documents")

# Main chat interface
st.header("Ask a Question")

# Create two columns for the question form
col1, col2 = st.columns([3, 1])

with col1:
    question = st.text_input(
        "Your Question:",
        placeholder="e.g., What is the main topic of the documents?",
        key="question_input"
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)  # Add some spacing
    # Option to use different LLM for this question
    use_different_llm = st.checkbox("Use different LLM", key="use_different_llm")

# Additional LLM selection for this specific question
question_provider = selected_provider
question_model = selected_model

if use_different_llm:
    col3, col4 = st.columns(2)
    with col3:
        question_provider = st.selectbox(
            "Provider for this question:",
            providers,
            key="question_provider"
        )
    with col4:
        if question_provider in llm_options["providers"]:
            q_models = llm_options["providers"][question_provider]["models"]
            question_model = st.selectbox(
                "Model for this question:",
                q_models,
                key="question_model"
            )

# Submit button
submitted = st.button("Ask Question", type="primary", use_container_width=True)

# Process question
if submitted and question:
    try:
        # Show loading spinner
        with st.spinner(f"Thinking with {question_provider} ({question_model})..."):
            # Prepare request payload
            payload = {
                "question": question,
                "llm_provider": question_provider,
                "model_name": question_model
            }
            
            # Make API call
            response = requests.post(
                f"{API_URL}/ask",
                json=payload,
                timeout=60  # Increased timeout for some models
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result["answer"]
                llm_used = result.get("llm_used", f"{question_provider} ({question_model})")
                
                # Add to chat history
                st.session_state.chat_history.append({
                    "question": question,
                    "answer": answer,
                    "llm_used": llm_used,
                    "timestamp": time.time()
                })
                
                st.success(f"Answer received from {llm_used}!")
                
            elif response.status_code == 503:
                st.error("RAG system not initialized. Please check if PDFs are loaded.")
            else:
                st.error(f"API Error: {response.status_code}")
                if response.text:
                    st.error(response.text)
                    
    except requests.exceptions.Timeout:
        st.error("Request timed out. This model might take longer to respond. Please try again.")
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {str(e)}")
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")

elif submitted and not question:
    st.warning("Please enter a question!")

# Display chat history
if st.session_state.chat_history:
    st.header("Chat History")
    
    # Reverse order to show latest first
    for i, chat in enumerate(reversed(st.session_state.chat_history)):
        with st.container():
            # Question
            col1, col2 = st.columns([1, 10])
            with col1:
                st.markdown("**Q:**")
            with col2:
                st.markdown(f"*{chat['question']}*")
            
            # Answer
            col1, col2 = st.columns([1, 10])
            with col1:
                st.markdown("**A:**")
            with col2:
                st.markdown(chat['answer'])
            
            # Show which LLM was used
            col1, col2 = st.columns([1, 10])
            with col1:
                st.markdown("")
            with col2:
                st.markdown(f"*🤖 Answered by: {chat.get('llm_used', 'Unknown')}*")
            
            st.markdown("---")
    
    # Clear history button
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# Footer
st.markdown("---")
st.markdown("**Note:** Make sure your PDF files are in the 'pdfs' folder and the FastAPI server is running.")
st.markdown("**API Keys:** Set your API keys as environment variables: `OPENAI_API_KEY`, `GROQ_API_KEY`, `GOOGLE_API_KEY`")

# Auto-refresh status (optional)
if st.checkbox("Auto-refresh status"):
    time.sleep(1)
    st.rerun()