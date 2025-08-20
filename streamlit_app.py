# streamlit_app.py - Streamlit Frontend
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
st.markdown("Ask questions about your PDF documents!")

# Sidebar for system status
with st.sidebar:
    st.header("System Status")
    
    # Check API status
    try:
        response = requests.get(f"{API_URL}/status", timeout=5)
        if response.status_code == 200:
            status_data = response.json()
            if status_data["status"] == "ready":
                st.success("✅ System Ready")
                st.info(status_data["message"])
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
    st.markdown("**Instructions:**")
    st.markdown("1. Make sure FastAPI server is running")
    st.markdown("2. Add PDF files to the 'pdfs' folder")
    st.markdown("3. Ask questions about your documents")

# Main chat interface
st.header("Ask a Question")

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Question input
with st.form("question_form", clear_on_submit=True):
    question = st.text_input(
        "Your Question:",
        placeholder="e.g., What is the main topic of the documents?",
        key="question_input"
    )
    submitted = st.form_submit_button("Ask Question", type="primary")

# Process question
if submitted and question:
    try:
        # Show loading spinner
        with st.spinner("Thinking..."):
            # Make API call
            response = requests.post(
                f"{API_URL}/ask",
                json={"question": question},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result["answer"]
                
                # Add to chat history
                st.session_state.chat_history.append({
                    "question": question,
                    "answer": answer,
                    "timestamp": time.time()
                })
                
                st.success("Answer received!")
                
            else:
                st.error(f"API Error: {response.status_code}")
                if response.text:
                    st.error(response.text)
                    
    except requests.exceptions.Timeout:
        st.error("Request timed out. Please try again.")
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
            col1, col2 = st.columns([1, 10])
            with col1:
                st.markdown("**Q:**")
            with col2:
                st.markdown(f"*{chat['question']}*")
            
            col1, col2 = st.columns([1, 10])
            with col1:
                st.markdown("**A:**")
            with col2:
                st.markdown(chat['answer'])
            
            st.markdown("---")
    
    # Clear history button
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# Footer
st.markdown("---")
st.markdown("**Note:** Make sure your PDF files are in the 'pdfs' folder and the FastAPI server is running.")

# Auto-refresh status every 30 seconds (optional)
if st.checkbox("Auto-refresh status"):
    time.sleep(1)  # Small delay to prevent too frequent refreshes
    st.rerun()