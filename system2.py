# system_gemini_modern.py
import os
import sys
import io
import json
import tempfile
import shutil
import warnings
import re
from typing import List, Optional

import streamlit as st

from pydantic import BaseModel, Field

# Vectorstore + loaders + splitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma


# NEW: Latest Google GenAI SDK
from google import genai
from google.genai import types
from google.genai import errors
# Silence noisy warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


import streamlit as st

GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
CHROMA_DB_DIR = "./"

GEMINI_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "text-embedding-004"

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not found in environment variables. Add to your .env file.")
    st.stop()

# Initialize modern GenAI client
client = genai.Client(api_key=GEMINI_API_KEY)


# -------------------------
# Pydantic schemas for quiz
# -------------------------
class Question(BaseModel):
    question: str = Field(description="The multiple choice question text.")
    options: List[str] = Field(description="A list of 4 unique and plausible possible answers.")
    correct_answer: str = Field(description="The exact text of the correct option.")
    user_selection: Optional[str] = Field(None)


class Quiz(BaseModel):
    questions: List[Question] = Field(description="A list of exactly 5 multiple choice questions.")


# -------------------------
# Streamlit caching helpers
# -------------------------
@st.cache_resource
def initialize_embeddings():
    """Return config for embeddings"""
    return {"model": EMBEDDING_MODEL, "client": client}

@st.cache_resource(show_spinner="Processing document and creating vector store...")
def setup_retriever(uploaded_file, _embeddings_config):
    """
    Load PDF, split into chunks, compute embeddings via Google GenAI and create Chroma DB retriever.
    """
    if uploaded_file is None:
        return None

    # Save uploaded file to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        try:
            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()
        except Exception:
            loader = UnstructuredPDFLoader(tmp_file_path)
            documents = loader.load()
    except Exception as e:
        st.error(f"Failed to load PDF: {e}")
        os.remove(tmp_file_path)
        return None

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    splits = splitter.split_documents(documents)
    st.success(f"Document split into {len(splits)} chunks.")
    os.remove(tmp_file_path)

    # MODERN: Updated embeddings wrapper with batching
    class GeminiEmbeddingsWrapper:
        def __init__(self, model_name, client):
            self.model_name = model_name
            self.client = client

        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            """Embed multiple documents with batching (max 100 per batch)"""
            all_embeddings = []
            batch_size = 100
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                response = self.client.models.embed_content(
                    model=self.model_name,
                    contents=batch
                )
                all_embeddings.extend([emb.values for emb in response.embeddings])
            
            return all_embeddings

        def embed_query(self, text: str) -> List[float]:
            """Embed a single query"""
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=[text]
            )
            return response.embeddings[0].values

    # Use _embeddings_config
    embedder = GeminiEmbeddingsWrapper(_embeddings_config["model"], _embeddings_config["client"])

    # Create Chroma vectorstore
    try:
        vectordb = Chroma.from_documents(documents=splits, embedding=embedder, persist_directory=CHROMA_DB_DIR)
    except InvalidArgumentError as e:
        if "Collection expecting embedding with dimension" in str(e):
            st.warning("Embedding dimension mismatch detected. Clearing Chroma DB and retrying.")
            if os.path.exists(CHROMA_DB_DIR):
                shutil.rmtree(CHROMA_DB_DIR)
            vectordb = Chroma.from_documents(documents=splits, embedding=embedder, persist_directory=CHROMA_DB_DIR)
        else:
            raise

    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    st.success("Vector store and retriever created successfully.")
    return retriever


# -------------------------
# Helper: assemble context
# -------------------------
def build_context_from_docs(docs):
    """Build context string from retrieved documents"""
    ctx_pieces = []
    for i, doc in enumerate(docs):
        content = getattr(doc, "page_content", str(doc))
        trimmed = content.strip()
        if len(trimmed) > 1500:
            trimmed = trimmed[:1500] + " ... (truncated)"
        # Clean for JSON safety
        trimmed = trimmed.replace('"', "'").replace('\n', ' ').replace('\r', ' ')
        ctx_pieces.append(f"--- Retrieved chunk {i+1} ---\n{trimmed}\n")
    return "\n\n".join(ctx_pieces)


# -------------------------
# Helper: call Gemini Chat with modern SDK
# -------------------------
def gemini_chat(messages, model, temperature=0.1, max_output_tokens=4096):
    """
    Calls the Gemini API to generate content.
    
    Args:
        messages: A list of dicts like [{"role": "user", "content": "text"}]
        model: The model name (e.g., "gemini-2.5-flash")
        temperature: Controls randomness (0.0 to 1.0)
        max_output_tokens: Maximum tokens for the response
        
    Returns:
        The assistant's response text (str), or a detailed error message (str).
    """
    
    # 1. Prepare contents and extract system instruction
    system_instruction = None
    contents = []
    
    for msg in messages:
        if msg.get("role") == "system":
            # Set system instruction and skip it from the main contents list
            system_instruction = msg["content"]
        else:
            # Map roles: user -> user, assistant -> model (for the new SDK)
            role = "user" if msg.get("role") in ["user", "human"] else "model"
            contents.append(types.Content(
                parts=[types.Part(text=msg["content"])],
                role=role
            ))
    
    # 2. Call the API with comprehensive error handling
    try:
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                system_instruction=system_instruction
            )
        )
        
        # CRITICAL CHECK 1: Did the model return any text?
        if response.text is not None and response.text.strip():
            return response.text
        
        # If text is empty, check for block reasons (e.g., safety policy)
        if response.candidates and response.candidates[0].finish_reason:
            reason = response.candidates[0].finish_reason.name
            
            # Check for safety blocks
            if reason == "SAFETY":
                return "ERROR: Response was blocked due to safety policy. Please try a different query."
            
            return f"ERROR: Model generation stopped unexpectedly (Reason: {reason})."
        
        # Default fallback for an empty response object
        return "ERROR: The model returned an empty or unparseable response object."

    # CRITICAL CHECK 2: Catch specific API errors (Rate Limit, Auth, Timeout)
    except errors.APIError as e:
        # e.g., HTTP 429 (RESOURCE_EXHAUSTED/Rate Limit) or 403 (PERMISSION_DENIED)
        return f"API_ERROR: Gemini request failed (Code: {e.code}). Please check your quota or network."
        
    except Exception as e:
        # Catch non-API exceptions (e.g., network issues, SDK bugs)
        return f"UNKNOWN_ERROR: An unexpected Python error occurred during generation: {e}"
# -------------------------
# Streamlit UI and main logic
# -------------------------
def display_chat_history(chat_history):
    for message in chat_history:
        role = message.get("role", "user")
        content = message.get("content", "")
        if role == "user":
            with st.chat_message("user"):
                st.markdown(content)
        else:
            with st.chat_message("assistant"):
                st.markdown(content)


def main():
    st.set_page_config(page_title="Conversational PDF Chatbot (Gemini-only)")
    st.title("📄 Conversational PDF Chatbot — Gemini 1.5 Flash (Modern SDK)")

    # Sidebar
    with st.sidebar:
        st.header("1) Upload PDF")
        uploaded_file = st.file_uploader("Choose PDF", type="pdf", key="pdf_uploader")
        st.header("2) Model")
        st.markdown(f"**Gemini model:** `{GEMINI_MODEL}`")
        st.markdown(f"**Embeddings model:** `{EMBEDDING_MODEL}`")

    # Initialize embedding config
    embeddings_config = initialize_embeddings()

    # Session state init
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "quiz_data" not in st.session_state:
        st.session_state.quiz_data = None
    if "quiz_responses" not in st.session_state:
        st.session_state.quiz_responses = {}
    if "quiz_score" not in st.session_state:
        st.session_state.quiz_score = None
    if "quiz_submitted" not in st.session_state:
        st.session_state.quiz_submitted = False
    if "prev_mode" not in st.session_state:
        st.session_state.prev_mode = "Conversational Chat (RAG)"

    mode = st.selectbox("Select Application Mode:", ["Conversational Chat (RAG)", "Targeted Summarization", "Exam Generator & Grader"], key="app_mode")

    # If mode changes, reset states
    if mode != st.session_state.prev_mode:
        st.session_state.chat_history = []
        st.session_state.quiz_data = None
        st.session_state.quiz_score = None
        st.session_state.quiz_submitted = False
        st.session_state.prev_mode = mode
        st.rerun()

    # Setup retriever if file uploaded
    if uploaded_file and st.session_state.retriever is None:
        with st.spinner("Processing PDF and creating vector store..."):
            retriever = setup_retriever(uploaded_file, embeddings_config)
            st.session_state.retriever = retriever
            if retriever:
                st.success("Ready to chat with your document!")
                st.session_state.chat_history = []
                st.session_state.quiz_data = None
                st.session_state.quiz_score = None
                st.session_state.quiz_submitted = False
                st.rerun()

    if st.session_state.retriever is None:
        st.info("Please upload a PDF file in the sidebar to begin.")
        return

    # Show working file
    st.subheader(f"Working with: {uploaded_file.name}")
    retriever = st.session_state.retriever

    # MODE 1: Conversational RAG
    if mode == "Conversational Chat (RAG)":
        display_chat_history(st.session_state.chat_history)
        prompt = st.chat_input("Ask a question about the PDF...")
        
        if prompt:
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            display_chat_history(st.session_state.chat_history)
            
            with st.chat_message("assistant"):
                with st.spinner("Retrieving context and asking Gemini..."):
                    docs = retriever.invoke(prompt)
                    context = build_context_from_docs(docs)
                    
                    messages = [
                        {"role": "system", "content": "You are an expert assistant. Use the DOCUMENT CONTEXT strictly to answer. If the answer is not in context, say 'I don't know'."},
                        {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{prompt}\n\nAnswer based only on CONTEXT. If not present, reply 'I don't know based on the provided document.'"}
                    ]
                    
                    answer = gemini_chat(messages, model=GEMINI_MODEL, temperature=0.1, max_output_tokens=4096)
                    st.markdown(answer)
            
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.rerun()

    # MODE 2: Targeted Summarization
    elif mode == "Targeted Summarization":
        st.warning("This mode retrieves relevant chunks and synthesizes them.")
        display_chat_history(st.session_state.chat_history)
        
        summary_request = st.chat_input("What part of the document would you like summarized?")
        if summary_request:
            st.session_state.chat_history.append({"role": "user", "content": f"Summary Request: {summary_request}"})
            display_chat_history(st.session_state.chat_history)
            
            with st.chat_message("assistant"):
                with st.spinner("Retrieving and synthesizing..."):
                    docs = retriever.invoke(summary_request)
                    context = build_context_from_docs(docs)
                    
                    messages = [
                        {"role": "system", "content": "You are an expert summarizer. Synthesize CONTEXT into 1-2 cohesive paragraphs. No bullet points."},
                        {"role": "user", "content": f"CONTEXT:\n{context}\n\nREQUEST: {summary_request}\n\nProvide a concise synthesized summary."}
                    ]
                    
                    summary = gemini_chat(messages, model=GEMINI_MODEL, temperature=0.0, max_output_tokens=4096)
                    st.markdown(summary)
            
            st.session_state.chat_history.append({"role": "assistant", "content": summary})
            st.rerun()

    # MODE 3: Exam Generator & Grader
    # --- MODE 3: Exam Generator & Grader (The 'else' block from main function) ---
  
    # --- MODE 3: Exam Generator & Grader ---
    # Assuming this is inside your main function's conditional logic (e.g., if MODE == 3)
        # MODE 3: Exam Generator & Grader
    else:
        st.header("📚 Exam Generator & Grader")
        
        # --- Input Field and Generation Button ---
        quiz_topic = st.text_input(
            "Enter topic for a 5-question exam:", 
            key="quiz_topic_input", 
            placeholder="e.g., 'Principles of Retrieval-Augmented Generation (RAG)'"
        )

        # 1. QUIZ GENERATION LOGIC (Only runs if a quiz hasn't been generated yet)
        if st.session_state.get('quiz_data') is None:
            
            # --- The Generate Quiz button is placed here ---
            generate_quiz_button = st.button("✨ Generate Quiz")
            # -----------------------------------------------

            if generate_quiz_button and quiz_topic:
                # Store topic for display later
                st.session_state.quiz_topic = quiz_topic
                
                # 1. Start generation (Use a spinner for user feedback)
                with st.spinner(f"Generating 5-question quiz on '{quiz_topic}'..."):
                    
                    # --- Retrieve context from the document ---
                    docs = retriever.invoke(quiz_topic)
                    context = build_context_from_docs(docs)
                    
                    # --- System Instruction to enforce strict JSON output ---
                    system_prompt = """You are an expert examiner. Generate exactly 5 multiple-choice questions.
                    
Rules:
1. Each question MUST have exactly 4 options (A, B, C, D)
2. The 'correct_answer' MUST exactly match one of the options
3. Focus ONLY on the provided context
4. Output ONLY valid JSON, no markdown, no explanations, no code blocks
5. Use this exact schema:
{"questions": [{"question": "...", "options": ["...", "...", "...", "..."], "correct_answer": "..."}]}"""

                    # --- Call Gemini with NATIVE STRUCTURED OUTPUT ---
                    try:
                        response = client.models.generate_content(
                            model=GEMINI_MODEL,
                            contents=f"Context:\n{context}\n\nTopic: {quiz_topic}\n\nGenerate 5 MCQs:",
                            config=types.GenerateContentConfig(
                                temperature=0.2,
                                max_output_tokens=4096,
                                system_instruction=system_prompt,
                                response_mime_type="application/json",
                                response_schema=Quiz  # ENFORCES schema directly!
                            )
                        )
                        
                        # --- Direct JSON parsing (no regex needed!) ---
                        quiz_obj = Quiz.model_validate_json(response.text)
                        st.session_state.quiz_data = quiz_obj.questions
                        st.success("✅ Exam generated successfully! Scroll down to begin.")
                        st.rerun()  # Rerun to show the quiz form
                        
                    except Exception as e:
                        # --- Enhanced error handling with debug output ---
                        st.error(f"❌ Failed to generate quiz: {e}")
                        with st.expander("🔍 Debug - Raw model output"):
                            st.code(response.text if 'response' in locals() else "No response received", language="json")
                        st.stop()  # Stop execution to prevent further errors

        # 2. QUIZ DISPLAY AND GRADING BLOCK (Only runs if quiz_data exists)
        if st.session_state.get('quiz_data') is not None:
            # Display the topic
            st.subheader(f"📝 Exam on: {st.session_state.get('quiz_topic', 'Topic')}")
            
            # Initialize responses if not exists
            if 'quiz_responses' not in st.session_state:
                st.session_state.quiz_responses = {}
            
            # --- Show Quiz Form (if not submitted) ---
            if not st.session_state.get('quiz_submitted', False):
                with st.form("quiz_input_form", clear_on_submit=False):
                    for i, q in enumerate(st.session_state.quiz_data):
                        st.subheader(f"Question {i+1}: {q.question}")
                        
                        # Radio buttons for answer selection
                        user_answer = st.radio(
                            f"Select your answer for Question {i+1}:",
                            q.options,
                            key=f"q_{i}_input",
                            index=0,
                            label_visibility='collapsed'
                        )
                        # Store the response
                        st.session_state.quiz_responses[f"q_{i}"] = user_answer
                        st.markdown("---")
                    
                    # Submit button
                    if st.form_submit_button("🎯 Submit Exam"):
                        # --- Grading Logic ---
                        total = len(st.session_state.quiz_data)
                        correct = sum(
                            1 for i, q in enumerate(st.session_state.quiz_data)
                            if st.session_state.quiz_responses.get(f"q_{i}") == q.correct_answer
                        )
                        st.session_state.quiz_score = f"{correct} / {total}"
                        st.session_state.quiz_submitted = True
                        st.rerun()  # Show results

            # --- Show Results (after submission) ---
            if st.session_state.get('quiz_submitted', False):
                score = st.session_state.quiz_score
                st.success(f"### Your Grade: {score}")
                
                # Display each question with feedback
                for i, q in enumerate(st.session_state.quiz_data):
                    st.subheader(f"Question {i+1}: {q.question}")
                    user_answer = st.session_state.quiz_responses.get(f"q_{i}")
                    
                    for option in q.options:
                        style = "padding: 8px; border-radius: 5px; margin-bottom: 5px;"
                        icon = ""
                        
                        # Highlight correct answer
                        if option == q.correct_answer:
                            style += "background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb;"
                            icon = " ✅ **Correct Answer**"
                        # Highlight user's wrong answer
                        elif option == user_answer and option != q.correct_answer:
                            style += "background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb;"
                            icon = " ❌ **Your Answer**"
                        
                        st.markdown(f'<div style="{style}">{option}{icon}</div>', unsafe_allow_html=True)
                    
                    st.markdown("---")
                
                # --- Final Feedback Message ---
                try:
                    score_parts = score.split(' / ')
                    score_num = int(score_parts[0])
                    total_num = int(score_parts[1])
                    
                    if score_num == total_num:
                        st.balloons()
                        st.markdown("🎉 **Perfect score! Outstanding understanding!**")
                    elif score_num >= total_num * 0.8:
                        st.markdown("🌟 **Great job! You have a solid understanding.**")
                    elif score_num >= total_num * 0.6:
                        st.markdown("👍 **Good effort! Review the material to improve.**")
                    else:
                        st.markdown("📖 **Keep studying! Focus on key concepts from the document.**")
                except:
                    st.warning("Could not calculate score-based feedback.")


if __name__ == "__main__":
    main()
