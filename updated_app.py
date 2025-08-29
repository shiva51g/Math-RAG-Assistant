# app.py
import datetime
import io
import os
import traceback
from typing import List, Optional

import streamlit as st

# ---- LangChain & model imports ----
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory,
)

# Document loading & vectorstore imports (try common modules; app will catch missing)
try:
    from langchain.document_loaders import PyPDFLoader, TextLoader
    from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
    from langchain.vectorstores import FAISS, Chroma
except Exception:
    # We'll handle missing libraries at runtime with a friendly message
    PyPDFLoader = None
    TextLoader = None
    OpenAIEmbeddings = None
    HuggingFaceInstructEmbeddings = None
    FAISS = None
    Chroma = None

# ---------------------------
# Streamlit page config
# ---------------------------
st.set_page_config(page_title="Math + RAG Assistant (Gemma2)", page_icon="📒", layout="wide")
st.title("📒 Math + RAG Assistant — Groq Gemma2 + LangChain")

# ---------------------------
# Sidebar: settings & controls
# ---------------------------
with st.sidebar:
    st.header("🔧 Settings")

    groq_api_key = st.text_input("Groq API Key", type="password")
    model_name = st.selectbox("Model", options=["Gemma2-9b-It"], index=0)
    st.markdown("---")

    st.subheader("Memory")
    memory_type = st.selectbox(
        "Memory Type",
        options=["ConversationBufferMemory", "ConversationBufferWindowMemory", "ConversationSummaryMemory"],
        index=0,
    )
    if memory_type == "ConversationBufferWindowMemory":
        window_k = st.number_input("Window size (K)", min_value=1, max_value=20, value=6, step=1)
    else:
        window_k = None

    st.markdown("---")
    st.subheader("RAG / Embeddings")
    emb_backend = st.selectbox("Embeddings backend (requires keys if OpenAI)", options=["HuggingFace", "OpenAI"], index=0)
    vector_backend = st.selectbox("Vector DB", options=["FAISS (local)", "Chroma (local)"], index=0)

    st.caption(
        "Upload documents (PDF/TXT). Documents are embedded and stored in a local vector DB for semantic search."
    )

    st.markdown("---")
    st.subheader("UI / Behavior")
    response_style = st.selectbox("Response style", options=["Detailed step-by-step", "Concise", "Final answer only"], index=0)
    show_tool_traces = st.checkbox("Show tool traces / streaming", value=True)
    st.markdown("---")

    st.button("🗑️ Clear Chat", key="clear_chat_button")
    st.markdown(" ")

# Basic guard: API key
if not groq_api_key:
    st.info("Enter your Groq API Key in the sidebar to initialize the model.")
    st.stop()

# ---------------------------
# Initialize LLM
# ---------------------------
try:
    llm = ChatGroq(model=model_name, groq_api_key=groq_api_key)
except Exception as e:
    st.error("Failed to initialize Groq Chat model. Check your key and internet connection.")
    st.exception(e)
    st.stop()

# ---------------------------
# Tools: Wikipedia wrapper (langchain-community), Math chain, Reasoning chain
# ---------------------------
# Wikipedia tool (langchain_community wrapper)
try:
    from langchain_community.utilities import WikipediaAPIWrapper

    wiki_wrapper = WikipediaAPIWrapper()
    wikipedia_tool = Tool(name="Wikipedia", func=wiki_wrapper.run,
                          description="Search Wikipedia for factual info.")
except Exception:
    wikipedia_tool = None

# Math tool
math_chain = LLMMathChain.from_llm(llm=llm)
calculator_tool = Tool(name="Calculator", func=math_chain.run,
                       description="Solve math expressions and numerical problems.")

# Reasoning tool (prompted LLMChain)
BASE_PROMPT = """You are a helpful math & reasoning assistant.
Follow the requested response style.
Style: {style}

Question:
{question}

Answer:
"""
prompt_template = PromptTemplate(input_variables=["question", "style"], template=BASE_PROMPT)
reasoning_chain = LLMChain(llm=llm, prompt=prompt_template)
reasoning_tool = Tool(name="ReasoningTool", func=lambda q: reasoning_chain.run({"question": q, "style": response_style}),
                      description="Logical reasoning and step-by-step explanations.")

# Collect base tools
base_tools = [t for t in (wikipedia_tool, calculator_tool, reasoning_tool) if t is not None]

# ---------------------------
# Session state: messages, memory, vectorstore
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Hi — I'm a Math + RAG assistant. Upload docs or ask a math question!"}
    ]

def build_memory(mem_type: str):
    if mem_type == "ConversationBufferWindowMemory":
        return ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=window_k or 6)
    if mem_type == "ConversationSummaryMemory":
        # ConversationSummaryMemory will use the LLM to summarize older messages
        return ConversationSummaryMemory(memory_key="chat_history", return_messages=True, llm=llm)
    return ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# create memory if not present or if type changed
if "memory" not in st.session_state or st.session_state.get("memory_type") != memory_type:
    st.session_state.memory = build_memory(memory_type)
    st.session_state.memory_type = memory_type

# Reset chat (sidebar button)
if st.session_state.get("clear_chat_button", False):
    st.session_state.messages = [{"role": "assistant", "content": "🧹 Chat cleared. Start fresh!"}]
    st.session_state.memory = build_memory(memory_type)
    # reset the flag so it doesn't keep clearing
    st.sidebar.session_state["clear_chat_button"] = False

# Vectorstore container in session_state
if "vectorstore_info" not in st.session_state:
    st.session_state.vectorstore_info = {
        "vectorstore": None,
        "index_is_built": False,
        "docs_count": 0,
        "embeddings_backend": None,
        "vector_backend": None,
    }

# ---------------------------
# Document upload (multiple)
# ---------------------------
with st.expander("📁 Upload documents for RAG (PDF, TXT)", expanded=True):
    uploaded_files = st.file_uploader("Upload one or more files", accept_multiple_files=True, type=["pdf", "txt"], key="rag_uploader")
    build_index_btn = st.button("🔨 Build / Update Vector Index")

    st.markdown("**Current vectorstore status:**")
    vs_info = st.session_state.vectorstore_info
    st.write(f"- Built: {vs_info['index_is_built']}")
    st.write(f"- Documents indexed: {vs_info['docs_count']}")
    st.write(f"- Embeddings backend: {vs_info.get('embeddings_backend')}")
    st.write(f"- Vector backend: {vs_info.get('vector_backend')}")

# Utility: load docs from uploaded files to list of LangChain Document objects
def load_documents_from_files(files) -> List:
    docs = []
    for f in files:
        filename = f.name.lower()
        try:
            if filename.endswith(".pdf"):
                if PyPDFLoader is None:
                    st.error("PyPDFLoader not available. Install langchain with document loaders support.")
                    return []
                # PyPDFLoader expects a path or file-like - we will write to temp buffer
                temp = io.BytesIO(f.read())
                # PyPDFLoader often accepts a path; write a temp file to disk
                tmp_path = f"/tmp/{f.name}"
                with open(tmp_path, "wb") as fh:
                    fh.write(temp.getbuffer())
                loader = PyPDFLoader(tmp_path)
                docs.extend(loader.load())
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            elif filename.endswith(".txt"):
                if TextLoader is None:
                    st.error("TextLoader not available. Install langchain with document loaders support.")
                    return []
                temp = io.StringIO(f.read().decode("utf-8"))
                # TextLoader can load from file-like objects in some versions; fallback to wrapping content
                docs.append({"page_content": temp.getvalue(), "metadata": {"source": f.name}})
            else:
                st.warning(f"Unsupported file type: {f.name}")
        except Exception as e:
            st.error(f"Failed to load {f.name}: {e}")
            st.exception(traceback.format_exc())
    return docs

# Utility: get embeddings instance
def get_embeddings(backend: str):
    if backend == "OpenAI":
        if OpenAIEmbeddings is None:
            st.error("OpenAIEmbeddings not available. Install the appropriate langchain extras.")
            return None
        # User must set OPENAI_API_KEY in env for OpenAI embeddings
        return OpenAIEmbeddings()
    else:  # HuggingFace
        if HuggingFaceInstructEmbeddings is None:
            st.error("HuggingFace embeddings not available. Install sentence-transformers / langchain extras.")
            return None
        # You can customize model here (small one to run locally)
        return HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")

# Build index on button click
if 'build_index_clicks' not in st.session_state:
    st.session_state['build_index_clicks'] = 0

if build_index_btn:
    st.session_state['build_index_clicks'] += 1
    if not uploaded_files:
        st.warning("No files uploaded.")
    else:
        with st.spinner("Loading documents and building index..."):
            docs = load_documents_from_files(uploaded_files)
            if not docs:
                st.error("No documents loaded. See error messages above.")
            else:
                embeddings = get_embeddings(emb_backend)
                if embeddings is None:
                    st.error("Embeddings backend not available. Cannot build index.")
                else:
                    # Try FAISS first, fallback to Chroma
                    try:
                        if vector_backend.startswith("FAISS"):
                            if FAISS is None:
                                raise Exception("FAISS lib not available")
                            vectorstore = FAISS.from_documents(docs, embeddings)
                            st.success("FAISS index created.")
                            backend_used = "FAISS"
                        else:
                            if Chroma is None:
                                raise Exception("Chroma lib not available")
                            vectorstore = Chroma.from_documents(docs, embeddings)
                            st.success("Chroma collection created.")
                            backend_used = "Chroma"

                        # store vectorstore in session
                        st.session_state.vectorstore_info.update({
                            "vectorstore": vectorstore,
                            "index_is_built": True,
                            "docs_count": len(docs),
                            "embeddings_backend": emb_backend,
                            "vector_backend": backend_used,
                        })

                    except Exception as e:
                        st.error("Failed to build vectorstore. Check installed libraries and available memory.")
                        st.exception(e)

# ---------------------------
# Build RAG tool (retriever-based)
# ---------------------------
def build_rag_tool():
    info = st.session_state.vectorstore_info
    vs = info.get("vectorstore")
    if vs is None:
        return None

    def rag_func(query: str) -> str:
        try:
            # get top k docs
            retriever = vs.as_retriever(search_kwargs={"k": 4})
            docs = retriever.get_relevant_documents(query)
            context = "\n\n".join([d.page_content if hasattr(d, "page_content") else d["page_content"] for d in docs])
            # Ask the LLM to answer using context
            prompt = f"Use the context below to answer the question. If context doesn't contain the answer, say you couldn't find it.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer concisely with steps if needed."
            # Use reasoning_chain or direct llm call
            return reasoning_chain.run({"question": prompt, "style": response_style})
        except Exception as e:
            return f"RAG tool error: {e}"

    return Tool(
        name="RAG",
        func=rag_func,
        description="Retrieve relevant passages from uploaded documents and answer using them."
    )

rag_tool = build_rag_tool()
if rag_tool:
    tools = base_tools + [rag_tool]
else:
    tools = base_tools

# ---------------------------
# Initialize agent with memory and tools
# ---------------------------
assistant_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True,
    memory=st.session_state.memory,
)

# ---------------------------
# Sidebar: download chat history
# ---------------------------
if st.session_state.get("messages"):
    chat_text = "\n\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages])
    filename = f"chat_history_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    st.sidebar.download_button(label="⬇️ Download Chat History", data=chat_text, file_name=filename, mime="text/plain")

# ---------------------------
# Render chat messages
# ---------------------------
for msg in st.session_state.messages:
    avatar = "🤖" if msg["role"] == "assistant" else "🧑"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# ---------------------------
# Chat input
# ---------------------------
user_input = st.chat_input("Type your question here…")

if user_input:
    # Add user message to history & UI
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    # Run agent: stream if traces enabled
    with st.chat_message("assistant", avatar="🤖"):
        try:
            cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            if show_tool_traces:
                response = assistant_agent.run(user_input, callbacks=[cb])
            else:
                response = assistant_agent.run(user_input)

            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

        except Exception as e:
            st.error("⚠️ Something went wrong while generating the answer.")
            st.exception(e)
            st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})

# ---------------------------
# Helpful examples & tips
# ---------------------------
with st.expander("💡 Examples / Tips", expanded=False):
    st.markdown(
        """
- Try math: `I have 5 bananas and 7 grapes... how many fruits at the end?`
- Upload PDFs (research notes, manuals) and press *Build / Update Vector Index*.
- Ask document-specific questions: `In the uploaded PDF, what does section 3 say about model architecture?`
- If the RAG result is empty, try rebuilding the index or uploading more documents.
"""
    )
