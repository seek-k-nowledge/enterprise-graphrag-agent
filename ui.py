import streamlit as st
import requests
from neo4j import GraphDatabase
from pyvis.network import Network
import streamlit.components.v1 as components
import io
import os

st.set_page_config(page_title="Enterprise GraphRAG", page_icon="🕸️", layout="wide")
st.title("🕸️ Enterprise GraphRAG Assistant")

# Lightweight provider validation (just confirm API key works, don't run full pipeline)
def validate_provider_key(provider: str, api_key: str) -> tuple[bool, str]:
    """
    Lightweight validation: send 1-token request to confirm API key works.
    Returns (success: bool, message: str)
    """
    try:
        if provider == "groq":
            from langchain_groq import ChatGroq
            llm = ChatGroq(model_name="openai/gpt-oss-120b", groq_api_key=api_key, temperature=0.0)
            # Single token completion to validate key
            llm.invoke([{"role": "user", "content": "x"}])
            return True, "✓ Groq API key is valid"

        elif provider == "cerebras":
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model_name="gpt-oss-120b",
                api_key=api_key,
                base_url="https://api.cerebras.ai/v1",
                temperature=0.0,
            )
            llm.invoke([{"role": "user", "content": "x"}])
            return True, "✓ Cerebras API key is valid"

        elif provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic(model="claude-haiku-4-5-20251001", api_key=api_key, temperature=0.0)
            llm.invoke([{"role": "user", "content": "x"}])
            return True, "✓ Anthropic API key is valid"

        else:
            return False, f"Unknown provider: {provider}"

    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "unauthorized" in error_msg.lower():
            return False, f"❌ {provider.capitalize()} API key is invalid or expired"
        elif "429" in error_msg:
            return False, f"⚠️ {provider.capitalize()} rate limit hit (try again in a moment)"
        else:
            return False, f"❌ Connection failed: {error_msg[:100]}"

# Check if a default provider is configured and working
@st.cache_data(ttl=300)
def check_default_provider():
    """
    Check if .env has a working default provider (Groq, Cerebras, or Anthropic).

    Validates each key with a lightweight 1-token test before confirming it works.
    Returns (is_working: bool, provider_name: str | None)
    """
    providers_to_check = [
        ("groq", os.getenv("GROQ_API_KEY")),
        ("cerebras", os.getenv("CEREBRAS_API_KEY")),
        ("anthropic", os.getenv("ANTHROPIC_API_KEY")),
    ]

    for provider_name, api_key in providers_to_check:
        if api_key:
            # Found a key in env, validate it works
            success, _ = validate_provider_key(provider_name, api_key)
            if success:
                return True, provider_name

    return False, None

# Sidebar - Document Ingestion
with st.sidebar:
    st.header("📄 Ingest Knowledge")

    ingestion_tab1, ingestion_tab2 = st.tabs(["📝 Text", "📕 PDF"])

    # Text Input Tab
    with ingestion_tab1:
        doc_id = st.text_input("Document ID", "doc_001", key="text_doc_id")
        doc_text = st.text_area("Document Content", height=200, key="text_doc_text")
        if st.button("Ingest Text", key="ingest_text_btn"):
            if not doc_text.strip():
                st.error("Document content cannot be empty")
            else:
                res = requests.post(
                    "http://fastapi:8000/api/v1/ingest",
                    json={
                        "source_id": doc_id,
                        "document_text": doc_text,
                        "priority": "normal",
                        "llm_provider": st.session_state.llm_provider,
                        "llm_api_key": st.session_state.llm_api_key,
                    }
                )
                if res.status_code == 200:
                    st.success("Ingestion job started!")
                else:
                    st.error(f"Error: {res.text}")

    # PDF Upload Tab
    with ingestion_tab2:
        pdf_doc_id = st.text_input("Document ID", "pdf_doc_001", key="pdf_doc_id")
        uploaded_file = st.file_uploader(
            "Upload PDF or Text File",
            type=["pdf", "txt"],
            key="pdf_uploader"
        )

        if uploaded_file is not None:
            st.write(f"**File:** {uploaded_file.name}")
            st.write(f"**Size:** {uploaded_file.size / 1024:.2f} KB")

            if st.button("Process & Ingest PDF", key="ingest_pdf_btn"):
                with st.spinner("Processing document..."):
                    try:
                        # Extract text from PDF or TXT
                        if uploaded_file.type == "application/pdf":
                            try:
                                from pypdf import PdfReader
                                pdf_reader = PdfReader(io.BytesIO(uploaded_file.read()))
                                text = ""
                                for page in pdf_reader.pages:
                                    text += page.extract_text()
                                st.success(f"✓ Extracted {len(pdf_reader.pages)} pages")
                            except ImportError:
                                st.error("pypdf not installed. Run: pip install pypdf")
                                text = None
                        else:
                            # Plain text file
                            text = uploaded_file.read().decode("utf-8")
                            st.success(f"✓ Read text file ({len(text)} characters)")

                        if text:
                            # Chunk the document
                            from langchain_text_splitters import RecursiveCharacterTextSplitter
                            splitter = RecursiveCharacterTextSplitter(
                                chunk_size=1000,
                                chunk_overlap=150,
                                length_function=len
                            )
                            chunks = splitter.split_text(text)
                            st.info(f"📦 Created {len(chunks)} chunks")

                            # Ingest all chunks
                            ingested = 0
                            failed = 0
                            for i, chunk in enumerate(chunks):
                                try:
                                    res = requests.post(
                                        "http://fastapi:8000/api/v1/ingest",
                                        json={
                                            "source_id": f"{pdf_doc_id}_chunk_{i}",
                                            "document_text": chunk,
                                            "priority": "normal",
                                            "llm_provider": st.session_state.llm_provider,
                                            "llm_api_key": st.session_state.llm_api_key,
                                        },
                                        timeout=10
                                    )
                                    if res.status_code == 200:
                                        ingested += 1
                                    else:
                                        failed += 1
                                except Exception as e:
                                    failed += 1

                            st.success(f"✅ Ingestion complete: {ingested} chunks processed")
                            if failed > 0:
                                st.warning(f"⚠️ {failed} chunks failed")

                    except Exception as e:
                        st.error(f"❌ Processing failed: {e}")

# Initialize session state for LLM settings
if "llm_provider" not in st.session_state:
    st.session_state.llm_provider = None
if "llm_api_key" not in st.session_state:
    st.session_state.llm_api_key = None

# Settings in sidebar
with st.sidebar:
    st.divider()
    st.subheader("⚙️ LLM Settings")

    provider_options = [
        ("Auto (Cerebras→Groq)", None),
        ("Groq", "groq"),
        ("Cerebras", "cerebras"),
        ("Anthropic", "anthropic"),
    ]
    provider_labels = [label for label, _ in provider_options]
    provider_values = [value for _, value in provider_options]

    current_index = 0
    if st.session_state.llm_provider is not None:
        try:
            current_index = provider_values.index(st.session_state.llm_provider)
        except ValueError:
            current_index = 0

    selected_label = st.selectbox(
        "Provider",
        options=provider_labels,
        index=current_index,
        key="settings_provider"
    )
    selected_provider = provider_values[provider_labels.index(selected_label)]

    api_key = st.text_input(
        "API Key",
        type="password",
        key="settings_api_key",
        placeholder="Paste your API key here",
        help="Your provider's API key. Only used for this session."
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Test Connection", key="test_conn_btn"):
            if api_key and selected_provider:
                with st.spinner("Testing API key..."):
                    success, message = validate_provider_key(selected_provider, api_key)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
            else:
                st.warning("Select provider and enter API key")

    with col2:
        if st.button("Save Settings", key="save_settings_btn"):
            st.session_state.llm_provider = selected_provider
            st.session_state.llm_api_key = api_key
            st.success("Settings saved for this session!")

    if st.session_state.llm_provider:
        st.caption(f"Using: {selected_label}")

# Check if provider configured (session state OR .env default)
# Also validate that any .env provider actually works before using it
has_default, default_provider = check_default_provider()
if has_default and not st.session_state.llm_provider:
    # Auto-load from env on first page load
    st.session_state.llm_provider = default_provider
    st.session_state.llm_api_key = os.getenv(f"{default_provider.upper()}_API_KEY")

# Provider is configured if either:
# 1. User set it in session state with a valid key, OR
# 2. A working provider is configured in .env
provider_configured = (
    bool(st.session_state.llm_api_key and st.session_state.llm_provider) or has_default
)

# Main Layout Tabs
tab1, tab2 = st.tabs(["💬 Chat Assistant", "🕸️ Interactive Graph Viewer"])

with tab1:
    # First-run onboarding
    if not provider_configured:
        st.warning("""
        ⚠️ **API Key Not Configured**

        To use this application, you need an LLM provider API key:
        - **Groq** (free tier, recommended): [Get key at console.groq.com](https://console.groq.com)
        - **Cerebras**: [Sign up at cerebras.ai](https://cerebras.ai)
        - **Anthropic**: [API key at console.anthropic.com](https://console.anthropic.com)

        Once you have a key, go to **⚙️ LLM Settings** above to configure it.
        """)
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Ask a question about your knowledge graph..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.spinner("Searching graph & synthesizing answer..."):
            res = requests.post(
                "http://fastapi:8000/api/v1/query",
                json={
                    "query": prompt,
                    "llm_provider": st.session_state.llm_provider,
                    "llm_api_key": st.session_state.llm_api_key,
                }
            )
            if res.status_code == 200:
                answer = res.json().get("answer", "No response received.")
            else:
                answer = f"Error querying backend: {res.text}"
            
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)

with tab2:
    st.header("Graph Explorer")
    col1, col2 = st.columns([3, 1])
    with col2:
        graph_limit = st.number_input("Limit", min_value=10, max_value=500, value=50, step=10)

    if st.button("Refresh Graph View"):
        try:
            res = requests.get(
                "http://fastapi:8000/api/v1/graph",
                params={"limit": graph_limit}
            )
            if res.status_code == 200:
                graph_data = res.json()
                nodes = graph_data.get("nodes", [])
                edges = graph_data.get("edges", [])

                if not nodes:
                    st.info("📊 No graph data available. Ingest documents to populate the graph.")
                else:
                    net = Network(height="500px", width="100%", bgcolor="#222222", font_color="white", directed=True)

                    # Add nodes
                    for node in nodes:
                        node_id = node.get("id", "Unknown")
                        label = node.get("label", "Unknown")
                        node_type = node.get("type", "Node")
                        color = "#97C2FC" if node_type == "Entity" else "#FFFF00"
                        net.add_node(node_id, label=label, title=f"{node_type}", color=color)

                    # Add edges
                    for edge in edges:
                        src = edge.get("source", "Unknown")
                        tgt = edge.get("target", "Unknown")
                        label = edge.get("label", "link")
                        net.add_edge(src, tgt, title=label, label=label)

                    net.save_graph("graph.html")
                    with open("graph.html", "r", encoding="utf-8") as f:
                        html_content = f.read()
                    components.html(html_content, height=520)
                    st.caption(f"✓ Displayed {net.num_nodes} nodes, {len(edges)} edges")
            else:
                st.error(f"❌ Failed to fetch graph data: {res.text}")

        except Exception as e:
            st.error(f"❌ Could not fetch graph: {e}")
            st.caption("Ensure the FastAPI service is running at http://localhost:8000")
