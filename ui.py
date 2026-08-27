import streamlit as st
import requests
from neo4j import GraphDatabase
from pyvis.network import Network
import streamlit.components.v1 as components
import io
import os
import time

# Friendly explanations for extraction quality checks
FRIENDLY_ERROR_MESSAGES = {
    "surface_form_check": {
        "title": "🔍 A detail couldn't be double-checked",
        "explanation": (
            "The AI mentioned something here, but I couldn't find the "
            "exact wording in your document to confirm it. Rather than "
            "guess, I skipped it — this keeps your knowledge graph "
            "accurate and trustworthy."
        ),
    },
    "evidence_check": {
        "title": "🔗 A connection couldn't be confirmed",
        "explanation": (
            "The AI thought two things in your document were related, "
            "but couldn't find the exact supporting text. To avoid "
            "adding a connection that might not be accurate, I left "
            "it out."
        ),
    },
    "relation_resolution": {
        "title": "❓ Something didn't quite line up",
        "explanation": (
            "One of the connections the AI found didn't fully make "
            "sense (it may have referenced something that wasn't "
            "clearly identified), so it was left out to keep things "
            "accurate."
        ),
    },
}

DEFAULT_FRIENDLY_MESSAGE = {
    "title": "ℹ️ A detail was double-checked",
    "explanation": (
        "The AI found this detail, but when double-checked against your "
        "document, something didn't quite match. To keep your knowledge "
        "graph accurate, it was skipped — this is normal and nothing to "
        "worry about."
    ),
}

st.set_page_config(page_title="Enterprise GraphRAG", page_icon="🕸️", layout="wide")

# Dark data-graph theme CSS
DARK_THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');

/* Root color variables */
:root {
    --bg-dark: #0a0e17;
    --bg-card: #0f1524;
    --accent-blue: #6d8cff;
    --border-color: #22304a;
    --text-muted: #6b7488;
}

/* Overall theme */
.stApp {
    background-color: var(--bg-dark);
}

/* Main container */
[data-testid="stAppViewContainer"] {
    background-color: var(--bg-dark);
    color: #e8eef5;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: var(--bg-dark);
    border-right: 1px solid var(--border-color);
}

[data-testid="stSidebar"] [data-testid="stVerticalBlockContainer"] {
    background-color: var(--bg-dark);
}

/* Cards and containers with left border accent */
.status-card {
    background-color: var(--bg-card);
    border-left: 4px solid var(--accent-blue);
    border-radius: 0 8px 8px 0;
    padding: 16px;
    margin: 8px 0;
    border-top: 1px solid var(--border-color);
    border-right: 1px solid var(--border-color);
    border-bottom: 1px solid var(--border-color);
    transition: all 0.3s ease;
}

.status-card:hover {
    background-color: #151f30;
    border-left-color: #8aa4ff;
    box-shadow: 0 4px 12px rgba(109, 140, 255, 0.1);
}

.status-card.success {
    border-left-color: #10b981;
}

.status-card.success:hover {
    border-left-color: #34d399;
    box-shadow: 0 4px 12px rgba(16, 185, 129, 0.1);
}

.status-card.warning {
    border-left-color: #f59e0b;
}

.status-card.warning:hover {
    border-left-color: #fbbf24;
    box-shadow: 0 4px 12px rgba(245, 158, 11, 0.1);
}

.status-card.info {
    border-left-color: var(--accent-blue);
}

.status-card.info:hover {
    border-left-color: #8aa4ff;
    box-shadow: 0 4px 12px rgba(109, 140, 255, 0.1);
}

/* Buttons */
.stButton > button {
    background-color: var(--accent-blue);
    color: #0a0e17;
    border: none;
    border-radius: 6px;
    font-weight: 600;
    padding: 8px 16px;
    transition: all 0.2s ease;
    font-family: 'Segoe UI', sans-serif;
}

.stButton > button:hover {
    background-color: #8aa4ff;
    box-shadow: 0 4px 12px rgba(109, 140, 255, 0.2);
    transform: translateY(-1px);
}

.stButton > button:active {
    transform: translateY(0);
}

/* Secondary buttons (less prominent) */
.stButton > button[data-testid="baseButton-secondary"] {
    background-color: var(--bg-card);
    color: var(--accent-blue);
    border: 1px solid var(--accent-blue);
}

.stButton > button[data-testid="baseButton-secondary"]:hover {
    background-color: rgba(109, 140, 255, 0.1);
    border-color: #8aa4ff;
}

/* Monospace font for technical details */
.tech-id, .doc-id, .chunk-id {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    color: var(--accent-blue);
    background-color: rgba(109, 140, 255, 0.08);
    padding: 2px 6px;
    border-radius: 4px;
    letter-spacing: 0.3px;
}

/* Code blocks */
.stCodeBlock {
    background-color: #080c13 !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 6px !important;
}

pre {
    background-color: #080c13 !important;
    color: #a1afc3 !important;
}

code {
    font-family: 'JetBrains Mono', monospace;
    color: var(--accent-blue);
    background-color: rgba(109, 140, 255, 0.08);
    padding: 2px 6px;
    border-radius: 3px;
}

/* Expanders */
.streamlit-expanderHeader {
    background-color: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 6px;
    color: #e8eef5;
    font-weight: 500;
}

.streamlit-expanderHeader:hover {
    background-color: rgba(109, 140, 255, 0.05);
    border-color: var(--accent-blue);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] button {
    color: var(--text-muted);
    border-bottom: 2px solid transparent;
    transition: all 0.3s ease;
}

.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
    color: var(--accent-blue);
    border-bottom-color: var(--accent-blue);
}

.stTabs [data-baseweb="tab-list"] button:hover {
    color: #a1afc3;
}

/* Text input & select boxes */
.stTextInput > div > div > input,
.stSelectbox > div > div > select,
.stTextArea > div > div > textarea {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    color: #e8eef5 !important;
    border-radius: 6px !important;
    font-family: 'Segoe UI', sans-serif;
}

.stTextInput > div > div > input:focus,
.stSelectbox > div > div > select:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--accent-blue) !important;
    box-shadow: 0 0 0 2px rgba(109, 140, 255, 0.2) !important;
}

/* Messages and alerts */
.stAlert {
    background-color: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 6px;
}

.stSuccess {
    background-color: rgba(16, 185, 129, 0.1) !important;
    border-left: 4px solid #10b981 !important;
    color: #d1fae5 !important;
}

.stError {
    background-color: rgba(239, 68, 68, 0.1) !important;
    border-left: 4px solid #ef4444 !important;
    color: #fee2e2 !important;
}

.stWarning {
    background-color: rgba(245, 158, 11, 0.1) !important;
    border-left: 4px solid #f59e0b !important;
    color: #fef3c7 !important;
}

.stInfo {
    background-color: rgba(109, 140, 255, 0.1) !important;
    border-left: 4px solid var(--accent-blue) !important;
    color: #dbeafe !important;
}

/* Progress indicators */
.stProgress > div > div > div {
    background-color: var(--accent-blue);
}

/* Spinner */
.stSpinner {
    color: var(--accent-blue);
}

/* Divider */
hr {
    background-color: var(--border-color);
    border: none;
    height: 1px;
}

/* Fade-in animation for ingestion progress messages */
@keyframes fadeInSlide {
    from {
        opacity: 0;
        transform: translateY(-10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.ingestion-progress {
    animation: fadeInSlide 0.4s ease-out forwards;
}

.ingestion-progress:nth-child(1) { animation-delay: 0s; }
.ingestion-progress:nth-child(2) { animation-delay: 0.1s; }
.ingestion-progress:nth-child(3) { animation-delay: 0.2s; }
.ingestion-progress:nth-child(4) { animation-delay: 0.3s; }
.ingestion-progress:nth-child(5) { animation-delay: 0.4s; }

/* Headings */
h1, h2, h3, h4, h5, h6 {
    color: #e8eef5;
}

h1 {
    border-bottom: 2px solid var(--accent-blue);
    padding-bottom: 8px;
}

/* Links */
a {
    color: var(--accent-blue);
    text-decoration: none;
}

a:hover {
    color: #8aa4ff;
    text-decoration: underline;
}

/* Scrollbar styling */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: var(--bg-dark);
}

::-webkit-scrollbar-thumb {
    background: var(--border-color);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--accent-blue);
}
</style>
"""

st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)
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
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**File:** <code class='tech-id'>{uploaded_file.name}</code>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"**Size:** {uploaded_file.size / 1024:.2f} KB", unsafe_allow_html=True)

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
                            st.markdown(
                                f'<div class="ingestion-progress" style="background-color: rgba(109, 140, 255, 0.1); border-left-color: var(--accent-blue); padding: 12px; border-radius: 0 6px 6px 0; margin: 8px 0;"><strong>📦 Created <code class="tech-id">{len(chunks)}</code> chunks</strong></div>',
                                unsafe_allow_html=True
                            )

                            # Ingest all chunks
                            ingested = 0
                            failed = 0
                            chunk_errors = []
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
                                        job_data = res.json()
                                        job_id = job_data.get("job_id")
                                        # Poll job status and collect errors
                                        if job_id:
                                            job_status_res = requests.get(
                                                f"http://fastapi:8000/api/v1/jobs/{job_id}",
                                                timeout=10
                                            )
                                            if job_status_res.status_code == 200:
                                                job_status = job_status_res.json()
                                                if job_status.get("result", {}).get("extraction_errors"):
                                                    chunk_errors.extend(
                                                        job_status["result"]["extraction_errors"]
                                                    )
                                        ingested += 1
                                    else:
                                        failed += 1
                                except Exception as e:
                                    failed += 1

                                # Rate limit protection: spread chunks over time to stay under
                                # provider TPM (tokens-per-minute) limits. Groq free tier: 8000 TPM.
                                # 56 chunks * 150 tokens ≈ 8400 tokens, so 8s delay keeps us safe.
                                if i < len(chunks) - 1:
                                    time.sleep(8)

                            # Render completion message with fade-in animation
                            st.markdown(
                                f'<div class="ingestion-progress"><strong>✅ Ingestion complete</strong><br>Processed {ingested} chunks with {len(chunk_errors)} quality checks</div>',
                                unsafe_allow_html=True
                            )
                            if failed > 0:
                                st.markdown(
                                    f'<div class="ingestion-progress" style="animation-delay: 0.1s;"><strong>⚠️ {failed} chunks failed</strong></div>',
                                    unsafe_allow_html=True
                                )
                            if chunk_errors:
                                with st.expander(f"ℹ️ {len(chunk_errors)} details were double-checked and skipped for accuracy"):
                                    st.info(
                                        "These are intentional quality checks—nothing went wrong. Your knowledge "
                                        "graph is more accurate because of these checks. Click each item below to learn more."
                                    )
                                    for i, error in enumerate(chunk_errors):
                                        stage = error.get("stage", "unknown")
                                        friendly = FRIENDLY_ERROR_MESSAGES.get(stage, DEFAULT_FRIENDLY_MESSAGE)
                                        chunk_id = error.get('chunk_id', 'chunk')

                                        # Determine status card color based on error type
                                        card_class = "status-card info"
                                        if stage == "surface_form_check":
                                            card_class = "status-card warning"
                                        elif stage == "evidence_check":
                                            card_class = "status-card warning"
                                        elif stage == "relation_resolution":
                                            card_class = "status-card info"

                                        # Render styled status card
                                        st.markdown(
                                            f'<div class="{card_class}"><strong>{friendly["title"]}</strong> — <code class="chunk-id">{chunk_id}</code></div>',
                                            unsafe_allow_html=True
                                        )
                                        st.markdown(friendly["explanation"], unsafe_allow_html=True)

                                        with st.expander("🔧 Show technical details"):
                                            col1, col2 = st.columns(2)
                                            with col1:
                                                st.caption(f"**Type:** `{stage}`")
                                            with col2:
                                                st.caption(f"**Error ID:** <code class='tech-id'>{error.get('chunk_id', 'N/A')}</code>", unsafe_allow_html=True)
                                            st.caption(f"**Message:** {error.get('message', 'N/A')}")
                                            if error.get("payload"):
                                                st.code(error['payload'], language="text")

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
