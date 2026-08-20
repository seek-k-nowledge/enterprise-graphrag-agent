import streamlit as st
import requests
from neo4j import GraphDatabase
from pyvis.network import Network
import streamlit.components.v1 as components
import io

st.set_page_config(page_title="Enterprise GraphRAG", page_icon="🕸️", layout="wide")
st.title("🕸️ Enterprise GraphRAG Assistant")

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
                    "http://localhost:8000/api/v1/ingest",
                    json={
                        "source_id": doc_id,
                        "document_text": doc_text,
                        "priority": "normal"
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
                                        "http://localhost:8000/api/v1/ingest",
                                        json={
                                            "source_id": f"{pdf_doc_id}_chunk_{i}",
                                            "document_text": chunk,
                                            "priority": "normal"
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

# Main Layout Tabs
tab1, tab2 = st.tabs(["💬 Chat Assistant", "🕸️ Interactive Graph Viewer"])

with tab1:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Ask a question about your knowledge graph..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.spinner("Searching graph & synthesizing answer..."):
            res = requests.post("http://localhost:8000/api/v1/query", json={"query": prompt})
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
            driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "graphrag_dev_password"))
            with driver.session() as session:
                # Query 1: Fetch nodes with relationships
                edges_result = session.run(
                    """MATCH (n)-[r]->(m)
                       RETURN labels(n)[0] AS source_type,
                              coalesce(n.name, n.id, n.canonical_name, n.text, 'Node') AS source,
                              type(r) AS rel,
                              labels(m)[0] AS target_type,
                              coalesce(m.name, m.id, m.canonical_name, m.text, 'Node') AS target
                       LIMIT $limit""",
                    limit=graph_limit
                )
                edges_records = list(edges_result)

                # Query 2: Fetch isolated nodes (only if we have room)
                remaining = max(1, graph_limit - len(edges_records) * 2)
                isolated_result = session.run(
                    """MATCH (n)
                       WHERE NOT EXISTS((n)-[]-())
                       RETURN labels(n)[0] AS type,
                              coalesce(n.name, n.id, n.canonical_name, n.text, 'Node') AS name
                       LIMIT $limit""",
                    limit=remaining
                )
                isolated_records = list(isolated_result)

                all_records = edges_records + isolated_records

                if not all_records:
                    st.info("📊 No graph data available. Ingest documents to populate the graph.")
                else:
                    net = Network(height="500px", width="100%", bgcolor="#222222", font_color="white", directed=True)
                    edge_count = 0

                    # Process edge records
                    for record in edges_records:
                        src = record.get("source", "Unknown")
                        src_type = record.get("source_type", "Node")
                        tgt = record.get("target", "Unknown")
                        tgt_type = record.get("target_type", "Node")
                        rel = record.get("rel", "link")

                        net.add_node(src, label=src, title=f"{src_type}", color="#97C2FC")
                        net.add_node(tgt, label=tgt, title=f"{tgt_type}", color="#FFFF00")
                        net.add_edge(src, tgt, title=rel, label=rel)
                        edge_count += 1

                    # Process isolated node records
                    for record in isolated_records:
                        name = record.get("name", "Unknown")
                        node_type = record.get("type", "Node")
                        if name not in [n for n in net.nodes]:
                            net.add_node(name, label=name, title=f"{node_type}", color="#97C2FC")

                    net.save_graph("graph.html")
                    with open("graph.html", "r", encoding="utf-8") as f:
                        html_content = f.read()
                    components.html(html_content, height=520)
                    st.caption(f"✓ Displayed {net.num_nodes} nodes, {edge_count} edges")

        except Exception as e:
            st.error(f"❌ Could not connect to Neo4j database: {e}")
            st.caption("Ensure Neo4j is running at bolt://localhost:7687 with credentials neo4j/graphrag_dev_password")
