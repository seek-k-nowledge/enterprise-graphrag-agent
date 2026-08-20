import streamlit as st
import requests
from neo4j import GraphDatabase
from pyvis.network import Network
import streamlit.components.v1 as components

st.set_page_config(page_title="Enterprise GraphRAG", page_icon="🕸️", layout="wide")
st.title("🕸️ Enterprise GraphRAG Assistant")

# Sidebar - Document Ingestion
with st.sidebar:
    st.header("📄 Ingest Knowledge")
    doc_id = st.text_input("Document ID", "doc_001")
    doc_text = st.text_area("Document Content", height=200)
    if st.button("Ingest Document"):
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
                result = session.run(
                    "MATCH (n) OPTIONAL MATCH (n)-[r]->(m) RETURN n, r, m LIMIT $limit",
                    limit=graph_limit
                )
                records = list(result)

                if not records:
                    st.info("📊 No graph data available. Ingest documents to populate the graph.")
                else:
                    net = Network(height="500px", width="100%", bgcolor="#222222", font_color="white", directed=True)
                    edge_count = 0

                    for record in records:
                        n_node = record.get("n")
                        rel = record.get("r")
                        m_node = record.get("m")

                        # Extract source node info
                        if n_node:
                            src_id = n_node.get("name") or n_node.get("id") or str(id(n_node))
                            src_label = str(src_id)
                            net.add_node(src_label, label=src_label, color="#97C2FC")

                            # Add target node and edge only if relationship exists
                            if rel and m_node:
                                tgt_id = m_node.get("name") or m_node.get("id") or str(id(m_node))
                                tgt_label = str(tgt_id)
                                rel_type = rel.type if hasattr(rel, "type") else "link"

                                net.add_node(tgt_label, label=tgt_label, color="#FFFF00")
                                net.add_edge(src_label, tgt_label, title=rel_type, label=rel_type)
                                edge_count += 1

                    net.save_graph("graph.html")
                    with open("graph.html", "r", encoding="utf-8") as f:
                        html_content = f.read()
                    components.html(html_content, height=520)
                    st.caption(f"✓ Displayed {net.num_nodes} nodes, {edge_count} edges")

        except Exception as e:
            st.error(f"❌ Could not connect to Neo4j database: {e}")
            st.caption("Ensure Neo4j is running at bolt://localhost:7687 with credentials neo4j/graphrag_dev_password")
