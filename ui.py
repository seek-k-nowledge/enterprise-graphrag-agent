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
    if st.button("Refresh Graph View"):
        try:
            driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "graphrag_dev_password"))
            with driver.session() as session:
                result = session.run(
                    "MATCH (n)-[r]->(m) RETURN coalesce(n.name, n.id, labels(n)[0]) AS source, type(r) AS rel, coalesce(m.name, m.id, labels(m)[0]) AS target LIMIT 50"
                )
                net = Network(height="500px", width="100%", bgcolor="#222222", font_color="white", directed=True)
                for record in result:
                    src = str(record["source"])
                    tgt = str(record["target"])
                    rel = str(record["rel"])
                    net.add_node(src, label=src, color="#97C2FC")
                    net.add_node(tgt, label=tgt, color="#FFFF00")
                    net.add_edge(src, tgt, title=rel, label=rel)
                net.save_graph("graph.html")
                with open("graph.html", "r", encoding="utf-8") as f:
                    html_content = f.read()
                components.html(html_content, height=520)
        except Exception as e:
            st.error(f"Could not connect to Neo4j database: {e}")
