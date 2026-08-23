# User-Facing Setup Guide: Implementation Status

## What's Implemented ✅

1. **Backend Provider Override** (Commit: `8f36a38`)
   - `get_llm()` accepts `api_key_override` parameter
   - All client factories (`_create_cerebras_client`, `_create_groq_client`, `_create_anthropic_client`) support key override
   - Request schemas updated to include `llm_provider` and `llm_api_key` fields (Commit: `9e8078b`)

2. **Windows Shortcuts** (Commit: `5305b5b`)
   - `start.bat`: One-click launch (requires Docker Desktop installed)
   - `stop.bat`: Clean shutdown

## What Still Needs Implementation 🚧

### #2: Frontend Settings Tab
**Location**: `ui.py`

Add to sidebar after ingestion tabs:
```python
# Settings Tab
with st.expander("⚙️ Settings", expanded=False):
    st.subheader("LLM Provider Configuration")
    
    provider = st.selectbox(
        "Select Provider",
        options=["Auto (Cerebras→Groq)", "Groq", "Cerebras", "Anthropic"],
        key="provider_select"
    )
    
    api_key = st.text_input(
        "API Key",
        type="password",
        key="api_key_input",
        help="Your provider's API key. Will be used for this session only."
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Test Connection"):
            # Test by calling /health with provider/key
            try:
                res = requests.get(
                    "http://localhost:8000/health",
                    params={"provider": provider, "api_key": api_key},
                    timeout=5
                )
                if res.status_code == 200:
                    st.success("✓ Connected!")
                else:
                    st.error(f"Error: {res.text}")
            except Exception as e:
                st.error(f"Connection failed: {e}")
    
    with col2:
        if st.button("Save"):
            st.session_state['llm_provider'] = provider
            st.session_state['llm_api_key'] = api_key
            st.success("Saved to session")

    st.session_state.setdefault('llm_provider', 'Auto (Cerebras→Groq)')
    st.session_state.setdefault('llm_api_key', '')
```

### #3: First-Run Onboarding
**Location**: `ui.py` (main page)

Add at the very top if no provider configured:
```python
if not st.session_state.get('llm_api_key'):
    st.warning("""
    ⚠️ **API Key Not Configured**
    
    To use this application, you need an LLM provider API key:
    - **Groq** (free tier recommended): [Get key at console.groq.com](https://console.groq.com)
    - Cerebras: [Sign up at cerebras.ai](https://cerebras.ai)
    - Anthropic: [API key at console.anthropic.com](https://console.anthropic.com)
    
    Once you have a key, go to **⚙️ Settings** in the sidebar to configure it.
    """)
```

### #4: Docker Compose Consolidation
**File**: `docker-compose.yml` (currently only has Neo4j)

Add FastAPI and Streamlit services:
```yaml
version: '3.8'

services:
  neo4j:
    image: neo4j:5-community
    ports:
      - "7474:7474"  # Neo4j Browser
      - "7687:7687"  # Bolt
    environment:
      NEO4J_AUTH: neo4j/graphrag_dev_password
      NEO4J_apoc_export_allow__all__imports: "true"
    volumes:
      - neo4j_data:/data

  fastapi:
    build:
      context: .
      dockerfile: Dockerfile.fastapi
    ports:
      - "8000:8000"
    environment:
      NEO4J_URI: bolt://neo4j:7687
      NEO4J_USER: neo4j
      NEO4J_PASSWORD: graphrag_dev_password
      PYTHONUNBUFFERED: 1
    depends_on:
      - neo4j
    command: uvicorn stages.fastapi_service.main:create_app --host 0.0.0.0 --port 8000

  streamlit:
    build:
      context: .
      dockerfile: Dockerfile.streamlit
    ports:
      - "8501:8501"
    environment:
      STREAMLIT_SERVER_HEADLESS: "true"
      NEO4J_URI: bolt://neo4j:7687
      NEO4J_USER: neo4j
      NEO4J_PASSWORD: graphrag_dev_password
    depends_on:
      - fastapi
    command: streamlit run ui.py --server.port 8501 --server.address 0.0.0.0

volumes:
  neo4j_data:
```

**New files needed**:
- `Dockerfile.fastapi`: 
  ```dockerfile
  FROM python:3.11-slim
  WORKDIR /app
  COPY requirements.txt .
  RUN pip install -r requirements.txt
  COPY . .
  ```

- `Dockerfile.streamlit`:
  ```dockerfile
  FROM python:3.11-slim
  WORKDIR /app
  COPY requirements.txt .
  RUN pip install -r requirements.txt
  COPY . .
  ```

### #1 (Complete): Backend Endpoint Integration
**File**: `stages/fastapi_service/main.py`

Update endpoints to pass provider/key through:
```python
# In query() endpoint:
answer_payload = answer_query(
    payload,
    app_state["graph_accessor"],
    provider=query_req.llm_provider,  # NEW
    api_key=query_req.llm_api_key,    # NEW
)

# In process_ingestion():
# Pass through to extract_document and any LLM calls
# (currently extract_document doesn't use LLM provider config)
```

## Getting a Free API Key

**Recommended: Groq** (most reliable during testing)
1. Go to [console.groq.com](https://console.groq.com)
2. Sign up with email
3. Copy API key
4. Paste in Settings tab → "Groq"
5. Click "Test Connection"

**Alternative: Anthropic**
1. Go to [console.anthropic.com](https://console.anthropic.com)
2. Create account
3. Copy API key
4. Paste in Settings tab → "Anthropic"

## Brand-New User Setup Checklist

```
Prerequisites:
☐ Docker Desktop installed (https://www.docker.com/products/docker-desktop)
☐ API key obtained (Groq or Anthropic)

Setup:
☐ Clone or download this repo
☐ Open terminal in project root
☐ Double-click start.bat (opens browser automatically)
☐ Go to Settings (⚙️) tab
☐ Select provider and paste API key
☐ Click "Test Connection"
☐ Start asking questions in Chat tab!

Stopping:
☐ Double-click stop.bat (or close terminal if running manually)
```

## Token Budget Tracking
These features were prioritized by user-impact and implementation complexity:
- ✅ Backend (3%) - minimal code changes, enables everything else
- ⚠️ Windows shortcuts (2%) - implemented, no Docker yet
- 🚧 UI Settings (15%) - straightforward Streamlit form
- 🚧 Docker (10%) - requires Dockerfile creation
- 🚧 Onboarding (5%) - simple conditional message

**Estimated implementation time for remaining items**: 2-4 hours
