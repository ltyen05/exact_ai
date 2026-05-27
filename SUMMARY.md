# EXACT 2026 Refactoring Summary

## 🎯 Mục Đích

Refactor project theo hướng **LangGraph** để:
- ✅ Xây dựng agentic workflows rõ ràng (nodes + edges)
- ✅ Thống nhất input/output giữa các agents
- ✅ Tích hợp LangSmith để trace & debug
- ✅ Tạo abstract interface cho LLM (dễ thay đổi provider)
- ✅ Cải thiện testability & maintainability

---

## 📁 File Thay Đổi & Tạo Mới

### New Files (Refactoring)

| File | Mục Đích |
|------|----------|
| `agents/models/state.py` | WorkflowState, AgentOutput, AgentInput |
| `agents/models/__init__.py` | Package init |
| `agents/llm/base.py` | Abstract LLMClientBase |
| `agents/llm/vllm_client.py` | VLLMClient (moved from llm_client.py) |
| `agents/llm/__init__.py` | Package init |
| `agents/nodes/router_node.py` | RouterNode for LangGraph |
| `agents/nodes/logic_node.py` | LogicNode for LangGraph |
| `agents/nodes/physics_node.py` | PhysicsNode for LangGraph |
| `agents/nodes/__init__.py` | Package init |
| `agents/graph.py` | **ExactGraph - Main LangGraph workflow** |
| `LANGSMITH_SETUP.md` | Hướng dẫn tạo API key & cấu hình |
| `QUICKSTART.md` | 5-minute setup guide |
| `ARCHITECTURE.md` | Detailed graph visualization |
| `REFACTORING.md` | Change documentation |
| `.env.example` | Environment template |

### Modified Files

| File | Thay Đổi |
|------|----------|
| `requirements.txt` | + langchain, langgraph, langsmith, python-dotenv |
| `main.py` | Use ExactGraph instead of ExactPipeline |
| `api.py` | Use ExactGraph instead of ExactPipeline |
| `agents/llm_client.py` | Backward compatibility wrapper |

### Unchanged (Legacy Support)

```
agents/
├── formatter.py     (still used)
├── logic_agent.py   (still used by LogicNode)
├── physics_agent.py (still used by PhysicsNode)
├── router.py        (legacy, replaced by RouterNode)
├── pipeline.py      (legacy, replaced by ExactGraph)
```

---

## 🚀 Architecture Changes

### Before (Linear Pipeline)
```
ExactPipeline
  ├─ RouterAgent → Classify
  ├─ LogicAgent → Solve
  └─ PhysicsAgent → Solve
```

### After (LangGraph)
```
ExactGraph (StateGraph)
  ├─ [Router Node] → Classify & route
  ├─ [Logic Node] → Solve (if logic)
  └─ [Physics Node] → Solve (if physics)
  
  State: WorkflowState (flows between nodes)
  Edges: Conditional routing based on state
```

---

## 🔄 Data Flow

### Unified I/O Format

**Before:**
- Logic → inconsistent output
- Physics → different format

**After:**
```python
# All agents return
AgentOutput(
    answer="...",
    reasoning="...",
    confidence=0.85,
    metadata={...},
    agent_name="LogicAgent",
)
```

---

## 🧩 New Components

### 1. WorkflowState

Central state object managing all data:
```python
@dataclass
class WorkflowState:
    original_payload: Dict
    question: str
    query_type: Optional[str]  # Set by router
    premises_nl: List[str]
    agent_output: Optional[AgentOutput]  # Set by agent nodes
    errors: List[str]
```

### 2. AgentOutput

Standardized output format:
```python
@dataclass
class AgentOutput:
    answer: str
    reasoning: str
    confidence: float
    metadata: Dict[str, Any]
    agent_name: str
```

### 3. Node Functions

Each node processes state:
```python
def router(state: WorkflowState) -> WorkflowState:
    state.query_type = classify(state.question)
    return state

def logic_node(state: WorkflowState) -> WorkflowState:
    state.agent_output = solve_logic(...)
    return state
```

### 4. Conditional Edges

Route based on state:
```python
graph.add_conditional_edges(
    "router",
    lambda s: s.route_to_agent(),  # "logic_agent" or "physics_agent"
    {"logic_agent": "logic", "physics_agent": "physics"}
)
```

---

## 🔌 Abstract LLM Interface

### Before
```python
class VLLMClient:
    def chat(self, messages, ...):
        # Hardcoded vLLM logic
```

### After
```python
class LLMClientBase(ABC):
    @abstractmethod
    def chat(self, messages, ...) -> str:
        pass

class VLLMClient(LLMClientBase):
    # vLLM implementation

class AnthropicClient(LLMClientBase):
    # Can add later
```

### Usage
```python
llm = VLLMClient()  # or AnthropicClient()
graph = ExactGraph(llm=llm)
```

---

## 📊 Observability with LangSmith

### Before
- No tracing
- Hard to debug

### After
```python
# Set environment
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "ls_..."

# All calls automatically traced
result = graph.predict(payload)
# ↓ Visible on LangSmith dashboard
```

**See on Dashboard:**
- Input/Output per node
- Latency
- Token counts
- Error traces

---

## 📚 Documentation Created

| Document | Content |
|----------|---------|
| `QUICKSTART.md` | 5-minute setup, common tasks |
| `ARCHITECTURE.md` | Graph topology, state flow, data classes |
| `REFACTORING.md` | Detailed before/after comparison |
| `LANGSMITH_SETUP.md` | LangSmith API key & tracing |
| `.env.example` | Environment variables template |

---

## ✅ Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment (Optional)
```bash
cp .env.example .env
# Edit with your LangSmith API key (optional)
```

### 3. Run Demo
```bash
python main.py demo
```

### 4. Run Evaluation
```bash
python main.py eval --max-records 100 --details results.json
```

### 5. Start API
```bash
python -m uvicorn api:app --reload
```

---

## 🔄 Migration Path

### Old Code
```python
from agents.pipeline import ExactPipeline
pipe = ExactPipeline()
result = pipe.predict(payload)
```

### New Code
```python
from agents.graph import ExactGraph
from agents.llm.vllm_client import VLLMClient

llm = VLLMClient()
graph = ExactGraph(llm=llm)
result = graph.predict(payload)
```

### Backward Compatibility
Old imports still work (for now):
```python
from agents.llm_client import VLLMClient  # ✓ Wrapper redirects
```

---

## 🎯 Benefits

| Aspect | Benefit |
|--------|---------|
| **Architecture** | Clear node/edge structure (like DAG) |
| **State** | Single source of truth (WorkflowState) |
| **Output** | Standardized format (AgentOutput) |
| **Testing** | Isolated node testing |
| **Extensibility** | Easy to add new nodes |
| **Debug** | Full trace visibility (LangSmith) |
| **Maintainability** | Less coupling, clear contracts |

---

## 🚀 Next Steps

1. **Verify Installation**
   ```bash
   python main.py demo
   ```

2. **Setup LangSmith** (Optional)
   - See [LANGSMITH_SETUP.md](./LANGSMITH_SETUP.md)

3. **Run Evaluation**
   ```bash
   python main.py eval --max-records 100
   ```

4. **Deploy API**
   ```bash
   python -m uvicorn api:app --reload
   ```

---

## 📖 Documentation Map

- 📄 **QUICKSTART.md** - Start here
- 📐 **ARCHITECTURE.md** - Graph visualization
- 🔄 **REFACTORING.md** - Technical details
- 🔬 **LANGSMITH_SETUP.md** - Observability
- ⚙️ **.env.example** - Configuration

---

## 🧪 Testing

### Unit Tests (To Do)
```python
def test_router_node():
    state = WorkflowState(question="...", payload={})
    result = router(state)
    assert result.query_type in ["logic", "physics"]
```

### Integration Tests (To Do)
```python
def test_full_graph():
    graph = ExactGraph(llm=MockLLM())
    result = graph.predict({"question": "...", "type": "logic"})
    assert result["answer"] != "Unknown"
```

---

## 📝 Version Info

| Component | Version |
|-----------|---------|
| LangGraph | 0.0.1+ |
| LangChain | 0.1.0+ |
| LangSmith | 0.1.0+ |
| Python | 3.10+ |

---

**Refactoring Date**: 2026-05-27  
**Status**: ✅ Complete  
**Breaking Changes**: None (backward compatible)  
**Recommended**: Update to use `ExactGraph` over `ExactPipeline`
