# EXACT 2026 Refactoring - LangGraph Implementation

## 🎯 Tổng Quan Cải Tiến

Project đã được refactor theo hướng **LangGraph** - một framework cho phép xây dựng agentic workflows dễ bảo trì và mở rộng.

### Các Thay Đổi Chính

## 1️⃣ **State Management** 

### Trước (Old)
Dữ liệu truyền từ class sang class không có cấu trúc rõ ràng.

### Sau (New)
```python
@dataclass
class WorkflowState:
    """Tất cả dữ liệu lưu trữ ở class này"""
    original_payload: Dict[str, Any]
    question: str
    query_type: Optional[str]
    agent_output: Optional[AgentOutput]
    # ... more fields
```

**Lợi ích:**
- ✅ Clear contract giữa các node
- ✅ Dễ debug: toàn bộ state visible
- ✅ Type safety: Pydantic validation

---

## 2️⃣ **LangGraph Workflow**

### Graph Structure

```
START
  ↓
[Router Node]  ← Classifies: logic or physics
  ↓
  ├─ Edge Function: route_to_agent()
  ↓
[Logic Node]  ← Z3 solver for logic problems
  ↓
END

[Physics Node] ← Formula + LLM for physics problems
  ↓
END
```

### Node = Agent, Edge = Function

- **Node**: Là một agent (routing, logic, physics)
- **Edge**: Là hàm quyết định (state → next node)
- **State**: Dữ liệu truyền qua các node

```python
# Edge function
def route(state: WorkflowState) -> str:
    if state.query_type == "logic":
        return "logic_agent"
    return "physics_agent"

# Node function
def logic_node(state: WorkflowState) -> WorkflowState:
    # Process state
    state.agent_output = ...
    return state
```

---

## 3️⃣ **Abstract LLM Client**

### Trước
```python
class VLLMClient:
    def chat(self, messages, ...):
        # Tightly coupled to vLLM
```

### Sau
```python
class LLMClientBase(ABC):
    @abstractmethod
    def chat(self, messages, ...) -> str:
        pass

class VLLMClient(LLMClientBase):
    # Implementation for vLLM
    
class AnthropicClient(LLMClientBase):
    # Can add later
```

**Lợi ích:**
- ✅ Dễ thay đổi LLM provider (vLLM → Claude → GPT-4)
- ✅ Dependency injection: `graph = ExactGraph(llm=VLLMClient())`

---

## 4️⃣ **Standardized Input/Output**

### Agent Output Format

Tất cả agent trả về cùng format:

```python
@dataclass
class AgentOutput:
    answer: str          # Final answer
    reasoning: str       # Explanation
    confidence: float    # 0.0 to 1.0
    metadata: Dict[str, Any]  # Extra data
    agent_name: str      # Which agent?
```

### Trước vs Sau

**Trước (Inconsistent):**
```python
# Logic agent
return {
    "answer": "Yes",
    "reasoning": "...",
    "confidence": 0.85,
    "metadata": {
        "kb_facts": 5,
        "kb_rules": 3,
    }
}

# Physics agent
return {
    "answer": "10 N",
    "unit": "N",
    "cot": [...],
    "confidence": 0.75,
    # Different format!
}
```

**Sau (Consistent):**
```python
# Both return AgentOutput
output = AgentOutput(
    answer="...",
    reasoning="...",
    confidence=0.8,
    metadata={
        "unit": "N",
        "cot": [...],
        "kb_facts": 5,
    },
    agent_name="PhysicsAgent",
)
```

---

## 5️⃣ **Observability với LangSmith**

### Trước
- Không có tracing
- Khó debug

### Sau
```python
# Tất cả automatically traced
from dotenv import load_dotenv
import os

load_dotenv()

# Set environment
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "ls_..."

# Run workflow
result = graph.predict(payload)
# ↓ Automatically logged to LangSmith dashboard
```

**Xem trên dashboard:**
- Input/Output của từng node
- Latency & performance metrics
- Token counts
- Error traces

---

## 6️⃣ **File Structure**

### Trước
```
agents/
├── llm_client.py
├── logic_agent.py
├── physics_agent.py
├── router.py
└── pipeline.py
```

### Sau
```
agents/
├── llm/
│   ├── base.py              # Abstract
│   ├── vllm_client.py       # Implementation
│   └── __init__.py
├── models/
│   ├── state.py             # WorkflowState, AgentOutput, AgentInput
│   └── __init__.py
├── nodes/
│   ├── router_node.py       # RouterNode (for LangGraph)
│   ├── logic_node.py        # LogicNode
│   ├── physics_node.py      # PhysicsNode
│   └── __init__.py
├── graph.py                 # ExactGraph (LangGraph workflow)
├── llm_client.py            # Backward compatibility wrapper
├── logic_agent.py           # Legacy (still used by nodes)
├── physics_agent.py         # Legacy (still used by nodes)
├── router.py                # Legacy (replaced by RouterNode)
└── pipeline.py              # Legacy (replaced by ExactGraph)
```

---

## 🚀 Quickstart

### 1. Setup Environment
```bash
# Copy template
cp .env.example .env

# Edit .env with your values
nano .env

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup LangSmith (Optional)

Xem [LANGSMITH_SETUP.md](./LANGSMITH_SETUP.md)

### 3. Run Demo
```bash
python main.py demo
# Logic problem ✓
# Physics problem ✓
# All traces logged to LangSmith!
```

### 4. Run API
```bash
python -m uvicorn api:app --reload

# Test
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "type": "logic",
    "question": "Does John graduate?",
    "premises-NL": ["If student has GPA > 3.5, graduates."]
  }'
```

---

## 📊 Comparison: Old vs New Architecture

| Aspect | Old | New |
|--------|-----|-----|
| **Routing** | `RouterAgent` class | `RouterNode` + conditional edges |
| **State** | Scattered dicts | `WorkflowState` dataclass |
| **LLM** | Hardcoded `VLLMClient` | Abstract + implementations |
| **Agents Output** | Inconsistent formats | Standardized `AgentOutput` |
| **Workflow** | Sequential pipeline | LangGraph with edges |
| **Observability** | None | LangSmith tracing |
| **Testability** | Difficult | Easy (isolated nodes) |
| **Extensibility** | Hard | Easy (add nodes) |

---

## 🔧 Adding New Nodes

### Example: Add a Math Solver Node

```python
# agents/nodes/math_node.py

class MathNode:
    def __init__(self, llm: LLMClientBase):
        self.llm = llm
    
    def __call__(self, state: WorkflowState) -> WorkflowState:
        # Process state.question
        result = self.solve_math(state.question)
        
        state.agent_output = AgentOutput(
            answer=result,
            reasoning="Math solver",
            confidence=0.9,
            agent_name="MathAgent",
        )
        return state
```

### Add to Graph

```python
# agents/graph.py

def _build_graph(self):
    graph = StateGraph(WorkflowState)
    
    # Add new node
    graph.add_node("math", MathNode(self.llm))
    
    # Add conditional edge
    graph.add_conditional_edges(
        "router",
        lambda s: s.route_to_agent(),
        {
            "logic_agent": "logic",
            "physics_agent": "physics",
            "math_agent": "math",  # New!
        }
    )
    
    # End edge
    graph.add_edge("math", "__end__")
```

---

## 🐛 Debugging with LangSmith

### View Traces

1. Go to https://smith.langchain.com/
2. Select project "exact-2026"
3. Click on a run to see:
   - Input payload
   - Output
   - Latency per node
   - Errors (if any)

### Add Custom Metadata

```python
def logic_node(state: WorkflowState) -> WorkflowState:
    # ... solve logic ...
    
    # Add metadata
    state.metadata = {
        "kb_size": len(kb.facts),
        "solving_time_ms": elapsed,
        "strategy": "z3_constraint",
    }
    return state
```

---

## ⚠️ Breaking Changes

### Old Code That Needs Update

```python
# OLD - No longer works
from agents.pipeline import ExactPipeline
pipe = ExactPipeline()
result = pipe.predict(payload)

# NEW - Use ExactGraph
from agents.graph import ExactGraph
from agents.llm.vllm_client import VLLMClient

llm = VLLMClient()
graph = ExactGraph(llm=llm)
result = graph.predict(payload)
```

### Backward Compatibility

Old imports still work (for now):
```python
from agents.llm_client import VLLMClient  # ✓ Still works
```

---

## 📚 References

- **LangGraph Docs**: https://python.langchain.com/docs/langgraph
- **LangSmith Docs**: https://docs.smith.langchain.com/
- **LangChain Docs**: https://python.langchain.com/docs

---

## ✅ Migration Checklist

- [x] Create `WorkflowState` dataclass
- [x] Create abstract `LLMClientBase`
- [x] Create node classes (Router, Logic, Physics)
- [x] Build `ExactGraph` workflow
- [x] Update `main.py` to use `ExactGraph`
- [x] Update `api.py` to use `ExactGraph`
- [x] Add LangSmith setup guide
- [x] Add `.env.example`
- [x] Update `requirements.txt`
- [ ] Add unit tests for new structure
- [ ] Add integration tests

---

Generated: 2026-05-27
