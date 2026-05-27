# EXACT 2026 - Advanced Router & Subgraph Implementation

## 🎯 Tổng Quan

Project đã được refactor với **advanced routing** và **specialized subgraphs** cho Physics và Logic problems.

### Kiến Trúc Mới

```
START
  ↓
[Router Node] - Keyword-based classification
  ├─ Physics keywords, symbols, formulas
  ├─ Logic keywords, patterns (if-then, premises)
  └─ Multiple choice detection
  ↓
  ├─ PHYSICS SUBGRAPH
  │  ├─ Extract Quantities (LLM)
  │  ├─ Classify Problem Type (LLM)
  │  ├─ Select Abstract / RAG (Retrieval)
  │  ├─ Compute (SymPy)
  │  └─ Explain (LLM with detailed prompt)
  │
  └─ LOGIC SUBGRAPH
     ├─ Extract Logic (Premises, Rules, Query)
     ├─ Convert to FOL (First Order Logic)
     ├─ Verify with Z3 (Constraint solving)
     └─ Explain (LLM with logical reasoning)
  ↓
[Formatter Node] - Unify output format
  ↓
END
```

---

## 🔍 Enhanced Router

### Routing Strategy

File: `agents/nodes/router_node.py`

**Physics Detection:**
- Keywords: calculate, find, force, energy, voltage, circuit, etc.
- Symbols: Ω, V, A, N, Hz, °C, etc.
- Formulas: F=, E=, V=, P=, etc.
- Threshold: ≥2 physics keywords

**Logic Detection:**
- Keywords: premise, reasoning, conclusion, if-then, etc.
- Patterns: "If...then", "All...are", "Every...is"
- Multiple choice detection: Options A/B/C/D

**Default:** Logic (if ambiguous)

### Code Example

```python
from agents.nodes.router_node import RouterNode

router = RouterNode()
query_type = router.classify({
    "question": "Calculate the resultant force...",
    "type": "physics"
})
# Returns: "physics"
```

---

## ⚡ Physics Subgraph

File: `agents/subgraphs/physics_subgraph.py`

### Pipeline Stages

#### 1️⃣ Extract Quantities
```python
# LLM extracts from problem statement
Input: "Two forces of 5N and 3N at 60° angle..."
Output: {
    "quantities": [
        {"name": "Force 1", "symbol": "F1", "value": 5, "unit": "N"},
        {"name": "Force 2", "symbol": "F2", "value": 3, "unit": "N"},
        {"name": "Angle", "symbol": "θ", "value": 60, "unit": "°"}
    ],
    "facts": ["Two forces act simultaneously"],
    "objective": "Find resultant force"
}
```

#### 2️⃣ Classify Problem Type
```python
# LLM categorizes problem
Output: {
    "type": "mechanics",  # or electricity, thermodynamics, etc.
    "confidence": 0.95,
    "reason": "Keywords: force, angle suggest mechanics"
}
```

#### 3️⃣ Select Abstract or RAG
```python
# Option A: Find similar problem in training data (RAG)
if similarity ≥ 0.75:
    return training_example_with_solution

# Option B: Use direct computation
else:
    return {"approach": "Direct computation"}
```

#### 4️⃣ Compute Solution
```python
# SymPy formulas or solver
solve_by_formula("Two forces... 60° angle...")
# Returns: answer, unit, confidence, cot (chain of thought)
```

#### 5️⃣ Explain
```python
# LLM generates detailed explanation
Input: All extracted data + computed result
Output: "Step 1: Given F1=5N, F2=3N, θ=60°
        Step 2: Use resultant formula: R = √(F1² + F2² + 2F1F2cosθ)
        Step 3: R = √(25 + 9 + 30×0.5) = √49 = 7N"
```

---

## 🧠 Logic Subgraph

File: `agents/subgraphs/logic_subgraph.py`

### Pipeline Stages

#### 1️⃣ Extract Logic
```python
Input Premises:
- "If a student completes all courses, they are eligible"
- "If eligible AND GPA > 3.5, graduates with honors"
- "John completed all courses and has GPA 3.8"

Query: "Does John graduate with honors?"

Output: Extracted premises list + query
```

#### 2️⃣ Convert to FOL (First Order Logic)
```python
Conversion Methods:
- LLM-based: Intelligent NL → FOL conversion
- Heuristic: Pattern matching fallback

Output FOL:
- Predicates: eligible(X), honors(X), completed(X)
- Rules: completed(X) → eligible(X)
- Facts: completed(John), gpa(John, 3.8)
- Query: honors(John)?
```

#### 3️⃣ Verify with Z3
```python
# Z3 constraint solver
Z3Engine.verify(kb, query)
# Returns: Yes/No/Unknown, with confidence

# Explanation:
honors(John) = True (because:
  - completed(John) is True
  - gpa(John, 3.8) > 3.5 is True
  - Rule: completed(X) ∧ gpa(X) > 3.5 → honors(X)
  - Therefore: honors(John) = True)
```

#### 4️⃣ Explain
```python
# LLM generates educational explanation
Output: "Given premises:
1. Completion of all courses → eligibility
2. Eligibility + GPA > 3.5 → honors with distinction
3. John has completed all courses (given)
4. John has GPA 3.8 (> 3.5) (given)

Logical deduction:
- From (3) and (1): John is eligible
- From eligible state + (4) and (2): John graduates with honors

Answer: Yes, John receives academic distinction"
```

---

## 📊 State Management

### WorkflowState Fields

```python
@dataclass
class WorkflowState:
    # Input
    original_payload: Dict[str, Any]
    question: str
    query_type: Optional[str]  # "physics" or "logic"
    premises_nl: List[str]  # For logic
    premises_fol: Optional[List[str]]  # For logic
    
    # Processing
    router_confidence: float  # Router certainty
    parsed_kb: Any  # HornKB object for logic
    
    # Output
    agent_output: Optional[AgentOutput]
    logic_output: Optional[AgentOutput]
    physics_output: Optional[AgentOutput]
    
    # Metadata
    errors: List[str]
    metadata: Dict[str, Any]  # All intermediate results
```

### AgentOutput (Unified)

```python
@dataclass
class AgentOutput:
    answer: str  # Final answer
    reasoning: str  # Explanation
    confidence: float  # 0.0 to 1.0
    metadata: Dict[str, Any]  # Extra data
    agent_name: str  # Which agent
```

---

## 🚀 Usage

### Run Demo

```bash
python main.py demo
```

Expected Output:
```json
{
  "type": "physics",
  "question": "Two electric forces...",
  "answer": "7.91 N",
  "reasoning": "Using vector addition of forces...",
  "confidence": 0.85,
  "metadata": {
    "unit": "N",
    "problem_type": "mechanics",
    "extract_quantities": {...},
    "abstract_source": "training" or "none"
  }
}

{
  "type": "logic",
  "question": "Does John receive academic distinction?",
  "answer": "Yes",
  "reasoning": "Logical deduction using Z3...",
  "confidence": 0.95,
  "metadata": {
    "kb_facts": 3,
    "kb_rules": 2,
    "fol_conversion": {...},
    "z3_result": {...}
  }
}
```

### Run Evaluation

```bash
python main.py eval --max-records 100 --details results.json
```

### Start API

```bash
python -m uvicorn api:app --reload

# Test
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "type": "physics",
    "question": "Calculate force...",
    "unit": "N"
  }'
```

---

## 📁 File Structure

```
agents/
├── nodes/
│   ├── router_node.py      # Enhanced keyword-based router
│   ├── logic_node.py       # Legacy (kept for compatibility)
│   └── physics_node.py     # Legacy (kept for compatibility)
│
├── subgraphs/
│   ├── physics_subgraph.py # 5-stage physics pipeline
│   ├── logic_subgraph.py   # 4-stage logic pipeline
│   └── __init__.py
│
├── models/
│   ├── state.py            # WorkflowState, AgentOutput
│   └── __init__.py
│
├── llm/
│   ├── base.py             # Abstract LLMClientBase
│   ├── vllm_client.py      # VLLMClient implementation
│   └── __init__.py
│
├── graph.py                # MAIN: ExactGraph orchestrates subgraphs
├── llm_client.py           # Backward compatibility
├── pipeline.py             # Legacy (ExactPipeline)
├── router.py               # Legacy
├── logic_agent.py          # Used by subgraph nodes
├── physics_agent.py        # Used by subgraph nodes
└── formatter.py            # Output formatting
```

---

## 🔧 Customization

### Add Custom Physics Problem Type

```python
# In physics_subgraph.py, extend ClassifyProblemNode

class CustomClassifyNode:
    def __call__(self, state):
        # Add your custom logic
        if "quantum" in state.question.lower():
            state.metadata["problem_type"] = "quantum"
        return state
```

### Add Custom Logic Extraction

```python
# In logic_subgraph.py, extend ExtractLogicNode

class CustomExtractNode:
    def __call__(self, state):
        # Pre-process premises
        custom_premises = self.preprocess(state.premises_nl)
        state.metadata["custom_premises"] = custom_premises
        return state
```

### Experiment: With vs Without RAG

```python
# In SelectAbstractNode - compare approaches

# Approach 1: With RAG
retriever = PhysicsRetriever(kb_path)
similar = retriever.best(question)  # Find example

# Approach 2: Without RAG
direct_computation = solve_by_formula(question)
```

---

## 📊 Metadata Structure

### Physics Subgraph Metadata

```python
state.metadata = {
    "router_classification": "physics",
    "extract_quantities": {
        "source": "llm",
        "data": {quantities, facts, objective}
    },
    "quantities": [...],
    "facts": [...],
    "objective": "...",
    "problem_type": "mechanics",
    "problem_confidence": 0.95,
    "abstract_source": "training" or "none",
    "abstract_similarity": 0.82,
    "abstract_data": {
        "approach": "RAG" or "Direct",
        "cot": [...]
    },
    "compute_source": "formula",
    "formula_matched": True,
    "compute_result": {
        "answer": "7N",
        "unit": "N",
        "confidence": 0.9,
        "cot": [...]
    },
    "explanation": "Step-by-step explanation...",
    "explanation_source": "llm"
}
```

### Logic Subgraph Metadata

```python
state.metadata = {
    "router_classification": "logic",
    "premises_nl": [...],
    "premises_fol": [...],
    "query": "...",
    "kb_facts": 5,
    "kb_rules": 3,
    "fol_conversion": {
        "method": "llm" or "heuristic",
        "facts": [...],
        "rules": [...],
        "query": "..."
    },
    "z3_result": {
        "answer": "Yes",
        "atom": "honors(John)",
        "confidence": 0.95
    },
    "explanation": "Detailed logical explanation..."
}
```

---

## ✅ Checklist

- [x] Enhanced RouterNode with detailed keyword matching
- [x] Physics Subgraph (5 stages)
- [x] Logic Subgraph (4 stages)
- [x] ExactGraph orchestration
- [x] Unified WorkflowState
- [x] Metadata collection
- [x] LangSmith integration (inherited from previous)
- [ ] Unit tests for nodes
- [ ] Integration tests for subgraphs
- [ ] Performance benchmarks

---

## 📖 References

- **Subgraph Files**: `agents/subgraphs/`
- **Router Logic**: `agents/nodes/router_node.py`
- **State Definition**: `agents/models/state.py`
- **Main Graph**: `agents/graph.py`

---

**Version**: 2.1 (Advanced Routing & Subgraphs)  
**Last Updated**: 2026-05-27
