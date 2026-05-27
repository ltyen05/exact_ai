# LangGraph Architecture Visualization

## Graph Topology

```
START
  │
  ▼
┌──────────────────────────────────────────┐
│ [Router Node]                            │
│                                          │
│ Input: WorkflowState                     │
│  - question: str                         │
│  - original_payload: Dict                │
│                                          │
│ Logic: classify query type               │
│  ├─ Check explicit type field            │
│  ├─ Check for logic keywords             │
│  ├─ Check for physics hints              │
│  └─ Default: logic                       │
│                                          │
│ Output: state.query_type = "logic"/"physics"
└──────────────────────────────────────────┘
  │
  │ Edge: conditional_edges()
  │ Function: state.route_to_agent()
  │
  ├──────────────────────┬──────────────────────┐
  │                      │                      │
  │ Return: "logic_agent" or "physics_agent"   │
  │                      │                      │
  ▼                      ▼                      ▼

┌────────────────────────┐    ┌────────────────────────┐
│ [Logic Node]           │    │ [Physics Node]         │
│                        │    │                        │
│ Agents:                │    │ Agents:                │
│ - LogicNLParserAgent   │    │ - PhysicsAgent         │
│ - Z3ReasonerAgent      │    │                        │
│                        │    │ Tools:                 │
│ Input State:           │    │ - Calculator           │
│ - question             │    │ - Retriever            │
│ - premises_nl          │    │ - LLM (if enabled)     │
│ - premises_fol         │    │                        │
│                        │    │ Input State:           │
│ Process:               │    │ - question             │
│ 1. Parse premises      │    │                        │
│ 2. Build KB            │    │ Process:               │
│ 3. Query with Z3       │    │ 1. Try formula match   │
│ 4. Find best choice    │    │ 2. Try LLM             │
│                        │    │ 3. Retrieval fallback  │
│ Output State:          │    │                        │
│ - agent_output         │    │ Output State:          │
│   ├─ answer: str       │    │ - agent_output         │
│   ├─ reasoning: str    │    │   ├─ answer: str       │
│   ├─ confidence: float │    │   ├─ reasoning: str    │
│   └─ metadata: Dict    │    │   ├─ confidence: float │
│                        │    │   └─ metadata: Dict    │
└────────────────────────┘    └────────────────────────┘
  │                              │
  │ Edge: graph.add_edge()       │
  │ To: "__end__"               │
  │                              │
  └──────────────────┬───────────┘
                     │
                     ▼
                   END

     result_state = graph.invoke(initial_state)
     output_dict = result_state.to_dict()
```

## State Flow

```
WorkflowState Object Lifecycle
═══════════════════════════════

1. Initial State (created in predict())
   ┌─────────────────────────────────┐
   │ WorkflowState(                  │
   │   question="...",               │
   │   query_type=None,              │
   │   premises_nl=[...],            │
   │   agent_output=None,            │
   │   ...                           │
   │ )                               │
   └─────────────────────────────────┘
            │
            │ graph.invoke(state)
            ▼
   
2. After Router Node
   ┌─────────────────────────────────┐
   │ state.query_type = "logic"      │
   │ state.router_confidence = 0.9   │
   └─────────────────────────────────┘
            │
            │ route_to_agent() → "logic_agent"
            ▼
   
3. After Logic/Physics Node
   ┌─────────────────────────────────┐
   │ state.agent_output = AgentOutput│
   │   answer="Yes",                 │
   │   reasoning="Z3 entails...",    │
   │   confidence=0.85,              │
   │   metadata={...},               │
   │                                 │
   │ state.logic_output = same       │
   └─────────────────────────────────┘
            │
            │ convert to dict
            ▼
   
4. Final Output
   ┌─────────────────────────────────┐
   │ {                               │
   │   "type": "logic",              │
   │   "answer": "Yes",              │
   │   "reasoning": "...",           │
   │   "confidence": 0.85,           │
   │   "metadata": {...}             │
   │ }                               │
   └─────────────────────────────────┘
```

## Node Function Signature

```python
def node_function(state: WorkflowState) -> WorkflowState:
    """
    Process state and return updated state.
    
    LangGraph Pattern:
    - Input: WorkflowState
    - Process: Modify fields
    - Output: Updated WorkflowState (return modified)
    - Side Effects: None (functional)
    
    Example (Logic Node):
    
    def logic_node(state: WorkflowState) -> WorkflowState:
        # 1. Extract relevant fields
        question = state.question
        premises = state.premises_nl
        
        # 2. Process
        kb = build_knowledge_base(premises)
        answer = query_with_z3(kb, question)
        
        # 3. Create output
        output = AgentOutput(
            answer=answer,
            reasoning="...",
            confidence=0.85,
            agent_name="LogicAgent",
        )
        
        # 4. Update state
        state.agent_output = output
        state.logic_output = output
        
        # 5. Return
        return state
    """
    pass
```

## Edge Routing

```python
# Conditional Edge
graph.add_conditional_edges(
    "router",                           # Source node
    lambda state: state.route_to_agent(),  # Edge function
    {                                   # Mapping: return value → next node
        "logic_agent": "logic",         # If returns "logic_agent", go to "logic" node
        "physics_agent": "physics",     # If returns "physics_agent", go to "physics" node
    }
)

# Normal Edge (always go to __end__)
graph.add_edge("logic", "__end__")
graph.add_edge("physics", "__end__")
```

## Data Classes

```
AgentInput (for reference)
────────────────────────
question: str
query_type: str ("logic" or "physics")
premises_nl: List[str]
premises_fol: Optional[List[str]]
payload: Dict[str, Any]
id: Optional[str]

AgentOutput (standardized agent response)
──────────────────────────────────────────
answer: str
reasoning: str
confidence: float (0.0 - 1.0)
metadata: Dict[str, Any]
agent_name: str

WorkflowState (graph state)
──────────────────────────
original_payload: Dict[str, Any]
question: str
query_type: Optional[str]
premises_nl: List[str]
premises_fol: Optional[List[str]]
record_id: Optional[str]

router_confidence: float
parsed_kb: Optional[Dict[str, Any]]

agent_output: Optional[AgentOutput]
logic_output: Optional[AgentOutput]
physics_output: Optional[AgentOutput]

errors: List[str]
metadata: Dict[str, Any]
```

## Execution Trace Example

```
Query Input
├─ question: "Does John graduate with honors?"
├─ type: "logic"
├─ premises-NL: ["If GPA > 3.5, graduate with honors.", "John has GPA 3.8."]
└─ id: "edu-001"
    │
    ▼ invoke(state)
    
Router Node
├─ detect: query_type = "logic"
├─ confidence: 0.95
└─ return: state (updated)
    │
    ▼ conditional_edges()
    
Route: "logic_agent"
├─ next node: "logic"
└─ continue
    │
    ▼ Logic Node
    
Logic Node Execution
├─ Build KB from premises
│  ├─ Fact: graduate_with_honors(John)
│  └─ Rule: high_gpa → graduate_with_honors
├─ Query Z3: entails(graduate_with_honors(John))?
├─ Result: True
├─ Confidence: 0.90
└─ Create AgentOutput
    │
    ▼ return: state (with agent_output)
    
Normal Edge: "logic" → "__end__"
├─ No more nodes
└─ graph returns final state
    │
    ▼ to_dict() conversion
    
Final Output
{
  "type": "logic",
  "question": "Does John graduate with honors?",
  "answer": "Yes",
  "reasoning": "Logic reasoning using formal KB and Z3",
  "confidence": 0.90,
  "metadata": {
    "atom": "graduate_with_honors(John)",
    "cot": [...],
    "kb_facts": 2,
    "kb_rules": 1
  }
}
```

---

**Graph Version**: 1.0 (LangGraph)  
**Created**: 2026-05-27
