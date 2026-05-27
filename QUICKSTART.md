# EXACT 2026 - Quick Start Guide

## 🚀 5-Minute Setup

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Create Environment Config (Optional)
```bash
cp .env.example .env
# Edit .env with your LangSmith API key (optional)
```

### 3️⃣ Run Demo
```bash
python main.py demo
```

Expected output:
```json
{
  "type": "physics",
  "answer": "...",
  "reasoning": "...",
  "confidence": 0.75
}
{
  "type": "logic",
  "answer": "Yes",
  "reasoning": "...",
  "confidence": 0.90
}
```

### 4️⃣ Run API Server
```bash
python -m uvicorn api:app --reload
```

Then test:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"type":"logic","question":"Is John eligible?","premises-NL":["If student has GPA > 3.5, eligible.","John has GPA 3.8."]}'
```

### 5️⃣ Run Full Evaluation
```bash
python main.py eval --max-records 100 --details results.json
```

---

## 📊 New Architecture Overview

### Graph Flow
```
Input Query
    ↓
[Router] Classifies logic/physics
    ↓
    ├─ [Logic Agent] → Z3 Solver
    │
    └─ [Physics Agent] → Formula/LLM/Retrieval
    ↓
Output (Standardized Format)
```

### Key Components

| Component | Purpose |
|-----------|---------|
| `WorkflowState` | Central state object |
| `AgentOutput` | Standardized output format |
| `ExactGraph` | LangGraph workflow |
| `LLMClientBase` | Abstract LLM interface |
| Nodes | Router, Logic, Physics agents |

---

## 🔧 Configuration

### Environment Variables (`.env`)

```bash
# LangSmith (optional)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls_your_key_here
LANGCHAIN_PROJECT=exact-2026

# LLM Server (vLLM)
EXACT_LLM_BASE_URL=http://localhost:8000/v1
EXACT_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
EXACT_LLM_API_KEY=EMPTY
```

---

## 📂 Project Structure

```
exact2026/
├── agents/
│   ├── llm/                # Abstract LLM + implementations
│   ├── models/             # State dataclasses
│   ├── nodes/              # Graph nodes
│   ├── graph.py            # LangGraph workflow
│   ├── logic_agent.py      # Legacy (used by nodes)
│   └── physics_agent.py    # Legacy (used by nodes)
├── data/                   # Training datasets
├── eval/                   # Evaluation scripts
├── tools/                  # Z3, calculator
├── main.py                 # CLI interface
├── api.py                  # FastAPI server
├── REFACTORING.md          # Detailed refactoring guide
├── LANGSMITH_SETUP.md      # LangSmith instructions
└── requirements.txt        # Dependencies
```

---

## 🎯 Common Tasks

### Run Inference on Custom Data
```bash
python main.py infer \
  --input my_queries.jsonl \
  --output results.jsonl
```

### Evaluate on Full Dataset
```bash
python main.py eval \
  --logic data/Logic_Based_Educational_Queries.json \
  --physics data/Physics_Problems_Text_Only_removeQA.json \
  --max-records 2000 \
  --details eval_results.json
```

### Debug with LangSmith
1. Set `LANGCHAIN_API_KEY` in `.env`
2. Run any command
3. Visit https://smith.langchain.com/ → "exact-2026" project
4. Click on a run to see detailed traces

---

## 🤔 Switching LLM Providers

### From vLLM to Anthropic Claude

```python
# agents/llm/anthropic_client.py

from anthropic import Anthropic
from agents.llm.base import LLMClientBase

class AnthropicClient(LLMClientBase):
    def __init__(self, api_key: str):
        self.client = Anthropic(api_key=api_key)
    
    @property
    def enabled(self) -> bool:
        return bool(self.client)
    
    def chat(self, messages, temperature=0.0, max_tokens=1024, **kwargs):
        response = self.client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
        )
        return response.content[0].text
```

Then use it:
```python
from agents.llm.anthropic_client import AnthropicClient

llm = AnthropicClient(api_key="sk-ant-...")
graph = ExactGraph(llm=llm)
```

---

## 📖 Learn More

- **Detailed Architecture**: See [REFACTORING.md](./REFACTORING.md)
- **LangSmith Guide**: See [LANGSMITH_SETUP.md](./LANGSMITH_SETUP.md)
- **LangGraph Docs**: https://python.langchain.com/docs/langgraph

---

## ✅ Troubleshooting

### "ModuleNotFoundError: No module named 'langgraph'"
```bash
pip install langgraph langchain --upgrade
```

### "LLM not configured" error
Set env vars:
```bash
export EXACT_LLM_BASE_URL=http://localhost:8000/v1
export EXACT_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
```

### Traces not appearing on LangSmith
Check:
```bash
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(f\"Key: {os.getenv('LANGCHAIN_API_KEY', 'NOT SET')[:20]}...\")"
```

---

**Version**: 2.0 (LangGraph-based)  
**Last Updated**: 2026-05-27
