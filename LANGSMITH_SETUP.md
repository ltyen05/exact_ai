# LangSmith Setup Guide

## Tại sao LangSmith?

LangSmith cung cấp observability cho LangGraph workflows:
- 🔍 Trace toàn bộ execution path của agents
- 📊 Visualize graph flow
- 🐛 Debug issues một cách dễ dàng
- 📈 Monitor performance metrics
- 💾 Lưu trữ toàn bộ logs

## Bước 1: Tạo LangSmith Account

1. Truy cập: https://smith.langchain.com/
2. Sign up bằng Google, GitHub, hoặc email
3. Xác nhận email

## Bước 2: Tạo API Key

1. Đăng nhập vào LangSmith dashboard
2. Vào **Settings** → **API Keys**
3. Click **Create API Key**
4. Copy API key (dạng `ls_...`)

## Bước 3: Cấu Hình Environment

Tạo file `.env` ở root của project:

```bash
# LangSmith Configuration
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=ls_YOUR_API_KEY_HERE
LANGCHAIN_PROJECT=exact-2026  # Tên project của bạn

# LLM Configuration (vLLM)
EXACT_LLM_BASE_URL=http://localhost:8000/v1
EXACT_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
EXACT_LLM_API_KEY=EMPTY
```

**Lưu ý:** Không commit file `.env` vào git. Thêm vào `.gitignore`:

```bash
echo ".env" >> .gitignore
```

## Bước 4: Load Environment Variables

Cập nhật `main.py` hoặc file khởi động:

```python
from dotenv import load_dotenv
import os

# Load từ .env file
load_dotenv()

# Hoặc set trực tiếp
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "ls_..."
```

## Bước 5: Verify Configuration

Chạy script kiểm tra:

```bash
python -c "
import os
from dotenv import load_dotenv

load_dotenv()

print('LangSmith Config:')
print(f\"  API Key: {os.getenv('LANGCHAIN_API_KEY', 'NOT SET')[:20]}...\")
print(f\"  Endpoint: {os.getenv('LANGCHAIN_ENDPOINT', 'NOT SET')}\")
print(f\"  Project: {os.getenv('LANGCHAIN_PROJECT', 'NOT SET')}\")
print(f\"  Tracing: {os.getenv('LANGCHAIN_TRACING_V2', 'NOT SET')}\")
"
```

## Bước 6: Chạy Workflow với Tracing

```bash
# Run inference - tất cả traces sẽ gửi lên LangSmith
python main.py eval --max-records 10

# Hoặc API
python -m uvicorn api:app --reload
```

## Bước 7: Xem Traces trên LangSmith

1. Vào https://smith.langchain.com/
2. Chọn project: **exact-2026**
3. Xem các traces đã ghi lại:
   - **Input/Output** của từng node
   - **Timing** và latency
   - **Errors** nếu có
   - **Token counts**

## Graph Visualization trên LangSmith

LangSmith sẽ tự động visualize graph structure:

```
                  START
                    ↓
              [Router Node]
                    ↓
        ┌───────────┴───────────┐
        ↓                       ↓
    [Logic Node]          [Physics Node]
        ↓                       ↓
      END ←───────────────────→ END
```

## Troubleshooting

### API Key không hợp lệ
```
Error: Invalid LANGCHAIN_API_KEY
```
→ Copy key từ dashboard (bắt đầu bằng `ls_`)

### Không thể kết nối
```
Error: Failed to connect to Smith
```
→ Check internet connection, LANGCHAIN_ENDPOINT có sẵn

### Traces không xuất hiện
```
No runs showing in dashboard
```
→ Kiểm tra LANGCHAIN_PROJECT name khớp nhau
→ Kiểm tra LANGCHAIN_TRACING_V2=true

## Thêm Custom Metadata

Bạn có thể thêm custom metadata để debug tốt hơn:

```python
# Trong node function
from langsmith import get_run_tree

def logic_node(state):
    # ... logic code ...
    
    # Add metadata
    get_run_tree().metadata = {
        "kb_size": len(kb.facts),
        "confidence": state.agent_output.confidence,
        "model": "z3-solver",
    }
    
    return state
```

## Xóa Traces (nếu cần)

Vào LangSmith dashboard → Settings → Project Settings → Delete Project

---

**Tài liệu tham khảo:**
- https://docs.smith.langchain.com/
- https://github.com/langchain-ai/langgraph
