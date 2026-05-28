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
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=ls_YOUR_API_KEY_HERE
LANGSMITH_PROJECT=exact-2026  # Tên project của bạn

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
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "ls_..."
```

## Bước 5: Verify Configuration

Chạy script kiểm tra:

```bash
python -c "
import os
from dotenv import load_dotenv

load_dotenv()

print('LangSmith Config:')
print(f\"  API Key: {os.getenv('LANGSMITH_API_KEY', 'NOT SET')[:20]}...\")
print(f\"  Endpoint: {os.getenv('LANGSMITH_ENDPOINT', 'NOT SET')}\")
print(f\"  Project: {os.getenv('LANGSMITH_PROJECT', 'NOT SET')}\")
print(f\"  Tracing: {os.getenv('LANGSMITH_TRACING', 'NOT SET')}\")
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
   - **Input/Output** nghiệp vụ tối giản của từng node
   - **Timing** và latency
   - **Errors** nếu có
   - Span `llm.http_attempt` để xem stage, model, attempt, kích thước và thời gian request

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
Error: Invalid LANGSMITH_API_KEY
```
→ Copy key từ dashboard (bắt đầu bằng `ls_`)

### Không thể kết nối
```
Error: Failed to connect to Smith
```
→ Check internet connection, LANGSMITH_ENDPOINT có sẵn

### Traces không xuất hiện
```
No runs showing in dashboard
```
→ Kiểm tra LANGSMITH_PROJECT name khớp nhau
→ Kiểm tra LANGSMITH_TRACING=true

## Dữ Liệu Trace Được Giữ Lại

Workflow dùng `@langsmith.traceable` với `process_inputs` và
`process_outputs`. Trace không lưu prompt đầy đủ, routing confidence,
`raw_type`, `source`, hoặc metadata nội bộ. Với lời gọi LLM, trace lưu:

- stage xử lý;
- provider và model;
- JSON mode, attempt number;
- số ký tự request/response và duration.

Thông tin này đủ để phát hiện Parsing chậm hoặc retry OpenRouter mà không
đưa input/output dư vào LangSmith.

## Xóa Traces (nếu cần)

Vào LangSmith dashboard → Settings → Project Settings → Delete Project

---

**Tài liệu tham khảo:**
- https://docs.smith.langchain.com/
- https://github.com/langchain-ai/langgraph
