# EXACT 2026 Multi-Agent QA Pipeline

Pipeline này bám theo hướng neurosymbolic / multi-agent:

1. **RouterAgent**: nhận một unified payload, phân loại `logic` hay `physics`.
2. **Logic pipeline**:
   - `LogicNLParserAgent`: chuyển premises/question thành Horn-rule JSON bằng local LLM qua vLLM nếu có; nếu không có LLM thì dùng `premises-FOL` hoặc regex fallback.
   - `Z3ReasonerAgent`: chạy forward chaining + Z3 entailment check.
   - `ExplanationAgent`: sinh explanation ngắn, có tool trace / premise evidence.
3. **Physics pipeline**:
   - `Formula calculator`: bắt các mẫu phổ biến như resultant force, capacitor energy/charge, parallel-plate capacitor, RLC resonance.
   - `Local LLM solver`: gọi model open-source <=8B qua vLLM nếu cấu hình.
   - `Retriever fallback`: lấy bài gần nhất từ training set khi không có LLM hoặc công thức chưa match.
4. **Formatter**: ép output về JSON gồm `answer`, `explanation`, và các trường khuyến khích: `fol`, `cot`, `premises`, `confidence`.
5. **P1 evaluator**: tính correctness cho Logic và Physics; Physics có numeric tolerance + unit matching.

## Cài đặt

```bash
cd exact_2026_agent
pip install -r requirements.txt
```

## Chạy nhanh không cần LLM

```bash
python main.py demo
python main.py eval --max-records 20 --details eval_details.json
```

## Chạy inference file JSON/JSONL

Input một record physics:

```json
{"type":"physics", "question":"Two electric forces, each with a magnitude of 5 N, act at an angle of 60° to each other. What is the resultant force?"}
```

Input một record logic:

```json
{
  "type":"logic",
  "premises-NL":["If a student completes all required courses, they are eligible for graduation.", "John has completed all required courses."],
  "question":"Is John eligible for graduation?"
}
```

Chạy:

```bash
python main.py infer --input input.jsonl --output predictions.jsonl
```

## Kết nối local open-source LLM qua vLLM

Theo luật cuộc thi, không gọi closed-source model ở inference. Chạy một model <=8B class bằng vLLM, ví dụ:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 --port 8000
```

Sau đó cấu hình:

```bash
export EXACT_LLM_BASE_URL=http://localhost:8000/v1
export EXACT_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
export EXACT_LLM_API_KEY=EMPTY
python main.py eval --max-records 20
```

## Output schema

```json
{
  "answer": "B",
  "explanation": "...",
  "fol": "optional symbolic evidence",
  "cot": ["tool-visible step 1", "tool-visible step 2"],
  "premises": ["formula or retrieved evidence"],
  "confidence": 0.82,
  "type": "logic"
}
```

## Ghi chú quan trọng

- File physics trong project là bản đã lọc QA-prefix, còn 1.354 bài.
- Khi đánh giá local Logic có thể bật `premises-FOL` từ training data để kiểm tra symbolic engine. Khi test thật, committee chỉ gửi premises-NL, nên nên bật local LLM parser qua vLLM.
- Không nên dùng API ChatGPT/Claude/Gemini ở inference vì trái luật cuộc thi.
- Code này ưu tiên cấu trúc pipeline, traceability và P1 evaluator; để tăng điểm P1 thật, nên fine-tune hoặc few-shot một model <=8B cho hai tác vụ: NL->logic JSON và physics formula selection.
