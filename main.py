import os
import io
import csv
from datetime import datetime
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from groq import Groq


# =========================
# 1. ENV & GROQ CLIENT
# =========================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set in environment or .env file")

# Groq client
groq_client = Groq(api_key=GROQ_API_KEY)

# Model Groq đang hoạt động (tránh model đã bị decommission)
GROQ_MODEL = os.getenv("GROQ_MODEL", "deepseek-r1-distill-qwen-32b")


# =========================
# 2. FASTAPI APP & CORS
# =========================

app = FastAPI(
    title="Wellbeing Companion Backend",
    description="Multi-agent style CBT wellbeing chatbot for first-year students.",
    version="0.7",
)

# CORS
allow_origins_env = os.getenv("ALLOW_ORIGINS", "")
if allow_origins_env.strip():
    ALLOW_ORIGINS = [o.strip() for o in allow_origins_env.split(",") if o.strip()]
else:
    ALLOW_ORIGINS = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# 3. DATA MODELS
# =========================

class ChatRequest(BaseModel):
    user_id: Optional[str] = None
    message: str

    # metadata từ UI
    student_type: Optional[str] = None  # e.g. "Domestic (Australia)", "International – SE Asia"
    student_region: Optional[str] = None  # e.g. "Australia", "Vietnam"
    language: Optional[str] = "vi"  # "vi" hoặc "en"


class ChatResponse(BaseModel):
    reply: str
    model: str
    meta: Dict[str, Any]


class MessageLog(BaseModel):
    timestamp: str
    user_id: Optional[str]
    student_type: Optional[str]
    student_region: Optional[str]
    language: Optional[str]
    user_message: str
    assistant_reply: str
    model: str


# Bộ nhớ tạm để log hội thoại (mất khi restart, nhưng đủ để demo /export/messages)
MESSAGE_LOGS: List[MessageLog] = []


# =========================
# 4. SYSTEM PROMPT (multi-agent style)
# =========================

BASE_SYSTEM_PROMPT = """
You are a CBT-informed wellbeing companion for first-year university students.

Your job:
- Listen with empathy.
- Reflect feelings and normalise the experience.
- Offer gentle, practical CBT-style strategies (thought reframing, problem solving, behavioural activation).
- Always stay within a **supportive, non-clinical** role (you are NOT a therapist).
- Encourage help-seeking: friends, family, student support, counsellors, emergency services when needed.
- Be culturally sensitive to both Vietnamese and international students studying in Australia.

Structure your reply in 3 short parts:
1) Empathic reflection (1–2 câu ngắn, có thể dùng song ngữ Anh–Việt nếu phù hợp).
2) Gợi ý cụ thể 2–3 bước nhỏ mà sinh viên có thể thử ngay (có thể gợi ý kiểu CBT).
3) Nhắc nhẹ về nguồn hỗ trợ (bạn bè, gia đình, dịch vụ hỗ trợ sinh viên, counsellor, emergency nếu nguy hiểm).

Safety rules:
- Nếu người dùng nhắc tới tự hại bản thân, tự tử, bạo lực, lạm dụng hoặc nguy cơ an toàn nghiêm trọng,
  hãy:
  (a) Phản hồi rất nghiêm túc và đồng cảm.
  (b) Khuyến khích họ liên hệ ngay với dịch vụ khẩn cấp tại quốc gia của họ (ví dụ: 000 ở Úc) hoặc tới bệnh viện gần nhất.
  (c) Khuyến khích liên hệ dịch vụ counselling / đường dây nóng tại trường hoặc quốc gia của họ.
  (d) Không đưa hướng dẫn cụ thể gây hại.

Always use a warm, clear and simple tone. 
If the student writes mostly in Vietnamese, ưu tiên trả lời bằng tiếng Việt (kèm 1–2 câu tiếng Anh đơn giản nếu phù hợp).
If they write in English, trả lời chủ yếu bằng tiếng Anh, có thể xen 1–2 câu tiếng Việt giản dị để tạo cảm giác gần gũi.
"""


def build_system_prompt(req: ChatRequest) -> str:
    profile_info = f"Student type: {req.student_type or 'Unknown'}; " \
                   f"Region: {req.student_region or 'Unknown'}; " \
                   f"Language preference: {req.language or 'vi'}."

    return BASE_SYSTEM_PROMPT + "\n\nExtra context about this student:\n" + profile_info


# =========================
# 5. GROQ CALL WRAPPER
# =========================

def call_groq(req: ChatRequest) -> str:
    """
    Gọi Groq để sinh câu trả lời. Nếu lỗi thì raise HTTPException 500.
    """
    system_prompt = build_system_prompt(req)

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": req.message.strip(),
        },
    ]

    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
        )
        content = completion.choices[0].message.content
        if not content:
            raise ValueError("Empty response from Groq")
        return content
    except Exception as e:
        # Đưa error vào log server, và trả HTTP 500 cho frontend
        print(f"[ERROR] Groq completion failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Wellbeing agent is temporarily unavailable. Please try again later."
        )


# =========================
# 6. ENDPOINTS
# =========================

@app.get("/health")
async def health():
    return {"status": "ok", "model": GROQ_MODEL}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Endpoint chính cho UI.
    """
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="Message must not be empty.")

    reply = call_groq(req)

    # Log lại hội thoại
    log_entry = MessageLog(
        timestamp=datetime.utcnow().isoformat(),
        user_id=req.user_id,
        student_type=req.student_type,
        student_region=req.student_region,
        language=req.language,
        user_message=req.message,
        assistant_reply=reply,
        model=GROQ_MODEL,
    )
    MESSAGE_LOGS.append(log_entry)

    return ChatResponse(
        reply=reply,
        model=GROQ_MODEL,
        meta={
            "student_type": req.student_type,
            "student_region": req.student_region,
            "language": req.language,
        },
    )


@app.get("/export/messages")
async def export_messages():
    """
    Xuất toàn bộ MESSAGE_LOGS thành file CSV.
    UI đang gọi /export/messages nên giữ endpoint này.
    """

    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow([
        "timestamp",
        "user_id",
        "student_type",
        "student_region",
        "language",
        "user_message",
        "assistant_reply",
        "model",
    ])

    for log in MESSAGE_LOGS:
        writer.writerow([
            log.timestamp,
            log.user_id or "",
            log.student_type or "",
            log.student_region or "",
            log.language or "",
            log.user_message.replace("\n", " ").strip(),
            log.assistant_reply.replace("\n", " ").strip(),
            log.model,
        ])

    output.seek(0)
    headers = {
        "Content-Disposition": 'attachment; filename="wellbeing_messages.csv"'
    }
    return StreamingResponse(output, media_type="text/csv", headers=headers)


# =========================
# 7. LOCAL DEV ENTRYPOINT
# =========================

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "10000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
