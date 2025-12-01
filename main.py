import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq

# =========================================================
#  Cấu hình Groq client + model
# =========================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set in environment or .env file")

# Model mặc định: llama-3.1-8b-instant (free, nhanh)
GROQ_MODEL_ID = os.getenv("GROQ_MODEL_ID", "llama-3.1-8b-instant")

groq_client = Groq(api_key=GROQ_API_KEY)


# =========================================================
#  System prompts cho 3 agents
# =========================================================

BASE_CONTEXT = """
You are part of a CBT-based wellbeing multi-agent system that supports first-year university students.
The student may be an international or domestic student, feeling lonely, stressed, or worried about study.

General rules for ALL agents:
- Be warm, validating, non-judgemental.
- Write in clear, simple language.
- If the user message is in Vietnamese, answer in Vietnamese.
- If the user message is in English, answer in English.
- Do NOT claim to be a therapist, doctor, or crisis service.
- If there is any hint of self-harm or serious risk, gently encourage the student to reach out to:
  - trusted people in their life, AND
  - professional or emergency services in their country.
"""

EMOTION_SYSTEM_PROMPT = (
    BASE_CONTEXT
    + """
You are the EMOTION agent.

Your job:
- Identify the student's key emotions and needs.
- Reflect their feelings back with high empathy.
- Name emotions explicitly (e.g. "lonely", "overwhelmed", "anxious", "homesick").
- Keep it short: 3–5 sentences.

Output format:
- A short paragraph (NO bullet list).
- No advice, no action plan yet. Just understanding and validation.
"""
)

COACH_SYSTEM_PROMPT = (
    BASE_CONTEXT
    + """
You are the COACH agent.

Your job:
- Offer practical, CBT-informed support for the student's situation.
- Use:
  - gentle cognitive reframing,
  - simple behavioural suggestions (small steps),
  - self-compassion and growth mindset.
- Tailor your response to first-year university students.
- Avoid medical language or diagnosing.

Output format:
- 2–4 short paragraphs.
- Very concrete and encouraging.
- At the end, add 2–3 reflective questions the student could think about or journal about.
"""
)

SAFETY_SYSTEM_PROMPT = (
    BASE_CONTEXT
    + """
You are the SAFETY agent.

Your job:
- Scan the student's message for any risk of self-harm, suicide, or serious harm.
- Be very sensitive but calm.
- If there is ANY possible risk, you must:
  - advise the student to contact local emergency services or crisis hotlines,
  - encourage them to reach out to trusted people (friends, family, university support, counsellors),
  - clearly state that online tools cannot handle emergencies.

Output format:
- 3–6 sentences.
- Always include at least one sentence reminding the student that if they are in immediate danger,
  they should contact emergency services in their country right away.
"""
)


# =========================================================
#  FastAPI app + CORS
# =========================================================

app = FastAPI(title="Wellbeing Agent – Groq Multi-Agent Backend")

allow_origins_env = os.getenv("ALLOW_ORIGINS", "*")
if allow_origins_env.strip() == "*":
    origins = ["*"]
else:
    origins = [origin.strip() for origin in allow_origins_env.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
#  Pydantic models
# =========================================================

class ChatRequest(BaseModel):
    message: str
    # metadata gửi từ frontend – để optional cho chắc
    student_type: Optional[str] = None
    student_region: Optional[str] = None
    lang: Optional[str] = None
    user_id: Optional[str] = None


class ChatResponse(BaseModel):
    # final message cho UI (UI đang dùng trường này)
    reply: str

    # thêm chi tiết 3 agents (UI có thể dùng sau này)
    emotion: str
    coach: str
    safety: str


# =========================================================
#  Hàm gọi Groq
# =========================================================

def call_groq_agent(system_prompt: str, user_message: str) -> str:
    """
    Gọi 1 agent Groq với system prompt riêng.
    Trả về content của assistant (string).
    """
    completion = groq_client.chat.completions.create(
        model=GROQ_MODEL_ID,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.7,
        max_tokens=800,
    )

    try:
        return completion.choices[0].message.content.strip()
    except Exception as exc:  # phòng lỗi bất ngờ trong cấu trúc response
        raise RuntimeError(f"Groq response parsing error: {exc}") from exc


# =========================================================
#  Endpoints
# =========================================================

@app.get("/health")
def health_check():
    return {"status": "ok", "model": GROQ_MODEL_ID}


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(payload: ChatRequest):
    if not payload.message or not payload.message.strip():
        raise HTTPException(status_code=400, detail="Message is required")

    user_text = payload.message.strip()

    # Thêm chút meta profile cho agents (nếu có)
    profile_lines = []
    if payload.student_type:
        profile_lines.append(f"Student type: {payload.student_type}")
    if payload.student_region:
        profile_lines.append(f"Region: {payload.student_region}")
    if payload.lang:
        profile_lines.append(f"Preferred language code (if provided): {payload.lang}")
    if payload.user_id:
        profile_lines.append(f"User id (for context only, not for identification): {payload.user_id}")

    profile_block = ""
    if profile_lines:
        profile_block = "\n\n[Profile]\n" + "\n".join(profile_lines)

    full_user_message = user_text + profile_block

    try:
        # 3 agents chạy lần lượt (đơn giản, dễ debug; nếu muốn sau có thể chuyển sang async/gather)
        emotion_text = call_groq_agent(EMOTION_SYSTEM_PROMPT, full_user_message)
        coach_text = call_groq_agent(COACH_SYSTEM_PROMPT, full_user_message)
        safety_text = call_groq_agent(SAFETY_SYSTEM_PROMPT, full_user_message)

    except Exception as exc:
        # Log nội bộ trên Render (stdout), UI sẽ nhận 500
        raise HTTPException(
            status_code=500,
            detail=f"Error while contacting Groq: {exc}",
        )

    # Final reply cho UI – gom lại 3 agent, nhưng vẫn dễ đọc cho sinh viên
    # UI hiện chỉ dùng `reply`, còn 3 trường kia để debug / future UI.
    if payload.lang and payload.lang.lower().startswith("vi"):
        final_reply = (
            f"**1. Cảm xúc của bạn**\n{emotion_text}\n\n"
            f"**2. Gợi ý hỗ trợ**\n{coach_text}\n\n"
            f"**3. An toàn & hỗ trợ khẩn cấp**\n{safety_text}"
        )
    else:
        final_reply = (
            f"**1. How you might be feeling**\n{emotion_text}\n\n"
            f"**2. Support & next steps**\n{coach_text}\n\n"
            f"**3. Safety & urgent support**\n{safety_text}"
        )

    return ChatResponse(
        reply=final_reply,
        emotion=emotion_text,
        coach=coach_text,
        safety=safety_text,
    )


# =========================================================
#  Local dev entrypoint (không dùng trên Render nhưng tiện test)
# =========================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "10000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
