import os
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv

# ---------------------------------------------------------
# 1. Load env & init Groq
# ---------------------------------------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set in environment or .env file")

groq_client = Groq(api_key=GROQ_API_KEY)

# ---------------------------------------------------------
# 2. FastAPI app + CORS
# ---------------------------------------------------------
app = FastAPI(title="Wellbeing Agent – Groq backend v0.7")

allow_origins = os.getenv("ALLOW_ORIGINS", "*")
origins = [o.strip() for o in allow_origins.split(",")] if allow_origins else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# 3. Pydantic models
# ---------------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    language: Optional[str] = "vi"
    user_id: Optional[str] = None
    student_type: Optional[str] = None
    student_region: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    language: str
    debug: Dict[str, Any]


# ---------------------------------------------------------
# 4. System prompts cho từng agent
# ---------------------------------------------------------
EMOTION_SYSTEM_PROMPT = """
You are an emotion detection assistant for first-year university students.
Your job is to label the student's main emotional tone and give a one-sentence
explanation. Answer in JSON with keys: emotion_label, explanation.
"""

COACH_SYSTEM_PROMPT = """
You are a gentle CBT-informed wellbeing coach for first-year university students.
Respond in a short, warm, practical way. Use simple language, max ~180 words.
If the input language is Vietnamese, reply in natural Vietnamese.
If English, reply in natural English.
You can suggest 2–3 small next steps and one reflective question.
Avoid clinical language and diagnosis.
"""

SAFETY_SYSTEM_PROMPT = """
You are a safety checker. You ONLY decide if the message needs escalation.
Return JSON with keys:
- needs_urgent_help: true/false
- reason: short text
Flag true if there is self-harm, suicide, serious harm to others, or abuse.
Otherwise false.
"""

AGGREGATOR_SYSTEM_PROMPT = """
You are the final wellbeing companion talking directly to the student.
You receive:
- the student's original message
- an emotion analysis
- a draft coaching reply
- a safety flag

Your task:
1. Keep the supportive tone of the coaching reply.
2. Briefly acknowledge the detected emotion.
3. If safety.needs_urgent_help is true, add a short, clear note encouraging
   the student to reach out to professional or emergency services, but do NOT
   panic or sound alarmist.
4. Answer in the same LANGUAGE as the student's message (Vietnamese or English).

Return ONLY the final message text, no JSON.
"""

# ---------------------------------------------------------
# 5. Helper: gọi Groq model
# ---------------------------------------------------------
def groq_chat(system_prompt: str, user_prompt: str, model: str = "deepseek-r1-distill-llama-70b") -> str:
    completion = groq_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.6,
        max_tokens=800,
    )
    return completion.choices[0].message.content.strip()


# ---------------------------------------------------------
# 6. Agents
# ---------------------------------------------------------
def run_emotion_agent(message: str) -> Dict[str, Any]:
    raw = groq_chat(EMOTION_SYSTEM_PROMPT, message)
    # cố gắng parse JSON nhưng nếu lỗi thì trả thẳng text
    import json

    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {"emotion_label": "unknown", "explanation": raw}


def run_coach_agent(message: str, language: str) -> str:
    user_prompt = f"Language: {language}\nStudent message: {message}"
    return groq_chat(COACH_SYSTEM_PROMPT, user_prompt)


def run_safety_agent(message: str) -> Dict[str, Any]:
    import json

    raw = groq_chat(SAFETY_SYSTEM_PROMPT, message)
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {"needs_urgent_help": False, "reason": raw}


def run_aggregator(
    original_message: str,
    language: str,
    emotion: Dict[str, Any],
    coach_reply: str,
    safety: Dict[str, Any],
) -> str:
    summary_prompt = f"""
Student language: {language}
Original message:
{original_message}

Emotion analysis (JSON):
{emotion}

Draft coaching reply:
{coach_reply}

Safety check result (JSON):
{safety}
"""
    return groq_chat(AGGREGATOR_SYSTEM_PROMPT, summary_prompt)


# ---------------------------------------------------------
# 7. Routes
# ---------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "model": "groq-deepseek-v0.7"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        message = req.message.strip()
        if not message:
            raise HTTPException(status_code=400, detail="Empty message")

        language = (req.language or "vi").lower()

        # 1) Emotion
        emotion = run_emotion_agent(message)

        # 2) Coaching
        coach_reply = run_coach_agent(message, language)

        # 3) Safety
        safety = run_safety_agent(message)

        # 4) Aggregator
        final_reply = run_aggregator(
            original_message=message,
            language=language,
            emotion=emotion,
            coach_reply=coach_reply,
            safety=safety,
        )

        return ChatResponse(
            reply=final_reply,
            language=language,
            debug={
                "emotion": emotion,
                "coach": coach_reply,
                "safety": safety,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        # log cho Render
        print("Error in /chat:", repr(e))
        raise HTTPException(status_code=500, detail="Internal error in wellbeing agent")
