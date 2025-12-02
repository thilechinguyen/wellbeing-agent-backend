import os
import logging
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wellbeing-backend")

# Environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set")

GROQ_MODEL_ID = os.getenv("GROQ_MODEL_ID", "llama-3.3-70b-versatile")

groq_client = Groq(api_key=GROQ_API_KEY)


# ============================================================
# University of Adelaide Support Pack
# ============================================================
ADELAIDE_SUPPORT = """
University of Adelaide – Student Wellbeing Support

• Counselling Support (free for all students)
  https://www.adelaide.edu.au/counselling/

• After-hours Crisis Line (5pm–9am weekdays, 24/7 weekends & holidays)
  Phone: 1300 167 654
  Text: 0488 884 197

• Student Life Support
  https://www.adelaide.edu.au/student/wellbeing

• International Student Support
  https://international.adelaide.edu.au/student-support

• Emergency (Australia-wide)
  Call 000 for urgent life-threatening emergencies.
"""


# ============================================================
# Pydantic Models
# ============================================================
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    student_id: Optional[str] = None
    message: str
    history: List[ChatMessage] = []

class ChatResponse(BaseModel):
    reply: str


# ============================================================
# AGENT 1 – INSIGHT AGENT
# phân tích tin nhắn → risk level, emotions, topics
# ============================================================
def run_insight_agent(message: str) -> Dict[str, Any]:
    prompt = f"""
You are an Insight Extraction Agent for a wellbeing system.
Analyse the student's message and extract:

- emotion: one-word emotion (e.g., joy, sadness, worry, stress)
- risk_level: low / medium / high
- positive_event: true/false
- topics: list of short tags (e.g., exam, loneliness, homesickness)

Return ONLY valid JSON.

Message:
{message}
"""
    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL_ID,
            messages=[{"role": "system", "content": prompt}],
            temperature=0.2
        )
        import json
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        logger.warning("Insight agent failed:", e)
        return {
            "emotion": "neutral",
            "risk_level": "low",
            "positive_event": False,
            "topics": []
        }


# ============================================================
# AGENT 2 – PROFILE AGENT
# nhận insight + history → tạo profile summary
# ============================================================
def run_profile_agent(student_id: str, insights: Dict[str, Any]) -> str:
    prompt = f"""
You are the Profile Agent. Your job is to summarise the student's
current wellbeing state from the extracted insights.

Student ID: {student_id}

Insights:
{insights}

Create a SHORT 2–3 sentence summary describing:
- the student's emotional tone
- the key concerns or topics
- whether this is a positive-mood or negative-mood moment
"""
    completion = groq_client.chat.completions.create(
        model=GROQ_MODEL_ID,
        messages=[{"role": "system", "content": prompt}],
        temperature=0.3
    )
    return completion.choices[0].message.content


# ============================================================
# BASE SYSTEM PROMPT for Agent 3 (Wellbeing Response Agent)
# ============================================================
BASE_SYSTEM_PROMPT = f"""
You are a warm wellbeing companion for University of Adelaide students.
You use CBT, Positive Psychology and supportive counselling skills,
but you are *not* a therapist and never diagnose.

You ALWAYS stay culturally sensitive and respond in the student's language.

You also ALWAYS include, when appropriate, the following official support information:

{ADELAIDE_SUPPORT}

RULES FOR GOOD NEWS:
- celebrate warmly
- respond like a natural friend
- light, joyful tone
- avoid clinical analysis unless asked
- keep it short and human

RULES FOR STRESS/SADNESS:
- validate feelings
- be calm, soft, gentle
- offer small practical next steps
- provide counselling/crisis info ONLY if relevant

NEVER:
- give legal/financial/medical advice
- promise confidentiality
"""

# ============================================================
# AGENT 3 – WELLBEING RESPONSE AGENT
# ============================================================
def run_response_agent(req: ChatRequest, profile_summary: str, insights: Dict[str, Any]) -> str:

    # Nếu là positive event → thêm tone vui
    joy_boost = ""
    if insights.get("positive_event"):
        joy_boost = """
The student is sharing GOOD NEWS.
Respond in a happy, natural, uplifting tone.
Do NOT be clinical. Celebrate with them warmly.
"""

    messages = [
        {"role": "system", "content": BASE_SYSTEM_PROMPT + joy_boost},
        {"role": "system", "content": f"Profile Summary:\n{profile_summary}"},
    ]

    for m in req.history:
        messages.append({"role": m.role, "content": m.content})

    messages.append({"role": "user", "content": req.message})

    completion = groq_client.chat.completions.create(
        model=GROQ_MODEL_ID,
        messages=messages,
        temperature=0.6,
        max_tokens=1024
    )
    return completion.choices[0].message.content


# ============================================================
# FastAPI setup
# ============================================================
app = FastAPI(title="Wellbeing Agent – 3 Agent Pipeline")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):

    student_id = req.student_id or "anonymous"
    msg = req.message

    # 1) INSIGHT AGENT
    insights = run_insight_agent(msg)

    # 2) PROFILE AGENT
    profile = run_profile_agent(student_id, insights)

    # 3) RESPONSE AGENT (final answer)
    reply = run_response_agent(req, profile, insights)

    return ChatResponse(reply=reply)


@app.get("/health")
def health():
    return {"status": "ok", "model": GROQ_MODEL_ID}
