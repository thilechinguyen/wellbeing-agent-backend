import os
import logging
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wellbeing-backend")

# ============================================================
# ENV
# ============================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set")

MODEL_ID = os.getenv("GROQ_MODEL_ID", "llama-3.3-70b-versatile")
groq_client = Groq(api_key=GROQ_API_KEY)

# ============================================================
# University of Adelaide Support (Will be conditionally injected)
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
  Call 000 for urgent emergencies.
"""

# ============================================================
# MODELS
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
# AGENT 1 — INSIGHT AGENT
# ============================================================
def run_insight_agent(message: str) -> Dict[str, Any]:
    prompt = f"""
You are the Insight Extraction Agent.
Analyse the student's message and extract:

- emotion: (one word)
- risk_level: low / medium / high
- positive_event: true/false
- topics: list of tags

Return ONLY valid JSON.

Message:
{message}
"""
    try:
        completion = groq_client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "system", "content": prompt}],
            temperature=0.2,
        )
        import json
        return json.loads(completion.choices[0].message.content)

    except Exception as e:
        logger.warning("Insight agent failed:", e)
        return {
            "emotion": "neutral",
            "risk_level": "low",
            "positive_event": False,
            "topics": [],
        }


# ============================================================
# AGENT 2 — PROFILE AGENT
# ============================================================
def run_profile_agent(student_id: str, insights: Dict[str, Any]) -> str:
    prompt = f"""
You are the Profile Agent.
Produce a SHORT 2–3 sentence summary describing:

- the student's emotional state
- key concerns/topics
- whether this moment is positive or negative

Insight data:
{insights}
"""
    completion = groq_client.chat.completions.create(
        model=MODEL_ID,
        messages=[{"role": "system", "content": prompt}],
        temperature=0.3,
    )
    return completion.choices[0].message.content


# ============================================================
# SYSTEM PROMPT — STRICT LANGUAGE + STRICT SUPPORT CONDITIONS
# ============================================================
BASE_SYSTEM_PROMPT = f"""
You are a warm wellbeing companion for University of Adelaide students.
Use CBT, Positive Psychology, supportive listening — but you are NOT a therapist.

LANGUAGE RULES (STRICT):
- If the student writes primarily in Vietnamese → ALWAYS reply 100% in Vietnamese.
- If the student writes primarily in English → ALWAYS reply 100% in English.
- NEVER reply in Chinese/Japanese/Korean unless the ENTIRE user message is in that language.
- If the user includes 1–2 Chinese characters in a Vietnamese/English sentence
  (e.g., “để庆祝”), treat them as normal text and DO NOT explain them unless the user explicitly asks:
  (“dịch chữ này”, “giải thích chữ này”, “what does this mean”).
- NEVER spontaneously switch to Chinese or give bilingual explanations.

SUPPORT RULES (STRICT):
- Only provide University of Adelaide wellbeing/counselling support information when:
  (1) The insight risk_level is medium or high, OR
  (2) The user expresses sadness, stress, anxiety, loneliness, overwhelm.
- DO NOT provide support info for neutral, positive, playful, or definition-type questions.

GOOD NEWS RULES:
- If user shares positive news → respond joyfully, naturally, friendly.
- Keep it light and human, no clinical tone.

NEGATIVE EMOTION RULES:
- Validate feelings softly.
- Offer small practical steps.
- Only then add support info (if support conditions are satisfied).

NEVER:
- Give medical/legal/financial advice.
- Promise confidentiality.
"""


# ============================================================
# AGENT 3 — RESPONSE AGENT
# ============================================================
def run_response_agent(req: ChatRequest, profile_summary: str, insights: Dict[str, Any]) -> str:

    # JOY MODE (if positive event)
    joy_boost = ""
    if insights.get("positive_event"):
        joy_boost = """
USER IS SHARING GOOD NEWS.
Respond with joyful, warm, natural tone. No clinical analysis.
"""

    # Determine if we should inject support info
    msg_low = req.message.lower()
    add_support = False

    if insights.get("risk_level") in ["medium", "high"]:
        add_support = True

    emotional_keywords = [
        "stress", "lo lắng", "buồn", "sad", "tired",
        "khó chịu", "overwhelmed", "burnout", "anxious"
    ]
    if any(kw in msg_low for kw in emotional_keywords):
        add_support = True

    support_block = ADELAIDE_SUPPORT if add_support else ""

    # Build messages
    messages = [
        {"role": "system", "content": BASE_SYSTEM_PROMPT + joy_boost},
        {"role": "system", "content": f"Profile Summary:\n{profile_summary}"},
        {"role": "system", "content": support_block},
    ]

    for m in req.history:
        messages.append({"role": m.role, "content": m.content})

    messages.append({"role": "user", "content": req.message})

    completion = groq_client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        temperature=0.6,
        max_tokens=1024,
    )
    return completion.choices[0].message.content


# ============================================================
# FASTAPI APP
# ============================================================
app = FastAPI(title="Wellbeing Agent — 3 Agent Pipeline (Strict Fix)")

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

    # 1) INSIGHT
    insights = run_insight_agent(req.message)

    # 2) PROFILE
    profile = run_profile_agent(student_id, insights)

    # 3) RESPONSE
    reply = run_response_agent(req, profile, insights)

    return ChatResponse(reply=reply)


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID}
