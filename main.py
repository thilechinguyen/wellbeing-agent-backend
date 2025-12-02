import os
import logging
from typing import List, Optional, Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq

# ============================================================
# Logging & ENV
# ============================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wellbeing-backend-7agents")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set")

MODEL_ID = os.getenv("GROQ_MODEL_ID", "llama-3.3-70b-versatile")

groq_client = Groq(api_key=GROQ_API_KEY)

# ============================================================
# University of Adelaide Support Pack
# (chỉ chèn khi phù hợp, theo rules ở dưới)
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
# Pydantic models
# ============================================================
class ChatMessage(BaseModel):
    role: str   # "user" / "assistant" / "system"
    content: str

class ChatRequest(BaseModel):
    student_id: Optional[str] = None
    message: str
    history: List[ChatMessage] = []

class ChatResponse(BaseModel):
    reply: str


# ============================================================
# AGENT 1 – INSIGHT AGENT
# - emotion
# - risk_level
# - positive_event
# - topics
# ============================================================
def run_insight_agent(message: str) -> Dict[str, Any]:
    prompt = f"""
You are the Insight Extraction Agent in a wellbeing system.

Task:
- Read the student's message.
- Extract a few key fields.
- Be concise and practical.

Return ONLY valid JSON with keys:
- "emotion": one word (e.g., "joy", "sadness", "worry", "stress", "anger", "neutral")
- "risk_level": "low", "medium", or "high"
- "positive_event": true or false
- "topics": a short list of 1–4 tags (e.g., ["exam", "loneliness", "homesickness"])

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
        logger.warning(f"Insight agent failed: {e}")
        return {
            "emotion": "neutral",
            "risk_level": "low",
            "positive_event": False,
            "topics": [],
        }


# ============================================================
# AGENT 2 – PROFILE AGENT
# - Tóm tắt trạng thái hiện tại (2–3 câu)
# ============================================================
def run_profile_agent(student_id: str, insights: Dict[str, Any]) -> str:
    prompt = f"""
You are the Profile Agent.

You receive:
- a pseudo student ID (not identifying)
- the current insight data

Produce a SHORT 2–3 sentence summary describing:
- the student's emotional tone
- the main concerns or topics
- whether this moment seems generally positive or negative.

Write in neutral English (this is an internal summary for the system).

Student ID: {student_id}
Insights: {insights}
"""
    completion = groq_client.chat.completions.create(
        model=MODEL_ID,
        messages=[{"role": "system", "content": prompt}],
        temperature=0.3,
    )
    return completion.choices[0].message.content


# ============================================================
# AGENT 3 – RISK TREND AGENT
# (Ở đây không có DB, nên trend chỉ dựa vào history hiện tại)
# - trend: "unknown" / "stable" / "worsening" / "improving"
# ============================================================
def run_trend_agent(student_id: str, insights: Dict[str, Any], history: List[ChatMessage]) -> Dict[str, Any]:
    """
    Đơn giản: nhìn qua history trên UI hiện tại để đoán trend.
    Nếu sau này bạn thêm DB, có thể thay thế agent này bằng bản đọc nhiều session.
    """
    history_text = "\n".join([f"{m.role}: {m.content}" for m in history[-8:]])  # lấy 8 lượt gần nhất

    prompt = f"""
You are the Risk Trend Agent.

You get:
- a student pseudo ID
- the latest insight (emotion, risk_level, topics)
- a short recent conversation history

Infer:
- "trend": one of "unknown", "stable", "worsening", "improving"
- "rationale": one short sentence explaining why

Return ONLY valid JSON with keys "trend" and "rationale".

Student ID: {student_id}
Latest insights: {insights}

Recent history:
{history_text}
"""
    try:
        completion = groq_client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "system", "content": prompt}],
            temperature=0.3,
        )
        import json
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        logger.warning(f"Trend agent failed: {e}")
        return {"trend": "unknown", "rationale": "No reliable trend from limited data."}


# ============================================================
# AGENT 4 – INTERVENTION AGENT
# - Gợi ý 0–2 "mini-exercises"
# ============================================================
def run_intervention_agent(insights: Dict[str, Any], trend: Dict[str, Any], message: str) -> str:
    prompt = f"""
You are the Intervention Recommender Agent for a wellbeing system.

You receive:
- current insights (emotion, risk_level, positive_event, topics)
- a trend label (unknown/stable/worsening/improving)
- the student's latest message

Task:
- If risk_level is low or medium, and the message is about stress, worry, or feeling down,
  suggest 1–2 VERY SMALL, PRACTICAL exercises the student could try
  (e.g., a 1-minute breathing exercise, a short grounding practice, a tiny journaling task).
- If risk_level is high, keep the suggestion extremely gentle and simple, and only if appropriate.
- If this is a positive/celebration message, you may suggest 1 small way to savour or celebrate.
- If the message is just informational or neutral, you may return an empty suggestion.

Output:
- A short paragraph (max 3 sentences), or an empty string if no intervention is needed.

Insights: {insights}
Trend: {trend}
Message: {message}
"""
    completion = groq_client.chat.completions.create(
        model=MODEL_ID,
        messages=[{"role": "system", "content": prompt}],
        temperature=0.4,
    )
    return completion.choices[0].message.content.strip()


# ============================================================
# AGENT 5 – SAFETY & ESCALATION AGENT
# - Bắt các trường hợp self-harm / harm others
# ============================================================
def run_safety_agent(message: str, insights: Dict[str, Any]) -> Dict[str, Any]:
    """
    Trả về:
    - "escalate": bool
    - "reason": short text
    - "override_risk_level": optional "medium"/"high"/null
    """
    base = {
        "escalate": False,
        "reason": "",
        "override_risk_level": None,
    }

    msg_low = message.lower()
    # Một số từ khoá rất thô sơ, bạn có thể mở rộng thêm
    danger_keywords = [
        "tự tử", "tự sát", "kết thúc cuộc đời", "không muốn sống nữa",
        "kill myself", "end my life", "suicide", "hurt myself",
        "giết người", "kill someone", "harm others",
    ]
    if any(kw in msg_low for kw in danger_keywords):
        base["escalate"] = True
        base["reason"] = "Message contains self-harm or harm-others phrases."
        base["override_risk_level"] = "high"
        return base

    # Nếu insight đã là high thì cũng đánh dấu escalate (nhưng mềm hơn)
    if insights.get("risk_level") == "high":
        base["escalate"] = True
        base["reason"] = "Insight agent assessed high risk."
        base["override_risk_level"] = "high"

    return base


# ============================================================
# AGENT 6 – STYLE / PERSONA AGENT
# - Ước lượng style mà user hợp (ngắn/dài, casual/formal, mềm/cứng)
# ============================================================
def run_style_agent(student_id: str, history: List[ChatMessage], insights: Dict[str, Any]) -> str:
    recent_user_msgs = "\n".join(
        [m.content for m in history if m.role == "user"][-5:]
    )
    prompt = f"""
You are the Style/Persona Agent.

You see:
- student pseudo ID
- a few recent user messages
- current insights

Task:
- Produce a SHORT note (2–3 bullet points) describing how the assistant
  should adapt its style for THIS student, for example:
  - shorter vs longer answers
  - more casual vs more formal
  - more examples vs more direct
- Do NOT mention this note to the student; it is internal only.

Student ID: {student_id}
Recent user messages:
{recent_user_msgs}

Insights: {insights}
"""
    completion = groq_client.chat.completions.create(
        model=MODEL_ID,
        messages=[{"role": "system", "content": prompt}],
        temperature=0.4,
    )
    return completion.choices[0].message.content


# ============================================================
# BASE SYSTEM PROMPT – STRICT LANGUAGE + SUPPORT RULES
# ============================================================
BASE_SYSTEM_PROMPT = f"""
You are a warm wellbeing companion for University of Adelaide students.
You use CBT, Positive Psychology, and supportive listening skills,
but you are NOT a therapist and do NOT diagnose.

LANGUAGE RULES (STRICT):
- If the student writes primarily in Vietnamese → ALWAYS reply 100% in Vietnamese.
- If the student writes primarily in English → ALWAYS reply 100% in English.
- NEVER reply in Chinese/Japanese/Korean unless the ENTIRE user message is in that language.
- If the user includes 1–2 Chinese characters in a Vietnamese/English sentence
  (e.g., "để庆祝"), treat them as normal text and DO NOT explain them unless the user explicitly asks:
  ("dịch chữ này", "giải thích chữ này", "what does this mean").
- NEVER spontaneously switch to Chinese or give bilingual explanations.

SUPPORT RULES (STRICT):
- Only provide University of Adelaide wellbeing/counselling support information when:
  (1) The risk_level or override_risk_level is "medium" or "high", OR
  (2) The user clearly expresses sadness, stress, anxiety, loneliness, overwhelm.
- DO NOT provide support info for neutral, playful, or purely positive/celebration questions.
- DO NOT provide support info for simple definition/explanation requests.

GOOD NEWS RULES:
- If the student shares positive news, respond joyfully, naturally, and like a friendly human.
- Keep it light and short, avoid clinical analysis unless they ask.

NEGATIVE EMOTION RULES:
- Validate feelings gently.
- Offer 1–2 small, practical next steps.
- Only then add support info if the SUPPORT RULES are satisfied.

NEVER:
- Give medical, legal, or financial advice.
- Promise confidentiality.
- Act as an emergency service.
"""


# ============================================================
# AGENT 7 – RESPONSE AGENT (FINAL REPLY)
# - dùng tất cả thông tin từ 1–6
# ============================================================
def run_response_agent(
    req: ChatRequest,
    insights: Dict[str, Any],
    profile_summary: str,
    trend: Dict[str, Any],
    interventions: str,
    safety: Dict[str, Any],
    style_hint: str,
) -> str:

    # Xử lý Joy Mode
    joy_boost = ""
    if insights.get("positive_event"):
        joy_boost = """
The user is sharing GOOD NEWS / a positive event.
Respond with a joyful, friendly, light tone.
Do NOT be clinical. Celebrate warmly with them.
"""

    # Xử lý risk level với safety override
    effective_risk = safety.get("override_risk_level") or insights.get("risk_level")

    # Có chèn Adelaide support hay không?
    msg_low = req.message.lower()
    emotional_keywords = [
        "stress", "lo lắng", "buồn", "sad", "tired",
        "khó chịu", "overwhelmed", "burnout", "anxious", "căng thẳng"
    ]
    add_support = False

    if effective_risk in ["medium", "high"]:
        add_support = True
    if any(kw in msg_low for kw in emotional_keywords):
        add_support = True
    if safety.get("escalate"):
        add_support = True

    support_block = ADELAIDE_SUPPORT if add_support else ""

    # Nếu có safety escalate thì nhắc rõ trong system prompt
    safety_block = ""
    if safety.get("escalate"):
        safety_block = f"""
SAFETY FLAG:
- The Safety Agent flagged this message as needing extra care.
- Reason: {safety.get("reason")}
Guidelines:
- Be extremely gentle.
- DO NOT provide detailed techniques that could be misused.
- Emphasise contacting real-world support and crisis services.
"""

    # Nếu có intervention gợi ý thì đưa cho assistant như hint
    interventions_block = ""
    if interventions:
        interventions_block = f"""
INTERVENTION SUGGESTIONS (for you to weave in naturally if appropriate):
{interventions}
"""

    # Style hint
    style_block = ""
    if style_hint:
        style_block = f"""
STYLE HINT for this student:
{style_hint}
"""

    system_content = (
        BASE_SYSTEM_PROMPT
        + joy_boost
        + safety_block
        + interventions_block
        + style_block
        + "\n\nEffective risk level: "
        + str(effective_risk)
        + "\n\nTrend info:\n"
        + str(trend)
        + "\n\n"
    )

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_content},
        {"role": "system", "content": f"Profile Summary:\n{profile_summary}"},
        {"role": "system", "content": support_block},
    ]

    # Thêm history
    for m in req.history:
        messages.append({"role": m.role, "content": m.content})

    # Thêm message hiện tại
    messages.append({"role": "user", "content": req.message})

    completion = groq_client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        temperature=0.6,
        max_tokens=1024,
    )
    return completion.choices[0].message.content


# ============================================================
# FastAPI app
# ============================================================
app = FastAPI(title="Wellbeing Agent – 7-Agent Pipeline")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # dev: mở, sau này khóa lại domain UI cũng được
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):

    student_id = req.student_id or "anonymous"
    logger.info(f"Incoming from {student_id}: {req.message!r}")

    # 1) Insight
    insights = run_insight_agent(req.message)

    # 2) Profile
    profile = run_profile_agent(student_id, insights)

    # 3) Trend
    trend = run_trend_agent(student_id, insights, req.history)

    # 4) Intervention
    interventions = run_intervention_agent(insights, trend, req.message)

    # 5) Safety
    safety = run_safety_agent(req.message, insights)

    # 6) Style
    style_hint = run_style_agent(student_id, req.history, insights)

    # 7) Final Response
    reply = run_response_agent(
        req=req,
        insights=insights,
        profile_summary=profile,
        trend=trend,
        interventions=interventions,
        safety=safety,
        style_hint=style_hint,
    )

    return ChatResponse(reply=reply)


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID}
