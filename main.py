import os
import re
import json
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
# JSON Extraction Helper (robust)
# ============================================================
def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Try to parse JSON from LLM output.
    1) Try full text
    2) Try first {...} block
    """
    if not text:
        return None

    text = text.strip()

    # Try full text
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try to find first {...}
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except Exception:
            pass

    return None

# ============================================================
# Adelaide support block
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
# Pydantic models
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
# AGENT 1 - Insight Agent
# ============================================================
def run_insight_agent(message: str) -> Dict[str, Any]:
    prompt = f"""
You are the Insight Extraction Agent in a wellbeing system.

Task:
- Read the student's message.
- Return ONLY a JSON object with the following keys:
  - "emotion": one word such as "joy", "sadness", "worry", "stress", "anger", "neutral"
  - "risk_level": "low", "medium", or "high"
  - "positive_event": true or false
  - "topics": a short list of 1-4 simple tags

Do not include any explanation or extra text. Only output the JSON object.

Message:
{message}
"""
    try:
        completion = groq_client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "system", "content": prompt}],
            temperature=0,
        )
        raw = completion.choices[0].message.content
        data = extract_json(raw)
        if data is not None:
            return data
        raise ValueError("Insight agent returned non-JSON output")
    except Exception as e:
        logger.warning("Insight agent failed: %s", e)
        return {
            "emotion": "neutral",
            "risk_level": "low",
            "positive_event": False,
            "topics": [],
        }

# ============================================================
# AGENT 2 - Profile Agent
# ============================================================
def run_profile_agent(student_id: str, insights: Dict[str, Any]) -> str:
    prompt = f"""
You are the Profile Agent.

Summarize the student's current emotional state in 2-3 sentences.
This summary is INTERNAL ONLY (never shown to the student).

Student ID: {student_id}
Insights: {insights}
"""
    completion = groq_client.chat.completions.create(
        model=MODEL_ID,
        messages=[{"role": "system", "content": prompt}],
        temperature=0.2,
    )
    return completion.choices[0].message.content

# ============================================================
# AGENT 3 - Trend Agent (simple, from history)
# ============================================================
def run_trend_agent(student_id: str, insights: Dict[str, Any], history: List[ChatMessage]) -> Dict[str, Any]:
    history_text = "\n".join([f"{m.role}: {m.content}" for m in history[-6:]])

    prompt = f"""
You are the Trend Agent.

Look at:
- the latest insight data
- a short recent conversation history

Return ONLY a JSON object with:
- "trend": one of "unknown", "stable", "worsening", "improving"
- "rationale": one short sentence

Student ID: {student_id}
Latest insights: {insights}

Recent history:
{history_text}
"""
    try:
        completion = groq_client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "system", "content": prompt}],
            temperature=0,
        )
        raw = completion.choices[0].message.content
        data = extract_json(raw)
        if data is not None:
            return data
        raise ValueError("Trend agent returned non-JSON output")
    except Exception as e:
        logger.warning("Trend agent failed: %s", e)
        return {"trend": "unknown", "rationale": "Not enough reliable data"}

# ============================================================
# AGENT 4 - Intervention Agent
# ============================================================
def run_intervention_agent(insights: Dict[str, Any], trend: Dict[str, Any], message: str) -> str:
    # For good news with low risk, skip interventions completely
    if insights.get("positive_event") and insights.get("risk_level") == "low":
        return ""

    prompt = f"""
You are the Intervention Agent.

If the student's message shows stress, sadness, worry, or feeling down:
- Suggest 1-2 very small, practical exercises (for example a 1-minute breathing
  exercise, a short grounding practice, or a tiny journaling task).
- Keep it at most 2 sentences.

If the message is neutral or only informational, or you are not sure,
return an empty string.

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
# AGENT 5 - Safety Agent
# ============================================================
def run_safety_agent(message: str, insights: Dict[str, Any]) -> Dict[str, Any]:
    base = {"escalate": False, "reason": "", "override_risk_level": None}

    msg = message.lower()
    danger_keywords = [
        "tự tử",
        "tự sát",
        "không muốn sống",
        "kill myself",
        "end my life",
        "suicide",
        "hurt myself",
        "giết người",
        "kill someone",
    ]

    if any(kw in msg for kw in danger_keywords):
        return {
            "escalate": True,
            "reason": "Self-harm or harm-others keywords detected",
            "override_risk_level": "high",
        }

    if insights.get("risk_level") == "high":
        return {
            "escalate": True,
            "reason": "Insight agent assessed high risk",
            "override_risk_level": "high",
        }

    return base

# ============================================================
# AGENT 6 - Style Agent
# ============================================================
def run_style_agent(student_id: str, history: List[ChatMessage], insights: Dict[str, Any]) -> str:
    recent_user_msgs = "\n".join([m.content for m in history if m.role == "user"][-5:])

    prompt = f"""
You are the Style Agent.

Based on the recent messages and insights, produce 2-3 bullet points
describing how the assistant should adapt its tone for this student
(for example: shorter answers, more casual, more examples, more direct).

This note is INTERNAL ONLY, never shown to the student.

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
# Base system prompt (strict language + rules)
# ============================================================
BASE_SYSTEM_PROMPT = """
You are a wellbeing companion for University of Adelaide students.

Language rules:
- If the student writes mainly in Vietnamese, always answer 100% in Vietnamese.
- If the student writes mainly in English, answer 100% in English.
- Do not reply in Chinese/Japanese/Korean unless the entire user message is in that language.
- If the user mixes 1-2 foreign characters inside Vietnamese, treat them as normal text
  and do not explain them unless they ask explicitly.

Good news rules:
- If the student's message is a clearly positive event and risk level is low,
  you must respond like a close Vietnamese friend.
- Tone: vui, thân mật, tự nhiên, giống bạn thân người Việt ở Adelaide.
- Do NOT use CBT, journaling, self-reflection, breathing exercises or any
  wellbeing techniques in that case.
- Do NOT mention wellbeing support services in that case.
- Keep the reply ngắn gọn, vui, ấm áp, có thể hỏi thêm 1 câu nhỏ kiểu
  "Giờ tính ăn mừng sao nè?".

Support rules:
- Only provide University of Adelaide wellbeing/counselling support information when:
  (1) risk_level or override_risk_level is "medium" or "high", OR
  (2) the user clearly expresses sadness, stress, anxiety, loneliness, or overwhelm.
- Do NOT provide support info for neutral, playful, or purely positive messages.

Never:
- Give medical, legal, or financial advice.
- Promise confidentiality.
- Act as an emergency service.
"""

# ============================================================
# AGENT 7 - Response Agent (final answer)
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

    joy_mode = insights.get("positive_event") and insights.get("risk_level") == "low"

    joy_block = ""
    if joy_mode:
        joy_block = """
The user is sharing clearly positive news with low risk.
For this message, you MUST talk like a close Vietnamese friend celebrating with them.

Tone requirements:
- Very warm, excited, natural, friendly, Vietnamese style.
- Use simple, everyday Vietnamese expressions like "Trời ơi, chúc mừng nha!",
  "Ghê vậy trời!", "Quá dữ luôn á!", "Vui giùm luôn đó!".
- Keep it short, fun, light.
- You may ask ONE playful follow-up question (for example "Giờ tính ăn mừng sao nè?").
- Absolutely do NOT:
  - suggest CBT, journaling, breathing, or reflection exercises
  - mention University support services
  - sound like a counsellor or teacher.
"""

    effective_risk = safety.get("override_risk_level") or insights.get("risk_level")

    msg_low = req.message.lower()
    emotional_keywords = ["stress", "lo lắng", "buồn", "sad", "anxious", "căng thẳng"]

    add_support = False
    if effective_risk in ["medium", "high"]:
        add_support = True
    if any(kw in msg_low for kw in emotional_keywords):
        add_support = True
    if safety.get("escalate"):
        add_support = True

    if joy_mode:
        add_support = False

    support_block = ADELAIDE_SUPPORT if add_support else ""

    system_content = (
        BASE_SYSTEM_PROMPT
        + joy_block
        + "\nSTYLE HINT (internal):\n"
        + style_hint
        + "\n"
    )

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_content},
        {"role": "system", "content": f"Profile summary:\n{profile_summary}"},
        {"role": "system", "content": f"Trend info:\n{trend}"},
        {"role": "system", "content": f"Internal intervention suggestions:\n{interventions}"},
        {"role": "system", "content": support_block},
    ]

    for m in req.history:
        messages.append({"role": m.role, "content": m.content})

    messages.append({"role": "user", "content": req.message})

    completion = groq_client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        temperature=0.65,
        max_tokens=800,
    )
    return completion.choices[0].message.content

# ============================================================
# FastAPI app
# ============================================================
app = FastAPI(title="Wellbeing Agent - 7 Agents with Joy Mode and JSON Fix")

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
    logger.info("Incoming from %s: %r", student_id, req.message)

    insights = run_insight_agent(req.message)
    profile = run_profile_agent(student_id, insights)
    trend = run_trend_agent(student_id, insights, req.history)
    interventions = run_intervention_agent(insights, trend, req.message)
    safety = run_safety_agent(req.message, insights)
    style_hint = run_style_agent(student_id, req.history, insights)

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
