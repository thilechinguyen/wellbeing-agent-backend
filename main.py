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
# Simple language fallback if agent does not return language
# ============================================================
def detect_language_fallback(message: str) -> str:
    msg = message.lower()

    # UI prefix, e.g. [lang=vi;...]
    if "[lang=vi" in msg:
        return "vi"
    if "[lang=en" in msg:
        return "en"
    if "[lang=zh" in msg:
        return "zh"
    if "[lang=ko" in msg:
        return "ko"
    if "[lang=ja" in msg or "[lang=jp" in msg:
        return "ja"

    # Vietnamese diacritics
    if re.search(
        r"[ăâêôơưđáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệóòỏõọốồổỗộớờởỡợúùủũụứừửữựíìỉĩị]",
        message,
    ):
        return "vi"

    # Default English
    return "en"


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
# AGENT 1 - Insight Agent (with context + language)
# ============================================================
def run_insight_agent(message: str, history: List[ChatMessage]) -> Dict[str, Any]:
    """
    Use both recent context and latest message to infer:
    - emotion
    - risk_level
    - positive_event
    - topics
    - language
    """
    recent = history[-4:]
    context_text = "\n".join([f"{m.role}: {m.content}" for m in recent])

    prompt = f"""
You are the Insight Extraction Agent in a wellbeing system.

You see a short conversation and the latest student message.
You must classify the student's CURRENT emotional state
by considering BOTH:
- the recent context
- the latest message

Recent conversation (most recent at the end):
{context_text}

Latest student message:
{message}

Return ONLY a JSON object with the following keys:
- "emotion": one word, for example "joy", "sadness", "worry",
  "stress", "anger", "neutral"
- "risk_level": "low", "medium", or "high"
- "positive_event": true or false
- "topics": a short list of 1-4 simple tags, for example ["exam", "friends"]
- "language": main language code of the student's message and context, one of:
    "vi" (Vietnamese), "en" (English), "zh" (Chinese),
    "ko" (Korean), "ja" (Japanese), or "other"

Do not include any explanation or extra text outside the JSON.
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
            "language": "other",
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
# AGENT 3 - Trend Agent
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
    # Skip for clearly positive events with low risk
    if insights.get("positive_event") and insights.get("risk_level") == "low":
        return ""

    prompt = f"""
You are the Intervention Agent.

If the student's message shows stress, sadness, worry, or feeling down:
- Suggest 1-2 very small, practical exercises (for example a 1-minute
  breathing exercise, a short grounding practice, or a tiny journaling task).
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
# AGENT 5 - Safety Agent (includes relationship violence)
# ============================================================
def run_safety_agent(message: str, insights: Dict[str, Any]) -> Dict[str, Any]:
    base = {"escalate": False, "reason": "", "override_risk_level": None}

    msg = message.lower()
    danger_keywords = [
        "tự tử",
        "tự sát",
        "không muốn sống",
        "khong muon song",
        "kill myself",
        "end my life",
        "suicide",
        "hurt myself",
        "giết người",
        "giet nguoi",
        "kill someone",
    ]

    relationship_violence = [
        "đánh em",
        "danh em",
        "bị đánh",
        "bi danh",
        "bạo lực",
        "bao luc",
        "hit me",
        "hurt me",
        "abused me",
        "domestic violence",
        "he hit me",
        "he slapped me",
        "he punched me",
    ]

    if any(kw in msg for kw in danger_keywords):
        return {
            "escalate": True,
            "reason": "Self-harm or harm-others keywords detected",
            "override_risk_level": "high",
        }

    if any(kw in msg for kw in relationship_violence):
        return {
            "escalate": True,
            "reason": "Relationship violence detected",
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
You are a wellbeing companion for university students.

Language rules:
- Always answer 100% in the same main language as the student.
- Do not mix languages unless the student clearly does so and asks for it.
- Do not translate unless the student explicitly asks for a translation.

For Vietnamese (vi):
- Use natural Vietnamese.
- You may use casual Southern style expressions if appropriate.

For English (en):
- Use natural conversational English.
- If the student is in Australia, you may sound like an Aussie uni friend.

For Chinese (zh), Korean (ko), Japanese (ja):
- Use casual, friendly style in that language.
- Keep sentences simple and supportive.

Good news rules:
- If the student's message is a clearly positive event and risk level is low,
  you must respond like a close friend (not a counsellor).
- No CBT, journaling, breathing exercises, or wellbeing techniques in that case.
- Do NOT mention wellbeing / counselling services in that case.

Support rules:
- Only provide university wellbeing/counselling support information when:
  (1) risk_level or override_risk_level is "medium" or "high", OR
  (2) the user clearly expresses sadness, stress, anxiety, loneliness, or overwhelm.
- Do NOT provide support info for neutral, playful, or purely positive messages.

Never:
- Give medical, legal, or financial advice.
- Promise confidentiality.
- Act as an emergency service.
"""


# ============================================================
# Joy-context detection (Joy Sticky)
# ============================================================
def is_celebration_context(all_msgs: List[ChatMessage]) -> bool:
    """
    Check if the recent conversation is mainly about a positive event
    like scholarship, lottery, getting a job, passing an exam, etc.
    """
    user_texts = [m.content.lower() for m in all_msgs if m.role == "user"]
    if not user_texts:
        return False

    # Look at last few user messages only
    text = " ".join(user_texts[-6:])

    celebration_keywords = [
        "trúng số",
        "trung so",
        "trúng học bổng",
        "trung hoc bong",
        "được học bổng",
        "duoc hoc bong",
        "đậu visa",
        "dau visa",
        "được nhận",
        "duoc nhan",
        "passed the exam",
        "pass the exam",
        "got a scholarship",
        "won the lottery",
        "got the job",
        "got an offer",
        "accepted into",
    ]
    negative_keywords = [
        "buồn",
        "buon",
        "stress",
        "lo lắng",
        "lo lang",
        "không muốn sống",
        "khong muon song",
        "sad",
        "anxious",
        "depressed",
        "bị đánh",
        "bi danh",
        "đánh em",
        "danh em",
        "bạo lực",
        "bao luc",
    ]

    if any(kw in text for kw in celebration_keywords) and not any(
        kw in text for kw in negative_keywords
    ):
        return True
    return False


# ============================================================
# Language-specific blocks
# ============================================================
def build_language_block(language: str) -> str:
    if language == "vi":
        return """
Detected language: Vietnamese (vi).

When replying in Vietnamese:
- Use simple, natural Vietnamese, can be Southern-style if the message suggests it.
- Understand slang such as:
  "ăn bể bụng" (eat a lot, very full),
  "bún có bèo" (too cheap or too basic for celebration),
  "bèo quá hong?" (is it too cheap?),
  "xỉu up xỉu down" (very shocked or excited),
  "lụm liền" (grab the opportunity immediately).
- Respond naturally as a close Vietnamese friend.
"""
    if language == "en":
        return """
Detected language: English (en).

When replying in English:
- Sound like a close uni friend.
- You may understand and use light Aussie slang such as:
  "mate", "stoked", "bloody legend", "no worries", "that's sick".
- Keep it warm, supportive, and conversational.
"""
    if language == "zh":
        return """
Detected language: Chinese (zh).

When replying in Chinese:
- Use friendly, casual Mandarin.
- Understand basic internet slang like "哈哈哈", "可以哦", "稳了", "太牛了".
- Sound like a close friend, not a therapist.
"""
    if language == "ko":
        return """
Detected language: Korean (ko).

When replying in Korean:
- Use casual but respectful style suitable for uni students.
- Understand common slang like "대박", "헐", "ㅋㅋㅋ", "미쳤다" used in a positive sense.
"""
    if language == "ja":
        return """
Detected language: Japanese (ja).

When replying in Japanese:
- Use friendly, casual style (but not rude).
- Understand simple slang like "やばい", "マジで", "すごい" used in a positive sense.
"""
    return """
Detected language: other or unknown.

Follow the language used by the student and keep a friendly, simple tone.
"""


def build_joy_block(language: str, joy_mode: bool) -> str:
    if not joy_mode:
        return ""

    if language == "vi":
        return """
The user is sharing clearly positive news with low risk and the conversation
is in Vietnamese.

Joy mode instructions:
- Respond exactly like a close Vietnamese friend (Southern casual style is OK).
- Tone: very warm, excited, funny and friendly.
- Use expressions like:
  "Trời ơi, chúc mừng nha!",
  "Ghê vậy trời!",
  "Quá dữ luôn á!",
  "Vui giùm luôn đó!".
- You MAY ask ONE playful follow-up question such as "Giờ tính ăn mừng sao nè?".
- Absolutely DO NOT:
  - suggest CBT, journaling, breathing, reflection
  - mention wellbeing or counselling services
  - sound like a counsellor or teacher.
"""

    if language == "en":
        return """
The user is sharing clearly positive news with low risk and the conversation
is in English.

Joy mode instructions:
- Respond like a close uni friend (Aussie style is OK).
- Tone: warm, excited, relaxed.
- You can use light slang such as:
  "Congrats mate, that's awesome!",
  "I'm so stoked for you!",
  "You're a legend!".
- You MAY ask ONE playful follow-up question such as
  "So how are you going to celebrate?".
- Do NOT bring up CBT or wellbeing techniques.
- Do NOT mention support services in this message.
"""

    if language == "zh":
        return """
The user is sharing clearly positive news with low risk and the conversation
is in Chinese.

Joy mode instructions:
- Respond like a close Chinese-speaking friend.
- Tone: warm, excited, casual.
- You can use expressions like:
  "哇，太厉害了！",
  "恭喜恭喜！",
  "真的很为你开心！".
- You MAY ask one light follow-up question about how they will celebrate.
- Do NOT mention wellbeing techniques or support services in this message.
"""

    if language == "ko":
        return """
The user is sharing clearly positive news with low risk and the conversation
is in Korean.

Joy mode instructions:
- Respond like a close Korean friend.
- Tone: warm, excited, casual.
- You can use expressions like:
  "와, 대박이다!",
  "진짜 축하해!",
  "너무 잘했다!".
- You MAY ask one light follow-up question about celebration.
- Do NOT mention wellbeing techniques or support services in this message.
"""

    if language == "ja":
        return """
The user is sharing clearly positive news with low risk and the conversation
is in Japanese.

Joy mode instructions:
- Respond like a close Japanese friend.
- Tone: warm, excited, casual.
- You can use expressions like:
  "うわ、すごいね！",
  "おめでとう！",
  "自分のことみたいに嬉しい！".
- You MAY ask one light follow-up question.
- Do NOT mention wellbeing techniques or support services in this message.
"""

    return """
The user is sharing clearly positive news with low risk (unknown language).

Joy mode instructions:
- Reply like a close friend in the same language as the student.
- Keep it short, warm, and excited.
- No CBT, no wellbeing techniques, no support services in this message.
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

    # Language from insight or fallback
    language = insights.get("language") or detect_language_fallback(req.message)

    # Joy Sticky: check whole conversation context for celebration flow
    all_msgs: List[ChatMessage] = list(req.history) + [
        ChatMessage(role="user", content=req.message)
    ]
    celebration_flow = is_celebration_context(all_msgs)

    # Base joy mode from insight (single turn)
    risk_level = insights.get("risk_level")
    joy_mode_from_insight = bool(
        insights.get("positive_event") and risk_level == "low"
    )

    # Final joy mode: either explicit positive_event OR celebration context with low risk
    joy_mode = bool(
        joy_mode_from_insight
        or (celebration_flow and (risk_level in [None, "low"]))
    )

    msg_low = req.message.lower()

    # If strong negative / violence keywords appear, force Joy OFF
    negative_break_keywords = [
        "buồn",
        "buon",
        "khóc",
        "khoc",
        "đau",
        "dau",
        "bị đánh",
        "bi danh",
        "đánh em",
        "danh em",
        "bạo lực",
        "bao luc",
        "hurt me",
        "hit me",
        "abused me",
        "he hit me",
        "he slapped me",
        "he punched me",
        "không muốn sống",
        "khong muon song",
        "tự tử",
        "tự sát",
        "tu tu",
        "tu sat",
    ]

    if any(kw in msg_low for kw in negative_break_keywords):
        joy_mode = False
        celebration_flow = False

    language_block = build_language_block(language)
    joy_block = build_joy_block(language, joy_mode)

    effective_risk = safety.get("override_risk_level") or risk_level

    emotional_keywords = [
        "stress",
        "lo lắng",
        "lo lang",
        "buồn",
        "buon",
        "khóc",
        "khoc",
        "sad",
        "anxious",
        "căng thẳng",
        "cang thang",
        "cô đơn",
        "co don",
    ]

    add_support = False
    if effective_risk in ["medium", "high"]:
        add_support = True
    if any(kw in msg_low for kw in emotional_keywords):
        add_support = True
    if safety.get("escalate"):
        add_support = True

    # In joy mode, never add support block
    if joy_mode:
        add_support = False

    support_block = ADELAIDE_SUPPORT if add_support else ""

    system_content = (
        BASE_SYSTEM_PROMPT
        + "\n\nCurrent language code: "
        + str(language)
        + "\n\n"
        + language_block
        + "\n"
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
app = FastAPI(
    title="Wellbeing Agent - 7 Agents, Joy Sticky Auto OFF for Violence/Negative"
)

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

    insights = run_insight_agent(req.message, req.history)
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
