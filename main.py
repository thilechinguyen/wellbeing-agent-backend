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
# JSON Extraction Helper
# ============================================================
def extract_json(text: str):
    """
    Extract JSON from LLM output using:
    ```json
    { ... }
    ```
    If not found, try to parse the full text.
    """
    if not text:
        return None

    # Case 1: triple-backtick JSON
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass

    # Case 2: pure JSON
    try:
        return json.loads(text)
    except:
        return None


# ============================================================
# University of Adelaide Support Pack
# (chỉ chèn khi phù hợp)
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
# ============================================================
def run_insight_agent(message: str) -> Dict[str, Any]:
    prompt = f"""
You are the Insight Extraction Agent.

You MUST return ONLY valid JSON wrapped in triple backticks like:

```json
{{
 "emotion": "joy",
 "risk_level": "low",
 "positive_event": true,
 "topics": ["example"]
}}
Extract:
emotion
risk_level
positive_event
topics
NO extra text. NO explanation.
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

    if data:
        return data

    raise ValueError("Invalid JSON")

except Exception as e:
    logger.warning("Insight agent failed: %s", e)
    return {
        "emotion": "neutral",
        "risk_level": "low",
        "positive_event": False,
        "topics": [],
    }
============================================================
AGENT 2 – PROFILE AGENT
Summary (internal)
============================================================
def run_profile_agent(student_id: str, insights: Dict[str, Any]) -> str:
prompt = f"""
You are the Profile Agent.
Summarize the student's emotional state in 2–3 sentences.
This is INTERNAL, not shown to user.

Student ID: {student_id}
Insights: {insights}
"""
completion = groq_client.chat.completions.create(
model=MODEL_ID,
messages=[{"role": "system", "content": prompt}],
temperature=0.2,
)
return completion.choices[0].message.content

============================================================
AGENT 3 – TREND AGENT (simple: from history only)
============================================================
def run_trend_agent(student_id: str, insights: Dict[str, Any], history: List[ChatMessage]):
history_text = "\n".join([f"{m.role}: {m.content}" for m in history[-6:]])
prompt = f"""
You are the Trend Agent.
Return ONLY valid JSON:

{{
 "trend": "stable",
 "rationale": "short text"
}}
Student ID: {student_id}
Latest insights:
{insights}

Recent history:
{history_text}
"""

try:
    comp = groq_client.chat.completions.create(
        model=MODEL_ID,
        messages=[{"role": "system", "content": prompt}],
        temperature=0,
    )
    raw = comp.choices[0].message.content
    data = extract_json(raw)

    if data:
        return data

    raise ValueError("Invalid JSON")

except Exception as e:
    logger.warning("Trend agent failed: %s", e)
    return {"trend": "unknown", "rationale": "Not enough data"}
============================================================
AGENT 4 – INTERVENTION AGENT
(skip completely for positive events)
============================================================
def run_intervention_agent(insights, trend, message):
# Skip for positive events (friend mode)
if insights.get("positive_event") and insights.get("risk_level") == "low":
return ""
prompt = f"""
You are the Intervention Agent.
Suggest 1–2 tiny exercises ONLY if the message shows stress or sadness.
Otherwise return an empty string.

Respond with max 2 sentences.

Insights: {insights}
Trend: {trend}
Message: {message}
"""
comp = groq_client.chat.completions.create(
model=MODEL_ID,
messages=[{"role": "system", "content": prompt}],
temperature=0.4,
)
return comp.choices[0].message.content.strip()

============================================================
AGENT 5 – SAFETY AGENT
Detect self-harm/harm-others
============================================================
def run_safety_agent(message: str, insights: Dict[str, Any]):
base = {"escalate": False, "reason": "", "override_risk_level": None}
msg = message.lower()
danger = [
    "tự tử", "tự sát", "kill myself", "end my life", "suicide",
    "hurt myself", "không muốn sống", "giết người", "kill someone",
]

if any(k in msg for k in danger):
    return {
        "escalate": True,
        "reason": "Self-harm or harm keywords detected",
        "override_risk_level": "high",
    }

if insights.get("risk_level") == "high":
    return {
        "escalate": True,
        "reason": "Insight says high risk",
        "override_risk_level": "high",
    }

return base
============================================================
AGENT 6 – STYLE AGENT
============================================================
def run_style_agent(student_id: str, history: List[ChatMessage], insights):
recent_user_msgs = "\n".join([m.content for m in history if m.role == "user"][-5:])
prompt = f"""
You are the Style Agent.
Produce 2–3 bullet points recommending tone adjustments for this user.
Internal only.

Recent user messages:
{recent_user_msgs}

Insights: {insights}
"""

comp = groq_client.chat.completions.create(
    model=MODEL_ID,
    messages=[{"role": "system", "content": prompt}],
    temperature=0.4,
)
return comp.choices[0].message.content
============================================================
BASE SYSTEM PROMPT (STRICT)
============================================================
BASE_SYSTEM_PROMPT = """
You are a wellbeing companion for University of Adelaide students.
STRICT LANGUAGE:

If user writes Vietnamese → reply 100% in Vietnamese.
No Chinese/Korean/Japanese unless user writes entire message in that language.
Never bilingual unless user asks.
GOOD NEWS MODE:
If positive_event = true and risk = low:
→ You MUST talk like a close Vietnamese friend.
→ Tone = ấm áp, vui mừng, tự nhiên, kiểu bạn thân Adelaide.
→ Example expressions: “Trời ơi ghê vậy trời!”, “Quá dữ luôn á!”, “Chúc mừng nha!”
→ DO NOT: CBT, journaling, reflection, breathing exercises.
→ DO NOT: mention wellbeing support.
→ DO NOT: phân tích tâm lý hay giọng counsellor.
→ Keep it fun, short, happy, human.
SUPPORT RULES:
Only mention Adelaide support if:
risk = medium/high OR user expresses sadness/stress/anxiety.
Never include support for positive/celebration messages.
NEGATIVE EMOTION RULES:
Validate feelings softly.
Offer 1–2 small steps.
Then add support IF allowed.
NEVER:
Give medical/legal/financial advice.
Promise confidentiality.
"""
============================================================
AGENT 7 – FINAL RESPONSE AGENT
============================================================
def run_response_agent(req: ChatRequest,
insights,
profile_summary,
trend,
interventions,
safety,
style_hint):
joy_block = ""
if insights.get("positive_event"):
    joy_block = """
The user is sharing GOOD NEWS.
Respond EXACTLY like a close Vietnamese friend.
TONE:

Vui, thân mật, tự nhiên
Dùng tiếng Việt kiểu bạn bè
1 câu follow-up dễ thương được phép
FORBIDDEN:
CBT
journaling
phân tích cảm xúc
bất kỳ hỗ trợ wellbeing
"""
effective_risk = safety.get("override_risk_level") or insights.get("risk_level")

msg_low = req.message.lower()
emotional_keywords = ["stress", "lo lắng", "buồn", "sad", "anxious", "căng thẳng"]

add_support = False
if effective_risk in ["medium", "high"]:
add_support = True
if any(k in msg_low for k in emotional_keywords):
add_support = True
if safety.get("escalate"):
add_support = True

block support if positive event & low risk
if insights.get("positive_event") and effective_risk == "low":
add_support = False
support_block = ADELAIDE_SUPPORT if add_support else ""

system_prompt = (
BASE_SYSTEM_PROMPT
+ joy_block
+ f"\nSTYLE HINT (internal):\n{style_hint}\n"
)

messages = [
{"role": "system", "content": system_prompt},
{"role": "system", "content": f"Profile Summary:\n{profile_summary}"},
{"role": "system", "content": f"Trend Info:\n{trend}"},
{"role": "system", "content": f"Interventions (internal): {interventions}"},
{"role": "system", "content": support_block},
]

for m in req.history:
messages.append({"role": m.role, "content": m.content})

messages.append({"role": "user", "content": req.message})

comp = groq_client.chat.completions.create(
model=MODEL_ID,
messages=messages,
temperature=0.65,
max_tokens=800,
)
return comp.choices[0].message.content

============================================================
FastAPI App
============================================================
app = FastAPI(title="Wellbeing Agent – 7 Agents + Joy Mode + JSON Safe")
app.add_middleware(
CORSMiddleware,
allow_origins=[""],
allow_credentials=True,
allow_methods=[""],
allow_headers=["*"],
)

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):

sid = req.student_id or "anonymous"
logger.info(f"Incoming from {sid}: {req.message}")

insights = run_insight_agent(req.message)
profile = run_profile_agent(sid, insights)
trend = run_trend_agent(sid, insights, req.history)
interventions = run_intervention_agent(insights, trend, req.message)
safety = run_safety_agent(req.message, insights)
style_hint = run_style_agent(sid, req.history, insights)

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