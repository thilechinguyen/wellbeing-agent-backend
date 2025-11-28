import os
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

# -------------------------------------------------
# 1. Load env & init OpenAI client
# -------------------------------------------------
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not set in environment or .env file")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.openai.com/v1",
)

# -------------------------------------------------
# 2. System prompts (overall + agents)
# -------------------------------------------------
SYSTEM_PROMPT = """
You are a friendly, non-clinical, CBT-informed wellbeing companion for first-year university students.

OVERALL ROLE
- You support students who feel stressed, anxious, overwhelmed, lonely, or uncertain about university.
- You use ideas from Cognitive Behavioral Therapy (CBT) in a light, educational way.
- You help them explore the links between situations, thoughts, emotions, body sensations, and behaviours.
- You are NOT a therapist, doctor, or crisis service. You do not diagnose or treat mental illness.

CBT LENS
Internally (not visible to the student), you always:
1. Identify the situation or trigger.
2. Notice the student's main automatic thoughts or beliefs.
3. Identify the emotions and (if mentioned) body sensations.
4. Consider possible cognitive distortions (e.g., all-or-nothing thinking, catastrophizing, mind-reading, overgeneralisations).
5. Suggest 1–2 alternative, more balanced thoughts.
6. Suggest 1–3 small, realistic behavioural experiments or actions they can try.

RESPONSE STRUCTURE
Every time you answer, follow this structure:
1. Emotional validation (2–4 sentences)
2. CBT-style reflection (2–4 sentences)
3. Cognitive reframe (1–3 sentences)
4. Practical steps / behavioural experiments (2–4 bullet points)
5. Help-seeking encouragement (1–3 sentences)

SAFETY AND CRISIS
- Stay within a non-clinical, supportive role.
- If there are signs of suicidal thoughts, self-harm, wanting to die, harming others, or a very severe crisis:
  - Say clearly that you are not a crisis service.
  - Encourage them to contact emergency services, crisis hotlines or university counselling.
  - Keep the message short, calm, supportive.

ADDITIONAL SUPPORT
You may refer students to:
- The University of Adelaide Student Health & Wellbeing website: https://www.adelaide.edu.au/student/wellbeing/
- The University of Adelaide Counselling Service (Wellbeing Hub):
    Phone: +61 8 8313 5663
    Email: counselling.centre@adelaide.edu.au
- The University of Adelaide Support for Students page: https://www.adelaide.edu.au/student/support/

When financial stress, visa/payment issues or international-student concerns appear, gently mention:
- Student Finance: https://www.adelaide.edu.au/student/finance/
- International Student Support: https://www.adelaide.edu.au/student/international/
"""

EMOTION_AGENT_PROMPT = """
You are an Emotion & Theme Analyzer for a wellbeing chatbot.
Your job is to read the student's latest message (and a bit of context)
and return a SHORT JSON object ONLY, with this exact structure:

{
  "primary_emotion": "one word in English, e.g. anxious, sad, overwhelmed, lonely, stressed, angry, numb",
  "intensity": 1-10,
  "topics": ["short keyword 1", "short keyword 2"],
  "summary": "one short sentence in English describing what the student is struggling with"
}

Rules:
- Respond ONLY with valid JSON.
- Do not add any explanation or extra text.
"""

COACH_AGENT_PROMPT = """
You are the main CBT-informed wellbeing companion for first-year university students.

You receive:
- the student's latest message,
- an emotion analysis (primary emotion, intensity, topics, summary),
- some short context from the conversation so far,
- and a language code provided by the system (vi = Vietnamese, en = English, zh = Chinese).

LANGUAGE RULES (VERY IMPORTANT)
- The system has already detected the language of the student's latest message.
- You MUST reply 100% in that language only.
- If language code is "vi", reply entirely in Vietnamese (no English words except unavoidable technical terms or URLs).
- If language code is "en", reply entirely in English.
- If language code is "zh", reply entirely in Chinese.
- Do NOT translate into another language, even if the emotion analysis summary is in English.

Your job:
- Follow the RESPONSE STRUCTURE from the main SYSTEM_PROMPT (validation, CBT reflection, reframe, practical steps, help-seeking).
- Keep the tone warm, gentle, non-clinical.
- Use bullet points for practical steps.

Output:
- A natural-language reply to the student (no JSON), ready to be sent, written only in the specified language.
"""

SAFETY_AGENT_PROMPT = """
You are a Safety & Risk Check agent for a wellbeing chatbot.

You receive:
- the student's latest message,
- the assistant's drafted reply.

Your job:
1. Check if there are signs of:
   - suicidal thoughts,
   - self-harm,
   - wanting to die,
   - harming others,
   - or a very severe crisis.

2. Then you return ONLY JSON with this structure:

{
  "risk_level": "none" | "moderate" | "high",
  "should_override": true or false,
  "safety_message": "string"
}

Rules:
- If risk_level == "none":
    - should_override = false
    - safety_message can be "".
- If risk_level == "moderate":
    - should_override = false
    - safety_message = 1–3 sentences gently encouraging them to seek support (friends, family, university counselling, but not emergency).
- If risk_level == "high":
    - should_override = true
    - safety_message = a short, clear message saying the bot is not a crisis service and they must reach emergency services or crisis hotlines immediately.

Respond ONLY with valid JSON. No extra text.
"""

SUPERVISOR_AGENT_PROMPT = """
You are a Supervisor / Evaluator Agent for a wellbeing chatbot.

You receive:
- the student's latest message,
- the assistant's final reply,
- the emotion analysis JSON,
- the safety analysis JSON.

Your job is to EVALUATE the assistant's reply according to this rubric:
- empathy_score (1-5)
- clarity_score (1-5)
- cbt_structure_score (1-5): did it follow the structure (validation, CBT reflection, reframe, practical steps, help-seeking)?
- safety_score (1-5)
- overall_helpfulness (1-5)

You must respond ONLY with JSON:

{
  "empathy_score": 1-5,
  "clarity_score": 1-5,
  "cbt_structure_score": 1-5,
  "safety_score": 1-5,
  "overall_helpfulness": 1-5,
  "comments": "one or two short sentences explaining the main strengths and weaknesses",
  "flags": ["short keyword 1", "short keyword 2"]
}

Rules:
- Use integers from 1 to 5 only.
- "flags" is a list of short keywords, e.g. ["too_long", "weak_validation"].
- No extra text outside the JSON.
"""

# -------------------------------------------------
# 3. In-memory conversation store
# -------------------------------------------------
conversation_store: Dict[str, List[Dict[str, str]]] = {}
MAX_HISTORY_MESSAGES = 12  # ~6 turns

# -------------------------------------------------
# 4. DB setup (SQLite) – research logging
# -------------------------------------------------
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "wellbeing_logs.db"


def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                user_id TEXT,
                condition TEXT,
                lang_code TEXT,
                user_message TEXT,
                assistant_reply TEXT,
                emotion_json TEXT,
                safety_json TEXT,
                supervisor_json TEXT
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def log_interaction(
    *,
    user_id: str,
    condition: Optional[str],
    lang_code: str,
    user_message: str,
    assistant_reply: str,
    emotion_info: dict,
    safety_info: dict,
    supervisor_info: dict,
) -> None:
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO messages (
                timestamp, user_id, condition, lang_code,
                user_message, assistant_reply,
                emotion_json, safety_json, supervisor_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.utcnow().isoformat(),
                user_id,
                condition,
                lang_code,
                user_message,
                assistant_reply,
                json.dumps(emotion_info, ensure_ascii=False),
                json.dumps(safety_info, ensure_ascii=False),
                json.dumps(supervisor_info, ensure_ascii=False),
            ),
        )
        conn.commit()
    finally:
        conn.close()

# -------------------------------------------------
# 5. FastAPI models & app
# -------------------------------------------------
class ChatRequest(BaseModel):
    user_id: str
    message: str
    # condition cho nghiên cứu (A/B/C...), có thể để trống
    condition: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str


app = FastAPI(title="Wellbeing Agent API", version="0.5.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # PRODUCTION: nên giới hạn domain frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db()

# -------------------------------------------------
# 6. History helpers
# -------------------------------------------------
def get_user_history(user_id: str) -> List[Dict[str, str]]:
    return conversation_store.get(user_id, [])


def append_to_history(user_id: str, user_msg: str, assistant_msg: str) -> None:
    history = conversation_store.get(user_id, [])
    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": assistant_msg})
    if len(history) > MAX_HISTORY_MESSAGES:
        history = history[-MAX_HISTORY_MESSAGES:]
    conversation_store[user_id] = history


def history_to_short_text(history: List[Dict[str, str]], max_chars: int = 800) -> str:
    texts = []
    for m in history[-8:]:
        role = m["role"]
        prefix = "User: " if role == "user" else "Assistant: "
        texts.append(prefix + m["content"])
    joined = "\n".join(texts)
    return joined[-max_chars:] if len(joined) > max_chars else joined

# -------------------------------------------------
# 7. Simple language detection
# -------------------------------------------------
def detect_language(text: str) -> str:
    # Chinese characters range
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff":
            return "zh"

    vi_keywords = [
        " không ", " ko ", "nhưng", "vì", "nên",
        "em ", "anh ", "chị ", "cảm", "buồn", "lo ",
        "lo lắng", "căng thẳng", "mệt", "bạn bè",
        "trường", "đại học",
    ]
    vi_diacritics = "ăâđêôơưáàảãạấầẩẫậéèẻẽẹóòỏõọúùủũụýỳỷỹỵ"

    lowered = " " + text.lower() + " "
    if any(k in lowered for k in vi_keywords) or any(ch in vi_diacritics for ch in text.lower()):
        return "vi"

    return "en"

# -------------------------------------------------
# 8. Agent helpers
# -------------------------------------------------
def run_emotion_agent(user_message: str, history_text: str) -> dict:
    messages = [
        {"role": "system", "content": EMOTION_AGENT_PROMPT},
        {
            "role": "user",
            "content": (
                "Conversation context (short):\n"
                f"{history_text}\n\n"
                "Latest student message:\n"
                f"{user_message}\n\n"
                "Return JSON only."
            ),
        },
    ]
    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        temperature=0.2,
        max_tokens=300,
    )
    raw = completion.choices[0].message.content.strip()
    try:
        return json.loads(raw)
    except Exception:
        return {
            "primary_emotion": "unclear",
            "intensity": 5,
            "topics": [],
            "summary": "Could not parse emotion JSON.",
        }


def run_coach_agent(
    user_message: str,
    history_text: str,
    emotion_info: dict,
    lang_code: str,
) -> str:
    emotion_summary = json.dumps(emotion_info, ensure_ascii=False)
    lang_name = {"vi": "Vietnamese", "en": "English", "zh": "Chinese"}.get(
        lang_code, "English"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": COACH_AGENT_PROMPT},
        {
            "role": "system",
            "content": (
                f"The language code for the student's latest message is '{lang_code}', "
                f"which means you MUST reply entirely in {lang_name} only."
            ),
        },
        {
            "role": "user",
            "content": (
                "Conversation context (short):\n"
                f"{history_text}\n\n"
                "Emotion analysis JSON:\n"
                f"{emotion_summary}\n\n"
                "Student's latest message (reply ONLY in the same language):\n"
                f"{user_message}\n\n"
                "Now write a helpful, CBT-informed reply following the structure."
            ),
        },
    ]
    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        temperature=0.6,
        max_tokens=900,
    )
    return completion.choices[0].message.content.strip()


def run_safety_agent(user_message: str, drafted_reply: str) -> dict:
    messages = [
        {"role": "system", "content": SAFETY_AGENT_PROMPT},
        {
            "role": "user",
            "content": (
                "Student's latest message:\n"
                f"{user_message}\n\n"
                "Draft assistant reply:\n"
                f"{drafted_reply}\n\n"
                "Return JSON only."
            ),
        },
    ]
    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        temperature=0.2,
        max_tokens=300,
    )
    raw = completion.choices[0].message.content.strip()
    try:
        return json.loads(raw)
    except Exception:
        return {
            "risk_level": "none",
            "should_override": False,
            "safety_message": "",
        }


def run_supervisor_agent(
    user_message: str,
    final_reply: str,
    emotion_info: dict,
    safety_info: dict,
) -> dict:
    payload = {
        "student_message": user_message,
        "assistant_reply": final_reply,
        "emotion": emotion_info,
        "safety": safety_info,
    }

    messages = [
        {"role": "system", "content": SUPERVISOR_AGENT_PROMPT},
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False),
        },
    ]

    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        temperature=0.1,
        max_tokens=400,
    )
    raw = completion.choices[0].message.content.strip()
    try:
        return json.loads(raw)
    except Exception:
        return {
            "empathy_score": 3,
            "clarity_score": 3,
            "cbt_structure_score": 3,
            "safety_score": 4,
            "overall_helpfulness": 3,
            "comments": "Parsing error – default neutral scores.",
            "flags": ["parse_error"],
        }

# -------------------------------------------------
# 9. Health endpoints
# -------------------------------------------------
@app.get("/")
async def root():
    return {"status": "ok", "message": "Wellbeing agent backend is running."}


@app.get("/health")
async def health():
    return {"status": "healthy"}

# -------------------------------------------------
# 10. Main chat endpoint
# -------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest):
    try:
        user_id = (payload.user_id or "").strip() or "anonymous"
        user_message = (payload.message or "").strip()
        condition = (payload.condition or "").strip() or None

        if not user_message:
            raise HTTPException(status_code=400, detail="Message must not be empty.")

        # 1) detect language
        lang_code = detect_language(user_message)

        # 2) history
        history = get_user_history(user_id)
        history_text = history_to_short_text(history)

        # 3) emotion agent
        emotion_info = run_emotion_agent(user_message, history_text)

        # 4) coach agent
        drafted_reply = run_coach_agent(
            user_message=user_message,
            history_text=history_text,
            emotion_info=emotion_info,
            lang_code=lang_code,
        )

        # 5) safety agent
        safety_info = run_safety_agent(user_message, drafted_reply)

        final_reply = drafted_reply
        risk_level = safety_info.get("risk_level", "none")
        should_override = safety_info.get("should_override", False)
        safety_message = (safety_info.get("safety_message") or "").strip()

        if should_override and safety_message:
            final_reply = safety_message
        elif risk_level in ("moderate", "high") and safety_message:
            final_reply = drafted_reply + "\n\n" + safety_message

        # 6) supervisor agent
        supervisor_info = run_supervisor_agent(
            user_message=user_message,
            final_reply=final_reply,
            emotion_info=emotion_info,
            safety_info=safety_info,
        )

        # 7) update history
        append_to_history(user_id, user_message, final_reply)

        # 8) log for research
        log_interaction(
            user_id=user_id,
            condition=condition,
            lang_code=lang_code,
            user_message=user_message,
            assistant_reply=final_reply,
            emotion_info=emotion_info,
            safety_info=safety_info,
            supervisor_info=supervisor_info,
        )

        return ChatResponse(reply=final_reply)

    except HTTPException:
        raise
    except Exception as e:
        print("Error in /chat:", repr(e))
        raise HTTPException(status_code=500, detail="Internal server error.")

# -------------------------------------------------
# 11. Export messages as CSV (for research)
# -------------------------------------------------
@app.get("/export/messages")
async def export_messages():
    """
    Export toàn bộ log dưới dạng CSV để phân tích (NVivo / Python / R).
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                timestamp,
                user_id,
                condition,
                lang_code,
                user_message,
                assistant_reply,
                emotion_json,
                safety_json,
                supervisor_json
            FROM messages
            ORDER BY id ASC
            """
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    header = [
        "timestamp",
        "user_id",
        "condition",
        "lang_code",
        "user_message",
        "assistant_reply",
        "emotion_json",
        "safety_json",
        "supervisor_json",
    ]

    lines = [",".join(header)]
    for r in rows:
        # Escape dấu phẩy + xuống dòng bằng cách bọc trong dấu ngoặc kép
        formatted = []
        for value in r:
            text = "" if value is None else str(value)
            text = text.replace('"', '""')
            formatted.append(f'"{text}"')
        lines.append(",".join(formatted))

    csv_data = "\n".join(lines)
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="wellbeing_messages.csv"'},
    )
