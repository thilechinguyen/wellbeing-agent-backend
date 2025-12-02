import os
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq

# Local modules
from export_data import (
    load_messages,
    export_csv,
    export_pdf,
    EXPORT_DIR,
    CSV_PATH,
)

from conversation_logging import (
    init_db,
    log_turn,
    attach_export_routes,
)

# ============================================================
# 1. Load environment variables (.env)
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# ============================================================
# 2. Logging
# ============================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wellbeing-backend")

# ============================================================
# 3. Environment variables
# ============================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_ID = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

if not GROQ_API_KEY:
    raise RuntimeError("❌ GROQ_API_KEY is not set in .env")

client = Groq(api_key=GROQ_API_KEY)

ALLOW_ORIGINS = os.getenv(
    "ALLOW_ORIGINS",
    "https://wellbeing-agent-ui.onrender.com,http://localhost:3000"
).split(",")


# ============================================================
# 4. FastAPI App
# ============================================================
app = FastAPI(
    title="Wellbeing Agent Backend (Groq Llama3.1-8B)"
)

init_db()
attach_export_routes(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOW_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 5. Routes: Export CSV + ZIP PDF
# ============================================================
@app.get("/export/csv")
def export_csv_endpoint():
    messages = load_messages()
    if not messages:
        raise HTTPException(404, "No data available")

    export_csv(messages)

    return FileResponse(
        CSV_PATH,
        media_type="text/csv",
        filename="wellbeing_conversations.csv",
    )


@app.get("/export/report")
def export_report_endpoint():
    """
    Sinh PDF (1 file/user_id) → trả về 1 file ZIP.
    """
    messages = load_messages()
    if not messages:
        raise HTTPException(404, "No messages found.")

    export_pdf(messages)

    reports_dir = EXPORT_DIR / "reports"
    if not reports_dir.exists():
        raise HTTPException(500, "Reports directory missing.")

    # Tạo file zip tạm
    import tempfile, shutil
    tmp_dir = tempfile.mkdtemp()
    zip_path = Path(tmp_dir) / "wellbeing_reports.zip"

    shutil.make_archive(
        base_name=str(zip_path.with_suffix("")),
        format="zip",
        root_dir=str(reports_dir),
    )

    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename="wellbeing_reports.zip",
    )


# ============================================================
# 6. Models
# ============================================================
class ChatRequest(BaseModel):
    user_id: str
    message: str

    # Metadata được UI nhúng vào message, nhưng ta hỗ trợ thêm:
    session_id: Optional[str] = None
    turn_index: Optional[int] = None
    lang_code: Optional[str] = "vi"

    # Lưu cho DB
    student_type: Optional[str] = "domestic"
    student_region: Optional[str] = "au"


class ChatResponse(BaseModel):
    reply: str
    emotion_summary: str


# ============================================================
# 7. Prompts
# ============================================================
EMOTION_ANALYST_PROMPT = """
You are an EMOTION ANALYST.
Summarise in 3–4 short sentences:
- Emotions
- Intensity
- Themes
"""

CBT_COACH_PROMPT = """
You are a CBT wellbeing companion for first-year University of Adelaide students.

Emotion summary:
{emotion_summary}

Rules:
- Warm, validating, simple language.
- Match student’s language.
- Provide 2–4 realistic next steps.
- Include UoA supports only if appropriate.
- If distress → mention crisis line 1300 167 654 or text 0488 884 197.
"""

SAFETY_REVIEW_PROMPT = """
Rewrite only if needed to make the reply safer.
Never mention being an AI.
Always keep the same language as the reply.
Return ONLY the final text.
"""


# ============================================================
# 8. Helper: Call Groq
# ============================================================
def call_groq(system, user):
    try:
        completion = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.4,
            max_tokens=900,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        logger.exception(e)
        raise HTTPException(500, f"Groq error: {e}")


# ============================================================
# 9. Health Check
# ============================================================
@app.get("/health")
def health():
    test = call_groq("Say 'ok'.", "Say ok.")
    return {"status": "ok", "model": MODEL_ID, "llm": test}


# ============================================================
# 10. Main Chat Orchestrator
# ============================================================
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):

    if not req.message.strip():
        raise HTTPException(400, "Empty message")

    # ---- Build profile text ----
    profile = (
        f"Student type: {req.student_type}. "
        f"Region: {req.student_region}. "
        "First-year University of Adelaide student."
    )

    # ---- 1) Emotion Analyst ----
    emotion_summary = call_groq(
        EMOTION_ANALYST_PROMPT,
        f"{profile}\n\nMessage:\n{req.message}"
    )

    # ---- 2) CBT Coach ----
    coach_prompt = CBT_COACH_PROMPT.format(
        emotion_summary=emotion_summary
    )

    candidate_reply = call_groq(
        coach_prompt,
        f"{profile}\n\nMessage:\n{req.message}"
    )

    # ---- 3) Safety Filter ----
    final_reply = call_groq(
        SAFETY_REVIEW_PROMPT,
        f"Student message:\n{req.message}\n\nCandidate reply:\n{candidate_reply}"
    )

    # ---- 4) Save to DB ----
    log_turn(
        session_id=req.session_id,
        turn_index=req.turn_index,
        user_id=req.user_id,
        user_text=req.message,
        agent_text=final_reply,
        emotion=emotion_summary,
        safety=None,
        supervisor=None,
        lang_code=req.lang_code,
    )

    return ChatResponse(
        reply=final_reply,
        emotion_summary=emotion_summary,
    )
