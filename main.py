import os
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq


# ---------------------------------------------------------
# 1. Logging
# ---------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wellbeing-backend")

# ---------------------------------------------------------
# 2. Environment
# ---------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set in environment")

MODEL_ID = "llama-3.1-8b-instant"

client = Groq(api_key=GROQ_API_KEY)

ALLOW_ORIGINS = os.getenv(
    "ALLOW_ORIGINS",
    "https://wellbeing-agent-ui.onrender.com,http://localhost:3000"
).split(",")

# ---------------------------------------------------------
# 3. FastAPI app + CORS
# ---------------------------------------------------------
app = FastAPI(title="Wellbeing Agent Backend (Groq + Llama 3.1 8B)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOW_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# 4. Data models
# ---------------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    student_type: Optional[str] = "Domestic (Australia)"
    student_region: Optional[str] = "Australia"
    language: Optional[str] = "vi"  # "vi" | "en" – cho phép tùy chọn nếu sau này cần


class ChatResponse(BaseModel):
    reply: str
    emotion_summary: str


# ---------------------------------------------------------
# 5. System + Agent Prompts
# ---------------------------------------------------------

SYSTEM_PROMPT = CBT_COACH_PROMPT = """
You are a CBT-style wellbeing companion for first-year University of Adelaide students.

Your role is to:
- Help students understand the link between situations → thoughts → emotions → behaviours.
- Offer small, practical “behavioural experiments”.
- Keep a warm, validating, human tone.
- Include University of Adelaide support options in a natural, contextual way—not as a list of links dumped abruptly.
- Always match the student's language (Vietnamese → reply in Vietnamese; English → reply in English).

=== RESPONSE STRUCTURE (SOFT, NATURAL CBT STYLE) ===

1) EMOTIONAL VALIDATION (2–3 sentences)
   - Acknowledge how the student feels.
   - Show empathy, normalisation, and emotional understanding.

2) CBT REFLECTION (2–3 sentences)
   - Gently reflect the situation: what might be triggering the emotion?
   - Identify possible automatic thoughts or interpretations.
   - Show how these thoughts might shape emotions/behaviour.

3) REFRAME (1–2 sentences)
   - Offer 1–2 balanced alternative ways of viewing the situation.
   - Soft wording (“perhaps”, “it might help to consider”, “one possible interpretation is…”).

4) PRACTICAL STEPS (2–4 items)
   - Suggest small actions (behavioural experiments):
       • journaling a thought
       • sending one message to a classmate
       • taking a short walk to reset attention
       • breaking a task into tiny steps
   - If relevant, naturally integrate 1–2 University of Adelaide supports:
       • Counselling Support – https://www.adelaide.edu.au/counselling
       • Wellbeing Hub – https://www.adelaide.edu.au/student/wellbeing
       • AskADEL – https://www.adelaide.edu.au/ask-adelaide/
       • International Student Support – https://international.adelaide.edu.au/student-support
       • Writing Centre – https://www.adelaide.edu.au/writingcentre
       • Maths Learning Centre – https://www.adelaide.edu.au/mathslearning/
       • AUU Clubs – https://auu.org.au/clubs/
   - DO NOT dump URLs in a block. Only include them when relevant to the student’s issue.

5) HELP-SEEKING MESSAGE (1–2 sentences)
   - Gentle reminder they’re not alone.
   - Encourage reaching out to someone trusted or appropriate UofA service.
   - For high distress: include crisis line (1300 167 654 / 0488 884 197).

=== IMPORTANT STYLE RULES ===
- Sound human, warm, supportive.
- Avoid generic self-help tone.
- No clinical or diagnostic language.
- No overwhelming number of links.
- Each reply should feel personalised to the student's specific emotion summary.

Emotion summary (from analyst):
{emotion_summary}
"""


EMOTION_ANALYST_PROMPT = """
You are an EMOTION ANALYST.

Goal:
- Read the student's message and briefly summarise:
  • main emotions (e.g. lonely, anxious, overwhelmed)
  • rough intensity (e.g. mild / moderate / strong)
  • main themes (friends, study load, exams, money, homesickness, family, etc.)

Output format (short, max 3–4 sentences, English):
- Emotions:
- Intensity:
- Themes:
"""

CBT_COACH_PROMPT = """
You are a CBT-STYLE COACH and student wellbeing companion at the University of Adelaide.

Use:
- The student's profile (type, region)
- The emotion summary from EMOTION ANALYST
- The original student message

Your reply must:
1) Start with a short empathetic validation (1–2 sentences).
2) Reflect the main emotions/themes in simple language.
3) Offer 2–4 very concrete next steps that a first-year student at UofA can actually do:
   • on-campus places (Hub Central, counselling, Writing Centre, Maths Learning Centre, clubs, peer mentoring)
   • online supports (Studiosity, wellbeing website)
4) If the student sounds very distressed or unsafe, gently mention crisis contacts 
   (1300 167 654 or text 0488 884 197) and University Counselling, but DO NOT panic or be dramatic.
5) Use a friendly, calm tone.

Write in the same language as the student's message (if Vietnamese, reply in Vietnamese; 
 if English, reply in English).

Emotion summary (from analyst):
{emotion_summary}
"""

SAFETY_REVIEW_PROMPT = """
You are a SAFETY FILTER.

You receive:
- The student's original message.
- A candidate supportive reply from a CBT-style coach.

Your job:
- Check if the candidate reply is:
  • non-judgmental
  • does NOT promise confidentiality or medical outcomes
  • does NOT give medical, psychiatric, or legal advice
  • does NOT encourage self-harm, substance abuse, or risky behaviour
- If the reply is fine, output it unchanged.
- If something is risky, rewrite the reply to be safer, 
  emphasising seeking help (University Counselling, crisis line, trusted adults, emergency services).

Always answer in the SAME language as the candidate reply.
Return ONLY the final safe reply text, no explanation.
"""


# ---------------------------------------------------------
# 6. Groq helper
# ---------------------------------------------------------
def call_groq_chat(system_prompt: str, user_prompt: str) -> str:
    """
    Call Groq ChatCompletion with a system + user prompt and return the text.
    """
    try:
        completion = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
            max_tokens=900,
        )
        return (completion.choices[0].message.content or "").strip()
    except Exception as e:
        logger.exception("Groq API error")
        raise HTTPException(status_code=500, detail=f"Groq API error: {e}")


# ---------------------------------------------------------
# 7. Health endpoint
# ---------------------------------------------------------
@app.get("/health")
def health():
    # Để frontend test nhanh xem backend có sống + model có chạy không
    try:
        test = call_groq_chat(
            SYSTEM_PROMPT,
            "Short health check: reply with exactly 'ok'.",
        )
    except HTTPException as e:
        raise e

    return {"status": "ok", "model": MODEL_ID, "llm_reply": test}


# ---------------------------------------------------------
# 8. Orchestrator endpoint /chat
# ---------------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Pipeline:
    1. Emotion analyst => emotion_summary (internal, dùng làm context).
    2. CBT coach => candidate_reply (câu trả lời chính).
    3. Safety filter => final_reply (chỉnh nếu cần).
    → Frontend chỉ nhận final_reply + emotion_summary (optional logging).
    """

    user_message = req.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Message is empty.")

    # ----- 1) Build profile text -----
    profile_text = (
        f"Student type: {req.student_type}. "
        f"Region: {req.student_region}. "
        "They are a first-year student at the University of Adelaide."
    )

    # ----- 2) Emotion Analyst (internal) -----
    emotion_input = (
        f"Student profile: {profile_text}\n\n"
        f"Student message:\n{user_message}"
    )
    emotion_summary = call_groq_chat(
        SYSTEM_PROMPT + "\n" + EMOTION_ANALYST_PROMPT,
        emotion_input,
    )

    # ----- 3) CBT Coach (main reply) -----
    cbt_prompt_filled = CBT_COACH_PROMPT.format(
        emotion_summary=emotion_summary
    )
    coach_input = (
        f"Student profile: {profile_text}\n\n"
        f"Student message:\n{user_message}"
    )
    candidate_reply = call_groq_chat(
        SYSTEM_PROMPT + "\n" + cbt_prompt_filled,
        coach_input,
    )

    # ----- 4) Safety Review (rewrite if needed) -----
    safety_input = (
        "Student message:\n"
        f"{user_message}\n\n"
        "Candidate reply from CBT coach:\n"
        f"{candidate_reply}"
    )
    final_reply = call_groq_chat(
        SAFETY_REVIEW_PROMPT,
        safety_input,
    )

    # → Quan trọng: CHỈ trả về final_reply, KHÔNG join 3 câu trả lời
    return ChatResponse(
        reply=final_reply,
        emotion_summary=emotion_summary,
    )
