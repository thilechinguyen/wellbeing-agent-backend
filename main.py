import os
import json
import io
import csv
from datetime import datetime, timezone
from typing import Dict, List, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import requests

# -------------------------------------------------
# 1. Load environment & init Groq client (OpenAI-compatible)
# -------------------------------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set in environment or .env file")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-70b-versatile"


def groq_chat(messages: List[Dict[str, str]],
              temperature: float = 0.6,
              max_tokens: int = 800) -> str:
    """
    Helper function to call Groq in OpenAI-compatible chat format.
    """
    try:
        resp = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": GROQ_MODEL,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=40,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print("Groq error:", repr(e))
        raise HTTPException(status_code=500, detail="LLM engine error (Groq)." )


# -------------------------------------------------
# 2. System prompt CBT + wellbeing + Uni of Adelaide (giữ bản gốc)
#    (hiện tại không dùng trực tiếp, nhưng giữ lại để tái sử dụng sau)
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

1. EMOTIONAL VALIDATION (2–4 sentences)
   - Acknowledge and normalise the student's emotions.
   - Show that their feelings make sense in context.

2. CBT-STYLE REFLECTION (2–4 sentences)
   - Briefly reflect the situation, the possible thoughts, and how those thoughts may affect their feelings and behaviour.
   - Use simple language, not technical jargon.

3. COGNITIVE REFRAME (1–3 sentences)
   - Offer 1–2 more balanced, compassionate ways of looking at the situation.
   - Phrase them as suggestions or possibilities, not as forced positivity.

4. PRACTICAL STEPS / BEHAVIOURAL EXPERIMENTS
   - Provide a bullet list of 2–4 concrete actions they can try in the next 24–72 hours.
   - Make these actions very small, realistic, and specific (e.g., 'write down your main worry and one alternative explanation', 'send one message to a classmate', 'take a 5-minute walk and notice your breathing').

5. HELP-SEEKING ENCOURAGEMENT
   - End with 1–3 sentences gently encouraging them to talk to supportive people (friends, family, mentors, university services).
   - Remind them that it is okay to ask for help and that they do not have to handle everything alone.

SAFETY AND CRISIS
You must stay within a non-clinical, supportive role.

If the student mentions:
- suicidal thoughts,
- self-harm,
- wanting to die,
- harming others,
- or a very severe crisis,

then you MUST:
- Clearly say that you are not a crisis service or a substitute for professional help.
- Encourage them to immediately contact local emergency services, crisis hotlines, or university counselling.
- Keep your message short, calm, supportive, and focused on helping them reach real-world support.
- Avoid giving detailed instructions or advice on self-harm or suicide.

ADDITIONAL SUPPORT INFORMATION
If the student expresses a need for help, you may refer them to:
- The University of Adelaide Student Health & Wellbeing website: https://www.adelaide.edu.au/student/wellbeing/
  This site contains self-help resources, tips for stress, anxiety, exam support and more.
- The University of Adelaide Counselling Service (Wellbeing Hub):
    Phone: +61 8 8313 5663
    Email: counselling.centre@adelaide.edu.au
- The University of Adelaide Support for Students page: https://www.adelaide.edu.au/student/support/

You should encourage the student with a sentence like:
'If you feel your wellbeing is seriously affected or you think you need to talk to someone, you can visit these links or contact the counselling centre. You are not alone.'

UNIVERSITY-SPECIFIC SUPPORT (IMPORTANT)
When the student mentions difficulties related to:
- tuition fees,
- overdue payments,
- financial stress,
- money worries,
- visa concerns tied to payments,
- or challenges specific to international students,

you may gently guide them to official support channels from the University of Adelaide. Do not force referrals; offer them as helpful options.

Use the following verified information:

1. FINANCIAL SUPPORT (Domestic + International Students)
   - Financial hardship support, payment plan options, grants and emergency assistance:
     https://www.adelaide.edu.au/student/finance/
   - Student finance enquiries email:
     studentfinance@adelaide.edu.au

2. INTERNATIONAL STUDENT SUPPORT
   - International Student Support homepage:
     https://www.adelaide.edu.au/student/international/
   - Contact email:
     iss@adelaide.edu.au
   - Support includes: enrolment issues, CoE/visa concerns, wellbeing support, financial difficulties, and navigating university processes.

When referring to these services:
- keep a warm, supportive tone,
- do not imply that the student must contact them,
- frame it as 'you might find it helpful to reach out…',
- remind them they are not alone and help is available.

Do NOT give legal or migration advice. Refer students only to official university services or to Migration Agents where appropriate.

If the student shows severe distress (e.g., panic, extreme overwhelm):
- acknowledge feelings first,
- then offer these support options as possible next steps.

STYLE
- Use warm, simple, non-judgemental language.
- Avoid clinical terminology and diagnoses.
- Do not talk about CBT theory explicitly unless the student asks. Show CBT through your questions and suggestions.
- Keep answers focused and not too long (about 3–6 short paragraphs plus bullets).
"""


# -------------------------------------------------
# 2b. System prompt cho từng AGENT (bản multi-agent cũ của ông)
# -------------------------------------------------

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
- some short context from the conversation so far.

Your job:
- Follow the RESPONSE STRUCTURE from the main SYSTEM_PROMPT (validation, CBT reflection, reframe, practical steps, help-seeking).
- Keep the tone warm, gentle, non-clinical.
- You can use bullet points for practical steps.
- You can respond in the same language that the student uses (Vietnamese, English, or Chinese), based on their latest message.

Output:
- A natural language reply to the student (no JSON), ready to be sent.
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
    - safety_message can be an empty string "".
- If risk_level == "moderate":
    - should_override = false
    - safety_message = 1–3 sentences gently encouraging them to seek support (friends, family, university counselling, but not emergency).
- If risk_level == "high":
    - should_override = true
    - safety_message = a short, clear message saying the bot is not a crisis service and they must reach emergency services or crisis hotlines immediately.

Respond ONLY with valid JSON. No extra text.
"""


# -------------------------------------------------
# 3. In-memory conversation store (per user) + research log
# -------------------------------------------------
conversation_store: Dict[str, List[Dict[str, str]]] = {}
MAX_HISTORY_MESSAGES = 12  # chỉ giữ ~6 lượt hỏi–đáp gần nhất

research_log: List[Dict[str, Any]] = []  # để export CSV cho nghiên cứu


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
    for m in history[-8:]:  # chỉ lấy vài lượt gần nhất
        role = m["role"]
        prefix = "User: " if role == "user" else "Assistant: "
        texts.append(prefix + m["content"])
    joined = "\n".join(texts)
    if len(joined) > max_chars:
        return joined[-max_chars:]
    return joined


# -------------------------------------------------
# 4. Multi-agent helpers (Groq-based)
# -------------------------------------------------
def run_emotion_agent(user_message: str, history_text: str) -> dict:
    """
    Gọi EMOTION_AGENT để phân tích cảm xúc, chủ đề, tóm tắt.
    """
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

    raw = groq_chat(messages, temperature=0.2, max_tokens=300)
    try:
        data = json.loads(raw)
        return data
    except Exception:
        # Nếu model trả về JSON lỗi, fallback đơn giản
        return {
            "primary_emotion": "unclear",
            "intensity": 5,
            "topics": [],
            "summary": "Could not parse emotion JSON.",
        }


def run_coach_agent(user_message: str, history_text: str, emotion_info: dict) -> str:
    """
    Gọi COACH_AGENT để soạn câu trả lời chính dựa trên phân tích cảm xúc.
    """
    emotion_summary = json.dumps(emotion_info, ensure_ascii=False)

    messages = [
        {"role": "system", "content": COACH_AGENT_PROMPT},
        {
            "role": "user",
            "content": (
                "Conversation context (short):\n"
                f"{history_text}\n\n"
                "Emotion analysis JSON:\n"
                f"{emotion_summary}\n\n"
                "Student's latest message:\n"
                f"{user_message}\n\n"
                "Now write a helpful, CBT-informed reply following the structure."
            ),
        },
    ]

    reply = groq_chat(messages, temperature=0.6, max_tokens=900)
    return reply


def run_safety_agent(user_message: str, drafted_reply: str) -> dict:
    """
    Gọi SAFETY_AGENT để kiểm tra nguy cơ.
    """
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

    raw = groq_chat(messages, temperature=0.2, max_tokens=300)
    try:
        data = json.loads(raw)
        return data
    except Exception:
        # Nếu parse lỗi, coi như không có nguy cơ
        return {
            "risk_level": "none",
            "should_override": False,
            "safety_message": "",
        }


# -------------------------------------------------
# 5. FastAPI models & app
# -------------------------------------------------
class ChatRequest(BaseModel):
    user_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str


app = FastAPI(title="Wellbeing Agent API (Groq multi-agent)", version="0.5.0")

# Cho phép gọi từ bất cứ frontend nào (tạm thời cho dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------
# 6. Health check
# -------------------------------------------------
@app.get("/")
async def root():
    return {"status": "ok", "message": "Wellbeing agent backend (Groq) is running."}


@app.get("/health")
async def health():
    return {"status": "healthy"}


# -------------------------------------------------
# 7. Main chat endpoint (có memory + logging nghiên cứu)
# -------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest):
    try:
        user_id = payload.user_id.strip() or "anonymous"
        user_message = payload.message.strip()

        if not user_message:
            raise HTTPException(status_code=400, detail="Message must not be empty.")

        # Lấy lịch sử hội thoại của user
        history = get_user_history(user_id)

        # Tạo bản tóm tắt ngắn từ history cho agent dùng
        history_text = history_to_short_text(history)

        # ---------- Agent 1: Emotion Analyzer ----------
        emotion_info = run_emotion_agent(user_message, history_text)

        # ---------- Agent 2: Wellbeing Coach ----------
        drafted_reply = run_coach_agent(user_message, history_text, emotion_info)

        # ---------- Agent 3: Safety & Risk Check ----------
        safety_info = run_safety_agent(user_message, drafted_reply)

        final_reply = drafted_reply

        risk_level = safety_info.get("risk_level", "none")
        should_override = safety_info.get("should_override", False)
        safety_message = safety_info.get("safety_message", "").strip()

        if should_override and safety_message:
            final_reply = safety_message
        else:
            if risk_level in ("moderate", "high") and safety_message:
                final_reply = drafted_reply + "\n\n" + safety_message

        # Lưu lại vào history (cho体验 chat)
        append_to_history(user_id, user_message, final_reply)

        # Ghi log cho research
        research_log.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "user_id": user_id,
                "user_message": user_message,
                "assistant_reply": final_reply,
                "emotion_info": emotion_info,
                "safety_info": safety_info,
            }
        )

        return ChatResponse(reply=final_reply)

    except HTTPException:
        raise
    except Exception as e:
        print("Error in /chat:", repr(e))
        raise HTTPException(status_code=500, detail="Internal server error.")


# -------------------------------------------------
# 8. Export research log as CSV
# -------------------------------------------------
@app.get("/export/messages")
async def export_messages():
    """
    Trả về toàn bộ research_log dưới dạng CSV để tải về.
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer)

    writer.writerow(
        [
            "timestamp",
            "user_id",
            "user_message",
            "assistant_reply",
            "primary_emotion",
            "intensity",
            "topics",
            "risk_level",
        ]
    )

    for row in research_log:
        emo = row.get("emotion_info") or {}
        safety = row.get("safety_info") or {}
        writer.writerow(
            [
                row.get("timestamp", ""),
                row.get("user_id", ""),
                row.get("user_message", "").replace("\n", " "),
                row.get("assistant_reply", "").replace("\n", " "),
                emo.get("primary_emotion", ""),
                emo.get("intensity", ""),
                ", ".join(emo.get("topics", []) or []),
                safety.get("risk_level", ""),
            ]
        )

    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=messages_log.csv"},
    )
