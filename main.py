import os
import json
import csv
import io
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq

# -------------------------------------------------
# 1. Load .env & init Groq client (DeepSeek qua Groq)
# -------------------------------------------------
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise RuntimeError("GROQ_API_KEY is not set in environment or .env file")

groq_client = Groq(api_key=groq_api_key)

# Chọn model Groq (DeepSeek distill – miễn phí / tốc độ cao)
GROQ_MODEL = "deepseek-r1-distill-llama-70b"

# -------------------------------------------------
# 2. System prompts cho các agent
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
   - Make these actions very small, realistic, and specific (e.g., "write down your main worry and one alternative explanation",
     "send one message to a classmate", "take a 5-minute walk and notice your breathing").

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

ADDITIONAL SUPPORT INFORMATION (University of Adelaide)
If the student expresses a need for help, you may refer them to:
- The University of Adelaide Student Health & Wellbeing website:
  https://www.adelaide.edu.au/student/wellbeing/
- The University of Adelaide Counselling Service (Wellbeing Hub):
    Phone: +61 8 8313 5663
    Email: counselling.centre@adelaide.edu.au
- The University of Adelaide Support for Students page:
  https://www.adelaide.edu.au/student/support/

When referring to these services:
- keep a warm, supportive tone,
- do not imply that the student must contact them,
- frame it as "you might find it helpful to reach out…",
- remind them they are not alone and help is available.

Do NOT give legal or migration advice. Refer students only to official university services
or to Migration Agents where appropriate.

LANGUAGE
- Always reply in the same language that the student uses in their latest message.
- If the language looks Vietnamese, answer in Vietnamese.
- If the language looks English, answer in English.
- If the language looks Chinese, answer in Chinese.
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
- optional metadata about the student's profile (e.g., domestic/international, region).

Your job:
- Follow the RESPONSE STRUCTURE from the main SYSTEM_PROMPT
  (validation, CBT reflection, reframe, practical steps, help-seeking).
- Keep the tone warm, gentle, non-clinical.
- IMPORTANT: Respond in the SAME LANGUAGE as the student's latest message
  (Vietnamese, English, or Chinese).

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
    - safety_message = 1–3 sentences gently encouraging them to seek support
      (friends, family, university counselling, but not emergency).
- If risk_level == "high":
    - should_override = true
    - safety_message = a short, clear message saying the bot is not a crisis service and
      they must reach emergency services or crisis hotlines immediately.

Respond ONLY with valid JSON. No extra text.
"""

# -------------------------------------------------
# 3. In-memory conversation store + research log
# -------------------------------------------------
conversation_store: Dict[str, List[Dict[str, str]]] = {}
MAX_HISTORY_MESSAGES = 12

research_logs: List[Dict[str, str]] = []  # để export CSV nghiên cứu

# -------------------------------------------------
# 4. FastAPI models & app
# -------------------------------------------------
class ChatRequest(BaseModel):
    user_id: str
    message: str
    student_type: Optional[str] = None   # "Domestic (Australia)", "International – SE Asia", ...
    student_region: Optional[str] = None # "Australia", "Vietnam", "Europe", ...
    language: Optional[str] = None       # cho tương lai nếu muốn gửi riêng


class ChatResponse(BaseModel):
    reply: str


app = FastAPI(
    title="Wellbeing Agent API (Groq / DeepSeek)",
    version="0.7.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev cho dễ, sau có thể khóa domain lại
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# 5. Helpers
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
    if len(joined) > max_chars:
        return joined[-max_chars:]
    return joined


def call_groq_chat(messages: List[Dict[str, str]],
                   temperature: float,
                   max_tokens: int) -> str:
    """
    Wrapper gọi Groq chat.completions.
    """
    completion = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return completion.choices[0].message.content.strip()


def run_emotion_agent(user_message: str, history_text: str, profile_text: str) -> dict:
    prompt_user = (
        "Conversation context (short):\n"
        f"{history_text}\n\n"
        "Student profile:\n"
        f"{profile_text}\n\n"
        "Latest student message:\n"
        f"{user_message}\n\n"
        "Return JSON only."
    )
    messages = [
        {"role": "system", "content": EMOTION_AGENT_PROMPT},
        {"role": "user", "content": prompt_user},
    ]
    raw = call_groq_chat(messages, temperature=0.2, max_tokens=300)

    try:
        data = json.loads(raw)
        return data
    except Exception:
        # fallback nếu JSON lỗi
        return {
            "primary_emotion": "unclear",
            "intensity": 5,
            "topics": [],
            "summary": "Could not parse emotion JSON."
        }


def run_coach_agent(
    user_message: str,
    history_text: str,
    profile_text: str,
    emotion_info: dict
) -> str:
    emotion_summary = json.dumps(emotion_info, ensure_ascii=False)
    prompt_user = (
        "Conversation context (short):\n"
        f"{history_text}\n\n"
        "Student profile metadata:\n"
        f"{profile_text}\n\n"
        "Emotion analysis JSON:\n"
        f"{emotion_summary}\n\n"
        "Student's latest message:\n"
        f"{user_message}\n\n"
        "Now write a helpful CBT-informed reply following the structure."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": COACH_AGENT_PROMPT},
        {"role": "user", "content": prompt_user},
    ]
    reply = call_groq_chat(messages, temperature=0.6, max_tokens=900)
    return reply


def run_safety_agent(user_message: str, drafted_reply: str) -> dict:
    prompt_user = (
        "Student's latest message:\n"
        f"{user_message}\n\n"
        "Draft assistant reply:\n"
        f"{drafted_reply}\n\n"
        "Return JSON only."
    )
    messages = [
        {"role": "system", "content": SAFETY_AGENT_PROMPT},
        {"role": "user", "content": prompt_user},
    ]
    raw = call_groq_chat(messages, temperature=0.1, max_tokens=300)
    try:
        data = json.loads(raw)
        return data
    except Exception:
        return {
            "risk_level": "none",
            "should_override": False,
            "safety_message": ""
        }


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
# 7. Main chat endpoint
# -------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest):
    try:
        user_id = payload.user_id.strip() or "anonymous"
        user_message = payload.message.strip()

        if not user_message:
            raise HTTPException(status_code=400, detail="Message must not be empty.")

        history = get_user_history(user_id)
        history_text = history_to_short_text(history)

        profile_text = (
            f"student_type={payload.student_type or 'unknown'}, "
            f"student_region={payload.student_region or 'unknown'}"
        )

        # -------- Agent 1: Emotion --------
        emotion_info = run_emotion_agent(user_message, history_text, profile_text)

        # -------- Agent 2: Coach --------
        drafted_reply = run_coach_agent(
            user_message=user_message,
            history_text=history_text,
            profile_text=profile_text,
            emotion_info=emotion_info,
        )

        # -------- Agent 3: Safety --------
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

        # Lưu history
        append_to_history(user_id, user_message, final_reply)

        # Log cho nghiên cứu
        research_logs.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "user_id": user_id,
                "student_type": payload.student_type or "",
                "student_region": payload.student_region or "",
                "user_message": user_message,
                "assistant_reply": final_reply,
                "emotion_json": json.dumps(emotion_info, ensure_ascii=False),
                "risk_level": risk_level,
            }
        )

        return ChatResponse(reply=final_reply)

    except HTTPException:
        raise
    except Exception as e:
        print("Error in /chat:", repr(e))
        raise HTTPException(status_code=500, detail="Internal server error.")


# -------------------------------------------------
# 8. Export research logs (CSV)
# -------------------------------------------------
@app.get("/export/messages")
async def export_messages():
    """
    Xuất toàn bộ research_logs thành CSV để tải về.
    """
    if not research_logs:
        # vẫn trả file rỗng cho dễ xử lý
        headers = ["timestamp", "user_id", "student_type", "student_region",
                   "user_message", "assistant_reply", "emotion_json", "risk_level"]
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=headers)
        writer.writeheader()
    else:
        headers = list(research_logs[0].keys())
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=headers)
        writer.writeheader()
        for row in research_logs:
            writer.writerow(row)

    csv_bytes = buf.getvalue().encode("utf-8-sig")
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=wellbeing_messages.csv"},
    )


# -------------------------------------------------
# 9. Chạy local (python main.py)
# -------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=10000,
        reload=True,
    )
