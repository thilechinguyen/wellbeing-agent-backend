# orchestrator.py

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from agents import CBTAgent

logger = logging.getLogger("wellbeing-orchestrator")


META_PREFIX_RE = re.compile(
    r"^\[(?:lang=(?P<lang>[a-z]{2});)?(?:profile_type=(?P<ptype>[^;\]]+);)?(?:profile_region=(?P<pregion>[^\]]+))?\]\s*",
    re.IGNORECASE,
)


def _safe_str(x: Any) -> str:
    return "" if x is None else str(x)


def _clamp_history(history: Any, max_turns: int = 12) -> List[Dict[str, str]]:
    if not isinstance(history, list):
        return []

    cleaned: List[Dict[str, str]] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        role = _safe_str(item.get("role", "")).strip()
        content = _safe_str(item.get("content", "")).strip()
        if role not in ("user", "assistant", "system"):
            continue
        if not content:
            continue
        cleaned.append({"role": role, "content": content})

    return cleaned[-max_turns:]


def _extract_meta_from_message(user_message: str) -> Tuple[Dict[str, str], str]:
    m = META_PREFIX_RE.match(user_message or "")
    if not m:
        return {}, user_message

    meta = {
        "language": (m.group("lang") or "").lower().strip(),
        "profile_type": (m.group("ptype") or "").strip(),
        "profile_region": (m.group("pregion") or "").strip(),
    }
    meta = {k: v for k, v in meta.items() if v}
    stripped = (user_message[m.end():] if user_message else "").lstrip()
    return meta, stripped


STRESS_HINTS = [
    "stress", "stressed", "overwhelmed", "burnout",
    "lo lắng", "khủng hoảng", "trầm cảm",
    "mất ngủ", "kiệt sức", "panic",
    "anxious", "anxiety", "hopeless", "worthless",
    "tệ quá", "vô dụng", "thất bại", "không làm được",
]

CBT_HINTS = [
    "mọi thứ đều tệ",
    "em vô dụng",
    "mình vô dụng",
    "tôi vô dụng",
    "không ai thích mình",
    "ai cũng giỏi hơn mình",
    "mình là đồ thất bại",
    "mình thất bại",
    "nếu trượt thì xong đời",
    "nếu fail thì hết rồi",
    "tất cả là lỗi của mình",
    "mình không làm được gì ra hồn",
    "i am worthless",
    "i am a failure",
    "everything is ruined",
    "everything is my fault",
    "nobody likes me",
    "i can't do anything right",
    "if i fail it is over",
    "i'm not good enough",
]

CRISIS_HINTS = [
    "tự tử",
    "muốn chết",
    "không muốn sống",
    "tự hại",
    "hại bản thân",
    "muốn biến mất",
    "suicide",
    "kill myself",
    "self-harm",
    "hurt myself",
    "don't want to live",
    "not safe",
]


def _stress_level_hint(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in STRESS_HINTS)


def _needs_cbt_agent(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in CBT_HINTS)


def _is_crisis(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in CRISIS_HINTS)


def _uoa_faculty_bucket(message: str) -> str:
    t = (message or "").lower()

    if any(x in t for x in ["engineering", "science", "computer", "it", "kỹ thuật", "khoa học"]):
        return "SET"
    if any(x in t for x in ["health", "medical", "medicine", "nursing", "y khoa"]):
        return "HMS"
    if any(x in t for x in ["education", "business", "law", "arts", "kinh tế", "luật"]):
        return "ABLE"
    return "UNKNOWN"


@dataclass
class StudentContext:
    student_id: str
    language: str = "vi"
    profile_type: str = "domestic"
    profile_region: str = "au"
    faculty: str = "UNKNOWN"
    university: str = "University of Adelaide"


class Orchestrator:
    def __init__(self, model_id: str, client: Any):
        self.model_id = model_id
        self.client = client
        self.cbt_agent = CBTAgent(model_id=model_id, client=client)

    def _build_student_context(
        self,
        student_id: str,
        profile_type: Optional[str],
        profile_region: Optional[str],
        meta: Dict[str, str],
        user_message: str,
    ) -> StudentContext:
        lang = (meta.get("language") or "vi").lower()
        if lang not in ("vi", "en", "zh"):
            lang = "vi"

        ptype = (meta.get("profile_type") or profile_type or "domestic").lower()
        preg = (meta.get("profile_region") or profile_region or "au").lower()

        faculty = _uoa_faculty_bucket(user_message)

        return StudentContext(
            student_id=student_id,
            language=lang,
            profile_type=ptype,
            profile_region=preg,
            faculty=faculty,
        )

    def _system_prompt(self, ctx: StudentContext, is_stress: bool) -> str:
        tone = (
            "Tone: extra gentle, validating, calm. "
            "Reflect feelings first. Ask ONE short question. "
            "Offer 2–4 small, doable next steps."
            if is_stress
            else
            "Tone: warm, supportive, practical. Ask ONE clarifying question if needed."
        )

        return (
            "You are xChatbot, an INTERNAL wellbeing and student-support assistant "
            "working for the University of Adelaide (Australia).\n\n"

            "Your users are FIRST-YEAR undergraduate students at the University of Adelaide.\n\n"

            "You speak AS IF you are part of the University of Adelaide’s internal "
            "student support system — not an external advisor and not a generic chatbot.\n\n"

            "CORE RULES:\n"
            "- Always speak from an internal perspective using phrases like "
            "'here at the University of Adelaide', 'at Adelaide', "
            "'our Student Services', 'our campus'.\n"
            "- Always prioritise University of Adelaide services first.\n\n"

            "LANGUAGE:\n"
            "- Respond in the student’s UI language "
            "(Vietnamese / English / Chinese).\n"
            "- You may keep official service names in English.\n\n"

            "SUPPORT STYLE:\n"
            f"- {tone}\n"
            "- Do NOT provide medical diagnoses.\n\n"

            "DEFAULT UNIVERSITY SERVICES TO REFER TO:\n"
            "- Ask Adelaide (Student Hub Central)\n"
            "- Student Care\n"
            "- University Counselling Support\n"
            "- Academic Skills & Learning Centre\n"
            "- Faculty Student Support Offices (ABLE / HMS / SET)\n"
            "- Student Finance & Scholarships\n"
            "- Student Emergency Fund\n\n"

            f"Faculty context (if known): {ctx.faculty}\n\n"

            "CRISIS SAFETY:\n"
            "- If self-harm or suicidal intent is mentioned, "
            "encourage immediate professional help and "
            "refer to University of Adelaide crisis support.\n\n"

            "FINAL RULE:\n"
            "You are not a general chatbot. "
            "You are a trusted internal assistant speaking "
            "on behalf of the University of Adelaide."
        )

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        try:
            completion = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=0.65,
                max_tokens=800,
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            logger.exception("LLM call failed: %s", e)
            return "Mình đang gặp lỗi kỹ thuật. Bạn thử gửi lại sau ít phút nhé."

    def _crisis_response(self, language: str) -> str:
        if language == "en":
            return (
                "I'm really glad you told me this. It sounds like you may not be safe right now.\n\n"
                "Please contact emergency help immediately or go to the nearest emergency department. "
                "If you can, reach out to someone near you right now — a friend, housemate, family member, or staff member — and let them stay with you.\n\n"
                "At the University of Adelaide, please also contact University Counselling Support or Student Care as soon as possible. "
                "If you want, send me your country/location and I can help you phrase a message to a trusted person right now."
            )
        elif language == "zh":
            return (
                "谢谢你愿意告诉我这些。你现在听起来可能并不安全。\n\n"
                "请立刻联系紧急帮助，或者马上去最近的急诊部门。"
                "如果可以，请现在就联系你身边一个可信任的人，让对方陪着你。\n\n"
                "在阿德莱德大学这边，也请尽快联系 University Counselling Support 或 Student Care。"
            )
        else:
            return (
                "Cảm ơn bạn vì đã nói ra điều này. Nghe như lúc này bạn có thể đang không an toàn.\n\n"
                "Bạn hãy liên hệ hỗ trợ khẩn cấp ngay hoặc đến khoa cấp cứu gần nhất nếu có thể. "
                "Nếu được, hãy nhắn ngay cho một người thật ở gần bạn lúc này như bạn bè, người ở cùng nhà, người thân hoặc staff để họ ở cạnh bạn.\n\n"
                "Ở University of Adelaide, bạn cũng nên liên hệ University Counselling Support hoặc Student Care càng sớm càng tốt. "
                "Nếu muốn, mình có thể giúp bạn soạn ngay một tin nhắn ngắn để gửi cho người bạn tin tưởng."
            )

    def run(
        self,
        student_id: str,
        user_message: str,
        history: Optional[List[Dict[str, str]]] = None,
        profile_type: Optional[str] = None,
        profile_region: Optional[str] = None,
    ) -> str:
        meta, cleaned_message = _extract_meta_from_message(user_message or "")
        cleaned_message = cleaned_message.strip()

        hist = _clamp_history(history or [])

        ctx = self._build_student_context(
            student_id=student_id,
            profile_type=profile_type,
            profile_region=profile_region,
            meta=meta,
            user_message=cleaned_message,
        )

        if _is_crisis(cleaned_message):
            return self._crisis_response(ctx.language)

        if _needs_cbt_agent(cleaned_message):
            try:
                return self.cbt_agent.run(
                    user_message=cleaned_message,
                    history=hist,
                    language=ctx.language,
                )
            except Exception as e:
                logger.exception("CBT agent failed, fallback to default LLM: %s", e)

        is_stress = _stress_level_hint(cleaned_message)
        system_prompt = self._system_prompt(ctx, is_stress)

        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        for m in hist:
            if m["role"] != "system":
                messages.append(m)
        messages.append({"role": "user", "content": cleaned_message})

        reply = self._call_llm(messages)
        return reply or "Mình ở đây với bạn. Bạn có thể chia sẻ thêm không?"