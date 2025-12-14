# orchestrator.py — V12.2 (University of Adelaide INTERNAL assistant)

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("wellbeing-orchestrator")


# -----------------------------
# Meta prefix (backward compatible)
# -----------------------------
META_PREFIX_RE = re.compile(
    r"^\[(?:lang=(?P<lang>[a-z]{2});)?(?:profile_type=(?P<ptype>[^;\]]+);)?(?:profile_region=(?P<pregion>[^;\]]+))?\]\s*",
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


# -----------------------------
# Faculty + stress helpers
# -----------------------------
STRESS_HINTS = [
    "stress", "stressed", "overwhelmed", "burnout",
    "lo lắng", "khủng hoảng", "trầm cảm",
    "mất ngủ", "kiệt sức", "panic",
]


def _stress_level_hint(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in STRESS_HINTS)


def _uoa_faculty_bucket(message: str) -> str:
    t = (message or "").lower()

    if any(x in t for x in ["engineering", "science", "computer", "it", "kỹ thuật", "khoa học"]):
        return "SET"
    if any(x in t for x in ["health", "medical", "medicine", "nursing", "y khoa"]):
        return "HMS"
    if any(x in t for x in ["education", "business", "law", "arts", "kinh tế", "luật"]):
        return "ABLE"
    return "UNKNOWN"


# -----------------------------
# Context model
# -----------------------------
@dataclass
class StudentContext:
    student_id: str
    language: str = "vi"
    profile_type: str = "domestic"
    profile_region: str = "au"
    faculty: str = "UNKNOWN"
    university: str = "University of Adelaide"   # 🔒 FIXED


# -----------------------------
# Orchestrator
# -----------------------------
class Orchestrator:
    def __init__(self, model_id: str, client: Any):
        self.model_id = model_id
        self.client = client

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

    # 🔒 SYSTEM PROMPT vFinal — INTERNAL UoA ASSISTANT
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
            "- NEVER generalise using phrases such as "
            "'ở các trường đại học', 'at universities in Australia', "
            "'many universities'.\n"
            "- ALWAYS prioritise University of Adelaide services first.\n\n"

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

        is_stress = _stress_level_hint(cleaned_message)

        system_prompt = self._system_prompt(ctx, is_stress)

        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        for m in hist:
            if m["role"] != "system":
                messages.append(m)
        messages.append({"role": "user", "content": cleaned_message})

        reply = self._call_llm(messages)
        return reply or "Mình ở đây với bạn. Bạn có thể chia sẻ thêm không?"
