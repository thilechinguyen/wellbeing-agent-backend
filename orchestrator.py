# orchestrator.py — V12.1 (UoA-aware + Faculty routing + softer CBT)

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
    """Ensure history is list[{'role','content'}], clamp to last max_turns."""
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

    if len(cleaned) > max_turns:
        cleaned = cleaned[-max_turns:]
    return cleaned


def _extract_meta_from_message(user_message: str) -> Tuple[Dict[str, str], str]:
    """
    Supports old format:
      [lang=vi;profile_type=domestic;profile_region=au] hello
    Returns (meta_dict, message_without_prefix)
    """
    m = META_PREFIX_RE.match(user_message or "")
    if not m:
        return {}, user_message

    meta = {
        "language": (m.group("lang") or "").lower().strip(),
        "profile_type": (m.group("ptype") or "").strip(),
        "profile_region": (m.group("pregion") or "").strip(),
    }
    meta = {k: v for k, v in meta.items() if v}
    stripped = (user_message[m.end() :] if user_message else "").lstrip()
    return meta, stripped


# -----------------------------
# UoA detection + faculty routing
# -----------------------------
UOA_HINTS = [
    "university of adelaide",
    "uofa",
    "adelaide.edu.au",
    "myadelaide",
    "ask adelaide",
    "hub central",
    "hubcentral",
    "student hub",
    "north terrace",
]

ACADEMIC_ADVICE_KEYWORDS = [
    # vi
    "cố vấn",
    "co van",
    "học vụ",
    "hoc vu",
    "chọn môn",
    "chon mon",
    "lộ trình",
    "lo trinh",
    "rớt môn",
    "rot mon",
    "nguy cơ rớt",
    "deadline",
    "quá tải",
    "qua tai",
    "học không kịp",
    "hoc khong kip",
    "cần hỗ trợ học",
    "can ho tro hoc",
    "support học",
    # en
    "academic advisor",
    "academic advising",
    "course plan",
    "program advice",
    "enrol",
    "enrollment",
    "failed a course",
    "struggling with study",
]

STRESS_HINTS = [
    "stress",
    "stressed",
    "overwhelmed",
    "burnout",
    "lo lắng",
    "lo au",
    "hoảng",
    "hoang",
    "khủng hoảng",
    "khung hoang",
    "trầm cảm",
    "tram cam",
    "mất ngủ",
    "mat ngu",
    "kiệt sức",
    "kiet suc",
    "panic",
]


def _looks_like_uoa(text: str) -> bool:
    t = (text or "").lower()
    return any(h in t for h in UOA_HINTS)


def _is_academic_advice_question(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in ACADEMIC_ADVICE_KEYWORDS)


def _stress_level_hint(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in STRESS_HINTS)


def _uoa_faculty_bucket(message: str) -> str:
    """
    Rough inference from message:
    - SET: sciences/engineering/tech
    - HMS: health/medical
    - ABLE: arts/business/law/econ
    """
    t = (message or "").lower()

    if any(x in t for x in ["engineering", "engineer", "kỹ thuật", "ki thuat", "science", "khoa học", "computer", "it", "set"]):
        return "SET"
    if any(x in t for x in ["health", "medical", "medicine", "nursing", "hms", "y khoa", "dược", "duoc", "điều dưỡng", "dieu duong"]):
        return "HMS"
    if any(x in t for x in ["business", "law", "economics", "arts", "able", "luật", "luat", "kinh tế", "kinh te", "commerce"]):
        return "ABLE"
    return "UNKNOWN"


def _uoa_academic_advice_reply(lang: str, faculty_bucket: str) -> str:
    """Deterministic UoA-specific reply when user asks academic advising / support."""
    faculty_line_vi = (
        f"2) **Student Success Team** theo Faculty của em (**{faculty_bucket}**) để hỗ trợ học vụ, kế hoạch học, và kết nối đúng cố vấn.\n"
        if faculty_bucket in ("SET", "HMS", "ABLE")
        else "2) **Student Success Team** theo Faculty (ABLE / HMS / SET) để hỗ trợ học vụ và kết nối đúng cố vấn.\n"
    )

    if lang == "en":
        fac = faculty_bucket if faculty_bucket != "UNKNOWN" else "ABLE / HMS / SET"
        return (
            "If you study at the **University of Adelaide**, the fastest pathway is:\n\n"
            "1) **Student Hub / Hub Central** (best starting point) – they will direct you to the right service.\n"
            f"2) **Student Success Team** for your Faculty (**{fac}**) for academic/program advice.\n"
            "3) **Academic Skills** (Writing Centre, Maths Learning Centre, PASS, English Assist) for study skills.\n"
            "4) If stress is affecting study, consider **Counselling Support**.\n\n"
            "One quick question: which subject or deadline is worrying you most right now?"
        )

    # VI default
    return (
        "Nếu em học tại **University of Adelaide**, em có thể đi theo lộ trình nhanh nhất như sau\n\n"
        "1) **Student Hub / Hub Central** (điểm bắt đầu tốt nhất) để họ chỉ đúng kênh hỗ trợ.\n"
        f"{faculty_line_vi}"
        "3) **Student Academic Skills & Support** (Writing Centre, Maths Learning Centre, PASS, English Assist…) để cải thiện kỹ năng học.\n"
        "4) Nếu stress/lo âu ảnh hưởng việc học, em có thể đặt lịch **Counselling Support**.\n\n"
        "Mình hỏi 1 câu ngắn thôi: hiện giờ em lo nhất là môn nào hoặc deadline nào?"
    )


# -----------------------------
# Context model
# -----------------------------
@dataclass
class StudentContext:
    student_id: str
    language: str = "vi"            # vi/en/zh
    profile_type: str = "unknown"   # domestic/international/unknown
    profile_region: str = "unknown" # au/sea/eu/other/unknown
    university: str = "unknown"     # "uoa" or "unknown"


# -----------------------------
# Orchestrator
# -----------------------------
class Orchestrator:
    """
    V12.1 Orchestrator:
    - UoA auto-detection + academic support fast-path
    - Faculty bucket inference (SET/HMS/ABLE)
    - Softer CBT tone when stress hints are present
    - Calls Groq chat.completions for general cases
    """

    def __init__(self, model_id: str, client: Any):
        self.model_id = model_id
        self.client = client

    def _build_student_context(
        self,
        student_id: str,
        profile_type: Optional[str],
        profile_region: Optional[str],
        meta: Dict[str, str],
        combined_text_for_uni_detect: str,
    ) -> StudentContext:
        lang = (meta.get("language") or "vi").lower()
        ptype = (meta.get("profile_type") or profile_type or "unknown").strip().lower()
        preg = (meta.get("profile_region") or profile_region or "unknown").strip().lower()

        # normalize profile type
        if ptype in ("domestic", "sv trong nuoc", "sv trong nước", "local"):
            ptype = "domestic"
        elif ptype in ("international", "sv quoc te", "sv quốc tế", "intl"):
            ptype = "international"
        elif not ptype:
            ptype = "unknown"

        # normalize region
        if preg in ("au", "australia"):
            preg = "au"
        elif preg in ("sea", "southeast asia", "south-east asia"):
            preg = "sea"
        elif preg in ("eu", "europe"):
            preg = "eu"
        elif preg in ("other",):
            preg = "other"
        elif not preg:
            preg = "unknown"

        if lang not in ("vi", "en", "zh"):
            lang = "vi"

        university = "uoa" if _looks_like_uoa(combined_text_for_uni_detect) else "unknown"

        return StudentContext(
            student_id=student_id,
            language=lang,
            profile_type=ptype,
            profile_region=preg,
            university=university,
        )

    def _system_prompt(self, ctx: StudentContext, is_stress: bool) -> str:
        # language rules
        if ctx.language == "en":
            lang_rule = "Respond in English by default (you may add 1–2 short Vietnamese clarifications if helpful)."
        elif ctx.language == "zh":
            lang_rule = "默认用中文回复（必要时可补充简短英文/越南语解释）。"
        else:
            lang_rule = "Trả lời ưu tiên bằng tiếng Việt (có thể kèm 1–2 câu tiếng Anh nếu hữu ích)."

        profile_hint = f"Student profile: type={ctx.profile_type}, region={ctx.profile_region}, university={ctx.university}."

        # softer CBT stance if stress signs
        if is_stress:
            style = (
                "Tone: extra gentle, validating, calm. "
                "Start by reflecting feelings in one sentence. "
                "Ask ONE short question. "
                "Then give 2–4 tiny next steps (10 minutes / today / this week)."
            )
        else:
            style = (
                "Tone: warm, validating, practical. "
                "Ask ONE short clarifying question when needed. "
                "Offer 2–4 concrete next steps."
            )

        return (
            "You are a non-judgmental, trauma-informed wellbeing support assistant for university students. "
            "Your job is to provide emotional support and gentle CBT-aligned guidance.\n"
            "Rules:\n"
            f"- {style}\n"
            "- Do NOT provide medical diagnosis.\n"
            "- Do NOT shame, blame, or judge the student.\n"
            "- If the user expresses self-harm intent, encourage immediate help and provide crisis resources.\n"
            f"- {lang_rule}\n"
            f"- {profile_hint}\n"
        )

    def _user_context_block(self, ctx: StudentContext) -> str:
        return (
            f"(Context) student_id={ctx.student_id}, "
            f"profile_type={ctx.profile_type}, profile_region={ctx.profile_region}, "
            f"ui_language={ctx.language}, university={ctx.university}."
        )

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        try:
            completion = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=0.7,
                max_tokens=700,
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            logger.exception("LLM call failed: %s", e)
            return (
                "Mình đang gặp lỗi khi kết nối mô hình để tạo phản hồi. "
                "Bạn thử gửi lại trong ít phút nữa nhé."
            )

    def run(
        self,
        student_id: str,
        user_message: str,
        history: Optional[List[Dict[str, str]]] = None,
        profile_type: Optional[str] = None,
        profile_region: Optional[str] = None,
    ) -> str:
        # 1) parse metadata prefix if any
        meta, cleaned_message = _extract_meta_from_message(user_message or "")
        cleaned_message = cleaned_message.strip() or (user_message or "").strip()

        # 2) normalize history
        hist = _clamp_history(history or [])

        # 3) create combined text for university detection (message + history)
        hist_text = " ".join(m.get("content", "") for m in hist if isinstance(m, dict))
        combined_text = f"{hist_text}\n{cleaned_message}"

        # 4) build context
        ctx = self._build_student_context(
            student_id=student_id,
            profile_type=profile_type,
            profile_region=profile_region,
            meta=meta,
            combined_text_for_uni_detect=combined_text,
        )

        # 5) UoA fast-path: academic advising/support
        if ctx.university == "uoa" and _is_academic_advice_question(cleaned_message):
            bucket = _uoa_faculty_bucket(cleaned_message)
            return _uoa_academic_advice_reply(ctx.language, bucket)

        # 6) stress hint → softer prompt
        is_stress = _stress_level_hint(cleaned_message)

        # 7) build prompt messages
        sys = self._system_prompt(ctx, is_stress=is_stress)
        ctx_block = self._user_context_block(ctx)

        messages: List[Dict[str, str]] = [{"role": "system", "content": sys}]
        messages.append({"role": "system", "content": ctx_block})

        # append history (skip any system role from client)
        for m in hist:
            if m["role"] == "system":
                continue
            messages.append(m)

        messages.append({"role": "user", "content": cleaned_message})

        # 8) call LLM
        reply = self._call_llm(messages)

        if not reply:
            reply = "Mình ở đây với bạn. Bạn có thể nói thêm một chút về điều đang xảy ra không?"

        return reply
