# orchestrator.py — V12 (wired, no stub)

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("wellbeing-orchestrator")


# -----------------------------
# Utilities
# -----------------------------
META_PREFIX_RE = re.compile(
    r"^\[(?:lang=(?P<lang>[a-z]{2});)?(?:profile_type=(?P<ptype>[^;\]]+);)?(?:profile_region=(?P<pregion>[^;\]]+))?\]\s*",
    re.IGNORECASE,
)


def _safe_str(x: Any) -> str:
    return "" if x is None else str(x)


def _clamp_history(history: Any, max_turns: int = 12) -> List[Dict[str, str]]:
    """
    Ensure history is list[{"role": "...", "content": "..."}]
    Clamp to last max_turns messages to avoid token blow-up.
    """
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
    Supports old format you used:
      [lang=vi;profile_type=domestic;profile_region=au] hello
    Returns: (meta_dict, message_without_prefix)
    """
    m = META_PREFIX_RE.match(user_message or "")
    if not m:
        return {}, user_message

    meta = {
        "language": (m.group("lang") or "").lower().strip(),
        "profile_type": (m.group("ptype") or "").strip(),
        "profile_region": (m.group("pregion") or "").strip(),
    }
    # remove empties
    meta = {k: v for k, v in meta.items() if v}
    stripped = (user_message[m.end() :] if user_message else "").lstrip()
    return meta, stripped


@dataclass
class StudentContext:
    student_id: str
    language: str = "vi"
    profile_type: str = "unknown"   # domestic/international/unknown
    profile_region: str = "unknown" # au/sea/eu/other/unknown


# -----------------------------
# Orchestrator
# -----------------------------
class Orchestrator:
    """
    Single-file "wired" orchestrator:
    - accepts history + profile fields
    - parses optional metadata prefix
    - calls Groq chat.completions
    - returns a CBT-ish supportive response (trauma-informed, non-judgemental)
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
    ) -> StudentContext:
        lang = (meta.get("language") or "vi").lower()
        ptype = (meta.get("profile_type") or profile_type or "unknown").strip().lower()
        preg = (meta.get("profile_region") or profile_region or "unknown").strip().lower()

        # normalize common values
        if ptype in ("domestic", "sv trong nuoc", "sv trong nước", "local"):
            ptype = "domestic"
        elif ptype in ("international", "sv quoc te", "sv quốc tế", "intl"):
            ptype = "international"
        elif not ptype:
            ptype = "unknown"

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
            # default to vi if unknown
            lang = "vi"

        return StudentContext(
            student_id=student_id,
            language=lang,
            profile_type=ptype,
            profile_region=preg,
        )

    def _system_prompt(self, ctx: StudentContext) -> str:
        """
        Keep prompt short but strong.
        """
        # bilingual-ish guardrails
        if ctx.language == "en":
            lang_rule = "Respond in English by default (you may include short Vietnamese clarifications if helpful)."
        elif ctx.language == "zh":
            lang_rule = "默认用中文回复（必要时可补充简短英文/越南语解释）。"
        else:
            lang_rule = "Trả lời ưu tiên bằng tiếng Việt (có thể kèm 1–2 câu tiếng Anh nếu hữu ích)."

        profile_hint = f"Student profile: type={ctx.profile_type}, region={ctx.profile_region}."

        return (
            "You are a non-judgmental, trauma-informed wellbeing support assistant for university students. "
            "Your job is to provide emotional support and gentle CBT-aligned guidance.\n"
            "Rules:\n"
            "- Be warm, validating, and practical.\n"
            "- Ask 1 short clarifying question when needed.\n"
            "- Offer 2-4 concrete next steps, small and doable.\n"
            "- Do NOT provide medical diagnosis.\n"
            "- If the user expresses self-harm intent, encourage seeking immediate help and provide crisis resources.\n"
            f"- {lang_rule}\n"
            f"- {profile_hint}\n"
        )

    def _user_context_block(self, ctx: StudentContext) -> str:
        """
        Provide minimal context.
        """
        return (
            f"(Context) student_id={ctx.student_id}, "
            f"profile_type={ctx.profile_type}, profile_region={ctx.profile_region}, ui_language={ctx.language}."
        )

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """
        Groq Python SDK: client.chat.completions.create(...)
        """
        try:
            completion = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=0.7,
                max_tokens=700,
            )
            # Groq response format similar to OpenAI
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

        # 3) build student context
        ctx = self._build_student_context(student_id, profile_type, profile_region, meta)

        # 4) build prompt messages
        sys = self._system_prompt(ctx)
        ctx_block = self._user_context_block(ctx)

        messages: List[Dict[str, str]] = [{"role": "system", "content": sys}]
        # add a small context block as system to help the model
        messages.append({"role": "system", "content": ctx_block})

        # append history (without system)
        for m in hist:
            if m["role"] == "system":
                continue
            messages.append(m)

        # append current user message
        messages.append({"role": "user", "content": cleaned_message})

        # 5) call LLM
        reply = self._call_llm(messages)

        # 6) minimal post-processing (optional)
        # prevent empty reply
        if not reply:
            reply = "Mình ở đây với bạn. Bạn có thể nói thêm một chút về điều đang xảy ra không?"

        return reply
