# agents/response_agent.py

from typing import Any, Dict, List
from groq import Groq


class ResponseAgent:
    """
    Final response agent for Wellbeing Companion V12.

    Orchestrator sẽ gọi:
      self.response.run(
          user_message=...,
          history=...,
          insights=...,
          trend=...,
          interventions=...,
          safety=...,
          style_hint=...,
          personality=...,
          memory_summary=...,
          pronoun_pref=...,
          joy_mode=...,
          cultural_block=...,
          language=...,
          support_block=...,
      )
    """

    def __init__(self, model_id: str, client: Groq, identity_prompt: str):
        self.model_id = model_id
        self.client = client
        self.identity_prompt = identity_prompt

    def run(
        self,
        *,
        user_message: str,
        history: List[Dict[str, str]],
        insights: Dict[str, Any],
        trend: Dict[str, Any],
        interventions: str,
        safety: Dict[str, Any],
        style_hint: str,
        personality: Dict[str, Any],
        memory_summary: str,
        pronoun_pref: str,
        joy_mode: bool,
        cultural_block: str,
        language: str,
        support_block: str,
    ) -> str:
        # Chuẩn hóa history sang format messages của Groq
        replay_messages: List[Dict[str, str]] = []
        for m in history or []:
            # m có dạng {"role": "...", "content": "..."} từ frontend
            role = m.get("role", "user")
            content = m.get("content", "")
            replay_messages.append({"role": role, "content": content})

        # Hướng dẫn đặc biệt cho joy mode
        joy_block = ""
        if joy_mode:
            joy_block = """
The student is currently sharing clearly positive or celebratory news.
Stay in a joyful, friendly, peer-like tone.
Do NOT introduce CBT techniques or university services in this reply.
Sound like a close uni friend celebrating with them.
"""

        # Hướng dẫn cho support block (khi nguy cơ cao)
        support_instruction = ""
        if support_block:
            support_instruction = f"""
The safety layer marked this situation as HIGH RISK.

In this reply you MUST:
1) First, in the SAME LANGUAGE as the student, write 1–2 short sentences
   gently suggesting they consider reaching out for extra support.
2) After your own message, append the following support information
   verbatim, without translating or modifying it:

{support_block}
"""

        # System prompt tổng hợp
        system_prompt = f"""
{self.identity_prompt}

You are part of a multi-agent wellbeing system for first-year university students.
Another set of agents has already analysed the situation for you.

Language code for this conversation: {language}

Conversation summary (internal memory):
{memory_summary}

Emotional insights:
{insights}

Short-term trend:
{trend}

Personality / resilience / coping style signals (approximate, internal only):
{personality}

Style hint (how you should talk to this student):
{style_hint}

Cultural profile block:
{cultural_block}

Direct interventions suggested by the Intervention Agent:
{interventions}

Pronoun preference info (if any, in Vietnamese): {pronoun_pref}

Safety info (internal):
{safety}

Joy mode state: {joy_mode}
{joy_block}
{support_instruction}

General rules:
- You are NOT a therapist and must not present yourself as one.
- You speak like a warm, respectful uni friend.
- Keep replies concise, concrete, and easy to read.
- Be emotionally validating, never judgmental.
- Follow the student's language: if they use Vietnamese, reply in Vietnamese;
  if they use English, reply in English, etc.
"""

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
        ]

        # thêm history rút gọn
        messages.extend(replay_messages)

        # thêm message mới nhất
        messages.append({"role": "user", "content": user_message})

        completion = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=0.65,
            max_tokens=800,
        )
        reply = completion.choices[0].message.content or ""
        return reply.strip()
