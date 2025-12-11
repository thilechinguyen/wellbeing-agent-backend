# agents/response_agent.py

from typing import List, Dict, Any
from groq import Groq


class ResponseAgent:
    """
    Final response composer cho Wellbeing Companion V12.
    Nhận toàn bộ thông tin từ Orchestrator và gọi LLM để sinh câu trả lời cuối.
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
        """
        Compose system prompt + history + user_message và gọi Groq LLM.
        """

        # Joy mode instructions
        joy_instructions = ""
        if joy_mode:
            joy_instructions = """
JOY MODE:
- The student is sharing clearly positive news with low risk.
- Reply like a close uni friend celebrating with them.
- Warm, playful, excited tone.
- You MAY ask ONE light follow-up question (e.g. “How will you celebrate?”).
- Do NOT mention coping skills, CBT, journaling, breathing, or services.
"""

        # Safety instructions
        safety_instructions = ""
        if safety.get("override_risk_level") == "high":
            safety_instructions = """
HIGH RISK:
- Be very gentle and caring.
- Validate their pain and distress.
- You may gently ask if they are in a safe place and if someone they trust is nearby.
- Encourage reaching out to trusted people or support services.
- Do NOT minimise or ignore their feelings.
"""

        # Personality block
        personality_block = f"""
PERSONALITY SNAPSHOT (internal only):
- Big Five: {personality.get("big_five")}
- Resilience: {personality.get("resilience")}
- Coping style: {personality.get("coping_style")}
- Dynamic modifiers: {personality.get("dynamic_modifiers")}
"""

        system_prompt = f"""
{self.identity_prompt}

You are replying in language code: {language}.

====================
INTERNAL MEMORY SUMMARY
====================
{memory_summary}

Pronoun preference (Vietnamese, if applicable): {pronoun_pref}

====================
PERSONALITY PROFILE
====================
{personality_block}

====================
CULTURAL PROFILE
====================
{cultural_block}

====================
LATEST INSIGHTS
====================
{insights}

Trend info:
{trend}

Style hint for tone:
{style_hint}

Interventions you MAY weave in (unless joy mode is active):
{interventions}

{joy_instructions}
{safety_instructions}

Rules:
- Talk like a close friend, not a therapist or doctor.
- Keep sentences short, clear, and warm.
- No moral lectures, no generic life-coach speeches.
- Stay within the emotional context of the conversation.
"""

        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

        # Add short history (role/content dicts đến từ orchestrator)
        for m in history:
            messages.append({"role": m["role"], "content": m["content"]})

        messages.append({"role": "user", "content": user_message})

        completion = self.client.chat.completions.create(
            model=self.model_id,
            temperature=0.65,
            max_tokens=800,
            messages=messages,
        )

        reply = completion.choices[0].message.content.strip()

        # Nếu high-risk và có support_block (Adelaide services) thì append ở cuối
        if safety.get("override_risk_level") == "high" and support_block:
            reply += "\n\n" + support_block

        return reply
