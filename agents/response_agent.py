# agents/response_agent.py

from groq import Groq
from identity import IDENTITY_PROMPT  # nếu cần, hoặc truyền qua constructor


class ResponseAgentV12:
    """
    Final response composer for V12.
    """

    def __init__(self, model_id: str, client: Groq, identity_prompt: str):
        self.model_id = model_id
        self.client = client
        self.identity_prompt = identity_prompt

    def run(
        self,
        *,
        user_message,
        history,
        insights,
        trend,
        interventions,
        safety,
        style_hint,
        personality,
        memory_summary,
        pronoun_pref,
        joy_mode,
        cultural_block,
        language,
        support_block,
    ):
        joy_instructions = ""
        if joy_mode:
            joy_instructions = """
JOY MODE ACTIVE:
- Celebrate with the student
- Excited, playful tone
- Do NOT mention coping skills or services
- You may ask ONE fun follow-up question
"""

        safety_instructions = ""
        if safety.get("override_risk_level") == "high":
            safety_instructions = """
HIGH RISK:
- Very gentle, caring tone
- Validate their pain
- You may gently ask if they are safe now
- Encourage reaching out to trusted people
"""

        personality_block = f"""
PERSONALITY SNAPSHOT:
Big Five: {personality.get("big_five")}
Resilience: {personality.get("resilience")}
Coping style: {personality.get("coping_style")}
Dynamic modifiers: {personality.get("dynamic_modifiers")}
"""

        system_prompt = f"""
{self.identity_prompt}

LANGUAGE: {language}

===================
MEMORY SUMMARY
===================
{memory_summary}

Pronoun preference: {pronoun_pref}

===================
PERSONALITY
===================
{personality_block}

===================
CULTURAL PROFILE
===================
{cultural_block}

===================
INSIGHTS
===================
{insights}

Trend:
{trend}

Style hint:
{style_hint}

{joy_instructions}
{safety_instructions}

Interventions (can be used, but ignore in joy mode):
{interventions}
"""

        messages = [
            {"role": "system", "content": system_prompt},
        ]

        # (optional) có thể replay short history nếu muốn
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

        if safety.get("override_risk_level") == "high" and support_block:
            reply += "\n\n" + support_block

        return reply
