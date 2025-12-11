# agents/response_agent_v12.py (replace old ResponseAgent)

from groq import Groq

class ResponseAgentV12:
    """
    Combines:
    - Identity
    - Memory Summary
    - Personality hybrid
    - Style hints
    - Joy mode rules
    - Safety escalation
    - Intervention (optional)
    - Cultural profile block
    - Adelaide Support block (only high-risk)
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
        support_block
    ):

        joy_instructions = ""
        if joy_mode:
            joy_instructions = """
JOY MODE ACTIVE:
- Use excited, playful tone
- Celebrate with the student
- Do NOT mention wellbeing strategies
- Do NOT mention support services
- You MAY ask one fun follow-up question
"""

        safety_instructions = ""
        if safety.get("override_risk_level") == "high":
            safety_instructions = """
HIGH RISK DETECTED:
- Speak slowly, gently
- Validate feelings with warmth
- Ask if they are safe
- Encourage contacting trusted people
- At the end of your message, append the UNIVERSITY SUPPORT BLOCK
"""

        personality_block = f"""
PERSONALITY SNAPSHOT:
Big Five: {personality.get("big_five")}
Resilience: {personality.get("resilience")}
Coping Style: {personality.get("coping_style")}
Dynamic Modifiers: {personality.get("dynamic_modifiers")}
"""

        prompt = f"""
{self.identity_prompt}

===================
SYSTEM MEMORY
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

Interventions (ignored in joy mode):
{interventions}

Support block (attach only if high-risk):
{support_block}

Your task:
Write a warm, natural, empathetic message to the student in their language ({language}),
following all identity rules.
"""

        completion = self.client.chat.completions.create(
            model=self.model_id,
            temperature=0.65,
            max_tokens=700,
            messages=[{"role": "system", "content": prompt}],
        )

        reply = completion.choices[0].message.content.strip()

        # Append support services only when needed
        if safety.get("override_risk_level") == "high":
            reply += "\n\n" + support_block

        return reply
