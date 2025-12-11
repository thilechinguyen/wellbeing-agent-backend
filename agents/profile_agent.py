from typing import List
from .llm_client import client  # chỗ bạn đang gọi Groq/OpenAI
from .profile_model import (
    WellbeingProfile, BigFiveProfile, ResilienceProfile, CopingStyle
)
from .profile_store import get_profile, save_profile
import json

SYSTEM_PROMPT_PROFILE = """
You are a psychologist assistant.
Your task is to INFER a light-weight wellbeing profile from the user's messages.

You must output STRICT JSON with this schema:
{
  "big_five": {
    "openness": 1-5,
    "conscientiousness": 1-5,
    "extraversion": 1-5,
    "agreeableness": 1-5,
    "neuroticism": 1-5
  },
  "resilience": {
    "emotional_regulation": 1-5,
    "cognitive_flexibility": 1-5,
    "meaning_purpose": 1-5,
    "social_support": 1-5,
    "self_compassion": 1-5
  },
  "coping_style": "problem_focused" | "emotion_focused" | "avoidant" | "mixed",
  "notes": "very short explanation (max 2 sentences)"
}

Use previous profile as a prior and only make small adjustments unless the conversation clearly contradicts it.
"""

def infer_profile(user_id: str, conversation_snippet: str) -> WellbeingProfile:
    """Gọi LLM để cập nhật profile dựa trên 5–10 message gần nhất."""
    prev = get_profile(user_id)

    prev_json = prev.json() if prev else "null"

    user_prompt = f"""
Previous_profile_json = {prev_json}

Conversation_snippet = \"\"\"{conversation_snippet}\"\"\"

Update or create a WellbeingProfile JSON.
If previous_profile_json is not null, treat it as prior information.
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # hoặc model Groq bạn đang dùng
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_PROFILE},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )

    content = response.choices[0].message.content
    data = json.loads(content)

    profile = WellbeingProfile(
        big_five=BigFiveProfile(**data["big_five"]),
        resilience=ResilienceProfile(**data["resilience"]),
        coping_style=data["coping_style"],
        notes=data.get("notes"),
    )
    save_profile(user_id, profile)
    return profile
