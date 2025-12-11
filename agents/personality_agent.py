# agents/personality_agent.py
from groq import Groq
import json

class PersonalityAgent:
    """
    Hybrid model:
    - baseline: inferred from long-term history
    - dynamic modifiers: inferred from last 1–3 messages
    """

    def __init__(self, model_id: str, client: Groq):
        self.model_id = model_id
        self.client = client

    def run(self, full_history: list, recent_msgs: list):
        baseline_text = "\n".join([f"{m['role']}: {m['content']}" for m in full_history])
        recent_text = "\n".join([m["content"] for m in recent_msgs])

        prompt = f"""
You are the Personality Agent.

Infer the student's stable personality (baseline) and short-term shifts (dynamic)
based on conversation history.

Return ONLY JSON with:

- big_five:
    openness: 0–1
    conscientiousness: 0–1
    extraversion: 0–1
    agreeableness: 0–1
    neuroticism: 0–1

- resilience:
    score: 0–1
    explanation: one short sentence

- coping_style:
    type: "problem-focused" | "emotion-focused" | "avoidant" | "mixed"
    rationale: one short sentence

- dynamic_modifiers:
    emotional_reactivity: 0–1
    social_needs: 0–1
    confidence_shift: 0–1

Baseline (long-term history):
{baseline_text}

Recent (short-term):
{recent_text}
"""

        completion = self.client.chat.completions.create(
            model=self.model_id,
            temperature=0,
            messages=[{"role": "system", "content": prompt}],
        )

        raw = completion.choices[0].message.content.strip()
        try:
            return json.loads(raw)
        except:
            return {
                "big_five": {
                    "openness": 0.5,
                    "conscientiousness": 0.5,
                    "extraversion": 0.5,
                    "agreeableness": 0.5,
                    "neuroticism": 0.5,
                },
                "resilience": {"score": 0.5, "explanation": ""},
                "coping_style": {"type": "mixed", "rationale": ""},
                "dynamic_modifiers": {
                    "emotional_reactivity": 0.5,
                    "social_needs": 0.5,
                    "confidence_shift": 0.5,
                },
            }
