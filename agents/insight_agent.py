# agents/insight_agent.py
import re
from typing import Dict, Any, List
from groq import Groq


class InsightAgent:
    """
    Extract emotion, risk, topics, language from the latest message + short context.
    """

    def __init__(self, model_id: str, client: Groq):
        self.model_id = model_id
        self.client = client

    @staticmethod
    def extract_json(text: str):
        text = text.strip()
        try:
            return eval(text)
        except Exception:
            pass

        # Try first {...} block
        import re
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return eval(match.group(0))
            except Exception:
                pass

        return None

    def run(self, message: str, history: List[Dict[str, str]]) -> Dict[str, Any]:
        # Keep last 4 messages
        context = "\n".join([f"{m['role']}: {m['content']}" for m in history[-4:]])

        prompt = f"""
You are the Insight Extraction Agent in a wellbeing system.

You see:
- Recent conversation context
- The latest student message

Your job: classify *current* emotional state.

Recent context:
{context}

Latest message:
{message}

Return ONLY a JSON with:
- emotion: "joy", "sadness", "worry", "stress", "anger", "neutral"
- risk_level: "low", "medium", "high"
- positive_event: true/false
- topics: ["exam", "family", ...] (1–4 items)
- language: "vi", "en", "zh", "ja", "ko", or "other"
"""

        try:
            completion = self.client.chat.completions.create(
                model=self.model_id,
                temperature=0,
                messages=[{"role": "system", "content": prompt}],
            )

            raw = completion.choices[0].message.content
            data = InsightAgent.extract_json(raw)
            if data:
                return data

        except Exception:
            pass

        return {
            "emotion": "neutral",
            "risk_level": "low",
            "positive_event": False,
            "topics": [],
            "language": "other",
        }
