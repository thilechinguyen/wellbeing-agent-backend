# agents/intervention_agent.py
from groq import Groq

class InterventionAgent:
    """
    Suggest small wellbeing actions ONLY when student is sad/stressed.
    Never activate during joy mode.
    """

    def __init__(self, model_id: str, client: Groq):
        self.model_id = model_id
        self.client = client

    def run(self, insights: dict, trend: dict, message: str):
        if insights.get("positive_event") and insights.get("risk_level") == "low":
            return ""

        if insights.get("emotion") in ["joy", "neutral"]:
            return ""

        prompt = """
You are the Intervention Agent.

If the user is sad, stressed, anxious, or overwhelmed,
return ONE very small actionable suggestion (1–2 sentences).

If not appropriate, return an EMPTY STRING.
"""

        completion = self.client.chat.completions.create(
            model=self.model_id,
            temperature=0.3,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": message},
            ],
        )
        return completion.choices[0].message.content.strip()
