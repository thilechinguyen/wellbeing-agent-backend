# agents/trend_agent.py
from groq import Groq

class TrendAgent:
    """
    Detect emotional trend: improving / worsening / stable / unknown.
    """

    def __init__(self, model_id: str, client: Groq):
        self.model_id = model_id
        self.client = client

    def run(self, student_id: str, insights: dict, history: list):
        history_text = "\n".join(
            [f"{m['role']}: {m['content']}" for m in history[-6:]]
        )

        prompt = f"""
You are the Trend Agent.

Look at:
- Latest insight
- Recent conversation history

Return ONLY JSON:
- trend: "improving", "worsening", "stable", "unknown"
- rationale: one sentence

Student ID: {student_id}
Insights: {insights}

Recent history:
{history_text}
"""

        try:
            completion = self.client.chat.completions.create(
                model=self.model_id,
                temperature=0,
                messages=[{"role": "system", "content": prompt}],
            )

            raw = completion.choices[0].message.content.strip()

            try:
                return eval(raw)
            except:
                pass

        except:
            pass

        return {"trend": "unknown", "rationale": "Insufficient data"}
