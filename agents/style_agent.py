# agents/style_agent.py
from groq import Groq

class StyleAgent:
    """
    Suggest tone adaptation style instructions:
    - shorter responses
    - casual tone
    - more examples
    - slower pace
    """

    def __init__(self, model_id: str, client: Groq):
        self.model_id = model_id
        self.client = client

    def run(self, student_id: str, history: list, insights: dict):
        recent_msgs = "\n".join(
            [m["content"] for m in history if m["role"] == "user"][-5:]
        )

        prompt = f"""
You are the Style Agent.

Based on recent messages and insights, create 2–3 bullet points describing
how the assistant should adapt tone for this student.

Student ID: {student_id}

Recent user messages:
{recent_msgs}

Insights: {insights}
"""

        completion = self.client.chat.completions.create(
            model=self.model_id,
            temperature=0.4,
            messages=[{"role": "system", "content": prompt}],
        )

        return completion.choices[0].message.content.strip()
