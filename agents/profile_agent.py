# agents/profile_agent.py

from groq import Groq


class ProfileAgent:
    """
    Internal emotional summary generator.
    This summary is NOT shown to the student.
    """

    def __init__(self, model_id: str, client: Groq):
        self.model_id = model_id
        self.client = client

    def run(self, student_id: str, insights: dict) -> str:
        prompt = f"""
You are the Profile Agent.

Summarize the student's CURRENT emotional state in 2–3 sentences.
This summary is for INTERNAL SYSTEM MEMORY ONLY and NEVER shown
directly to the student.

Student ID: {student_id}
Insights: {insights}
"""

        completion = self.client.chat.completions.create(
            model=self.model_id,
            temperature=0.2,
            messages=[{"role": "system", "content": prompt}],
        )
        return completion.choices[0].message.content.strip()
