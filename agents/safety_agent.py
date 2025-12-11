# agents/safety_agent.py
import re

class SafetyAgent:
    """
    Keyword-based safety detection for:
    - Self-harm
    - Violence
    - High emotional risk
    """

    def __init__(self):
        pass

    def run(self, message: str, insights: dict):
        msg = message.lower()

        danger_keywords = [
            "tự tử","tự sát","không muốn sống","kill myself","end my life",
            "suicide","hurt myself"
        ]

        violence_keywords = [
            "đánh","bi danh","anh danh em","hit me","abuse","violence","hurt me"
        ]

        # self-harm
        if any(kw in msg for kw in danger_keywords):
            return {
                "escalate": True,
                "override_risk_level": "high",
                "self_harm": True,
                "violence": False,
                "reason": "Self-harm keywords detected"
            }

        # violence
        if any(kw in msg for kw in violence_keywords):
            return {
                "escalate": True,
                "override_risk_level": "high",
                "self_harm": False,
                "violence": True,
                "reason": "Violence keywords detected"
            }

        # generic high risk
        if insights.get("risk_level") == "high":
            return {
                "escalate": True,
                "override_risk_level": "high",
                "self_harm": False,
                "violence": False,
                "reason": "Insight agent flagged high risk"
            }

        return {
            "escalate": False,
            "override_risk_level": None,
            "self_harm": False,
            "violence": False,
            "reason": "low risk"
        }
