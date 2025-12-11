# agents/joy_agent.py

class JoyAgent:
    """
    Detects celebration mode:
    - scholarships
    - passing exams
    - job offers
    - lottery-like excitement
    - planning celebration
    AND maintains joy mode for multiple turns (sticky).
    """

    CELEBRATION_KEYWORDS = [
        "trúng", "trung", "học bổng", "hoc bong",
        "đậu", "pass", "passed", "got the job",
        "scholarship", "offer", "accepted", "trúng số",
        "ăn mừng", "celebrate"
    ]

    NEGATIVE_BREAK = [
        "buồn","sad","stress","không muốn sống","hurt","đánh"
    ]

    def detect(self, message: str) -> bool:
        msg = message.lower()
        return any(kw in msg for kw in self.CELEBRATION_KEYWORDS)

    def break_joy(self, message: str) -> bool:
        msg = message.lower()
        return any(kw in msg for kw in self.NEGATIVE_BREAK)

    def update_state(self, history: list, latest_message: str):
        """
        Returns joy_mode: True/False
        """
        # Check break condition
        if self.break_joy(latest_message):
            return False

        # If latest message is joy → enable joy mode
        if self.detect(latest_message):
            return True

        # If recent history contains joy → keep joy
        recent = " ".join([m["content"].lower() for m in history[-6:]])
        if any(kw in recent for kw in self.CELEBRATION_KEYWORDS):
            return True

        return False
