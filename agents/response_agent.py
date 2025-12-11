# agents/response_agent.py
# Stub đơn giản để đảm bảo import hoạt động

class ResponseAgent:
    def __init__(self, model_id, client, identity_prompt: str):
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
        support_block,
    ):
        # Tạm thời trả về 1 câu đơn giản để test backend
        return (
            "Hi, this is the temporary ResponseAgent stub. "
            "Backend V12 is running, but the final reply logic has not been wired yet."
        )
