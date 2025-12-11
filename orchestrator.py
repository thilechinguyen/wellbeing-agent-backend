# orchestrator.py

from agents.insight_agent import InsightAgent
from agents.profile_agent import ProfileAgent
from agents.trend_agent import TrendAgent
from agents.safety_agent import SafetyAgent
from agents.intervention_agent import InterventionAgent
from agents.style_agent import StyleAgent
from agents.personality_agent import PersonalityAgent
from agents.joy_agent import JoyAgent
from agents.cultural_agent import CulturalAgent
from agents.response_agent import ResponseAgent  # <── LƯU Ý DÒNG NÀY

from identity import IDENTITY_PROMPT
from support_blocks import ADELAIDE_SUPPORT_BLOCK
from memory import (
    append_message,
    get_summary_or_default,
    set_summary,
    update_emotional_state,
)
from conversation_logging import log_turn


class Orchestrator:
    """
    V12 orchestrator – gộp tất cả agents lại.
    main.py đang gọi Orchestrator, nên class này đặt tên Orchestrator cho khớp.
    """

    def __init__(self, model_id, client):
        self.insight = InsightAgent(model_id, client)
        self.profile = ProfileAgent(model_id, client)
        self.trend = TrendAgent(model_id, client)
        self.safety = SafetyAgent()
        self.intervention = InterventionAgent(model_id, client)
        self.style = StyleAgent(model_id, client)
        self.personality = PersonalityAgent(model_id, client)
        self.joy = JoyAgent()
        self.culture = CulturalAgent()
        self.response = ResponseAgentV12(model_id, client, IDENTITY_PROMPT)

    def run(
        self,
        student_id: str,
        user_message: str,
        history: list,
        profile_type=None,
        profile_region=None,
    ):
        # 1) lưu user message
        append_message(student_id, "user", user_message)

        # 2) Insight
        insights = self.insight.run(user_message, history)

        # 3) update emotional state
        update_emotional_state(
            session_id=student_id,
            primary_emotion=insights.get("emotion"),
            stress_level=insights.get("risk_level"),
            risk_level=insights.get("risk_level"),
            notes="; ".join(insights.get("topics", [])),
        )

        # 4) Profile summary (internal)
        profile_summary = self.profile.run(student_id, insights)
        set_summary(student_id, profile_summary)

        # 5) Trend
        trend = self.trend.run(student_id, insights, history)

        # 6) Safety
        safety = self.safety.run(user_message, insights)

        # 7) Intervention
        interventions = self.intervention.run(insights, trend, user_message)

        # 8) Style
        style_hint = self.style.run(student_id, history, insights)

        # 9) Personality
        personality = self.personality.run(
            full_history=history,
            recent_msgs=history[-3:],
        )

        # 10) Joy mode
        joy_mode = self.joy.update_state(history, user_message)

        # 11) Cultural profile
        cultural_block = self.culture.run(profile_type, profile_region)

        # 12) Memory summary
        memory_summary = get_summary_or_default(student_id)

        # 13) Support block (high risk)
        support_block = (
            ADELAIDE_SUPPORT_BLOCK
            if safety.get("override_risk_level") == "high"
            else ""
        )

        # 14) Gọi ResponseAgent
        reply = self.response.run(
            user_message=user_message,
            history=history,
            insights=insights,
            trend=trend,
            interventions=interventions,
            safety=safety,
            style_hint=style_hint,
            personality=personality,
            memory_summary=memory_summary,
            pronoun_pref="none",
            joy_mode=joy_mode,
            cultural_block=cultural_block,
            language=insights.get("language", "en"),
            support_block=support_block,
        )

        # 15) lưu reply
        append_message(student_id, "assistant", reply)

        # 16) log để nghiên cứu
        try:
            log_turn(
                session_id=student_id,
                turn_index=None,
                user_id=student_id,
                condition=None,
                lang_code=insights.get("language", "en"),
                user_text=user_message,
                agent_text=reply,
                emotion=insights,
                safety=safety,
                supervisor=None,
            )
        except Exception as e:
            print("log_turn failed:", e)

        return reply
