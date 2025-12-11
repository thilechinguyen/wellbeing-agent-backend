from pydantic import BaseModel
from typing import Literal, Optional

class BigFiveProfile(BaseModel):
    openness: int           # 1-5
    conscientiousness: int  # 1-5
    extraversion: int       # 1-5
    agreeableness: int      # 1-5
    neuroticism: int        # 1-5

class ResilienceProfile(BaseModel):
    emotional_regulation: int   # 1-5
    cognitive_flexibility: int  # 1-5
    meaning_purpose: int        # 1-5
    social_support: int         # 1-5
    self_compassion: int        # 1-5

CopingStyle = Literal["problem_focused", "emotion_focused", "avoidant", "mixed"]

class WellbeingProfile(BaseModel):
    big_five: BigFiveProfile
    resilience: ResilienceProfile
    coping_style: CopingStyle
    notes: Optional[str] = None

