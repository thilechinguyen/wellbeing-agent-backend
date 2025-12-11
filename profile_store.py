# profile_store.py
from typing import Optional
from .profile_model import WellbeingProfile, BigFiveProfile, ResilienceProfile

# tạm thời dùng dict in-memory (sau thay bằng DB)
PROFILE_CACHE: dict[str, WellbeingProfile] = {}

def get_profile(user_id: str) -> Optional[WellbeingProfile]:
    return PROFILE_CACHE.get(user_id)

def save_profile(user_id: str, profile: WellbeingProfile) -> None:
    PROFILE_CACHE[user_id] = profile
