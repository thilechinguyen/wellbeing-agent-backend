# agents/cultural_agent.py
from cultural_profiles import build_profile_block

class CulturalAgent:
    """
    Apply cultural & regional adaptation.
    For now: uses your existing build_profile_block() from V10.
    """

    def run(self, profile_type: str, profile_region: str):
        return build_profile_block(profile_type, profile_region)
