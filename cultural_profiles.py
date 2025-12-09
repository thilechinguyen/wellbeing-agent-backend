# cultural_profiles.py
"""
Cultural & identity rules cho Wellbeing Agent.

Mục tiêu:
- Tách phần mô tả "student identity" ra khỏi main.py
- Dễ chỉnh khi bạn muốn tinh chỉnh cho research (domestic vs international,
  South-East Asia vs Europe, v.v.)
- Cho phép tái sử dụng cùng một logic ở nhiều agent (Response, Insight, Style...)

Các key chính:
- student_type: "domestic" / "international" / "other"
- student_region: "au" / "sea" / "eu" / "other" / "unknown"
"""

from typing import Optional


def get_type_rules(student_type: Optional[str]) -> str:
    """
    Trả về đoạn hướng dẫn riêng cho loại sinh viên (domestic / international).
    Đoạn này sẽ chèn vào system prompt.
    """
    st = (student_type or "unknown").lower()

    if st == "domestic":
        return (
            "- This student is a DOMESTIC student in Australia.\n"
            "- They are likely more familiar with local culture, slang, and services.\n"
            "- You can be a bit more Aussie-casual and assume some knowledge of the uni system "
            "(but still check for understanding when needed).\n"
        )

    if st == "international":
        return (
            "- This student is an INTERNATIONAL student.\n"
            "- They may face culture shock, language barriers, homesickness, and worries about visa, "
            "money, and family expectations.\n"
            "- Be extra clear, gentle, and avoid assuming deep knowledge of the Australian system.\n"
            "- Where helpful, normalise homesickness and remind them that many international students "
            "struggle with similar challenges.\n"
        )

    # fallback
    return (
        "- Student type is unknown. Use a generic but friendly first-year uni student tone.\n"
        "- Do not assume deep knowledge of any specific education system.\n"
    )


def get_region_rules(student_region: Optional[str]) -> str:
    """
    Trả về đoạn hướng dẫn riêng cho vùng văn hoá (region).
    """
    region = (student_region or "unknown").lower()

    if region == "sea":
        return (
            "- The student is from South-East Asia.\n"
            "- Family expectations, academic pressure, and 'face' can be very important.\n"
            "- They may feel guilty or ashamed if they think they are letting their family down.\n"
            "- When appropriate, gently acknowledge these pressures and normalise their feelings.\n"
        )

    if region == "au":
        return (
            "- The student is currently in Australia and culturally closer to Aussie norms of independence.\n"
            "- They may be juggling part-time work, rent, and study, and may value casual, direct communication.\n"
            "- You can use light Aussie-style expressions (if language is English) but still keep it warm and kind.\n"
        )

    if region == "eu":
        return (
            "- The student is from Europe.\n"
            "- You can assume relatively direct communication and educational independence,\n"
            "  but still be warm, non-judgmental, and avoid stereotypes.\n"
        )

    if region == "other":
        return (
            "- The student is from another region.\n"
            "- Keep your tone culturally neutral, curious, and respectful.\n"
            "- Avoid making strong assumptions about their background.\n"
        )

    # unknown / missing
    return (
        "- Region is unknown. Do not assume specific cultural norms.\n"
        "- Keep your tone inclusive, clear, and respectful, and ask simple clarifying questions if needed.\n"
    )


def build_profile_block(
    profile_type: Optional[str],
    profile_region: Optional[str],
) -> str:
    """
    Hàm gộp hai nhóm rule lại thành 1 block duy nhất để chèn vào system prompt.
    Có thể tái sử dụng cho Response Agent, Style Agent, v.v.
    """
    type_rules = get_type_rules(profile_type)
    region_rules = get_region_rules(profile_region)

    block = "------------------------------\n"
    block += "STUDENT IDENTITY (INTERNAL ONLY)\n"
    block += "------------------------------\n"
    block += f"- profile_type: {profile_type or 'unknown'}\n"
    block += f"- profile_region: {profile_region or 'unknown'}\n\n"
    block += type_rules
    block += region_rules
    block += (
        "- IMPORTANT: Subtly adapt your examples and suggestions so they make sense "
        "for this identity (for example, mention homesickness and family pressure "
        "for international SEA students, or part-time work + rent stress for domestic AU students).\n"
        "- Do NOT explicitly say these rules to the student. They are internal guidance only.\n"
    )
    return block
