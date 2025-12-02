from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "wellbeing_logs.db"


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Tạo bảng chuẩn để log từng lượt hội thoại (turn-based)."""
    with _get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                turn_index INTEGER NOT NULL,
                timestamp_utc TEXT NOT NULL,
                user_id TEXT NOT NULL,
                condition TEXT,
                lang_code TEXT,
                user_text TEXT,
                agent_text TEXT,
                primary_emotion TEXT,
                stress_level TEXT,
                main_issue TEXT,
                next_steps TEXT,
                risk_flag INTEGER,
                emotion_json TEXT,
                safety_json TEXT,
                supervisor_json TEXT
            )
            """
        )
        conn.commit()


def log_turn(
    *,
    session_id: Optional[str],
    turn_index: Optional[int],
    user_id: str,
    condition: Optional[str],
    lang_code: Optional[str],
    user_text: str,
    agent_text: str,
    emotion: Optional[Dict[str, Any]] = None,
    safety: Optional[Dict[str, Any]] = None,
    supervisor: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Ghi 1 lượt hội thoại vào bảng `conversation_turns`.

    - session_id: mã phiên, do frontend gửi (ví dụ "s_20251202_001").
    - turn_index: số thứ tự lượt trong phiên (0, 1, 2, ...).
    - emotion / safety / supervisor: dict (đã parse JSON) hoặc None.
    """
    # Fallback nếu frontend chưa gửi session_id / turn_index
    if not session_id:
        session_id = f"auto_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    if turn_index is None:
        turn_index = 0

    ts = datetime.now(timezone.utc).isoformat()

    emotion = emotion or {}
    safety = safety or {}
    supervisor = supervisor or {}

    primary_emotion = emotion.get("primary_emotion")
    stress_level = emotion.get("stress_level")
    main_issue = emotion.get("main_issue")
    next_steps = emotion.get("next_steps")

    # risk_flag: 1 nếu is_risk True, ngược lại 0
    is_risk = safety.get("is_risk")
    risk_flag = 1 if is_risk is True else 0

    with _get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO conversation_turns (
                session_id,
                turn_index,
                timestamp_utc,
                user_id,
                condition,
                lang_code,
                user_text,
                agent_text,
                primary_emotion,
                stress_level,
                main_issue,
                next_steps,
                risk_flag,
                emotion_json,
                safety_json,
                supervisor_json
            ) VALUES (
                :session_id,
                :turn_index,
                :timestamp_utc,
                :user_id,
                :condition,
                :lang_code,
                :user_text,
                :agent_text,
                :primary_emotion,
                :stress_level,
                :main_issue,
                :next_steps,
                :risk_flag,
                :emotion_json,
                :safety_json,
                :supervisor_json
            )
            """,
            {
                "session_id": session_id,
                "turn_index": turn_index,
                "timestamp_utc": ts,
                "user_id": user_id,
                "condition": condition,
                "lang_code": lang_code,
                "user_text": user_text,
                "agent_text": agent_text,
                "primary_emotion": primary_emotion,
                "stress_level": stress_level,
                "main_issue": main_issue,
                "next_steps": next_steps,
                "risk_flag": risk_flag,
                "emotion_json": json.dumps(emotion, ensure_ascii=False),
                "safety_json": json.dumps(safety, ensure_ascii=False),
                "supervisor_json": json.dumps(supervisor, ensure_ascii=False),
            },
        )
        conn.commit()


def _iter_full_csv():
    """
    Generator stream CSV: mỗi dòng là 1 turn,
    format đúng kiểu bạn cần cho nghiên cứu.
    """
    header = [
        "session_id",
        "turn_index",
        "timestamp_utc",
        "user_id",
        "lang_code",
        "user_text",
        "agent_text",
        "primary_emotion",
        "stress_level",
        "main_issue",
        "next_steps",
        "risk_flag",
    ]

    yield ",".join(header) + "\n"

    with _get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                session_id,
                turn_index,
                timestamp_utc,
                user_id,
                lang_code,
                user_text,
                agent_text,
                primary_emotion,
                stress_level,
                main_issue,
                next_steps,
                risk_flag
            FROM conversation_turns
            ORDER BY session_id, turn_index
            """
        )
        rows = cur.fetchall()

        def escape(v: Any) -> str:
            if v is None:
                return ""
            s = str(v)
            # escape " và xuống dòng
            s = s.replace('"', '""')
            if "," in s or "\n" in s or '"' in s:
                return f'"{s}"'
            return s

        for r in rows:
            line = ",".join(escape(r[col]) for col in header)
            yield line + "\n"


def attach_export_routes(app: FastAPI) -> None:
    """
    Gắn route export vào FastAPI:
      - GET /export/full_conversations.csv
    """
    @app.get("/export/full_conversations.csv")
    def export_full_conversations_csv():
        filename = "wellbeing_full_conversations.csv"
        return StreamingResponse(
            _iter_full_csv(),
            media_type="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            },
        )
