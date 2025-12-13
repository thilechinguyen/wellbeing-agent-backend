# main.py — V12 (Fixed schema + metadata compatibility)

import os
import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from groq import Groq

from orchestrator import Orchestrator
from conversation_logging import init_db, attach_export_routes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wellbeing-v12")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_ID = os.getenv("GROQ_MODEL_ID", "llama-3.3-70b-versatile")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set in environment")

client = Groq(api_key=GROQ_API_KEY)
orchestrator = Orchestrator(model_id=MODEL_ID, client=client)

init_db()

app = FastAPI(title="Wellbeing Agent V12 – Multi-Agent Hybrid Personality System")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

attach_export_routes(app)


class ChatRequest(BaseModel):
    # Tương thích cả 2 kiểu payload
    # Frontend có thể gửi student_id hoặc user_id
    student_id: Optional[str] = None
    user_id: Optional[str] = None

    message: str
    history: List[Dict[str, Any]] = Field(default_factory=list)

    # Kiểu cũ (đang dùng trong orchestrator.run)
    profile_type: Optional[str] = None
    profile_region: Optional[str] = None

    # Kiểu mới (UI gửi kèm metadata)
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        extra = "allow"


class ChatResponse(BaseModel):
    reply: str


def _extract_profile(req: ChatRequest) -> tuple[Optional[str], Optional[str]]:
    profile_type = req.profile_type
    profile_region = req.profile_region

    if req.metadata and isinstance(req.metadata, dict):
        md = req.metadata

        # hỗ trợ nhiều key naming khác nhau
        profile_type = profile_type or md.get("profile_type") or md.get("student_type") or md.get("type")
        profile_region = profile_region or md.get("profile_region") or md.get("region") or md.get("country")

    return profile_type, profile_region


def _extract_student_id(req: ChatRequest) -> str:
    sid = req.student_id or req.user_id
    if not sid:
        raise HTTPException(status_code=422, detail="Missing student_id (or user_id)")
    return sid


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        student_id = _extract_student_id(req)
        profile_type, profile_region = _extract_profile(req)

        logger.info(
            "chat request student_id=%s profile_type=%s profile_region=%s",
            student_id,
            profile_type,
            profile_region,
        )

        reply = orchestrator.run(
            student_id=student_id,
            user_message=req.message,
            history=req.history,
            profile_type=profile_type,
            profile_region=profile_region,
        )
        return ChatResponse(reply=reply)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("chat error")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID}
