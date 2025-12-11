# main.py — V12

import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from orchestrator import Orchestrator
from conversation_logging import init_db, attach_export_routes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wellbeing-v12")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_ID = os.getenv("GROQ_MODEL_ID", "llama-3.3-70b-versatile")

client = Groq(api_key=GROQ_API_KEY)
orchestrator = OrchestratorV12(model_id=MODEL_ID, client=client)

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
    student_id: str
    message: str
    history: list = []
    profile_type: str | None = None
    profile_region: str | None = None


class ChatResponse(BaseModel):
    reply: str


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    reply = orchestrator.run(
        student_id=req.student_id,
        user_message=req.message,
        history=req.history,
        profile_type=req.profile_type,
        profile_region=req.profile_region,
    )
    return ChatResponse(reply=reply)


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID}
