"""
AI Meeting Intelligence Agent — Phase 2
FastAPI app: audio upload + Whisper transcription + LangChain agent + FAISS RAG
"""

import os
import uuid
import shutil
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from transcription_service import transcribe_audio, TranscriptResult
from memory_service import index_meeting
from agent_service import chat

# ── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Meeting Intelligence Agent",
    description="Upload meeting audio → transcribe → chat with your meeting using AI",
    version="2.0.0",
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".mp3", ".mp4", ".wav", ".m4a", ".webm", ".ogg"}

# ── In-memory store (swap for PostgreSQL in Phase 3+) ────────────────────────

meeting_store: dict[str, dict] = {}


# ── Pydantic models ───────────────────────────────────────────────────────────

class MeetingUploadResponse(BaseModel):
    meeting_id: str
    filename: str
    status: str
    message: str


class MeetingTranscriptResponse(BaseModel):
    meeting_id: str
    duration_seconds: float
    language: str
    speakers: list[str]
    segments: list[dict]
    full_transcript: str
    word_count: int
    created_at: str


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None   # optional — generated server-side if omitted


class ChatResponse(BaseModel):
    meeting_id: str
    session_id: str
    message: str
    response: str


# ── Helpers ───────────────────────────────────────────────────────────────────

def validate_file(file: UploadFile) -> None:
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )


def save_upload(file: UploadFile, meeting_id: str) -> Path:
    suffix = Path(file.filename).suffix.lower()
    dest = UPLOAD_DIR / f"{meeting_id}{suffix}"
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    return dest


async def process_meeting(meeting_id: str, file_path: Path) -> None:
    """
    Background task — three steps:
      1. Transcribe with Whisper + pyannote
      2. Store results
      3. Embed + index into FAISS  (new in Phase 2)
    """
    try:
        meeting_store[meeting_id]["status"] = "processing"

        # Step 1: Transcribe
        result: TranscriptResult = await transcribe_audio(file_path)

        # Step 2: Store
        meeting_store[meeting_id].update({
            "status":           "indexing",
            "duration_seconds": result.duration_seconds,
            "language":         result.language,
            "speakers":         result.speakers,
            "segments":         result.segments,
            "full_transcript":  result.full_transcript,
            "word_count":       len(result.full_transcript.split()),
            "created_at":       datetime.utcnow().isoformat(),
        })

        # Step 3: Embed + index into FAISS
        index_meeting(meeting_id, result.segments)

        meeting_store[meeting_id]["status"] = "completed"

    except Exception as exc:
        meeting_store[meeting_id]["status"] = "failed"
        meeting_store[meeting_id]["error"] = str(exc)
        raise


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post(
    "/meetings/upload",
    response_model=MeetingUploadResponse,
    status_code=202,
    tags=["meetings"],
)
async def upload_meeting(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """Upload a meeting recording. Returns a meeting_id for polling."""
    validate_file(file)
    meeting_id = str(uuid.uuid4())
    file_path  = save_upload(file, meeting_id)

    meeting_store[meeting_id] = {
        "meeting_id": meeting_id,
        "filename":   file.filename,
        "status":     "queued",
        "file_path":  str(file_path),
    }

    background_tasks.add_task(process_meeting, meeting_id, file_path)

    return MeetingUploadResponse(
        meeting_id=meeting_id,
        filename=file.filename,
        status="queued",
        message="Upload received. Transcription + indexing running in background.",
    )


@app.get(
    "/meetings/{meeting_id}/transcript",
    response_model=MeetingTranscriptResponse,
    tags=["meetings"],
)
async def get_transcript(meeting_id: str):
    """Retrieve the transcript for a completed meeting."""
    record = meeting_store.get(meeting_id)
    if not record:
        raise HTTPException(status_code=404, detail="Meeting not found.")

    status = record["status"]

    if status in ("queued", "processing", "indexing"):
        return JSONResponse(
            status_code=202,
            content={"meeting_id": meeting_id, "status": status, "message": f"Meeting is {status}..."}
        )
    if status == "failed":
        raise HTTPException(status_code=500, detail=f"Processing failed: {record.get('error')}")

    return MeetingTranscriptResponse(**{k: record[k] for k in MeetingTranscriptResponse.model_fields})


@app.post(
    "/meetings/{meeting_id}/chat",
    response_model=ChatResponse,
    tags=["chat"],
)
async def chat_with_meeting(meeting_id: str, request: ChatRequest):
    """
    Chat with your meeting using natural language.

    Example questions:
    - "Who owns the API redesign task?"
    - "What did Sarah say about the deadline?"
    - "Summarise the key decisions made"
    - "Has this topic come up in past meetings?"
    """
    record = meeting_store.get(meeting_id)
    if not record:
        raise HTTPException(status_code=404, detail="Meeting not found.")
    if record["status"] != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Meeting is still {record['status']}. Chat available once processing completes."
        )

    session_id = request.session_id or str(uuid.uuid4())

    response_text = await chat(
        meeting_id=meeting_id,
        session_id=session_id,
        user_message=request.message,
        meeting_store=meeting_store,
    )

    return ChatResponse(
        meeting_id=meeting_id,
        session_id=session_id,
        message=request.message,
        response=response_text,
    )


@app.get("/meetings/{meeting_id}/status", tags=["meetings"])
async def get_status(meeting_id: str):
    record = meeting_store.get(meeting_id)
    if not record:
        raise HTTPException(status_code=404, detail="Meeting not found.")
    return {"meeting_id": meeting_id, "status": record["status"]}


@app.get("/health", tags=["meta"])
async def health():
    return {"status": "ok", "version": "2.0.0"}
