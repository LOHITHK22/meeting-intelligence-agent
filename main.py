"""
AI Meeting Intelligence Agent — Phase 3
FastAPI app: upload → transcribe → index → extract structured insights → chat
"""

import os
import uuid
import shutil
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from transcription_service  import transcribe_audio, TranscriptResult
from memory_service         import index_meeting
from agent_service          import chat
from extraction_service     import extract_meeting_insights
from schemas                import MeetingExtraction

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Meeting Intelligence Agent",
    description="Upload → transcribe → extract insights → chat with your meeting",
    version="3.0.0",
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {".mp3", ".mp4", ".wav", ".m4a", ".webm", ".ogg"}

meeting_store: dict[str, dict] = {}


# ── Pydantic models ───────────────────────────────────────────────────────────

class MeetingUploadResponse(BaseModel):
    meeting_id: str
    filename:   str
    status:     str
    message:    str


class MeetingTranscriptResponse(BaseModel):
    meeting_id:       str
    duration_seconds: float
    language:         str
    speakers:         list[str]
    segments:         list[dict]
    full_transcript:  str
    word_count:       int
    created_at:       str


class ChatRequest(BaseModel):
    message:    str
    session_id: str | None = None


class ChatResponse(BaseModel):
    meeting_id: str
    session_id: str
    message:    str
    response:   str


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
    Full processing pipeline — 4 steps:
      1. Transcribe  (Whisper + pyannote)
      2. Store raw transcript
      3. Index into FAISS
      4. Extract structured insights  (new in Phase 3)
    """
    try:
        meeting_store[meeting_id]["status"] = "processing"

        # Step 1: Transcribe
        result: TranscriptResult = await transcribe_audio(file_path)

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

        # Step 2: Index into FAISS
        index_meeting(meeting_id, result.segments)

        # Step 3: Extract structured insights
        meeting_store[meeting_id]["status"] = "extracting"

        extraction: MeetingExtraction = await extract_meeting_insights(
            meeting_id=meeting_id,
            transcript=result.full_transcript,
            speakers=result.speakers,
        )

        meeting_store[meeting_id]["extraction"]      = extraction.model_dump()
        meeting_store[meeting_id]["markdown_report"] = extraction.to_markdown()
        meeting_store[meeting_id]["status"]          = "completed"

    except Exception as exc:
        meeting_store[meeting_id]["status"] = "failed"
        meeting_store[meeting_id]["error"]  = str(exc)
        raise


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/meetings/upload", response_model=MeetingUploadResponse, status_code=202, tags=["meetings"])
async def upload_meeting(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload a meeting recording. Processing runs in the background."""
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
        message="Upload received. Processing pipeline started.",
    )


@app.get("/meetings/{meeting_id}/transcript", response_model=MeetingTranscriptResponse, tags=["meetings"])
async def get_transcript(meeting_id: str):
    """Get the raw transcript for a meeting."""
    record = meeting_store.get(meeting_id)
    if not record:
        raise HTTPException(status_code=404, detail="Meeting not found.")
    status = record["status"]
    if status != "completed":
        return JSONResponse(status_code=202, content={"meeting_id": meeting_id, "status": status})
    return MeetingTranscriptResponse(**{k: record[k] for k in MeetingTranscriptResponse.model_fields})


@app.get("/meetings/{meeting_id}/insights", tags=["insights"])
async def get_insights(meeting_id: str, format: str = "json"):
    """
    Get structured insights from the meeting.
    ?format=json (default) or ?format=markdown
    """
    record = meeting_store.get(meeting_id)
    if not record:
        raise HTTPException(status_code=404, detail="Meeting not found.")
    if record["status"] != "completed":
        return JSONResponse(status_code=202, content={"meeting_id": meeting_id, "status": record["status"]})
    if "extraction" not in record:
        raise HTTPException(status_code=404, detail="No insights found.")

    if format == "markdown":
        return JSONResponse(content={"meeting_id": meeting_id, "report": record["markdown_report"]})

    return JSONResponse(content={"meeting_id": meeting_id, **record["extraction"]})


@app.get("/meetings/{meeting_id}/action-items", tags=["insights"])
async def get_action_items(meeting_id: str, owner: str | None = None):
    """
    Get action items. Optional filter: ?owner=Sarah
    """
    record = meeting_store.get(meeting_id)
    if not record or record["status"] != "completed":
        raise HTTPException(status_code=404, detail="Meeting not found or not yet processed.")

    items = record.get("extraction", {}).get("action_items", [])
    if owner:
        items = [a for a in items if owner.lower() in a.get("owner", "").lower()]

    return {"meeting_id": meeting_id, "count": len(items), "action_items": items}


@app.post("/meetings/{meeting_id}/chat", response_model=ChatResponse, tags=["chat"])
async def chat_with_meeting(meeting_id: str, request: ChatRequest):
    """Chat with the meeting agent in natural language."""
    record = meeting_store.get(meeting_id)
    if not record:
        raise HTTPException(status_code=404, detail="Meeting not found.")
    if record["status"] != "completed":
        raise HTTPException(status_code=409, detail=f"Meeting is still {record['status']}.")

    session_id    = request.session_id or str(uuid.uuid4())
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
    return {"status": "ok", "version": "3.0.0"}
