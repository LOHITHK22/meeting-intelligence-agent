"""
AI Meeting Intelligence Agent — Phase 1
FastAPI app: audio upload + Whisper transcription + speaker diarization
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

# ── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Meeting Intelligence Agent",
    description="Upload meeting audio → get transcript, speaker labels, and structured insights",
    version="1.0.0",
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".mp3", ".mp4", ".wav", ".m4a", ".webm", ".ogg"}
MAX_FILE_SIZE_MB = 500


# ── Pydantic models ───────────────────────────────────────────────────────────

class MeetingUploadResponse(BaseModel):
    meeting_id: str
    filename: str
    status: str              # "processing" | "completed" | "failed"
    message: str


class MeetingTranscriptResponse(BaseModel):
    meeting_id: str
    duration_seconds: float
    language: str
    speakers: list[str]
    segments: list[dict]     # [{speaker, start, end, text}, ...]
    full_transcript: str
    word_count: int
    created_at: str


# ── In-memory job store (swap for Redis or DB in production) ──────────────────

meeting_store: dict[str, dict] = {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def validate_file(file: UploadFile) -> None:
    """Raise HTTPException if the uploaded file is not a supported audio/video type."""
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )


def save_upload(file: UploadFile, meeting_id: str) -> Path:
    """Stream the uploaded file to disk and return its path."""
    suffix = Path(file.filename).suffix.lower()
    dest = UPLOAD_DIR / f"{meeting_id}{suffix}"
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    return dest


async def process_meeting(meeting_id: str, file_path: Path, original_filename: str) -> None:
    """
    Background task:
      1. Transcribe audio with Whisper
      2. Run speaker diarization
      3. Merge transcript + speaker labels
      4. Store results
    """
    try:
        meeting_store[meeting_id]["status"] = "processing"

        result: TranscriptResult = await transcribe_audio(file_path)

        meeting_store[meeting_id].update({
            "status": "completed",
            "duration_seconds": result.duration_seconds,
            "language": result.language,
            "speakers": result.speakers,
            "segments": result.segments,
            "full_transcript": result.full_transcript,
            "word_count": len(result.full_transcript.split()),
            "created_at": datetime.utcnow().isoformat(),
        })

    except Exception as exc:
        meeting_store[meeting_id]["status"] = "failed"
        meeting_store[meeting_id]["error"] = str(exc)
        raise


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post(
    "/meetings/upload",
    response_model=MeetingUploadResponse,
    status_code=202,
    summary="Upload a meeting recording",
    tags=["meetings"],
)
async def upload_meeting(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Audio or video recording of the meeting"),
):
    """
    Upload a meeting recording (mp3 / mp4 / wav / m4a / webm / ogg).

    Returns a `meeting_id` immediately. Transcription runs in the background.
    Poll `GET /meetings/{meeting_id}/transcript` to check status and retrieve results.
    """
    validate_file(file)

    meeting_id = str(uuid.uuid4())
    file_path = save_upload(file, meeting_id)

    # Seed the store so polling works right away
    meeting_store[meeting_id] = {
        "meeting_id": meeting_id,
        "filename": file.filename,
        "status": "queued",
        "file_path": str(file_path),
    }

    # Kick off transcription without blocking the HTTP response
    background_tasks.add_task(process_meeting, meeting_id, file_path, file.filename)

    return MeetingUploadResponse(
        meeting_id=meeting_id,
        filename=file.filename,
        status="queued",
        message="Upload received. Transcription is running in the background.",
    )


@app.get(
    "/meetings/{meeting_id}/transcript",
    response_model=MeetingTranscriptResponse,
    summary="Get the transcript for a meeting",
    tags=["meetings"],
)
async def get_transcript(meeting_id: str):
    """
    Retrieve the transcription result for a given meeting.

    - **202** if still processing
    - **200** with full transcript once complete
    - **404** if meeting_id is unknown
    - **500** if transcription failed
    """
    record = meeting_store.get(meeting_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Meeting '{meeting_id}' not found.")

    status = record["status"]

    if status == "queued":
        return JSONResponse(status_code=202, content={"meeting_id": meeting_id, "status": "queued", "message": "Waiting to start transcription."})

    if status == "processing":
        return JSONResponse(status_code=202, content={"meeting_id": meeting_id, "status": "processing", "message": "Transcription in progress — check back in a moment."})

    if status == "failed":
        raise HTTPException(status_code=500, detail=f"Transcription failed: {record.get('error', 'unknown error')}")

    # status == "completed"
    return MeetingTranscriptResponse(
        meeting_id=meeting_id,
        duration_seconds=record["duration_seconds"],
        language=record["language"],
        speakers=record["speakers"],
        segments=record["segments"],
        full_transcript=record["full_transcript"],
        word_count=record["word_count"],
        created_at=record["created_at"],
    )


@app.get(
    "/meetings/{meeting_id}/status",
    summary="Lightweight status check (no transcript body)",
    tags=["meetings"],
)
async def get_status(meeting_id: str):
    """Quick poll endpoint — returns status only, no heavy payload."""
    record = meeting_store.get(meeting_id)
    if not record:
        raise HTTPException(status_code=404, detail="Meeting not found.")
    return {"meeting_id": meeting_id, "status": record["status"]}


@app.get("/health", tags=["meta"])
async def health():
    return {"status": "ok"}
