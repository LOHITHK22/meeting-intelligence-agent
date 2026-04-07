"""
transcription_service.py
Handles:
  1. Whisper transcription  (via openai-whisper or OpenAI API)
  2. Speaker diarization    (via pyannote.audio)
  3. Merging both into timed, speaker-labelled segments
"""

import os
import asyncio
from dataclasses import dataclass, field
from pathlib import Path

import whisper                          # pip install openai-whisper
from pyannote.audio import Pipeline    # pip install pyannote.audio


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Segment:
    speaker: str        # "Speaker A", "Speaker B", ...
    start: float        # seconds
    end: float          # seconds
    text: str           # transcribed words for this segment

    def to_dict(self) -> dict:
        return {
            "speaker": self.speaker,
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "text": self.text.strip(),
        }


@dataclass
class TranscriptResult:
    duration_seconds: float
    language: str
    speakers: list[str]
    segments: list[dict]
    full_transcript: str


# ── Model loading (loaded once, reused across requests) ───────────────────────

_whisper_model = None
_diarization_pipeline = None


def _get_whisper_model(model_size: str = "base") -> whisper.Whisper:
    """
    Load Whisper model lazily. Options: tiny / base / small / medium / large.
    "base" is a good balance of speed vs. accuracy for demos.
    Swap to "large" for production quality.
    """
    global _whisper_model
    if _whisper_model is None:
        print(f"[Whisper] Loading '{model_size}' model...")
        _whisper_model = whisper.load_model(model_size)
        print("[Whisper] Model loaded.")
    return _whisper_model


def _get_diarization_pipeline() -> Pipeline:
    """
    Load pyannote speaker diarization pipeline.
    Requires a HuggingFace token with access to:
      pyannote/speaker-diarization-3.1
    Set HF_TOKEN in your environment.
    """
    global _diarization_pipeline
    if _diarization_pipeline is None:
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise EnvironmentError(
                "HF_TOKEN environment variable not set. "
                "Get a token at https://hf.co/settings/tokens and accept the "
                "pyannote/speaker-diarization-3.1 model terms."
            )
        print("[pyannote] Loading diarization pipeline...")
        _diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
        print("[pyannote] Pipeline loaded.")
    return _diarization_pipeline


# ── Core logic ────────────────────────────────────────────────────────────────

def _run_whisper(file_path: Path) -> dict:
    """
    Run Whisper transcription and return the raw result dict.
    The result includes:
      - result["text"]     : full transcript string
      - result["language"] : detected language code ("en", "es", etc.)
      - result["segments"] : list of {id, start, end, text, ...}
    """
    model = _get_whisper_model(model_size=os.getenv("WHISPER_MODEL", "base"))
    result = model.transcribe(
        str(file_path),
        verbose=False,
        word_timestamps=True,   # needed for fine-grained speaker alignment
    )
    return result


def _run_diarization(file_path: Path) -> list[tuple[float, float, str]]:
    """
    Run speaker diarization and return a list of (start, end, speaker_label) tuples.
    Example: [(0.0, 4.2, 'SPEAKER_00'), (4.3, 9.8, 'SPEAKER_01'), ...]
    """
    pipeline = _get_diarization_pipeline()
    diarization = pipeline(str(file_path))

    turns = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns.append((turn.start, turn.end, speaker))
    return turns


def _merge_transcript_with_speakers(
    whisper_segments: list[dict],
    diarization_turns: list[tuple[float, float, str]],
) -> tuple[list[Segment], list[str]]:
    """
    Align Whisper word-level segments with diarization speaker turns.

    Strategy:
      For each Whisper segment, find the diarization turn with the
      maximum overlap — that speaker "owns" the segment.
    """

    # Build a readable speaker name map: SPEAKER_00 → "Speaker A"
    speaker_ids = sorted({turn[2] for turn in diarization_turns})
    label_map = {sid: f"Speaker {chr(65 + i)}" for i, sid in enumerate(speaker_ids)}

    def find_speaker(seg_start: float, seg_end: float) -> str:
        best_speaker = "Unknown"
        best_overlap = 0.0
        for (t_start, t_end, spk) in diarization_turns:
            overlap = min(seg_end, t_end) - max(seg_start, t_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = label_map.get(spk, spk)
        return best_speaker

    merged: list[Segment] = []
    for ws in whisper_segments:
        speaker = find_speaker(ws["start"], ws["end"])
        # Merge consecutive segments from the same speaker
        if merged and merged[-1].speaker == speaker and (ws["start"] - merged[-1].end) < 1.5:
            merged[-1].end = ws["end"]
            merged[-1].text += " " + ws["text"]
        else:
            merged.append(Segment(
                speaker=speaker,
                start=ws["start"],
                end=ws["end"],
                text=ws["text"],
            ))

    speakers_present = list(dict.fromkeys(s.speaker for s in merged))
    return merged, speakers_present


def _build_full_transcript(segments: list[Segment]) -> str:
    """
    Build a readable transcript string.
    Example:
      [Speaker A | 0:00] Hi team, let's get started...
      [Speaker B | 0:12] Sounds good, I'll share my screen.
    """
    lines = []
    for seg in segments:
        minutes = int(seg.start // 60)
        seconds = int(seg.start % 60)
        timestamp = f"{minutes}:{seconds:02d}"
        lines.append(f"[{seg.speaker} | {timestamp}] {seg.text.strip()}")
    return "\n".join(lines)


# ── Public API ────────────────────────────────────────────────────────────────

async def transcribe_audio(file_path: Path) -> TranscriptResult:
    """
    Full pipeline: Whisper transcription + pyannote diarization → TranscriptResult.
    Runs CPU-bound work in a thread pool so the async event loop stays unblocked.
    """
    loop = asyncio.get_event_loop()

    # Run both in parallel using thread pool (both are CPU/IO bound)
    whisper_result, diarization_turns = await asyncio.gather(
        loop.run_in_executor(None, _run_whisper, file_path),
        loop.run_in_executor(None, _run_diarization, file_path),
    )

    segments, speakers = _merge_transcript_with_speakers(
        whisper_result["segments"],
        diarization_turns,
    )

    # Estimate duration from last segment end time
    duration = segments[-1].end if segments else 0.0

    full_transcript = _build_full_transcript(segments)

    return TranscriptResult(
        duration_seconds=round(duration, 2),
        language=whisper_result.get("language", "unknown"),
        speakers=speakers,
        segments=[s.to_dict() for s in segments],
        full_transcript=full_transcript,
    )
