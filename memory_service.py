"""
memory_service.py
Handles:
  1. Embedding meeting transcript segments using OpenAI embeddings
  2. Indexing them into FAISS for fast similarity search
  3. Retrieving relevant context for a given query
"""

import os
import json
import pickle
from pathlib import Path
from dataclasses import dataclass

import faiss
import numpy as np
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "text-embedding-3-small"   # cheap + fast; swap to -large for quality
EMBEDDING_DIM   = 1536                        # dimensions for text-embedding-3-small
INDEX_DIR       = Path("faiss_indexes")
INDEX_DIR.mkdir(exist_ok=True)

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class MemoryChunk:
    meeting_id: str
    speaker:    str
    start:      float
    end:        float
    text:       str

    def to_dict(self) -> dict:
        return {
            "meeting_id": self.meeting_id,
            "speaker":    self.speaker,
            "start":      self.start,
            "end":        self.end,
            "text":       self.text,
        }


# ── Embedding helpers ─────────────────────────────────────────────────────────

def _embed(texts: list[str]) -> np.ndarray:
    """
    Embed a list of strings using OpenAI and return a float32 numpy array
    of shape (len(texts), EMBEDDING_DIM).
    Batches automatically — OpenAI allows up to 2048 inputs per call.
    """
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    vectors = [item.embedding for item in response.data]
    return np.array(vectors, dtype=np.float32)


# ── Per-meeting FAISS index ───────────────────────────────────────────────────

class MeetingMemory:
    """
    One FAISS index per meeting.
    Persisted to disk so memory survives server restarts.
    """

    def __init__(self, meeting_id: str):
        self.meeting_id = meeting_id
        self.index_path  = INDEX_DIR / f"{meeting_id}.faiss"
        self.chunks_path = INDEX_DIR / f"{meeting_id}.chunks"
        self.index:  faiss.IndexFlatIP | None = None   # Inner Product (cosine after normalisation)
        self.chunks: list[MemoryChunk]         = []

    # ── Build ──────────────────────────────────────────────────────────────────

    def build(self, segments: list[dict]) -> None:
        """
        Embed all transcript segments and store them in a FAISS index.
        Call this once after transcription completes.

        segments: list of {speaker, start, end, text} dicts (from TranscriptResult)
        """
        if not segments:
            return

        self.chunks = [
            MemoryChunk(
                meeting_id=self.meeting_id,
                speaker=seg["speaker"],
                start=seg["start"],
                end=seg["end"],
                text=seg["text"],
            )
            for seg in segments
        ]

        texts   = [c.text for c in self.chunks]
        vectors = _embed(texts)

        # Normalise for cosine similarity via inner product
        faiss.normalize_L2(vectors)

        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.index.add(vectors)

        self._save()

    # ── Query ──────────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 5) -> list[MemoryChunk]:
        """
        Return the top_k most relevant transcript chunks for a query.
        Loads from disk if not already in memory.
        """
        if self.index is None:
            self._load()

        if self.index is None or self.index.ntotal == 0:
            return []

        q_vec = _embed([query])
        faiss.normalize_L2(q_vec)

        scores, indices = self.index.search(q_vec, min(top_k, self.index.ntotal))

        return [
            self.chunks[i]
            for i in indices[0]
            if i >= 0   # FAISS returns -1 for empty slots
        ]

    # ── Persistence ────────────────────────────────────────────────────────────

    def _save(self) -> None:
        faiss.write_index(self.index, str(self.index_path))
        with open(self.chunks_path, "wb") as f:
            pickle.dump(self.chunks, f)

    def _load(self) -> None:
        if self.index_path.exists() and self.chunks_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            with open(self.chunks_path, "rb") as f:
                self.chunks = pickle.load(f)

    def exists(self) -> bool:
        return self.index_path.exists()


# ── Global registry ───────────────────────────────────────────────────────────

_memory_registry: dict[str, MeetingMemory] = {}


def get_memory(meeting_id: str) -> MeetingMemory:
    """Get or create a MeetingMemory instance for the given meeting."""
    if meeting_id not in _memory_registry:
        _memory_registry[meeting_id] = MeetingMemory(meeting_id)
    return _memory_registry[meeting_id]


def index_meeting(meeting_id: str, segments: list[dict]) -> None:
    """Convenience function — embed + index a meeting's segments after transcription."""
    memory = get_memory(meeting_id)
    memory.build(segments)


def search_meeting(meeting_id: str, query: str, top_k: int = 5) -> list[dict]:
    """Search a single meeting's memory. Returns serialisable dicts."""
    memory = get_memory(meeting_id)
    chunks = memory.search(query, top_k=top_k)
    return [c.to_dict() for c in chunks]


def search_all_meetings(query: str, top_k_per_meeting: int = 3) -> list[dict]:
    """
    Search across ALL indexed meetings.
    Useful for questions like 'When did we last discuss the API redesign?'
    """
    results = []
    for index_file in INDEX_DIR.glob("*.faiss"):
        mid = index_file.stem
        memory = get_memory(mid)
        chunks = memory.search(query, top_k=top_k_per_meeting)
        results.extend([c.to_dict() for c in chunks])
    return results
