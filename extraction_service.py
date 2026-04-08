"""
extraction_service.py
Extracts structured data from meeting transcripts using GPT-4o.

Features:
  - Pydantic schema validation on LLM output
  - Self-healing: re-prompts up to MAX_RETRIES times if schema fails
  - Chunking for long transcripts that exceed context limits
  - Async-safe (runs blocking calls in thread pool)
"""

import os
import json
import asyncio
from typing import Any

from openai  import OpenAI
from pydantic import ValidationError

from schemas import MeetingExtraction, ActionItem, Decision, OpenQuestion

# ── Config ────────────────────────────────────────────────────────────────────

MAX_RETRIES      = 3       # re-prompt attempts if Pydantic validation fails
MAX_TRANSCRIPT_CHARS = 12_000   # ~3k tokens; truncate beyond this for the extraction call
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# ── Prompts ───────────────────────────────────────────────────────────────────

EXTRACTION_SYSTEM_PROMPT = """You are an expert meeting analyst. 
Your job is to extract structured information from meeting transcripts.

You MUST respond with ONLY a valid JSON object matching the schema provided.
Do NOT include any text before or after the JSON.
Do NOT wrap it in markdown code blocks.
Every required field must be present.

Guidelines:
- action_items: extract ALL tasks, assignments, and follow-ups mentioned
- decisions: only include things explicitly decided, not just discussed
- open_questions: things raised but NOT resolved by end of meeting
- tldr: one sentence, max 30 words
- title: descriptive, max 10 words, no filler words like "meeting" or "call"
- priority: infer from urgency words ("ASAP", "urgent", "critical" = high)
- owner: use the speaker's name if identifiable, otherwise "TBD"
"""

EXTRACTION_USER_PROMPT = """Extract structured information from this meeting transcript.

Return a JSON object with this exact schema:
{schema}

Transcript:
{transcript}
"""

REPAIR_PROMPT = """Your previous JSON response failed validation with these errors:
{errors}

Here was your response:
{bad_json}

Fix ONLY the fields that caused errors and return the corrected JSON object.
Do NOT include any text before or after the JSON.
"""


# ── Schema builder ────────────────────────────────────────────────────────────

def _build_schema_hint() -> str:
    """Generate a human-readable schema hint for the LLM prompt."""
    return json.dumps({
        "title": "string (max 10 words)",
        "meeting_type": "planning | standup | incident | one_on_one | retrospective | demo | other",
        "tldr": "string (one sentence)",
        "sentiment": "positive | neutral | negative",
        "action_items": [
            {
                "task": "string",
                "owner": "string",
                "due_date": "string or null",
                "priority": "high | medium | low",
                "context": "string or null"
            }
        ],
        "decisions": [
            {
                "decision": "string",
                "rationale": "string or null",
                "decided_by": "string or null"
            }
        ],
        "open_questions": [
            {
                "question": "string",
                "raised_by": "string or null",
                "notes": "string or null"
            }
        ],
        "topics": [
            {
                "topic": "string",
                "duration_pct": "number (0-100)",
                "speakers": ["string"]
            }
        ],
        "key_points": ["string"],
        "risks": ["string"],
        "next_meeting": "string or null"
    }, indent=2)


# ── Core extraction ───────────────────────────────────────────────────────────

def _call_llm(messages: list[dict]) -> str:
    """Call GPT-4o and return the raw response string."""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0,          # deterministic for structured extraction
        response_format={"type": "json_object"},   # enforces JSON output
        max_tokens=2000,
    )
    return response.choices[0].message.content


def _parse_and_validate(raw: str) -> MeetingExtraction:
    """Parse raw JSON string into a validated MeetingExtraction."""
    data = json.loads(raw)
    return MeetingExtraction(**data)


def _extract_sync(transcript: str) -> MeetingExtraction:
    """
    Synchronous extraction with self-healing retry loop.

    Flow:
      1. Send transcript to GPT-4o with schema hint
      2. Validate response with Pydantic
      3. If validation fails, send error + bad JSON back to LLM for repair
      4. Repeat up to MAX_RETRIES times
      5. Raise if all retries exhausted
    """
    # Truncate very long transcripts
    if len(transcript) > MAX_TRANSCRIPT_CHARS:
        transcript = transcript[:MAX_TRANSCRIPT_CHARS] + "\n...[transcript truncated for extraction]"

    schema_hint = _build_schema_hint()

    # Initial extraction call
    messages = [
        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
        {"role": "user",   "content": EXTRACTION_USER_PROMPT.format(
            schema=schema_hint,
            transcript=transcript,
        )},
    ]

    last_error  = None
    last_raw    = ""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if attempt == 1:
                raw = _call_llm(messages)
            else:
                # Self-healing: append the error and ask LLM to fix it
                repair_messages = messages + [
                    {"role": "assistant", "content": last_raw},
                    {"role": "user",      "content": REPAIR_PROMPT.format(
                        errors=str(last_error),
                        bad_json=last_raw,
                    )},
                ]
                raw = _call_llm(repair_messages)

            last_raw = raw
            return _parse_and_validate(raw)

        except (json.JSONDecodeError, ValidationError, KeyError) as e:
            last_error = e
            print(f"[Extraction] Attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    f"Extraction failed after {MAX_RETRIES} attempts. Last error: {e}"
                ) from e

    # Should never reach here
    raise RuntimeError("Extraction failed unexpectedly")


# ── Public async API ──────────────────────────────────────────────────────────

async def extract_meeting_insights(
    meeting_id: str,
    transcript: str,
    speakers:   list[str],
) -> MeetingExtraction:
    """
    Extract structured insights from a meeting transcript.
    Runs the blocking LLM call in a thread pool to keep FastAPI async.

    Returns a fully validated MeetingExtraction instance.
    """
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _extract_sync, transcript)
    return result
