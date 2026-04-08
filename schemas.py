"""
schemas.py
Pydantic models for all structured outputs extracted from meetings.
Used by extraction_service.py and returned via the API.
"""

from datetime import date
from enum   import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ── Enums ─────────────────────────────────────────────────────────────────────

class Priority(str, Enum):
    high   = "high"
    medium = "medium"
    low    = "low"


class MeetingType(str, Enum):
    planning    = "planning"
    standup     = "standup"
    incident    = "incident"
    one_on_one  = "one_on_one"
    retrospect  = "retrospective"
    demo        = "demo"
    other       = "other"


class Sentiment(str, Enum):
    positive = "positive"
    neutral  = "neutral"
    negative = "negative"


# ── Sub-models ────────────────────────────────────────────────────────────────

class ActionItem(BaseModel):
    task:        str            = Field(..., description="Clear description of what needs to be done")
    owner:       str            = Field(..., description="Name of the person responsible")
    due_date:    Optional[str]  = Field(None, description="Due date if mentioned, e.g. 'next Friday' or '2024-06-15'")
    priority:    Priority       = Field(Priority.medium, description="Inferred priority")
    context:     Optional[str]  = Field(None, description="Brief context from the meeting for this task")

    @field_validator("task", "owner")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()


class Decision(BaseModel):
    decision:    str            = Field(..., description="What was decided")
    rationale:   Optional[str]  = Field(None, description="Why this decision was made")
    decided_by:  Optional[str]  = Field(None, description="Who made or drove the decision")


class OpenQuestion(BaseModel):
    question:    str            = Field(..., description="Unresolved question from the meeting")
    raised_by:   Optional[str]  = Field(None, description="Who raised it")
    notes:       Optional[str]  = Field(None, description="Any partial answers or context")


class TopicSegment(BaseModel):
    topic:       str            = Field(..., description="Topic discussed")
    duration_pct: float         = Field(..., description="Approximate % of meeting time spent on this topic", ge=0, le=100)
    speakers:    list[str]      = Field(default_factory=list, description="Speakers who contributed to this topic")


# ── Main extraction schema ────────────────────────────────────────────────────

class MeetingExtraction(BaseModel):
    """
    Full structured extraction from a meeting transcript.
    Every field is populated by the LLM and validated by Pydantic.
    """

    # Overview
    title:          str             = Field(..., description="Auto-generated meeting title (max 10 words)")
    meeting_type:   MeetingType     = Field(..., description="Type of meeting")
    tldr:           str             = Field(..., description="One-sentence summary of the entire meeting")
    sentiment:      Sentiment       = Field(..., description="Overall meeting sentiment")

    # Core extractions
    action_items:   list[ActionItem]    = Field(default_factory=list, description="All action items mentioned")
    decisions:      list[Decision]      = Field(default_factory=list, description="Key decisions made")
    open_questions: list[OpenQuestion]  = Field(default_factory=list, description="Unresolved questions")
    topics:         list[TopicSegment]  = Field(default_factory=list, description="Main topics covered")

    # Key highlights
    key_points:     list[str]       = Field(default_factory=list, description="3-5 most important points from the meeting")
    risks:          list[str]       = Field(default_factory=list, description="Any risks or blockers mentioned")
    next_meeting:   Optional[str]   = Field(None, description="Next meeting date/time if discussed")

    @field_validator("tldr", "title")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()

    @field_validator("title")
    @classmethod
    def title_length(cls, v: str) -> str:
        words = v.split()
        if len(words) > 12:
            return " ".join(words[:12])
        return v

    def to_markdown(self) -> str:
        """Render the extraction as a clean markdown report."""
        lines = [
            f"# {self.title}",
            f"**Type:** {self.meeting_type.value.replace('_', ' ').title()} | "
            f"**Sentiment:** {self.sentiment.value.title()}",
            "",
            f"## TL;DR",
            self.tldr,
            "",
        ]

        if self.key_points:
            lines += ["## Key points", *[f"- {p}" for p in self.key_points], ""]

        if self.action_items:
            lines.append("## Action items")
            for i, a in enumerate(self.action_items, 1):
                due = f" · due {a.due_date}" if a.due_date else ""
                lines.append(f"{i}. **{a.task}** — @{a.owner} `{a.priority.value}`{due}")
                if a.context:
                    lines.append(f"   > {a.context}")
            lines.append("")

        if self.decisions:
            lines.append("## Decisions")
            for d in self.decisions:
                by = f" *(by {d.decided_by})*" if d.decided_by else ""
                lines.append(f"- {d.decision}{by}")
                if d.rationale:
                    lines.append(f"  > {d.rationale}")
            lines.append("")

        if self.open_questions:
            lines.append("## Open questions")
            for q in self.open_questions:
                by = f" *(raised by {q.raised_by})*" if q.raised_by else ""
                lines.append(f"- {q.question}{by}")
            lines.append("")

        if self.risks:
            lines += ["## Risks & blockers", *[f"- {r}" for r in self.risks], ""]

        if self.next_meeting:
            lines += ["## Next meeting", self.next_meeting, ""]

        return "\n".join(lines)
