"""
agent_service.py
LangChain agent with three custom tools:
  - search_this_meeting   : RAG over a specific meeting's transcript
  - search_all_meetings   : RAG across all past meetings
  - get_meeting_summary   : retrieve structured summary + action items
"""

import os
from langchain.agents         import AgentExecutor, create_openai_functions_agent
from langchain.prompts        import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools          import tool
from langchain_openai          import ChatOpenAI
from langchain.memory         import ConversationBufferWindowMemory
from langchain.schema         import SystemMessage

from memory_service import search_meeting, search_all_meetings

# ── LLM ───────────────────────────────────────────────────────────────────────

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    api_key=os.environ["OPENAI_API_KEY"],
)


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are Meeting Intelligence — an AI assistant that helps users 
get the most out of their meeting recordings.

You have access to:
- Full transcripts with speaker labels and timestamps
- Semantic search over current and past meetings
- Structured summaries and action items

When answering:
- Always cite the speaker and timestamp when referencing something said in a meeting
- Be concise and direct — users are busy professionals
- If you can't find something in the transcript, say so clearly
- Format action items as a numbered list with owner and due date when available
"""


# ── Tool factory ──────────────────────────────────────────────────────────────
# We use a factory so the meeting_id is baked into the tool at request time.
# This avoids global state and makes each agent session meeting-aware.

def make_tools(meeting_id: str, meeting_store: dict):
    """
    Create a set of LangChain tools scoped to a specific meeting.
    meeting_store: the in-memory dict from main.py (or your DB layer)
    """

    @tool
    def search_this_meeting(query: str) -> str:
        """
        Search the current meeting's transcript for information relevant to the query.
        Use this for questions about what was said, who said it, or decisions made
        in THIS meeting.

        Args:
            query: Natural language question about the meeting content
        """
        chunks = search_meeting(meeting_id, query, top_k=5)
        if not chunks:
            return "No relevant content found in this meeting's transcript."

        results = []
        for c in chunks:
            mins = int(c["start"] // 60)
            secs = int(c["start"] % 60)
            results.append(f"[{c['speaker']} @ {mins}:{secs:02d}] {c['text']}")

        return "\n\n".join(results)

    @tool
    def search_past_meetings(query: str) -> str:
        """
        Search across ALL past meeting transcripts for information relevant to the query.
        Use this for questions like 'When did we last discuss X?' or 
        'Has this topic come up before?'

        Args:
            query: Natural language question to search across all meetings
        """
        chunks = search_all_meetings(query, top_k_per_meeting=3)
        if not chunks:
            return "No relevant content found across past meetings."

        results = []
        for c in chunks:
            mins = int(c["start"] // 60)
            secs = int(c["start"] % 60)
            results.append(
                f"[Meeting {c['meeting_id'][:8]}... | {c['speaker']} @ {mins}:{secs:02d}] {c['text']}"
            )

        return "\n\n".join(results)

    @tool
    def get_meeting_overview(dummy: str = "") -> str:
        """
        Get a structured overview of the current meeting including:
        - Duration and speakers present
        - Full transcript (truncated to first 3000 chars for context)
        Use this when the user asks for a summary, agenda recap, or general overview.

        Args:
            dummy: Ignored — no input needed
        """
        record = meeting_store.get(meeting_id)
        if not record or record.get("status") != "completed":
            return "Meeting transcript is not yet available."

        speakers = ", ".join(record.get("speakers", []))
        duration = record.get("duration_seconds", 0)
        mins = int(duration // 60)
        secs = int(duration % 60)
        transcript_preview = record.get("full_transcript", "")[:3000]

        return f"""Meeting ID: {meeting_id}
Duration: {mins}m {secs}s
Speakers: {speakers}
Word count: {record.get('word_count', 0)}

--- Transcript (preview) ---
{transcript_preview}
{'...[truncated]' if len(record.get('full_transcript', '')) > 3000 else ''}
"""

    return [search_this_meeting, search_past_meetings, get_meeting_overview]


# ── Agent builder ─────────────────────────────────────────────────────────────

def build_agent(meeting_id: str, meeting_store: dict) -> AgentExecutor:
    """
    Build a LangChain OpenAI Functions agent for a specific meeting session.
    Each chat session gets its own agent + short-term memory (last 10 turns).
    """
    tools = make_tools(meeting_id, meeting_store)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=10,   # remember last 10 exchanges
    )

    return AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,           # logs tool calls to console — great for debugging
        max_iterations=5,       # prevent runaway loops
        handle_parsing_errors=True,
    )


# ── Session store ─────────────────────────────────────────────────────────────
# One agent executor per (meeting_id, session_id) pair.
# In production, replace with Redis-backed session management.

_agent_sessions: dict[str, AgentExecutor] = {}


def get_or_create_agent(meeting_id: str, session_id: str, meeting_store: dict) -> AgentExecutor:
    """Get an existing agent session or create a new one."""
    key = f"{meeting_id}:{session_id}"
    if key not in _agent_sessions:
        _agent_sessions[key] = build_agent(meeting_id, meeting_store)
    return _agent_sessions[key]


async def chat(
    meeting_id:  str,
    session_id:  str,
    user_message: str,
    meeting_store: dict,
) -> str:
    """
    Send a message to the meeting agent and get a response.
    Maintains conversation history per session.
    """
    agent = get_or_create_agent(meeting_id, session_id, meeting_store)
    result = await agent.ainvoke({"input": user_message})
    return result["output"]
