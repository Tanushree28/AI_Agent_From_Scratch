# app/agent.py
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

# New agent API + structured output
try:
    from langchain.agents import create_agent
except ImportError:
    # Some versions don't re-export it from langchain.agents
    from langchain.agents.agent import create_agent  # type: ignore

from langchain.agents.structured_output import ToolStrategy

# NOTE: import Tool objects
from tools import (
    search_tools,
    wiki_tool,
    savejson_tool,
    loadurl_tool,
    save_tools,
)

load_dotenv()


class ResearchResponse(BaseModel):
    topic: str
    style: str = Field(description="One of: explainer, news, pros & cons, timeline")
    summary: str
    sources: list[str]
    tools_used: list[str]
    why_these_sources: list[str] = Field(default_factory=list, description="2-3 bullets")
    freshness_notes: str = Field(default="", description="What is fresh/stale")


SYSTEM_PROMPT = """You are a careful research assistant. You can use tools to search, fetch a URL, and save JSON/TXT.

Return a ResearchResponse object.

Rules:
- Include at least 3 source URLs in "sources" (deduplicated, working links).
- Add 2-3 bullets in "why_these_sources" explaining selection (authority, recency, diversity).
- In "freshness_notes", indicate which sources are <30 days old and which look older.
- Keep "summary" concise but factual and cite sources inline by [#] indices when natural.
- Respect "style": explainer, news, pros & cons, timeline.
- If query looks medical/financial/legal advice, respond conservatively (safe mode).
"""

tools = [search_tools, wiki_tool, savejson_tool, loadurl_tool, save_tools]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
    response_format=ToolStrategy(ResearchResponse),
)


def _extract_tool_names(messages) -> list[str]:
    """Best-effort extraction of called tool names from the run trace."""
    used: list[str] = []
    for m in messages or []:
        tool_calls = getattr(m, "tool_calls", None)
        if not tool_calls:
            continue
        for tc in tool_calls:
            if isinstance(tc, dict):
                name = tc.get("name")
            else:
                name = getattr(tc, "name", None)
            if name:
                used.append(name)

    # de-dupe preserve order
    seen = set()
    deduped = []
    for n in used:
        if n not in seen:
            seen.add(n)
            deduped.append(n)
    return deduped


def run_research(query: str) -> dict:
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})

    structured = (
        result.get("structured_response")
        or result.get("structuredResponse")
        or result.get("structured")
    )

    used_tools = _extract_tool_names(result.get("messages", []))

    try:
        if hasattr(structured, "model_dump"):
            data = structured.model_dump()
        elif isinstance(structured, dict):
            data = structured
        else:
            data = {}

        if not data.get("tools_used"):
            data["tools_used"] = used_tools

        return data

    except Exception:
        return {
            "topic": query,
            "style": "explainer",
            "summary": "",
            "sources": [],
            "tools_used": used_tools,
            "why_these_sources": [],
            "freshness_notes": ""
        }
