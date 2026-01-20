# app/agent.py
import os
import json
import re
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

# New agent API
try:
    from langchain.agents import create_agent
except ImportError:
    from langchain.agents.agent import create_agent  # type: ignore

from tools import (
    search_tools,
    wiki_tool,
    savejson_tool,
    loadurl_tool,
    save_tools,
)

load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN") or ""


class ResearchResponse(BaseModel):
    topic: str
    style: str = Field(description="One of: explainer, news, pros & cons, timeline")
    summary: str
    sources: list[str]
    tools_used: list[str]
    why_these_sources: list[str] = Field(default_factory=list, description="2-3 bullets")
    freshness_notes: str = Field(default="", description="What is fresh/stale")


SYSTEM_PROMPT = """You are a careful research assistant. You can use tools to search, fetch a URL, and save JSON/TXT.

You MUST return ONLY valid JSON (no markdown, no extra text) that matches this schema:
{
  "topic": "...",
  "style": "explainer|news|pros & cons|timeline",
  "summary": "...",
  "sources": ["https://...", "https://...", "https://..."],
  "tools_used": ["Search", "Wikipedia", ...],
  "why_these_sources": ["...", "..."],
  "freshness_notes": "..."
}

Rules:
- Include at least 3 source URLs in "sources" (deduplicated, working links).
- Add 2-3 bullets in "why_these_sources" explaining selection (authority, recency, diversity).
- In "freshness_notes", indicate which sources are <30 days old and which look older (best-effort).
- Keep "summary" concise but factual and cite sources inline by [#] indices when natural.
- Respect "style": explainer, news, pros & cons, timeline.
- If query looks medical/financial/legal advice, respond conservatively (safe mode).
- Output JSON only. No markdown. No commentary.
"""

tools = [search_tools, wiki_tool, savejson_tool, loadurl_tool, save_tools]

# ✅ HF endpoint + chat wrapper (required by your installed versions)
hf_llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    provider="sambanova",  # change to "scaleway" or "ovhcloud" if sambanova fails
    task="text-generation",
    huggingfacehub_api_token=HF_TOKEN,
    max_new_tokens=700,
    temperature=0.2,
    do_sample=False,
)

llm2 = ChatHuggingFace(llm=hf_llm)

# ✅ IMPORTANT: No ToolStrategy / no forced tool_choice
agent = create_agent(
    model=llm2,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
)


# -----------------------
# Helpers
# -----------------------

def _extract_tool_names(messages) -> list[str]:
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

    seen = set()
    out = []
    for n in used:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(item.get("text") or item.get("content") or "")
            else:
                parts.append(str(item))
        return "\n".join([p for p in parts if p])
    return str(content)


def _get_final_ai_text(result: dict) -> str:
    messages = result.get("messages") or []

    # find last assistant/ai message
    for m in reversed(messages):
        mtype = getattr(m, "type", None)
        role = getattr(m, "role", None)
        if mtype in ("ai", "assistant") or role == "assistant":
            return _content_to_text(getattr(m, "content", ""))

    out = result.get("output")
    if isinstance(out, str):
        return out

    if messages:
        return _content_to_text(getattr(messages[-1], "content", ""))
    return ""


def _extract_json(text: str) -> dict:
    if not text:
        return {}

    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```$", "", t)

    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass

    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    return {}


def _extract_links_from_search_tool(messages) -> list[str]:
    links: list[str] = []
    for m in messages or []:
        if getattr(m, "type", None) != "tool":
            continue
        if getattr(m, "name", "") != "Search":
            continue

        content = _content_to_text(getattr(m, "content", ""))
        try:
            data = json.loads(content)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        link = item.get("link") or item.get("url")
                        if isinstance(link, str) and link.startswith("http"):
                            links.append(link)
        except Exception:
            pass

    seen = set()
    out = []
    for u in links:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _extract_search_snippets(messages) -> list[str]:
    snippets: list[str] = []
    for m in messages or []:
        if getattr(m, "type", None) != "tool":
            continue
        if getattr(m, "name", "") != "Search":
            continue

        content = _content_to_text(getattr(m, "content", ""))
        try:
            data = json.loads(content)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        snip = item.get("snippet") or ""
                        if isinstance(snip, str) and snip.strip():
                            snippets.append(snip.strip())
        except Exception:
            pass

    seen = set()
    out = []
    for s in snippets:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _build_summary_from_snippets(snippets: list[str], max_chars: int = 900) -> str:
    if not snippets:
        return ""
    text = " ".join(snippets)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def _ensure_schema(
    data: dict,
    query: str,
    used_tools: list[str],
    fallback_sources: list[str],
    fallback_snippets: list[str],
    raw_text: str,
) -> dict:
    if not isinstance(data, dict):
        data = {}

    data.setdefault("topic", query)
    data.setdefault("style", "explainer")
    data.setdefault("summary", "")
    data.setdefault("sources", [])
    data.setdefault("why_these_sources", [])
    data.setdefault("freshness_notes", "")
    data.setdefault("tools_used", used_tools)

    # normalize types
    if not isinstance(data["sources"], list):
        data["sources"] = []
    if not isinstance(data["why_these_sources"], list):
        data["why_these_sources"] = []
    if not isinstance(data["tools_used"], list):
        data["tools_used"] = used_tools

    # sources fallback
    if not data["sources"] and fallback_sources:
        data["sources"] = fallback_sources[:8]

    # summary fallback
    if not data["summary"]:
        s = _build_summary_from_snippets(fallback_snippets)
        if s:
            data["summary"] = s
        else:
            # last fallback: raw model text, but avoid dumping JSON
            if raw_text and not raw_text.strip().startswith("{"):
                data["summary"] = raw_text.strip()[:1500]

    # de-dupe sources
    seen = set()
    dedup = []
    for s in data["sources"]:
        if isinstance(s, str) and s and s not in seen:
            seen.add(s)
            dedup.append(s)
    data["sources"] = dedup

    return data


def run_research(query: str) -> dict:
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})

    messages = result.get("messages", []) or []
    used_tools = _extract_tool_names(messages)

    raw_text = _get_final_ai_text(result)
    parsed = _extract_json(raw_text)

    fallback_sources = _extract_links_from_search_tool(messages)
    fallback_snippets = _extract_search_snippets(messages)

    data = _ensure_schema(
        parsed,
        query=query,
        used_tools=used_tools,
        fallback_sources=fallback_sources,
        fallback_snippets=fallback_snippets,
        raw_text=raw_text,
    )

    # debug: remove later if you want
    data["_raw_model_output"] = raw_text

    return data
