"""
Tools the agent/LLM can call.
Includes: Wikipedia, DuckDuckGo search, Save to TXT, Save to JSON, and LoadURL.
"""

from datetime import datetime
import json
import os
import re
from typing import Optional, Union
import concurrent.futures

import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# Safer imports across LangChain versions
try:
    from langchain_core.tools import StructuredTool
except Exception:
    from langchain.tools import StructuredTool  # fallback

# ---------------------------
# Save to JSON (structured)
# ---------------------------

class SaveJSONArgs(BaseModel):
    data: Union[dict, str] = Field(
        ..., description="Structured research output to save (dict or JSON string)"
    )
    filename: Optional[str] = Field(
        default=None,
        description="Optional filename like research_YYYYmmdd.json (saved under ./exports)",
    )


def save_json(data: Union[dict, str], filename: Optional[str] = None):
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception:
            data = {"raw": data}

    os.makedirs("exports", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = filename or f"research_{ts}.json"
    path = os.path.join("exports", fname)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return {"saved_to": path}


savejson_tool = StructuredTool.from_function(
    name="SaveJSON",
    description=(
        "Save structured research output as JSON into ./exports folder. "
        "Pass the full final JSON in the 'data' field, optionally a 'filename'."
    ),
    func=save_json,
    args_schema=SaveJSONArgs,
)


# ---------------------------
# Save to TXT (structured)
# ---------------------------

class SaveTextArgs(BaseModel):
    data: str = Field(..., description="Text to append to the file.")
    filename: str = Field(default="research_output.txt", description="Path to the .txt file")


def save_to_txt(data: str, filename: str = "research_output.txt"):
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_data = f"--- Research Output --- \nTimestamp: {timestamp}\n\n{data}\n\n"
    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_data)
    return {"saved_to": os.path.abspath(filename)}


save_tools = StructuredTool.from_function(
    name="Save_Text_File",
    description="Append text to a .txt file. Provide 'data' and optional 'filename'.",
    func=save_to_txt,
    args_schema=SaveTextArgs,
)


# ---------------------------
# Load URL (fetch & clean)
# ---------------------------

def _clean_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for s in soup(["script", "style", "noscript"]):
        s.decompose()
    text = soup.get_text("\n")
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()


def fetch_url(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        return "ERROR: URL must start with http:// or https://"
    try:
        headers = {"User-Agent": "Mozilla/5.0 (research-agent demo)"}
        r = requests.get(url, headers=headers, timeout=12)
        r.raise_for_status()
        return _clean_text(r.text)[:8000]
    except Exception as e:
        return f"ERROR fetching URL: {e}"


class LoadURLArgs(BaseModel):
    url: str = Field(..., description="Public URL starting with http:// or https://")


loadurl_tool = StructuredTool.from_function(
    name="LoadURL",
    description="Fetch and return cleaned text content from a public URL.",
    func=fetch_url,
    args_schema=LoadURLArgs,
)


# ---------------------------
# Search (DuckDuckGo) - HARD timeout so it never hangs
# ---------------------------

class SearchArgs(BaseModel):
    query: str = Field(..., description="Search query")
    max_results: int = Field(default=5, ge=1, le=10, description="Max number of results")
    timeout_s: int = Field(default=8, ge=2, le=30, description="Timeout seconds")


def _ddg_search_blocking(query: str, max_results: int):
    """
    Blocking search implementation.
    Uses duckduckgo_search directly (more predictable than DuckDuckGoSearchRun).
    """
    try:
        from ddgs import DDGS

    except Exception as e:
        return [{"title": "", "link": "", "snippet": f"ERROR: duckduckgo_search not installed: {e}"}]

    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            title = r.get("title", "") or ""
            link = r.get("href") or r.get("url") or ""
            snippet = r.get("body", "") or ""
            if link:
                results.append({"title": title, "link": link, "snippet": snippet})

    return results


def search_web(query: str, max_results: int = 5, timeout_s: int = 8):
    """
    Returns a JSON list of {title, link, snippet}.
    Guaranteed to return within timeout_s (or returns an error item).
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_ddg_search_blocking, query, max_results)
        try:
            return fut.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError:
            return [{"title": "", "link": "", "snippet": f"ERROR: Search timed out after {timeout_s}s"}]
        except Exception as e:
            return [{"title": "", "link": "", "snippet": f"ERROR: Search failed: {e}"}]


search_tools = StructuredTool.from_function(
    name="Search",
    description="Search the web and returns a JSON list of {title, link, snippet}. Use these links; do not invent new ones.",
    func=search_web,
    args_schema=SearchArgs,
)


# ---------------------------
# Wikipedia
# ---------------------------

_api_wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=2000)
wiki_tool = WikipediaQueryRun(api_wrapper=_api_wrapper)
wiki_tool.name = "Wikipedia"
wiki_tool.description = "Search Wikipedia and return relevant summaries."
