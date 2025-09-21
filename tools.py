"""Tools are the things that the LLM/agents can use that we can either write ourself 
or we can bring in from things like the Langchain Community HUB"""

"""
here we will see how to write 3 different tools:
1. From the wikipedia
2. Go to duckduckgo and search
3. Custom tool that we will write ourself which can be any python function
"""

# tools.py

"""Tools the agent/LLM can call.
Includes: Wikipedia, DuckDuckGo search, Save to TXT, Save to JSON, and LoadURL.
"""

from datetime import datetime
import json
import os
import re
from typing import Optional, Union

import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool, StructuredTool


# ---------------------------
# Save to JSON (structured)
# ---------------------------

class SaveJSONArgs(BaseModel):
    # allow either a dict or JSON string; agent sometimes passes strings
    data: Union[dict, str] = Field(..., description="Structured research output to save (dict or JSON string)")
    filename: Optional[str] = Field(
        default=None,
        description="Optional filename like research_YYYYmmdd.json (saved under ./exports)"
    )

def save_json(data: Union[dict, str], filename: Optional[str] = None):
    # If data is a JSON string, parse it
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception:
            # if it isn't valid JSON, wrap it so we still save something useful
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
    description="Save structured research output as JSON into ./exports folder. "
                "Pass the full final JSON in the 'data' field, optionally a 'filename'.",
    func=save_json,
    args_schema=SaveJSONArgs,
)


# ---------------------------
# Save to TXT (original)
# ---------------------------

def save_to_txt(data: str, filename: str = "research_output.txt"):
    """Append plain-text research output to a .txt file."""
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_data = f"--- Research Output --- \nTimestamp: {timestamp}\n\n{data}\n\n"
    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_data)
    return {"saved_to": os.path.abspath(filename)}

save_tools = Tool(
    name="Save_Text_File",
    func=save_to_txt,
    description="Append text to a .txt file. Input should be a single string.",
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
    """Fetch a public URL and return cleaned text (first ~8k chars).
    Intended for quick recency/content checks.
    """
    if not url.startswith(("http://", "https://")):
        return "ERROR: URL must start with http:// or https://"
    try:
        headers = {"User-Agent": "Mozilla/5.0 (research-agent demo)"}
        r = requests.get(url, headers=headers, timeout=12)
        r.raise_for_status()
        return _clean_text(r.text)[:8000]
    except Exception as e:
        return f"ERROR fetching URL: {e}"

loadurl_tool = Tool(
    name="LoadURL",
    func=fetch_url,
    description="Fetch and return cleaned text content from a public URL.",
)


# ---------------------------
# Search (DuckDuckGo)
# ---------------------------

_search = DuckDuckGoSearchRun()
search_tools = Tool(
    name="Search",
    func=_search.run,
    description="Searches the web and returns a JSON list of {title, link, snippet}. Use these links; do not invent new ones.",
)


# ---------------------------
# Wikipedia (bumped limits)
# ---------------------------

# Increase depth vs. original (top_k_results=1, chars_max=100)
# to get more usable context for the agent.
_api_wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=2000)
wiki_tool = WikipediaQueryRun(api_wrapper=_api_wrapper)