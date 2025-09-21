# app/agent.py
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor

# NOTE: import the Tool objects (save_tools, savejson_tool, etc.)
from tools import (
    search_tools,
    wiki_tool,
    savejson_tool,
    loadurl_tool,
    save_tools,   # <-- this is the TXT-saving Tool (not the function)
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

parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         """You are a careful research assistant. You can use tools to search, fetch a URL, and save JSON/TXT.
Return ONLY valid JSON matching {format_instructions} (no prose outside JSON).

Rules:
- Include at least 3 source URLs in "sources" (deduplicated, working links).
- Add 2-3 bullets in "why_these_sources" explaining selection (authority, recency, diversity).
- In "freshness_notes", indicate which sources are <30 days old and which look older.
- Keep "summary" concise but factual and cite sources inline by [#] indices when natural.
- Respect "style": explainer, news, pros & cons, timeline.
- If query looks medical/financial/legal advice, respond conservatively (safe mode).
"""),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# Expose all tools (Search, Wikipedia, Save JSON, Load URL, Save TXT)
tools = [search_tools, wiki_tool, savejson_tool, loadurl_tool, save_tools]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)

agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=tools)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def run_research(query: str) -> dict:
    raw = agent_executor.invoke({"query": query})
    try:
        obj = parser.parse(raw["output"])
        return obj.model_dump()
    except Exception:
        # graceful fallback if the LLM returns slightly off-format JSON
        return {
            "topic": query,
            "style": "explainer",
            "summary": raw.get("output", ""),
            "sources": [],
            "tools_used": [],
            "why_these_sources": [],
            "freshness_notes": ""
        }
