# CiteMint — Research Agent (Streamlit)

**CiteMint** is a tiny agentic research app. It uses an LLM + web tools to search, read, and produce a concise summary with **clickable citations**, **“why these sources”** bullets, and **freshness notes**. One-click **JSON/Markdown** export included.

- **Live demo:** https://citemint.streamlit.app  
- **Source:** https://github.com/<your-username>/AI_Agent_From_Scratch  
- **Screenshot:**  
<img width="1137" height="561" alt="image" src="https://github.com/user-attachments/assets/4a5bd587-3b31-430e-9697-d12aafb5a387" />

<!--
---

## Table of Contents
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [Repo Layout](#repo-layout)
- [Quick Start (Local)](#quick-start-local)
- [Environment Variables](#environment-variables)
- [Deploy on Streamlit Cloud](#deploy-on-streamlit-cloud)
- [Configuration Notes](#configuration-notes)
- [Usage Tips](#usage-tips)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [License](#license)

---
-->

---

## Features

- **Agentic workflow** (LangChain tool-calling).
- **Structured search** (DuckDuckGo) → real `{title, link, snippet}` results (no invented URLs).
- **Wikipedia** + **LoadURL** (fetch & clean page text).
- **Citations you can click** (deduped).
- **Why these sources?** 2–3 bullets (authority / recency / diversity).
- **Freshness notes** (what’s current vs. older).
- **Safe mode** toggle (conservative tone on sensitive topics).
- **Export** to `.json` / `.md`.
- **Retry & backoff** for transient errors / rate limits.

---

## Tech Stack

- **Python**, **Streamlit**
- **LangChain** (agents + tool calling)
- `duckduckgo-search`, `wikipedia`
- `requests`, `beautifulsoup4`
- **Pydantic** (strict JSON parsing), **Tenacity** (retries)

Works with **one** provider:
- **OpenAI** (`gpt-4o-mini`) **or**
- **Anthropic** (`claude-3-5-sonnet-20241022`)

---

