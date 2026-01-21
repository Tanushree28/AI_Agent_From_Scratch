# streamlit_app.py
import os, json, textwrap
import streamlit as st

from app.agent import run_research

st.set_page_config(page_title="Research Agent", page_icon="🔎", layout="centered")
st.title("🔎 Research Agent ")

# --- session state for the input so preset buttons can populate it ---
if "query_input" not in st.session_state:
    st.session_state.query_input = ""

def set_preset(val: str):
    st.session_state.query_text = val

col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_input(
        "What can I help you research?",
        value=st.session_state.query_input,
        key="query_text",
        placeholder="e.g., impact of CRISPR in agriculture",
    )
with col2:
    style = st.selectbox("Style", ["explainer", "news", "pros & cons", "timeline"])

safe = st.toggle("Safe mode (avoid risky advice)", value=True)

preset_cols = st.columns(4)
presets = [
    "What is quantum computing?",
    "AI in healthcare 2025",
    "Pros & Cons of remote work",
    "Timeline of Web3",
]
for i, p in enumerate(presets):
    preset_cols[i].button(p, on_click=set_preset, args=(p,))

# ✅ HF token check
hf_token = (
    os.getenv("HUGGINGFACEHUB_API_TOKEN")
    or os.getenv("HF_TOKEN")
    or st.secrets.get("HUGGINGFACEHUB_API_TOKEN", None)
    or st.secrets.get("HF_TOKEN", None)
)

if not hf_token:
    st.warning("No Hugging Face token found. Add HUGGINGFACEHUB_API_TOKEN to .env or Streamlit Secrets.")

with st.form("run_form", clear_on_submit=False):
    submitted = st.form_submit_button("Run")
    user_query_clean = query.strip()

if submitted and user_query_clean:
    user_q = f"[style={style}; safe={str(safe).lower()}] {user_query_clean}"
    with st.spinner("Researching..."):
        result = run_research(user_q)

    st.subheader("Summary")
    st.write(result.get("summary", ""))

    yt = result.get("why_these_sources", []) or []
    if yt:
        st.subheader("Why these sources?")
        for b in yt:
            st.markdown(f"- {b}")

    fn = result.get("freshness_notes", "") or ""
    if fn:
        st.info(fn)

    st.subheader("Sources")
    srcs = result.get("sources", []) or []
    if not srcs:
        st.info("No sources returned.")
    else:
        for i, s in enumerate(srcs, start=1):
            st.markdown(f"{i}. [{s}]({s})")

    st.caption("Tools used: " + (", ".join(result.get("tools_used", [])) or "—"))

    st.subheader("Download")
    json_bytes = json.dumps(result, indent=2).encode("utf-8")
    st.download_button(
        "Download JSON",
        data=json_bytes,
        file_name="research.json",
        mime="application/json",
    )

    md = f"# {result.get('topic','Research')}\n\n" + textwrap.dedent(f"""
    **Style:** {result.get('style','explainer')}

    ## Summary
    {result.get('summary','')}

    ## Why these sources?
    {chr(10).join([f"- {b}" for b in yt])}

    ## Sources
    {chr(10).join([f"- {s}" for s in srcs])}

    ## Freshness notes
    {fn}
    """)
    st.download_button(
        "Download Markdown",
        data=md.encode("utf-8"),
        file_name="research.md",
        mime="text/markdown",
    )

    with st.expander("Show raw JSON"):
        st.code(json.dumps(result, indent=2), language="json")

st.markdown("---")
st.caption("Built with LangChain + HF • Streamlit demo for portfolio")
