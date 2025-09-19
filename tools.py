"""Tools are the things that the LLM/agents can use that we can either write ourself 
or we can bring in from things like the Langchain Community HUB"""

"""
here we will see how to write 3 different tools:
1. From the wikipedia
2. Go to duckduckgo and search
3. Custom tool that we will write ourself which can be any python function
"""

from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime

# Custom tool to save data to a text file
def save_to_txt(data: str, filename: str = "research_output.txt"):
    """Saves the given data to a text file with the specified filename."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_data = f"--- Research Output --- \nTimestamp: {timestamp}\n\n{data}\n\n"
    
    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_data)

save_tools = Tool(
    name="Save_Text_File",
    func=save_to_txt,
    description="Saves the given data to a text file. Input should be a string.",
)

search = DuckDuckGoSearchRun()
search_tools = Tool(
    name="Search",
    func=search.run,
    description="Searches the web for information",
)


api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
