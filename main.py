import os
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tools, wiki_tool, save_tools

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")

# Here specify all the fields that you wants as an output from LLM call
class ResearchResponse(BaseModel):
    topic : str
    summary : str
    sources : list[str]
    tools_used : list[str]

# llm = ChatOpenAI(model="gpt-4o-mini")
# # llm2 = ChatAnthropic(model="claude-3-5-sonnet-20241022")
# llm2 = ChatGoogleGenerativeAI(
#     model="gemini-2.0-flash",  # good default
#     google_api_key=(os.getenv("GOOGLE_API_KEY") or "").strip(),
#     temperature=0.2,
# )


llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=HF_TOKEN,
    max_new_tokens=700,
    temperature=0.2,
    do_sample=False,
    provider="sambanova",
)

llm2 = ChatHuggingFace(llm=llm, temperature=0.2)



# response = llm.invoke("What is the meaning of life")
# print(response)

parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# system is information to LLM so it know what it is supposed to do
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant. You will be provided with a topic to research paper.
            Answer the user query and use necessary tools.
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# .parser, partially fill in the promt by passing the format instructions, so now it used the parser and take the pydantic model and turn it to a stirn then give it to the promt

tools = [search_tools, wiki_tool, save_tools]

# Creating n running the agent
agent = create_tool_calling_agent(
    llm = llm2,
    prompt = prompt,
    tools = tools
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
query = input("What can i help you research?")
# verbose is to see the thought process of the agent

# promt varibale are in brackets
# char_history and agent_scratchpad will be handeled by the agent executor


raw_response = agent_executor.invoke({"query": query})

try:
    # this will get the o/p then the text from the 1st value and parse with our parser into a python object
    # structured_response = parser.parse(raw_response.get("output")[0]["text"])
    structured_response = parser.parse(raw_response["output"])
    print("Structured response:", structured_response)
except Exception as e:
    print("Error parsing response:", e, "Raw response:", raw_response)