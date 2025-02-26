from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

LANGSMITH_TRACING=True
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_PROJECT="my-first-langsmith-project"

llm = ChatOpenAI()
#response = llm.invoke("Hello, world!")
response = llm.invoke("What is todays date? What are the top 5 current world events happening?")
print(response.content)
