'''
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
import pandas as pd

df = pd.read_csv("titanic.csv")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
agent_executor = create_pandas_dataframe_agent(
    llm,
    df,
    agent_type="tool-calling",
    verbose=True
)
'''
from langchain_ollama import OllamaLLM
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_core.exceptions import OutputParserException
import pandas as pd
import warnings

# Load and clean data once
df = pd.read_csv("data-raw/event_data_sherdog.csv")
df.columns = df.columns.str.lower().str.strip()
df["weight class"].fillna("unknown", inplace=True)
df["event date"] = pd.to_datetime(df["event date"], errors="coerce")
df = df[df["event date"].dt.year >= 2010]
df.to_csv('data-raw/cleaned.csv')
df.dtypes()

# Initialize LLM
llm = OllamaLLM(model="deepseek-coder-v2",
                system=("You are an expert data analyst who specializes in MMA/UFC fighter and fight research and analysis like a professional handicapper in Vegas would. "
                        "Every time you are asked to do something, first ALWAYS inspect the dataset by listing its column names and analyzing some example rows so you can understand the data and learn the structure to query correctly. "
                        "Do some general data integrity checks to ensure the data is clean and consistent and fix it if needed. "
                        "Notice and remember how all the column names are lowercase for consitency for example. "
                        "Before executing any action, first validate that the required columns exist. "
                        "Then lastly proceed to use the dataset and your knowledge to answer the questions. "
                        "Don't just suggest code - run it and show the results. Always execute your working code using the python_repl_ast Action to see real results."))

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning, module='langchain_experimental.agents.agent_toolkits.pandas')

# Create agent
agent = create_pandas_dataframe_agent(
    llm, 
    df, 
    verbose=True, 
    allow_dangerous_code=True, 
    handle_parsing_errors=True
)

# Run query
# response = agent.invoke("Find and list me the most recent three event names and a single fight from each one")
# response = agent.invoke("Find and list me the most recent three UFC EVENTS (not individual fights in those events)")
# response = agent.invoke("Find and list me the most recent three UFC EVENTS (not individual fights in those events) and a single fight from each one")
response = agent.invoke("Find and analyze the results of Jon Jones most recent five fights chronologically, and summarize the outcomes for me of each.")

# print(response["output"])
print(response)
