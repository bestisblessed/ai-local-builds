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
df.to_csv('cleaned.csv')

# Initialize LLM
llm = OllamaLLM(model="deepseek-coder-v2",
                system=(
                    "You are an expert data analyst who specializes in MMA/UFC fighter and fight research like a professional handicapper in Vegas would. "
                    "CRITICAL: This dataset has converted ALL column names to lowercase. "
                    "Every time you are asked to do something, first ALWAYS inspect the dataset by listing its column names and analyzing some example rows "
                    "so you can understand the data and learn the structure to query correctly. "
                    "Do some general data integrity checks to ensure the data is clean and consistent and fix it if needed. "
                    "When referencing column names in code, use them exactly as they appear without adding quotes - for example use df[event date] not df['event date']. "                    "Before executing any action, first validate that the required columns exist. "
                    "Then lastly proceed to use the dataset and your knowledge to answer the questions."
                ))

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
response = agent.invoke("Find and list me the most recent three event names and a single fight from each one")
# response = agent.invoke("Find and list me the most recent three UFC EVENTS (not individual fights in those events)")
# response = agent.invoke("Find and list me the most recent three UFC EVENTS (not individual fights in those events) and a single fight from each one")

# print(response["output"])
print(response)
