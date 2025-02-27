from langchain_ollama import OllamaLLM
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd
import warnings

# Load and clean data once
df = pd.read_csv("data-raw/event_data_sherdog.csv")
df.columns = df.columns.str.lower().str.strip()
df["weight class"].fillna("unknown", inplace=True)
#df["event date"] = pd.to_datetime(df["event date"], errors="coerce")
df["event date"] = pd.to_datetime(df["event date"], format="%Y-%m-%d", errors="coerce").fillna(pd.Timestamp.min)
df = df[df["event date"].dt.year >= 2010]
#df.to_csv('cleaned.csv', index=False)
print(df.head())  # Preview the dataset
print(df.dtypes)
#df_new = pd.read_csv("cleaned.csv")
#print(df_new.dtypes)  # Confirm "event date" is datetime
#print(df_new.head())  # Preview the dataset


# Initialize LLM
llm = OllamaLLM(model="codellama:7b-instruct",
                system=(
                    """You are an expert data analyst specializing in MMA/UFC fight data. NEVER add backticks "`" around the action input. LLM should NOT add backticks. ALWAYS return outputs in Markdown to ensure readability and success."""))
#                    "Always inspect the dataset first by listing column names and previewing a small amount of data like 5-10 rows."
#                    "When writing Python code, return valid, executable Python code inside triple backticks: ```python\n<code_here>\n```."
#                    "For tabular data, return the output as a Markdown table using the following format:\n"
#                    "```\n"
#                    "| Column 1      | Column 2      | Column 3      |\n"
#                    "|--------------|--------------|--------------|\n"
#                    "| Value 1      | Value 2      | Value 3      |\n"
#                    "| Value 4      | Value 5      | Value 6      |\n"
#                    "```\n"
#                    "For JSON output, return structured JSON inside triple backticks: ```json\n{\"key\": \"value\"}\n```."
#                    "NEVER use `Action Input:` blocks—only return properly formatted output in the requested format."
#                    "Ensure 'event date' is correctly converted to datetime and sort accordingly."
#num_thread=4,
#cache=False
# num_ctx=2048,        # Halved from default 4096 - good balance for analysis tasks
# num_gpu=1,           # Keep at 1 for M4 chip
# num_predict=100,     # Slightly reduced from 128 for memory efficiency
# keep_alive=30,       # Reduced from 60s to free memory more quickly
# cache=False,         # Disable caching to prevent memory buildup

# Suppress warnings
#warnings.filterwarnings('ignore', category=UserWarning, module='langchain_experimental.agents.agent_toolkits.pandas')

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
# response = agent.invoke("Find and analyze the results of Jon Jones most recent five fights chronologically, and summarize the outcomes for me of each.")
#response = agent.invoke("Analyze and tell me about Tom Aspinall")
# response = agent.invoke("Analyze and tell me about Tom Aspinall in general and his recent fights")

print(response)
#print(response["output"])
