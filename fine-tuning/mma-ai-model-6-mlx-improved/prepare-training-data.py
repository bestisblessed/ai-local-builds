# import pandas as pd
# import json
# from datetime import datetime
# fighter_df = pd.read_csv("data-raw/fighter_info.csv")
# event_df = pd.read_csv("data-raw/event_data_sherdog.csv")
# def safe_to_datetime(date_str):
#     try:
#         return pd.to_datetime(date_str)
#     except:
#         return pd.NaT  
# fighter_df["Birth Date"] = fighter_df["Birth Date"].apply(safe_to_datetime)
# event_df["Event Date"] = pd.to_datetime(event_df["Event Date"], utc=True).dt.tz_localize(None)
# merged = pd.merge(
#     event_df,
#     fighter_df,
#     left_on=["Fighter 1", "Fighter 1 ID"],
#     right_on=["Fighter", "Fighter_ID"],
#     suffixes=('', '_fighter1')
# )
# merged = pd.merge(
#     merged,
#     fighter_df,
#     left_on=["Fighter 2", "Fighter 2 ID"],
#     right_on=["Fighter", "Fighter_ID"],
#     suffixes=('', '_fighter2')
# )
# print("Columns after merging:")
# print(merged.columns)
# merged = merged.drop(columns=["Fighter", "Fighter_fighter2"])
# merged["Age_fighter1"] = (merged["Event Date"] - merged["Birth Date"]).dt.days // 365
# merged["Age_fighter2"] = (merged["Event Date"] - merged["Birth Date_fighter2"]).dt.days // 365
# def height_to_inches(height):
#     try:
#         feet, inches = height.split("'")
#         return int(feet) * 12 + int(inches.replace('"', ''))
#     except:
#         return None  
# merged["Height_fighter1"] = merged["Height"].apply(height_to_inches)
# merged["Height_fighter2"] = merged["Height_fighter2"].apply(height_to_inches)
# def get_recent_fights(fighter_id, event_df):
#     fights = event_df[(event_df["Fighter 1 ID"] == fighter_id) | (event_df["Fighter 2 ID"] == fighter_id)]
#     fights = fights.sort_values(by="Event Date", ascending=False).head(5)
#     recent_fights = []
#     for _, fight in fights.iterrows():
#         opponent = fight["Fighter 2"] if fight["Fighter 1 ID"] == fighter_id else fight["Fighter 1"]
#         result = "Win" if fight["Winning Fighter"] == fighter_id else "Loss"
#         recent_fights.append(f"{opponent} ({result} by {fight['Winning Method']} in Round {fight['Winning Round']})")
#     return recent_fights
# def generate_prompt(row):
#     recent_fights_fighter1 = get_recent_fights(row["Fighter 1 ID"], event_df)
#     recent_fights_fighter1_str = "\n    - ".join(recent_fights_fighter1) if recent_fights_fighter1 else "No recent fights"
#     recent_fights_fighter2 = get_recent_fights(row["Fighter 2 ID"], event_df)
#     recent_fights_fighter2_str = "\n    - ".join(recent_fights_fighter2) if recent_fights_fighter2 else "No recent fights"
#     prompt = f"""
#     Analyze the fight between {row['Fighter 1']} and {row['Fighter 2']}:
#     - {row['Fighter 1']}: Wins={row['Wins']}, Losses={row['Losses']}, Height={row['Height_fighter1']}in, Age={row['Age_fighter1']}, Weight Class={row['Weight Class']}
#       Recent Fights:
#       - {recent_fights_fighter1_str}
#     - {row['Fighter 2']}: Wins={row['Wins_fighter2']}, Losses={row['Losses_fighter2']}, Height={row['Height_fighter2']}in, Age={row['Age_fighter2']}, Weight Class={row['Weight Class']}
#       Recent Fights:
#       - {recent_fights_fighter2_str}
#     Predict the winner, method, and round:
#     """
#     completion = f"{row['Winning Fighter']} wins by {row['Winning Method']} in Round {row['Winning Round']} at {row['Winning Time']}."
#     return {"prompt": prompt, "completion": completion}
# training_data = merged.apply(generate_prompt, axis=1).tolist()
# with open("data/training_data.jsonl", "w") as f:
#     for entry in training_data:
#         f.write(json.dumps(entry) + "\n")
# print("Training data generated and saved to 'training_data.jsonl'.")






# import pandas as pd
# import json
# from datetime import datetime
# from sklearn.model_selection import train_test_split

# # Load datasets
# fighter_df = pd.read_csv("data-raw/fighter_info.csv")
# event_df = pd.read_csv("data-raw/event_data_sherdog.csv")

# # Convert date columns to datetime with error handling
# def safe_to_datetime(date_str):
#     try:
#         return pd.to_datetime(date_str)
#     except:
#         return pd.NaT  # Return NaT (Not a Time) for invalid dates

# # Convert Birth Date to datetime (timezone-naive)
# fighter_df["Birth Date"] = fighter_df["Birth Date"].apply(safe_to_datetime)

# # Convert Event Date to datetime (timezone-naive)
# event_df["Event Date"] = pd.to_datetime(event_df["Event Date"], utc=True).dt.tz_localize(None)

# # Merge Fighter 1 stats
# merged = pd.merge(
#     event_df,
#     fighter_df,
#     left_on=["Fighter 1", "Fighter 1 ID"],
#     right_on=["Fighter", "Fighter_ID"],
#     suffixes=('', '_fighter1')
# )

# # Merge Fighter 2 stats
# merged = pd.merge(
#     merged,
#     fighter_df,
#     left_on=["Fighter 2", "Fighter 2 ID"],
#     right_on=["Fighter", "Fighter_ID"],
#     suffixes=('', '_fighter2')
# )

# # Debugging: Print column names to verify
# print("Columns after merging:")
# print(merged.columns)

# # Drop redundant columns (adjust based on actual column names)
# # Example: If the columns are named "Fighter_x" and "Fighter_y", drop them
# merged = merged.drop(columns=["Fighter", "Fighter_fighter2"])

# # Calculate age at fight time
# merged["Age_fighter1"] = (merged["Event Date"] - merged["Birth Date"]).dt.days // 365
# merged["Age_fighter2"] = (merged["Event Date"] - merged["Birth Date_fighter2"]).dt.days // 365

# # Convert height to inches
# def height_to_inches(height):
#     try:
#         feet, inches = height.split("'")
#         return int(feet) * 12 + int(inches.replace('"', ''))
#     except:
#         return None  # Return None for invalid height values

# merged["Height_fighter1"] = merged["Height"].apply(height_to_inches)
# merged["Height_fighter2"] = merged["Height_fighter2"].apply(height_to_inches)

# # Function to get the most recent 5 fights for a fighter
# def get_recent_fights(fighter_id, event_df):
#     fights = event_df[(event_df["Fighter 1 ID"] == fighter_id) | (event_df["Fighter 2 ID"] == fighter_id)]
#     fights = fights.sort_values(by="Event Date", ascending=False).head(5)
#     recent_fights = []
#     for _, fight in fights.iterrows():
#         opponent = fight["Fighter 2"] if fight["Fighter 1 ID"] == fighter_id else fight["Fighter 1"]
#         result = "Win" if fight["Winning Fighter"] == fighter_id else "Loss"
#         recent_fights.append(f"{opponent} ({result} by {fight['Winning Method']} in Round {fight['Winning Round']})")
#     return recent_fights

# # Generate prompt-completion pairs
# def generate_prompt(row):
#     # Get recent fights for Fighter 1
#     recent_fights_fighter1 = get_recent_fights(row["Fighter 1 ID"], event_df)
#     recent_fights_fighter1_str = "\n    - ".join(recent_fights_fighter1) if recent_fights_fighter1 else "No recent fights"

#     # Get recent fights for Fighter 2
#     recent_fights_fighter2 = get_recent_fights(row["Fighter 2 ID"], event_df)
#     recent_fights_fighter2_str = "\n    - ".join(recent_fights_fighter2) if recent_fights_fighter2 else "No recent fights"

#     prompt = f"""
#     Analyze the fight between {row['Fighter 1']} and {row['Fighter 2']}:
#     - {row['Fighter 1']}: Wins={row['Wins']}, Losses={row['Losses']}, Height={row['Height_fighter1']}in, Age={row['Age_fighter1']}, Weight Class={row['Weight Class']}
#       Recent Fights:
#       - {recent_fights_fighter1_str}
#     - {row['Fighter 2']}: Wins={row['Wins_fighter2']}, Losses={row['Losses_fighter2']}, Height={row['Height_fighter2']}in, Age={row['Age_fighter2']}, Weight Class={row['Weight Class']}
#       Recent Fights:
#       - {recent_fights_fighter2_str}
#     Predict the winner, method, and round:
#     """
#     completion = f"{row['Winning Fighter']} wins by {row['Winning Method']} in Round {row['Winning Round']} at {row['Winning Time']}."
#     return {"prompt": prompt, "completion": completion}

# # Apply the function to generate training data
# training_data = merged.apply(generate_prompt, axis=1).tolist()

# # Split the data into train, validation, and test sets
# train_data, test_data = train_test_split(training_data, test_size=0.2, random_state=42)  # 80% train, 20% test
# train_data, valid_data = train_test_split(train_data, test_size=0.125, random_state=42)  # 70% train, 10% validation

# # Export to JSONL files
# def export_to_jsonl(data, filename):
#     with open(filename, "w") as f:
#         for entry in data:
#             f.write(json.dumps(entry) + "\n")

# export_to_jsonl(train_data, "data/train.jsonl")
# export_to_jsonl(valid_data, "data/valid.jsonl")
# export_to_jsonl(test_data, "data/test.jsonl")

# print("Training data generated and saved to 'train.jsonl', 'valid.jsonl', and 'test.jsonl'.")





from sklearn.model_selection import train_test_split
import pandas as pd
import json
import shutil
import os
import re

shutil.rmtree("data")
os.makedirs("data", exist_ok=True)

# Load datasets
fighter_df = pd.read_csv("data-raw/fighter_info.csv")
event_df = pd.read_csv("data-raw/event_data_sherdog.csv")
print("Length of fighter_df:", len(fighter_df))
print("Length of event_df:", len(event_df))

# Remove rows with missing values
print("Dropping rows with missing values")
fighter_df = fighter_df.dropna()
event_df = event_df.dropna()
print("Length of fighter_df:", len(fighter_df))
print("Length of event_df:", len(event_df))

# Convert date columns to datetime with error handling
def safe_to_datetime(date_str):
    try:
        return pd.to_datetime(date_str)
    except:
        return pd.NaT  # Return NaT (Not a Time) for invalid dates

# Convert Birth Date to datetime (timezone-naive)
fighter_df["Birth Date"] = fighter_df["Birth Date"].apply(safe_to_datetime)

# Convert Event Date to datetime (timezone-naive)
event_df["Event Date"] = pd.to_datetime(event_df["Event Date"], utc=True).dt.tz_localize(None)

# Merge Fighter 1 stats
merged = pd.merge(
    event_df,
    fighter_df,
    left_on=["Fighter 1", "Fighter 1 ID"],
    right_on=["Fighter", "Fighter_ID"],
    suffixes=('', '_fighter1')
)

# Merge Fighter 2 stats
merged = pd.merge(
    merged,
    fighter_df,
    left_on=["Fighter 2", "Fighter 2 ID"],
    right_on=["Fighter", "Fighter_ID"],
    suffixes=('', '_fighter2')
)

# Debugging: Print column names to verify
print("Columns after merging:")
print(merged.columns)

# Drop redundant columns (adjust based on actual column names)
merged = merged.drop(columns=["Fighter", "Fighter_fighter2"])

# Remove rows with missing values after merging
print("Length of merged dataset:", len(merged))
merged = merged.dropna()
print("Dropping rows with missing values")
print("Length of merged dataset:", len(merged))

# Calculate age at fight time
merged["Age_fighter1"] = (merged["Event Date"] - merged["Birth Date"]).dt.days // 365
merged["Age_fighter2"] = (merged["Event Date"] - merged["Birth Date_fighter2"]).dt.days // 365

# Convert height to inches
def height_to_inches(height):
    try:
        feet, inches = height.split("'")
        return int(feet) * 12 + int(inches.replace('"', ''))
    except:
        return None  # Return None for invalid height values

merged["Height_fighter1"] = merged["Height"].apply(height_to_inches)
merged["Height_fighter2"] = merged["Height_fighter2"].apply(height_to_inches)

# Function to get the most recent 5 fights for a fighter
def get_recent_fights(fighter_id, event_df):
    fights = event_df[(event_df["Fighter 1 ID"] == fighter_id) | (event_df["Fighter 2 ID"] == fighter_id)]
    fights = fights.sort_values(by="Event Date", ascending=False).head(5)
    recent_fights = []
    for _, fight in fights.iterrows():
        opponent = fight["Fighter 2"] if fight["Fighter 1 ID"] == fighter_id else fight["Fighter 1"]
        result = "Win" if fight["Winning Fighter"] == fighter_id else "Loss"
        recent_fights.append(f"{opponent} ({result} by {fight['Winning Method']} in Round {fight['Winning Round']})")
    return recent_fights

# Generate prompt-completion pairs
def generate_prompt(row):
    recent_fights_fighter1 = get_recent_fights(row["Fighter 1 ID"], event_df)
    recent_fights_fighter1_str = "\n    - ".join(recent_fights_fighter1) if recent_fights_fighter1 else "No recent fights"

    recent_fights_fighter2 = get_recent_fights(row["Fighter 2 ID"], event_df)
    recent_fights_fighter2_str = "\n    - ".join(recent_fights_fighter2) if recent_fights_fighter2 else "No recent fights"

    prompt = f"""
    Analyze the fight between {row['Fighter 1']} and {row['Fighter 2']}:
    - {row['Fighter 1']}: Wins={row['Wins']}, Losses={row['Losses']}, Height={row['Height_fighter1']}in, Age={row['Age_fighter1']}, Weight Class={row['Weight Class']}
      Recent Fights:
      - {recent_fights_fighter1_str}
    - {row['Fighter 2']}: Wins={row['Wins_fighter2']}, Losses={row['Losses_fighter2']}, Height={row['Height_fighter2']}in, Age={row['Age_fighter2']}, Weight Class={row['Weight Class']}
      Recent Fights:
      - {recent_fights_fighter2_str}
    Predict the winner, method, and round:
    """
    completion = f"{row['Winning Fighter']} wins by {row['Winning Method']} in Round {row['Winning Round']} at {row['Winning Time']}."
    # return {"prompt": prompt, "completion": completion}
    return {"text": f"{prompt} {completion}"}

train_df, temp_df = train_test_split(merged, test_size=0.4, random_state=42)
valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

def save_jsonl(df, file_path):
    with open(file_path, "w") as outfile:
        for _, row in df.iterrows():
            json_obj = generate_prompt(row)
            outfile.write(json.dumps(json_obj) + "\n")

train_path = "data/train.jsonl"
valid_path = "data/valid.jsonl"
test_path = "data/test.jsonl"

save_jsonl(train_df, train_path)
save_jsonl(valid_df, valid_path)
save_jsonl(test_df, test_path)

# Clean JSONL files
def clean_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\s*:\s*', ': ', text)
    return text

def clean_jsonl(file_path):
    with open(file_path, "r") as infile:
        lines = [json.loads(line) for line in infile]
    with open(file_path, "w") as outfile:
        for data in lines:
            data["text"] = clean_text(data["text"])
            data["text"] = re.sub(r'Weight Class=nan', 'Weight Class=Unknown', data["text"])
            data["text"] = re.sub(r'-\s+', '- ', data["text"])
            outfile.write(json.dumps(data) + "\n")

clean_jsonl(train_path)
clean_jsonl(valid_path)
clean_jsonl(test_path)

print(train_path, valid_path, test_path)
