from sklearn.model_selection import train_test_split
import pandas as pd
import json
import shutil
import os
import subprocess

shutil.rmtree("data")
os.makedirs("data", exist_ok=True)
# os.rmdir("data")
# os.rmdir("data-raw")
# os.makedirs("data", exist_ok=True)
# os.makedirs("data-raw", exist_ok=True)
# shutil.copy("../../Scrapers/data/event_data_sherdog.csv", "data-raw/")
# shutil.copy("../../Scrapers/data/fighter_info.csv", "data-raw/")

# Prepare Data
input_csv_path = "data-raw/event_data_sherdog.csv"  # Ensure this path is correct
df = pd.read_csv(input_csv_path)
train_df, temp_df = train_test_split(df, test_size=0.4, random_state=42)
valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
def save_jsonl(df, file_path):
    with open(file_path, "w") as outfile:
        for _, row in df.iterrows():
            prompt = (f"[INST] Analyze and predict the outcome of a potential fight between {row['Fighter 1']} vs {row['Fighter 2']} "
                      f"({row['Weight Class']}), Event: {row['Event Name']} [/INST]")

            answer = (f"Event Location: {row['Event Location']}; "
                      f"Event Date: {row['Event Date']}; "
                      f"Fighter 1 ID: {row['Fighter 1 ID']}; "
                      f"Fighter 2 ID: {row['Fighter 2 ID']}; "
                      f"Weight Class: {row['Weight Class']}; "
                      f"Winning Fighter: {row['Winning Fighter']}; "
                      f"Winning Method: {row['Winning Method']}; "
                      f"Winning Round: {row['Winning Round']}; "
                      f"Winning Time: {row['Winning Time']}; "
                      f"Referee: {row['Referee']}; "
                      f"Fight Type: {row['Fight Type']}")

            json_obj = {"text": f"{prompt} {answer}"}
            outfile.write(json.dumps(json_obj) + "\n")
            
train_path = "data/train.jsonl"
valid_path = "data/valid.jsonl"
test_path = "data/test.jsonl"
save_jsonl(train_df, train_path)
save_jsonl(valid_df, valid_path)
save_jsonl(test_df, test_path)
print(train_path, valid_path, test_path)

# subprocess.run(["mlx_lm.lora", "--model", "mistralai/Mistral-7B-Instruct-v0.3", "--train", "--data", "data", "--batch-size", "1", "--iters", "50"])
# subprocess.run(["mlx_lm.lora", "--model", "meta-llama/Llama-3.2-3B-Instruct", "--train", "--data", "data", "--batch-size", "2", "--iters", "100"])
# subprocess.run(["mlx_lm.lora --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --train --data data --batch-size 2 --iters 100"])
subprocess.run(["mlx_lm.lora", "--model", "mistralai/Mistral-7B-Instruct-v0.3", "--train", "--data", "data", "--batch-size", "2", "--iters", "300"])
# https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3
# # Create Modelfile
# with open("Modelfile", "w") as f:
#     f.write('''
# FROM mistral
# ADAPTER ./adapters
# ''')

# subprocess.run(["ollama create deepseek-r1:7b -f Modelfile"])
