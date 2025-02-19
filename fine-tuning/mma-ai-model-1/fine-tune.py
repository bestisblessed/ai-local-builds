import pandas as pd
import torch
import pandas as pd
from datasets import Dataset
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import shutil
import os
# from datasets import load_dataset, concatenate_datasets
# from unsloth import FastLanguageModel, is_bfloat16_supported

torch.mps.empty_cache()
torch.mps.synchronize()
print("MPS memory cleared!")


shutil.rmtree('data')
os.makedirs('data')
shutil.copy('/Users/td/Code/mma-ai/Scrapers/data/event_data_sherdog.csv', 'data/')
shutil.copy('/Users/td/Code/mma-ai/Scrapers/data/fighter_info.csv', 'data/')


# Prepare event_data_sherdog.csv 
# Using Event Summaries

df = pd.read_csv("data/event_data_sherdog.csv")

# Fill missing values in 'Weight Class' with 'Unknown'
df["Weight Class"] = df["Weight Class"].fillna("Unknown")

# Define function to format data into prompt style
def format_fight_summary(row):
    return (
        f"Event Summary:\n"
        f"- Event: {row['Event Name']} at {row['Event Location']} on {row['Event Date']}\n"
        f"- Fighters: {row['Fighter 1']} vs. {row['Fighter 2']}\n"
        f"- Weight Class: {row['Weight Class']}\n"
        f"- Referee: {row['Referee']}\n\n"
        f"Fight Outcome:\n"
        f"- Winner: {row['Winning Fighter']}\n"
        f"- Method: {row['Winning Method']}\n"
        f"- Round: {row['Winning Round']}\n"
        f"- Time: {row['Winning Time']}\n\n"
        f"Additional Details:\n"
        f"- Fight Type: {row['Fight Type']}\n\n"
        f"Can you provide an in-depth analysis of this fight for betting purposes?"
        # f"Can you provide an in-depth analysis of this fight for betting purposes?"
    )
df["prompt"] = df.apply(format_fight_summary, axis=1)

# Select only the formatted prompts for training
df_prompts = df[["prompt"]]

# Save the formatted dataset for fine-tuning
formatted_file_path = "data/fight_summaries.csv"
df_prompts.to_csv(formatted_file_path, index=False)

# Display the first few formatted prompts
print(df_prompts.head())

# Output path for further fine-tuning
print(formatted_file_path)




# Fine Tune Mistral-7B-Instruct-v0.1

# Check if Apple MPS is available
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Load the event data
file_path = "data/fight_summaries.csv"
df = pd.read_csv(file_path)

# Convert to Hugging Face dataset
dataset = Dataset.from_pandas(df[["prompt"]])

# # Model configuration (LLaMA or Mistral)
# model_name = "meta-llama/Llama-2-7b-hf"  # Change to a model compatible with MPS
# model_name = "mistralai/Mistral-7B-Instruct-v0.1"  # Open-access alternative
model_name = "cognitivecomputations/TinyDolphin-2.8-1.1b"  # Open-access alternative
# model_name = "TheBloke/dolphin-2.7-mixtral-8x7b-GGUF"  # Open-access alternative
# model_name = "cognitivecomputations/Dolphin3.0-Llama3.1-8B-GGUF"  # Open-access alternative
#model_name = "meta-llama/Llama-3.1-8B"  # Open-access alternative
# max_seq_length = 1024
max_seq_length = 2048

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Ensure a valid padding token
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # torch_dtype=torch.float16,  # Use float16 for Apple MPS
    torch_dtype=torch.bfloat16,  # Use bfloat16 instead of float16 for MPS compatibility
    device_map={"": device},  # Ensures compatibility with MPS
)

# Tokenize dataset correctly
def tokenize_function(examples):
    return tokenizer(examples["prompt"], truncation=True, padding="max_length", max_length=max_seq_length)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Apply LoRA (Parameter Efficient Fine-Tuning)
lora_config = LoraConfig(
    r=16,  # LoRA rank
    lora_alpha=16,
    # r=32,  # LoRA rank
    # lora_alpha=32,
    lora_dropout=0,
    target_modules=["q_proj", "v_proj"],  # Adapt to your model
    bias="none",
)
model = get_peft_model(model, lora_config)

# Training Configuration
training_args = TrainingArguments(
    output_dir="outputs",
    per_device_train_batch_size=2,
    # per_device_train_batch_size=2,
    # per_device_train_batch_size=4,
    # gradient_accumulation_steps=8,
    gradient_accumulation_steps=4,
    # max_steps=60,  # Adjust for real training
    max_steps=200,
    learning_rate=2e-4,
    logging_steps=1,
    # fp16=True,  # Float16 for MPS efficiency
    bf16=True,  # Use bfloat16 instead of float16 for MPS compatibility
    save_total_limit=2,
    save_strategy="epoch",
    seed=42,
)

# Trainer setup
trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_dataset,
    # max_seq_length=max_seq_length,
    args=training_args,
)
# trainer = SFTTrainer(
#     model=model,
#     train_dataset=dataset,
#     dataset_text_field="prompt",
#     max_seq_length=max_seq_length,
#     tokenizer=tokenizer,
#     args=training_args,
# )

# Start fine-tuning
print("Starting fine-tuning...")
trainer.train()

# Save the fine-tuned model
model.save_pretrained("mma-ai-model-1")
tokenizer.save_pretrained("mma-ai-model-1")
print("Fine-tuning complete. Model saved to 'mma-ai-model-1'.")

