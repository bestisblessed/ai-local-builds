# Post Process

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model_name = "cognitivecomputations/TinyDolphin-2.8-1.1b"
fine_tuned_model = "mma-ai-model-1"

# Load base model
model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype="auto")

# Merge LoRA adapters
model = PeftModel.from_pretrained(model, fine_tuned_model)
model = model.merge_and_unload()

# Save merged model
model.save_pretrained("mma-ai-model-1-merged")
tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model)
tokenizer.save_pretrained("mma-ai-model-1-merged")

print("LoRA model merged successfully!")