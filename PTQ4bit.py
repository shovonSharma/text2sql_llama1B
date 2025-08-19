# PTQ to convert the weights to 4bit

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# Define model and quantization parameters
model_name = "unsloth/Llama-3.2-1B-Instruct"  # Base model or your fine-tuned model path
output_dir = "llama-3.2-1b-4bit-quantized"  # Directory to save quantized model
device_map = "auto"  # Automatically map to available devices

# Configure 4-bit quantization with bitsandbytes
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Enable 4-bit quantization
    bnb_4bit_quant_type="nf4",  # Use NF4 (4-bit NormalFloat) quantization
    bnb_4bit_compute_dtype=torch.bfloat16,  # Compute in BF16 for better precision
    bnb_4bit_use_double_quant=True,  # Use double quantization for better accuracy
)

# Load the model with quantization
print("Loading model with 4-bit quantization...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map=device_map,
    trust_remote_code=True,
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# If fine-tuned LoRA adapters, merge it into the base model
# (Optional, only if only LoRA adapters from previous fine-tuning)
from peft import PeftModel
lora_adapter_path = "llama-3.2-1b-spider-sql"  # Path to your fine-tuned LoRA adapters
if lora_adapter_path:
    print("Merging LoRA adapters...")
    model = PeftModel.from_pretrained(model, lora_adapter_path)
    model = model.merge_and_unload()  # Merge LoRA weights into the base model

# Save the quantized model
print("Saving quantized model...")
model.save_pretrained(output_dir, safe_serialization=True)
tokenizer.save_pretrained(output_dir)

print(f"Quantized 4-bit model saved to {output_dir}")