# %%capture
# import os
# if "COLAB_" not in "".join(os.environ.keys()):
#     !pip install unsloth
# else:
#     # Do this only in Colab notebooks! Otherwise use pip install unsloth
#     !pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft trl triton cut_cross_entropy unsloth_zoo
#     !pip install sentencepiece protobuf "datasets>=3.4.1,<4.0.0" "huggingface_hub>=0.34.0" hf_transfer
#     !pip install --no-deps unsloth


# !pip install trl


from unsloth import FastLanguageModel
import torch
max_seq_length = 1024 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",      # Llama-3.1 2x faster
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",    # 4bit for 405b!
    "unsloth/Mistral-Small-Instruct-2409",     # Mistral 22b 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct",           # Phi-3.5 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",            # Gemma 2x faster!

    "unsloth/Llama-3.2-1B-bnb-4bit",           # NEW! Llama 3.2 models
    "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    "unsloth/Llama-3.2-3B-bnb-4bit",
    "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",

    "unsloth/Llama-3.3-70B-Instruct-bnb-4bit" # NEW! Llama 3.3 70B!
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct", # or choose "unsloth/Llama-3.2-3B-Instruct"
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)


model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)


from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from transformers import TrainingArguments
import json

# Load the Spider dataset
print("Loading Spider dataset...")
dataset = load_dataset("spider")

# Print dataset info
print(f"Dataset keys: {dataset.keys()}")
print(f"Train samples: {len(dataset['train'])}")
print(f"Validation samples: {len(dataset['validation'])}")

# Look at a sample
print("\nSample data:")
sample = dataset['train'][0]
for key, value in sample.items():
    print(f"{key}: {value}")


# Data preprocessing for Spider dataset
def format_spider_data(examples):
    """Format Spider dataset for instruction tuning"""
    texts = []
    for i in range(len(examples['question'])):
        # Create instruction-response pairs
        instruction = f"Convert the following natural language question to SQL:\nQuestion: {examples['question'][i]}"
        response = examples['query'][i]
        
        # Format as chat template
        text = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant that converts natural language questions to SQL queries.<|eot_id|><|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{response}<|eot_id|><|end_of_text|>"""
        texts.append(text)
    
    return {"text": texts}

# Apply formatting to the dataset
print("Formatting dataset...")
formatted_train = dataset['train'].map(format_spider_data, batched=True, remove_columns=dataset['train'].column_names)
formatted_val = dataset['validation'].map(format_spider_data, batched=True, remove_columns=dataset['validation'].column_names)

print(f"Formatted train samples: {len(formatted_train)}")
print(f"Formatted validation samples: {len(formatted_val)}")

# Check a formatted sample
print("\nFormatted sample:")
print(formatted_train[0]['text'][:500] + "...")


# Training configuration
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=formatted_train,
    eval_dataset=formatted_val,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=True,  # Enable for faster training
    args=TrainingArguments(
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=200,
        max_steps=200,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=100,
        eval_steps=200,
        save_steps=200,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        eval_strategy="steps",  # Changed from evaluation_strategy
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
    ),
)

# Start training
print("Starting training...")
trainer.train()

# Save the model
print("Saving model...")
model.save_pretrained("llama-3.2-1b-spider-sql")
tokenizer.save_pretrained("llama-3.2-1b-spider-sql")



# Evaluation and testing
def generate_sql(model, tokenizer, question, max_length=512):
    """Generate SQL query from natural language question"""
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant that converts natural language questions to SQL queries.<|eot_id|><|start_header_id|>user<|end_header_id|>

Convert the following natural language question to SQL:
Question: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode and extract the SQL query
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    sql_query = full_response.split("assistant\n\n")[-1].strip()
    
    return sql_query

# Test on validation samples
print("Testing model on validation samples...")
test_samples = dataset['validation'].select(range(5))  # Test first 5 samples

for i, sample in enumerate(test_samples):
    question = sample['question']
    ground_truth = sample['query']
    
    print(f"\n--- Sample {i+1} ---")
    print(f"Question: {question}")
    print(f"Ground Truth: {ground_truth}")
    
    predicted = generate_sql(model, tokenizer, question)
    print(f"Predicted: {predicted}")
    print("-" * 50)

# Function to calculate exact match accuracy
def calculate_exact_match(predictions, ground_truths):
    """Calculate exact match accuracy"""
    correct = 0
    for pred, truth in zip(predictions, ground_truths):
        if pred.strip().lower() == truth.strip().lower():
            correct += 1
    return correct / len(predictions)

# Evaluate on a larger subset
print("\nEvaluating on larger validation subset...")
eval_subset = dataset['validation'].select(range(10))  # Evaluate on first 10 samples
predictions = []
ground_truths = []

for sample in eval_subset:
    pred = generate_sql(model, tokenizer, sample['question'])
    predictions.append(pred)
    ground_truths.append(sample['query'])

accuracy = calculate_exact_match(predictions, ground_truths)
print(f"Exact Match Accuracy: {accuracy:.4f}")