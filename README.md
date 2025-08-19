# text2sql_llama1B

The objective of this project is to produce SQL Query from Natural language.
For this task, i am finetuning Llama-3.2-1B-Instruct-bnb-4bit
on Spyder dataset (a large-scale complex and cross-domain semantic parsing and text-to-SQL dataset annotated by 11 Yale students)
using Unsloth AI.
Additionally, i am doing 4bit quantization (PTQ) to speed up inference, reduce memory usage and compute.
