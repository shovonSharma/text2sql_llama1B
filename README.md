# text2sql_llama1B

The objective of this project is to produce SQL Query from Natural language.
For this task, i am finetuning __Llama-3.2-1B-Instruct-bnb-4bit__
on __Spyder__ dataset (a large-scale complex and cross-domain semantic parsing and text-to-SQL dataset annotated by 11 Yale students)
using __Unsloth AI__.
Additionally, i am applying __4bit quantization (PTQ)__ on this model to speed up inference, reduce memory usage and compute.
