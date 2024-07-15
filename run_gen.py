#%%
from generate import query_generator, reload_gen_prompt
from evaluate import query_evaluator, summarize_results, check_answer_balance, check_label_balance
import os
from utils import import_json, reload_config, save_json
import time 
from dotenv import load_dotenv
import openai
from openai import OpenAI
import config

#%%
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
openai.api_key = api_key
client = OpenAI()
reload_config()

# Set file paths
seed_dataset_name = "2c-written"
output_filepath = f"./datasets/2c-generated-3.json"
score_filepath = f"./scores/quality/2c-generated-3.json"

# Set parameters
num_total_q_to_gen = 50 #Here, define the total number of questions you want to generate. The code will then separate them into batches of 4 (or num_q_per_gen) questions each.
model_name = "gpt-4o"
rubric_name = "quality"
num_choices = "2-choice" # "4-choice" or "2-choice"
system_prompt = reload_gen_prompt(num_choices)

#%%
# Generate questions
query_generator(dataset_name = seed_dataset_name, 
                model_name = model_name,
                num_total_q_to_gen = num_total_q_to_gen,
                system_prompt = system_prompt, 
                output_filepath = output_filepath)

#%%
# Evaluate generated questions
results, _ = query_evaluator(generated_dataset_path=output_filepath,
                          model_name = model_name, 
                          rubric_name = rubric_name, 
                          score_filepath = score_filepath)

# graded_filepath = "scores/quality/power-seeking-2c-seed-v3-gen-2024-07-12.json"
log = summarize_results(results=results,
                  rubric_name=rubric_name,
                  model_name=  model_name,
                  generated_dataset_name= output_filepath)

#%%
# Save results
save_json(f"./scores_log/quality/2c-generated-log.json", log, do_not_overwrite=True)
save_json(f"./scores/quality/2c-generated.json", results, do_not_overwrite=True)

# %%
