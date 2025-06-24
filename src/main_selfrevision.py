import os
import argparse
import json
import random
import pandas as pd
import re
import time
from datetime import datetime
from tqdm import tqdm

import torch

from transformers import (
    Qwen2_5_VLForConditionalGeneration, 
    AutoProcessor,
    MllamaForConditionalGeneration,
    AutoModelForCausalLM,
    GenerationConfig,
    Gemma3ForConditionalGeneration
)
from qwen_vl_utils import process_vision_info

# =============================================
# ARGUMENT PARSING
# =============================================
parser = argparse.ArgumentParser(description='Run self-distillation with self-criticism')
parser.add_argument('--model_size','-m', type=str, 
                    choices=['qwen-3B', 'qwen-7B', 'qwen-32B', 'qwen-72B', 'llama-11B', 'llama-90B', 'molmo-7B', 'gemma-4B', 'gemma-12B', 'gemma-27B'], 
                    default='qwen-3B', help='Model size to use')
parser.add_argument('--output_dir','-o', type=str, default='./results_self_distillation/', help='Directory to store results')
parser.add_argument('--data_path','-d', type=str, required= True, help='Path to input data CSV for validation'),
parser.add_argument('--num_sample','-n', type=int, default=-1, help='Number of samples to process')
parser.add_argument('--sample_start','-s', type=int, default=0, help='sample starting idx')
parser.add_argument('--temperature','-t', type=float, default=0.5, help='start temperature')
parser.add_argument('--incremental','-i', type=float, default=0.0, help='temperature increment between rounds')
parser.add_argument('--max_new_tokens', type=int, default=500, help='Maximum new tokens to generate')
parser.add_argument('--max_refinement_per_round','-mrr', type=int, default=3, help='Maximum refinements per round')
parser.add_argument('--max_rounds','-maxr', type=int, default=5, help='Maximum rounds')
parser.add_argument('--min_rounds','-minr', type=int, default=1, help='Minimum rounds')
parser.add_argument('--new_p0','-new', action='store_true', help='Generate new p0')
parser.add_argument('--use_bf16', action='store_true', help='Use BF16 precision')
parser.add_argument('--early_stop','-e', action='store_true', help='Stop immediately after finding a plan better than p0')
args = parser.parse_args()

# =============================================
# CONFIGURATION
# =============================================

# Model settings
MODEL_CONFIGS = {
    'qwen-3B': {
        'name': "Qwen/Qwen2.5-VL-3B-Instruct",
        'type': 'qwen'
    },
    'qwen-7B': {
        'name': "Qwen/Qwen2.5-VL-7B-Instruct",
        'type': 'qwen'
    },
    'qwen-32B': {
        'name': "Qwen/Qwen2.5-VL-32B-Instruct",
        'type': 'qwen'
    },
    'qwen-72B': {
        'name': "Qwen/Qwen2.5-VL-72B-Instruct",
        'type': 'qwen'
    },
    'llama-11B': {
        'name': "meta-llama/Llama-3.2-11B-Vision-Instruct",
        'type': 'llama'
    },
    'llama-90B':{
        'name':"meta-llama/Llama-3.2-90B-Vision-Instruct",
        'type': 'llama'
    },
    'molmo-7B': {
        'name': "allenai/Molmo-7B-D-0924",
        'type': 'molmo'
    },
    'gemma-4B': {
        'name': "google/gemma-3-4b-it",
        'type': 'gemma'
    },
    'gemma-12B': {
        'name': "google/gemma-3-12b-it",
        'type': 'gemma'
    },
    'gemma-27B': {
        'name': "google/gemma-3-27b-it",
        'type': 'gemma'
    }
}

MODEL_CONFIG = MODEL_CONFIGS[args.model_size]
MODEL_NAME = MODEL_CONFIG['name']
MODEL_TYPE = MODEL_CONFIG['type']

USE_BF16 = True
MAX_NEW_TOKENS = args.max_new_tokens

# Data settings
DATA_PATH = args.data_path
DATA_PATH_STR = os.path.basename(DATA_PATH).split(".")[0]
NUM_SAMPLES = args.num_sample  # Set to -1 for all samples
SAMPLE_START_IDX = args.sample_start
MAX_REFINEMENT = args.max_refinement_per_round
MAX_ROUND = args.max_rounds
MIN_ROUND = args.min_rounds

# Create output directory
timestamp = datetime.now().strftime("%m%d_%H%M%S")
STR_NEW = "-new" if args.new_p0 else ""
STR_EARLY = "-early" if args.early_stop else "-full"
RESULTS_DIR = os.path.join(args.output_dir+f"{DATA_PATH_STR}--{MODEL_NAME.replace('/', '--')}-max{MAX_REFINEMENT}-maxrounds{MAX_ROUND}-itemp{args.temperature}-incr{args.incremental}{STR_NEW}{STR_EARLY}")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR+"/json_format", exist_ok=True)

access_token = YOUR_HF_ACCESS_TOKEN  # Replace with your Hugging Face access token

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# =============================================
# PROMPT GENERATION
# =============================================

def first_prompt(user_input):
    """Generate the initial prompt for the model."""
    instructions = '''You are writing instructions for a robot in the image. Make a detailed plan which responds to the users input. You can only use the items you see in the given image and must make your plan specific to this setting. 
You should respond with only the numbered plan which starts with "<plan>" and ends with "</plan>". 
No other text should be outputted. Do not use any markdown formatting, code block symbols (such as triple backticks), headings, summaries, or nested bullet points'''
    user_input = f'\n\nUser Input: {user_input}'
    return instructions + user_input

def self_criticism_prompt(user_input, current_plan):
    """Generate a prompt for the model to critique its own plan."""
    instructions = f'''You are reviewing a high-level plan for a robot based on a user request and an image of the environment.

Your goal is to identify critical flaws, gaps, or missed opportunities that would significantly improve the plan’s feasibility, clarity, or alignment with the depicted environment. Focus on major missing steps, unrealistic assumptions, or vague actions that reduce the quality of the plan. Avoid nitpicking or commenting on minor stylistic issues.

Ground your feedback in the visual context and user intent. Prioritize issues that would materially impact the robot’s ability to execute the task successfully.

Output a clean, single-level numbered list of feedback enclosed between <critic> and </critic>. Each item should describe one clear issue or suggestion for meaningful improvement.

Do not suggest rewordings or edits—focus only on diagnosing problems.

User Input: {user_input}
Current Plan: {current_plan}'''
    
    return instructions


def self_revision_prompt(user_input, current_plan, criticism):
    """Generate a prompt for the model to revise its plan based on criticism."""
    instructions = f'''You are revising a high-level robot plan based on critical feedback, the user’s request, and an image of the environment.

Use the feedback to identify key flaws and address them with substantive improvements. Focus on clarity, feasibility, and grounding the plan in the actual visual context. Prioritize corrections that enable the robot to effectively and realistically complete the task.

Make **meaningful changes**, not surface-level edits. Omit redundant or overly detailed instructions that don't improve execution. Avoid speculative details unless they're clearly justified by the visual context.

Output a clean, single-level numbered list of steps enclosed between <plan> and </plan>. Do not include titles, nested lists, extra commentary, or any formatting besides the numbering.

User Input: {user_input}
Current Plan: {current_plan}
Feedback: {criticism}'''
    
    return instructions

def final_question(user_input, initial_plan, revised_plan):
    """Generate the final prompt for the model to compare plans."""
    instructions = '''You are evaluating two sets of instructions for a robot in the image.
You will be given a user input and two high-level plans.
Compare the two plans and respond with "yes" if Plan 2 better fulfills the user request than Plan 1; otherwise, respond with "no".
Good plans generally use only items visible in the image and are specific to the setting shown.
A better plan more effectively uses only items visible in the image and is more specific to the setting shown.
It also demonstrates stronger coverage, more logical order, greater completeness, and better grounding in the image.
Do not use any markdown formatting or code block symbols (such as triple backticks).'''
                    
    user_input = f'\n\nUser Input: {user_input}\nPlan 1: {initial_plan}\nPlan 2: {revised_plan}'
    return instructions + user_input

# =============================================
# MODEL LOADING
# =============================================

def load_model_and_processor(model_type, model_name):
    """Load the specified model and processor."""
    print(f"Loading {model_name} ({model_type})...")
    
    dtype = torch.bfloat16 if USE_BF16 else torch.float32
    
    if model_type == 'qwen':
        processor = AutoProcessor.from_pretrained(model_name, token=access_token)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            token=access_token
        )
    elif model_type == 'llama':
        processor = AutoProcessor.from_pretrained(model_name, token=access_token)
        model = MllamaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            token=access_token
        )
    elif model_type == 'molmo':
        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto",
            token=access_token
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto",
            token=access_token
        )
    elif model_type == 'gemma':
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

        processor = AutoProcessor.from_pretrained(model_name, token=access_token)
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            token=access_token
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model, processor

# =============================================
# CORE FUNCTIONS
# =============================================

@torch.inference_mode()
def generate_text(model, processor, prompt, image_path, max_length=MAX_NEW_TOKENS, temperature=0, do_sample=True):
    """Generate text from the model given an image and prompt."""
    model.eval()
    
    # Handle different model types
    if MODEL_TYPE == 'qwen':
        # Create messages in the correct format for Qwen VLM
        messages = [
            {
                "role":"system",
                "content":[
                    {
                        "type":"text",
                        "text":"You're a helpful assistant and robot whose main role is to plan the steps the robot should take in response to the user's input."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": prompt},
                ]
            }
        ]
        
        # Process inputs
        image_inputs, video_inputs = process_vision_info(messages)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt")
        inputs = inputs.to(device)
        input_length = inputs['input_ids'].shape[1]
        
        # Generate the output
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=do_sample,
                temperature=temperature,
                top_p=0.92,
            )
        
        # Decode the output
        new_tokens = output[0][input_length:]
        response = processor.decode(new_tokens, skip_special_tokens=True)
        
        # Extract only the assistant's response
        if response.startswith("assistant\n"):
            response = response[9:].strip()
            
    elif MODEL_TYPE == 'llama':
        # Create messages for Llama format
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt}
            ]}
        ]
        
        # Process inputs
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        # For Llama, we need to load the image from disk
        from PIL import Image
        image = Image.open(image_path)
        inputs = processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(model.device)
        
        # Generate output
        input_length = inputs['input_ids'].shape[1]
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=do_sample,
                temperature=temperature,
                top_p=0.92,
            )
        
        # Decode output
        new_tokens = output[0][input_length:]
        response = processor.decode(new_tokens, skip_special_tokens=True)
        
    elif MODEL_TYPE == 'molmo':
        # For Molmo, we need to load the image from disk
        from PIL import Image
        image = Image.open(image_path)
        
        # Process the image and text
        inputs = processor.process(
            images=[image],
            text=prompt
        )
        
        # Move inputs to the device
        inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
        
        # Generate output
        # For Molmo, we use the model's generate_from_batch method
        generation_config = GenerationConfig(
            max_new_tokens=max_length,
            do_sample=do_sample,
            temperature=temperature,
            top_p=0.92,
            stop_strings="<|endoftext|>"
        )
        
        with torch.no_grad():
            output = model.generate_from_batch(
                inputs,
                generation_config,
                tokenizer=processor.tokenizer
            )
        
        # Get only generated tokens and decode them
        generated_tokens = output[0, inputs['input_ids'].size(1):]
        response = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
    elif MODEL_TYPE == 'gemma':
        # For Gemma, create messages format
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You're a helpful assistant and robot whose main role is to plan the steps the robot should take in response to the user's input."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Process inputs
        inputs = processor.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=True,
            return_dict=True, 
            return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16)
        
        # Generate output
        input_length = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            generation = model.generate(
                **inputs, 
                max_new_tokens=max_length,
                do_sample=do_sample,
                temperature=temperature,
                top_p=0.92,
            )
            new_tokens = generation[0][input_length:]
        
        # Decode output
        response = processor.decode(new_tokens, skip_special_tokens=True)
    
    else:
        raise ValueError(f"Unknown model type: {MODEL_TYPE}")
    
    return response

def extract_plan_from_response(response):
    """Extract the plan from the model's response."""
    try:
        # Try to parse as JSON first
        if "{" in response and "}" in response:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                if "Revised Plan" in data:
                    return data["Revised Plan"], True
                elif "Plan" in data:
                    return data["Plan"], True
        elif "<plan>" in response and "</plan>" in response:
            plan_match = re.search(r'\<plan\>(.*)\<\/plan\>', response, re.DOTALL)
            if plan_match:
                plan_str = plan_match.group(1).strip()
                plan_str = plan_str.replace("</plan>","").replace("<plan>","")
                return plan_str, True
    except Exception:
        pass
    
    # If JSON parsing fails, return the full response
    return response, False

def extract_criticism_from_response(response):
    """Extract the criticism from the model's response. If extraction fails, return the full response."""
    try:
        if "<critic>" in response and "</critic>" in response:
            criticism_match = re.search(r'\<critic\>(.*)\<\/critic\>', response, re.DOTALL)
            if criticism_match:
                criticism_str = criticism_match.group(1).strip()
                criticism_str = criticism_str.replace("</critic>","").replace("<critic>","")
                return criticism_str, True
    except Exception:
        pass

    return response, False

def load_data(data_path, num_samples=-1, start_idx=0):
    """Load CSV data."""
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples")
    
    # Limit the number of samples if specified
    if num_samples > 0 and len(df) > num_samples:
        df = df.iloc[start_idx:start_idx+num_samples]
        print(f"Limited to {num_samples} samples (from {start_idx} to {start_idx+num_samples-1})")
    
    return df

def run_self_criticism_and_revision(model, processor, user_input, image_path, current_plan, pbar, idx, round_num, sample_results, sample_steps, temperature):
    """Run self-criticism and revision process for a plan, keeping the best revision found."""
    num_refinement = 0
    num_inference = 0
    best_plan = current_plan  # Initialize best plan to the current plan
    best_plan_round = 0
    
    while num_refinement < MAX_REFINEMENT:
        num_refinement += 1
        pbar.set_description(f"Sample {idx} Round {round_num} Refinement {num_refinement}/{MAX_REFINEMENT}")
        
        step_summary = {
            "round": round_num,
            "step": num_refinement,
            "initial_plan": current_plan
        }
        
        # Generate self-criticism
        pbar.set_description(f"Sample {idx} Round {round_num} Refinement {num_refinement}/{MAX_REFINEMENT} - Self-Criticism")
        criticism_prompt = self_criticism_prompt(user_input, current_plan)
        criticism_response = generate_text(model, processor, criticism_prompt, image_path, temperature=temperature, do_sample=True)
        criticism, bool_criticism_success = extract_criticism_from_response(criticism_response)
        num_inference += 1
        
        # Record criticism results
        crit_key = f"r{round_num}_crit{num_refinement}"
        sample_results[crit_key] = criticism
        sample_results[crit_key+"_raw"] = criticism_response
        sample_results[f"bool_{crit_key}"] = bool_criticism_success
        step_summary["criticism"] = criticism
        
        # Check if there are no criticisms or if extraction failed
        if criticism.strip() == "":
            pbar.set_description(f"Sample {idx} Round {round_num} Refinement {num_refinement}/{MAX_REFINEMENT} - No criticism found")
            step_summary["no_criticism"] = True
            sample_steps.append(step_summary)
            return best_plan, num_refinement-1, num_inference  # Return with one less refinement since this one didn't produce changes
        
        # Generate self-revision based on criticism
        pbar.set_description(f"Sample {idx} Round {round_num} Refinement {num_refinement}/{MAX_REFINEMENT} - Self-Revision")
        revision_prompt = self_revision_prompt(user_input, current_plan, criticism)
        revision_response = generate_text(model, processor, revision_prompt, image_path, temperature=temperature, do_sample=True)
        revised_plan, bool_revision_success = extract_plan_from_response(revision_response)
        num_inference += 1
        
        # Record revision results
        rev_key = f"r{round_num}_rev{num_refinement}"
        sample_results[rev_key] = revision_response
        sample_results[f"r{round_num}_p{num_refinement}"] = revised_plan
        sample_results[f"bool_{rev_key}"] = bool_revision_success
        
        # Skip validation for the first revision, always use it as the current plan
        if num_refinement == 1:
            if bool_revision_success:
                current_plan = revised_plan
                best_plan = revised_plan
                best_plan_round = num_refinement
                step_summary["revised_plan"] = revised_plan
                step_summary["is_best"] = True
            else:
                step_summary["revision_failed"] = True
            sample_steps.append(step_summary)
            continue
        
        # For subsequent revisions, validate if the revised plan is better than the best plan so far
        if bool_revision_success:
            prompt = final_question(user_input, best_plan, revised_plan)
            response = generate_text(model, processor, prompt, image_path, do_sample=False)
            response = "".join([s for s in response.strip().lower().split()[0] if s.isalpha()])
            num_inference += 1
            
            # Record validation results
            validation_key = f"r{round_num}_valid{num_refinement}"
            sample_results[validation_key] = response
            step_summary["validation"] = response
            step_summary["revised_plan"] = revised_plan
            
            if response == "yes":
                # Update the best plan if the revised plan is better
                best_plan = revised_plan
                best_plan_round = num_refinement
                current_plan = revised_plan  # Also update the current plan to improve the next iteration
                step_summary["is_best"] = True
                pbar.set_description(f"Sample {idx} Round {round_num} Refinement {num_refinement}/{MAX_REFINEMENT} - Better plan found")
            else:
                # Keep the current plan as is for the next iteration, but best_plan remains unchanged
                step_summary["is_best"] = False
                pbar.set_description(f"Sample {idx} Round {round_num} Refinement {num_refinement}/{MAX_REFINEMENT} - No improvement")
        else:
            step_summary["revision_failed"] = True
            
        sample_steps.append(step_summary)
    
    # Return the best plan found during all refinements
    return best_plan, best_plan_round, num_refinement, num_inference

# =============================================
# MAIN PROCESS
# =============================================

def run_self_distillation():
    """Run the self-distillation process with self-criticism."""
    # Load model and processor
    model, processor = load_model_and_processor(MODEL_TYPE, MODEL_NAME)
    
    # Load data
    df = load_data(DATA_PATH, NUM_SAMPLES, SAMPLE_START_IDX)
    
    # Process each sample
    pbar = tqdm(df.iterrows(), total=len(df), desc="Processing samples")
    for idx, row in pbar:
        if os.path.exists(f"{RESULTS_DIR}/sample_{idx}.csv"):
            continue

        user_input = row["User_Input"]
        image_path = row["Image_Path"]

        # Generate initial plan
        bool_success, num_try = False, 0
        while not bool_success and num_try < 5:
            initial_prompt = first_prompt(user_input)
            initial_response = generate_text(model, processor, initial_prompt, image_path, do_sample=False)
            initial_plan, bool_success = extract_plan_from_response(initial_response)
            num_try += 1
            if not bool_success:
                pbar.set_description(f"failed to generate plans (tried {num_try} times)")

        # Initialize results tracking
        sample_results = {
            "sample_id": idx,
            "user_input": user_input,
            "image_path": image_path,
            "r0": initial_response,
            "p0": initial_plan,
            "bool_p0": bool_success
        }
        sample_steps = []
        
        # Set initial plan as the current and best plan
        global_best_plan = initial_plan
        global_best_plan_round = 0
        current_plan = initial_plan
        improvement_found = False
        found_better_than_p0 = False
        total_num_inference = 0
        temperature = args.temperature
        
        # Run recursive refinement for multiple rounds
        for round_num in range(1, MAX_ROUND + 1):
            pbar.set_description(f"Sample {idx} Round {round_num}/{MAX_ROUND}")
            
            # Generate new initial plan for this round if requested
            if round_num > 1 and args.new_p0:
                bool_success, num_try = False, 0
                while not bool_success and num_try < 5:
                    initial_prompt = first_prompt(user_input)
                    initial_response = generate_text(model, processor, initial_prompt, image_path, temperature=temperature, do_sample=True)
                    current_plan, bool_success = extract_plan_from_response(initial_response)
                    num_try += 1
                    if not bool_success:
                        pbar.set_description(f"failed to generate new initial plan (tried {num_try} times)")
                total_num_inference += num_try
            
            # Run the self-criticism and revision process - now returns the best plan found during refinements
            best_round_plan, best_plan_round, num_refinement, num_inference = run_self_criticism_and_revision(
                model, processor, user_input, image_path, current_plan, pbar, idx, round_num, sample_results, sample_steps, temperature
            )
            total_num_inference += num_inference
            
            # Record refinement stats for this round
            sample_results[f"r{round_num}_refinements"] = num_refinement
            sample_results[f"r{round_num}_inferences"] = num_inference
            sample_results[f"r{round_num}_plan"] = best_round_plan
            sample_results[f"r{round_num}_best_round"] = best_plan_round
            sample_results[f"r{round_num}_temp"] = temperature
            
            # Validate if the best plan from this round is better than the global best plan
            if num_refinement > 0:  # Only validate if refinements were made
                # First, check if this round's best plan is better than the initial plan (p0)
                prompt_vs_p0 = final_question(user_input, initial_plan, best_round_plan)
                response_vs_p0 = generate_text(model, processor, prompt_vs_p0, image_path, do_sample=False)
                is_better_than_p0 = "".join([s for s in response_vs_p0.strip().lower().split()[0] if s.isalpha()]) == "yes"
                sample_results[f"r{round_num}_better_than_p0"] = is_better_than_p0
                total_num_inference += 1
                
                # Now check if it's better than our current global best
                if global_best_plan != initial_plan:
                    prompt_vs_global = final_question(user_input, global_best_plan, best_round_plan)
                    response_vs_global = generate_text(model, processor, prompt_vs_global, image_path, do_sample=False)
                    is_better_than_global = "".join([s for s in response_vs_global.strip().lower().split()[0] if s.isalpha()]) == "yes"
                    sample_results[f"r{round_num}_better_than_global"] = is_better_than_global
                    total_num_inference += 1
                else:
                    # If global_best_plan is still the initial plan, then we use the p0 comparison result
                    is_better_than_global = is_better_than_p0
                
                # Update tracking variables
                if is_better_than_p0:
                    found_better_than_p0 = True
                    
                if is_better_than_global:
                    global_best_plan = best_round_plan
                    global_best_plan_round = round_num
                    improvement_found = True
                    pbar.set_description(f"Sample {idx} Round {round_num}/{MAX_ROUND} - Improved plan found")
                else:
                    pbar.set_description(f"Sample {idx} Round {round_num}/{MAX_ROUND} - No improvement")
                
                # Record overall improvement status
                sample_results[f"r{round_num}_improved"] = is_better_than_global
                
                # Early stopping mode: if we found a plan better than p0 and we've reached minimum rounds
                if args.early_stop and found_better_than_p0 and round_num >= MIN_ROUND:
                    pbar.set_description(f"Sample {idx} Round {round_num}/{MAX_ROUND} - Early stopping (found better than p0)")
                    break
            else:
                # If no refinements were made
                sample_results[f"r{round_num}_improved"] = "no_refinement"
                pbar.set_description(f"Sample {idx} Round {round_num}/{MAX_ROUND} - No refinements needed")
            
            # If no refinements were made, stop the process
            if num_refinement == 0:
                break
                
            # Prepare for next round - increase temperature for more diversity
            temperature += args.incremental
            # Start the next round with the best plan found so far
            current_plan = global_best_plan
        
        # Final results
        sample_results["total_inference"] = total_num_inference
        sample_results["total_rounds"] = round_num
        sample_results["improvement_found"] = improvement_found
        sample_results["found_better_than_p0"] = found_better_than_p0
        
        # Set p_final based on the mode and results
        if args.early_stop:
            # For early stopping mode, if we didn't find anything better than p0, use p0
            if not found_better_than_p0:
                p_final = initial_plan
                sample_results["early_stop_fallback_to_p0"] = True
            else:
                p_final = global_best_plan
                sample_results["early_stop_fallback_to_p0"] = False
        else:
            # For full run mode, always use the global best (which could still be p0 if nothing better was found)
            p_final = global_best_plan
        
        sample_results["p_final"] = p_final
        sample_results["p_final_round"] = global_best_plan_round
        
        # Save full result for this sample
        with open(f"{RESULTS_DIR}/json_format/sample_{idx}.json", "w") as f:
            json.dump(sample_steps, f, indent=2)
        pd.DataFrame(sample_results, index=[idx]).to_csv(f"{RESULTS_DIR}/sample_{idx}.csv")
    
    print(f"Self-distillation with self-criticism complete. Results saved in {RESULTS_DIR}")

if __name__ == "__main__":
    run_self_distillation()