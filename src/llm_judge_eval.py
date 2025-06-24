import os
import json
import pandas as pd
import sys
import base64
import random
import argparse
from tqdm import tqdm
from openai import OpenAI

# Replace with your actual API key
OPENAI_KEY=YOUR_OPENAI_API_KEY  # Set your OpenAI API key here
rate_completion = 10/1000000
rate_prompt = 2.50/1000000

# =============================================
# ARGUMENT PARSING
# =============================================
parser = argparse.ArgumentParser(description='Run self-distillation on partitioned data')
parser.add_argument('--data_path','-d', type=str, required=True, help='Path to input data CSV'),
parser.add_argument('--output_path','-o', type=str, default=None, help='Path to store results')
parser.add_argument('--initial_plan_path','-ip', type=str, default=None, help='Path to input data CSV'),
parser.add_argument('--target_plan','-t', type=str, default="p_final", help='Path to input data CSV'),
parser.add_argument('--baseline_plan','-b', type=str, default="p0", help='Path to input data CSV'),
parser.add_argument('--postfix','-p', type=str, default="", help='postfix for the output file'),
args = parser.parse_args()


def winrate_evaluation_prompt():
    """
    Create a prompt for evaluating two plans in a head-to-head comparison.
    """
    return """You will be given a user input and two corresponding plans (Plan A and Plan B) with high-level steps that can be used by a robot to respond to the user input in a specific setting. I will also provide an image of the setting when available.

Your task is to evaluate which plan is better based on the following criteria:

### Coverage (Does the plan fully address the user input?)
- Does the plan thoroughly address all aspects of the user input without omissions?
- Does the plan cover the main points of the user input, or does it miss details?

### Ordering (Is the plan well-ordered?)
- Is the sequence of steps logical and efficient?
- Would any reordering of steps improve the plan?

### Completeness (Is the plan complete and informative?)
- Does the plan provide a complete picture of what needs to be done?
- Are the steps specific and detailed enough?
- Are there any gaps in the plan?

### Image Grounded (Can this plan be carried out in the specific setting shown in the image?)**
- Are all objects and actions mentioned clearly present or possible in the given setting in the image? 
- Is the plan specific and well grounded to the setting seen in the image?

### Overall Assessment
- Considering all criteria above, which plan is better overall?

For each of the five criteria (Coverage, Ordering, Completeness, Image-Grounded, and Overall), please:
1. Determine which plan is better (A, B, or Tie)
2. Provide a brief explanation (1-2 sentences) for your decision

Respond strictly in JSON format with the following structure:
{
  "Coverage": "A or B or Tie",
  "Coverage_Explanation": "Brief explanation",
  "Ordering": "A or B or Tie",
  "Ordering_Explanation": "Brief explanation",
  "Completeness": "A or B or Tie",
  "Completeness_Explanation": "Brief explanation",
  "Image-Grounded": "A or B or Tie", 
  "Image-Grounded_Explanation": "Brief explanation",
  "Overall": "A or B or Tie",
  "Overall_Explanation": "Brief explanation"
}

Do not use any markdown formatting or code block symbols. Only output a valid JSON object.
"""

def initialize_openai_client(api_key):
    """
    Initialize the OpenAI client.
    """
    try:
        return OpenAI(api_key=api_key)
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        return None

def extract_win_results(response_text):
    """
    Extract win results from the LLM response.
    """
    try:
        # Parse JSON response
        data = json.loads(response_text)
        
        # Extract results for each criterion
        results = {}
        for criterion in ["Coverage", "Ordering", "Completeness", "Image-Grounded", "Overall"]:
            winner = data.get(criterion, "")
            explanation = data.get(f"{criterion}_Explanation", "")
            results[criterion] = {"winner": winner, "explanation": explanation}
        
        return results
    except Exception as e:
        print(f"Error extracting win results: {e}")
        print(f"Response text: {response_text}")
        return None

def evaluate_with_winrate(client, user_input, plan_a, plan_b, is_llm_plan_a, model="gpt-4o", image_path=None):
    """
    Use LLM to evaluate which plan is better in a head-to-head comparison.
    """
    # Prepare prompt for evaluation
    prompt = winrate_evaluation_prompt()
    
    # Format the plans for presentation
    content = f"""\n### **User Input**
"{user_input}"

### **Plan A**
{plan_a}

### **Plan B**
{plan_b}
"""
    
    # # Prepare messages for API call
    # messages = [{"role": "user", "content": prompt + content}]
    
    # Handle image if path is provided
    try:
        # Create a message that includes the image
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
            # Update messages to include image
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": prompt + content
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
    except Exception as e:
        print(f"Error loading image from {image_path}: {e}")
        # Continue without image if there's an error
    
    # Call LLM API
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages
        )
        
        # Process response
        response_text = response.choices[0].message.content
        cost = response.usage.completion_tokens*rate_completion + response.usage.prompt_tokens*rate_prompt
        win_results = extract_win_results(response_text)
        
        # Map results back to original model identities (LLM vs Teacher)
        mapped_results = {}
        if win_results:
            for criterion, result in win_results.items():
                winner = result["winner"]
                explanation = result["explanation"]
                
                if winner == "A":
                    mapped_winner = "Refined" if is_llm_plan_a else "Baseline"
                elif winner == "B":
                    mapped_winner = "Baseline" if is_llm_plan_a else "Refined"
                else:  # Tie
                    mapped_winner = "Tie"
                
                mapped_results[criterion] = {"winner": mapped_winner, "explanation": explanation}
        
        return {
            "raw_response": response_text,
            "results": mapped_results,
            "cost": cost
        }
    except Exception as e:
        print(f"Error in LLM evaluation: {e}")
        return None

def parse_input_data(data):
    """
    Parse input data into a standardized format.
    """
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except:
            print("Error parsing JSON string")
            return []
    
    # Ensure data is a list
    if not isinstance(data, list):
        data = [data]
    
    return data

def evaluate_dataset_with_winrate(data, use_llm=True, api_key=None, model="gpt-4o"):
    """
    Evaluate an entire dataset using win rates between LLM and teacher model.
    """
    # Parse input data
    plans = parse_input_data(data)
    
    # Initialize OpenAI client if using LLM
    client = None
    if use_llm and api_key:
        client = initialize_openai_client(api_key)
    
    results = []
    total_cost = 0
    pbar = tqdm(plans, desc="Evaluating plans")
    
    for i, plan in enumerate(pbar):
        try:
            # Extract data
            user_input = plan.get("user_input", "") or plan.get("prompt", "")
            
            # Extract generated text (LLM plan)
            if "target" in plan:
                llm_plan = plan["target"]
            else:
                print(f"Warning: Could not find generated text for plan {i+1}")
                continue
            
            # Extract ground truth (baseline plan)
            if "baseline" in plan:
                baseline_plan = plan["baseline"]
            else:
                print(f"Warning: Could not find ground truth for plan {i+1}")
                continue
            
            # Extract image path if available
            image_path = plan.get("image_path", None)
            
            # Extract domain type if available
            domain = None
            for key in ["domain_type", "type", "domain"]:
                if key in plan:
                    domain = plan.get(key, "")
                    break
            
            # Randomize order of plans to avoid position bias
            is_llm_plan_a = random.choice([True, False])
            plan_a = llm_plan if is_llm_plan_a else baseline_plan
            plan_b = baseline_plan if is_llm_plan_a else llm_plan
            
            # Win rate evaluation
            win_results = None
            if use_llm and client:
                if plan_a.strip() == plan_b.strip():
                    win_results = {
                    "raw_response": "Plans are identical",
                    "results": {criterion:{"winner": "Tie", "explanation": "Plans are identical"} for criterion in ["Coverage", "Ordering", "Completeness", "Image-Grounded", "Overall"]},
                    "cost": 0
                    }
                else:
                    win_results = evaluate_with_winrate(client, user_input, plan_a, plan_b, is_llm_plan_a, model, image_path)
            
            # Combine results
            result = {
                "index": i,
                "user_input": user_input,
                "llm_plan": llm_plan,
                "baseline_plan": baseline_plan,
                "domain": domain,
                "image_path": image_path,
                "is_llm_plan_a": is_llm_plan_a
            }
            
            # Add win rate results if available
            if win_results:
                for criterion in ["Coverage", "Ordering", "Completeness", "Image-Grounded", "Overall"]:
                    if criterion in win_results["results"]:
                        result[f"{criterion.lower()}_winner"] = win_results["results"][criterion]["winner"]
                        result[f"{criterion.lower()}_explanation"] = win_results["results"][criterion]["explanation"]
                
                result["raw_response"] = win_results["raw_response"]
                total_cost += win_results["cost"]
                pbar.set_description(f"Cost: ${total_cost:.3f}")
            
            results.append(result)
        except Exception as e:
            print(f"Error processing plan {i+1}: {e}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    return results_df

def print_results(stats):
    # Print summary statistics
    print("\nWin Rate Evaluation Summary:")
    print(f"Number of examples evaluated: {stats["num_examples"]}")
    
    # Display win rates for each criterion
    for criterion, stat in stats.items():
        if criterion == "num_examples":
            continue
        print(f"\n{criterion.capitalize()} Statistics:")
        print(f"  Refined Answer Win Rate: {stat['refined_win_rate']:.2%} ({stat['refined_wins']} wins)")
        print(f"  Baseline Answer Win Rate: {stat['baseline_win_rate']:.2%} ({stat['baseline_wins']} wins)")
        print(f"  Tie Rate: {stat['tie_rate']:.2%} ({stat['ties']} ties)")

    for criterion, stat in stats.items():
        if criterion == "num_examples":
            continue
        # print(f"{stat['baseline_win_rate']:.0%},{stat['refined_win_rate']:.0%}", end=",")
        print(f"{stat['baseline_win_rate']:.0%} <-> {stat['refined_win_rate']:.0%}", end=",")
    print("")
        

def calculate_win_statistics(results_df, bool_exclude_identical = False):
    """
    Calculate win rate statistics from evaluation results.
    """
    if bool_exclude_identical:
        results_df = results_df[results_df["overall_explanation"] != "Plans are identical"]
    stats = {"num_examples": len(results_df)}
    
    for criterion in ["coverage", "ordering", "completeness", "image-grounded", "overall"]:
        if f"{criterion}_winner" in results_df.columns:
            # Count occurrences
            total = len(results_df)
            refined_wins = len(results_df[results_df[f"{criterion}_winner"] == "Refined"])
            baseline_wins = len(results_df[results_df[f"{criterion}_winner"] == "Baseline"])
            ties = len(results_df[results_df[f"{criterion}_winner"] == "Tie"])
            
            # Calculate win rates
            refined_win_rate = refined_wins / total
            baseline_win_rate = baseline_wins / total
            tie_rate = ties / total
            
            # Add to stats
            stats[criterion] = {
                "total": total,
                "refined_wins": refined_wins,
                "baseline_wins": baseline_wins,
                "ties": ties,
                "refined_win_rate": refined_win_rate,
                "baseline_win_rate": baseline_win_rate,
                "tie_rate": tie_rate
            }
    
    return stats

def cleanup_format(data):
    def _cleanup(text):
        text = str(text)
        if "</think>" in text:
            text = text.split("</think>")[-1]
        if "<plan>" in text:
            text = [s.strip() for s in text.split("<plan>") if s.strip() != ""][-1]
        if "</plan>" in text:
            text = [s.strip() for s in text.split("</plan>") if s.strip() != ""][-1]
        if "|" in text:
            text = "\n".join([line.strip() for line in text.split("|")])
        return text.strip()
            
    for d in data:
        d["target"] = _cleanup(d["target"])
        d["baseline"] = _cleanup(d["baseline"])
    return data

def main():
    # Parse command line arguments
    filename = args.data_path
    if not args.output_path:
        output_file = filename.replace(".csv", f"_{args.target_plan}_winrate_results{args.postfix}.csv").replace(".json",f"_{args.target_plan}_winrate_results{args.postfix}.csv")
    else:
        output_file = args.output_path
    print(f"Output file: {output_file}")
    if os.path.isfile(output_file):
        print(f"Output evaluation result already exists: {output_file}\nLoading the file")
        results = pd.read_csv(output_file)
        stats = calculate_win_statistics(results)
        stats_only_refined = calculate_win_statistics(results, bool_exclude_identical=True)
        print_results(stats)
        if stats["num_examples"] != stats_only_refined["num_examples"]:
            print("---------------------")
            print_results(stats_only_refined)
        return 
    
    print(f"Reading data from {filename}")
    try:
        with open(filename, 'r') as f:
            if filename.endswith('.json'):
                data = json.load(f)
            elif filename.endswith('.csv'):
                df = pd.read_csv(filename)
                if "Image_Path" in df.columns:
                    df = df.rename(columns={"Image_Path": "image_path","User_Input":"user_input"})
                data = df.to_dict('records')
            else:
                print("Unsupported file format. Please use .json or .csv")
                return
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    if args.initial_plan_path:
        initial_plan_file = args.initial_plan_path
        if initial_plan_file.endswith('.json'):
            with open(args.initial_plan_path, 'r') as f:
                data_initial = pd.DataFrame(json.load(f))
        elif initial_plan_file.endswith('.csv'):
            data_initial = pd.read_csv(initial_plan_file)
        if "sample_id" in data_initial.columns:
            data_initial.sort_values(by=["sample_id"], inplace=True)
    else:
        if filename.endswith('.json'):
            print("Warning: No initial plan file provided. Using ground truth as baseline.")

    for idx, d in enumerate(data):
        if args.initial_plan_path:
            data[idx]["target"] = d['generated_text'] if "generated_text" in d else d[args.target_plan]
            data[idx]["baseline"] = data_initial.iloc[idx][args.baseline_plan]
            assert data[idx]["image_path"] == data_initial.iloc[idx]["image_path"], f"Image paths do not match for index {idx}"
        else:
            data[idx]["target"] = d[args.target_plan]
            data[idx]["baseline"] = d[args.baseline_plan]
    
    # clean up the format (<think>, <plan>, </think>, </plan>)
    data = cleanup_format(data)
    print({k:v for k,v in data[0].items() if not pd.isna(v)})

    # Run evaluation and save
    results = evaluate_dataset_with_winrate(data, use_llm=True, api_key=OPENAI_KEY, model="gpt-4o")
    results.to_csv(output_file, index=False)
    
    # Calculate win statistics
    stats = calculate_win_statistics(results)
    stats_only_refined = calculate_win_statistics(results, bool_exclude_identical=True)
    print_results(stats)
    print("---------------------")
    print_results(stats_only_refined)

    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    data = main()