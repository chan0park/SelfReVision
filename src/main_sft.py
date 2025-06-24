import os
import argparse
import json
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    Qwen2_5_VLForConditionalGeneration, 
    AutoProcessor,
    MllamaForConditionalGeneration,
    Gemma3ForConditionalGeneration,
    get_linear_schedule_with_warmup
)
from PIL import Image
import gc
import matplotlib.pyplot as plt
from qwen_vl_utils import process_vision_info

# =============================================
# ARGUMENT PARSING
# =============================================
parser = argparse.ArgumentParser(description='Run SFT with self-distilled data')
parser.add_argument('--model_type', '-mt', type=str, choices=['qwen', 'llama', 'gemma'], default='qwen', 
                    help='Model type to use')
parser.add_argument('--model_size','-m', type=str, 
                    choices=['3B', '7B', '11B', '32B', '72B', '90B', '4B', '12B', '27B'], 
                    default='3B', help='Model size to use')
parser.add_argument('--max_refinement_per_round','-mrr', type=int, default=2, help='max_refinement_per_round')
parser.add_argument('--output_dir','-o', type=str, required=True, help='Directory to store results')
parser.add_argument('--data_path','-d', type=str, required=True, help='Path to input data dir')
parser.add_argument('--test_data_path','-td', type=str, required=True, help='Path to test data dir')
parser.add_argument('--data_keyword','-dk', type=str, choices=["vlm","simulation"],default='vlm', help='keyword to filter training data')
parser.add_argument('--val_data_path','-vd', type=str, required=True, help='Path to dev data')
parser.add_argument('--dev_size','-dev', type=int, default=100, help='Number of samples to use for dev set')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument('--data_version','-v', type=str, default='v10', help='version of the training data')
parser.add_argument('--target_plan','-plan', type=str, default='p5', help='version of the training data')
parser.add_argument('--permutation','-p', type=str, default="", help='set if specific permutation')
parser.add_argument('--num_sample','-num', type=int, default=-1, help='Number of samples to use for training')
parser.add_argument('--num_epoch','-ep', type=int, default=4, help='Number of training epochs')
parser.add_argument('--batch_size','-bs', type=int, default=4, help='batch size')
parser.add_argument('--gradient_accumulation','-ga', type=int, default=4, help='gradient accumulation steps')
parser.add_argument('--learning_rate','-lr', type=float, default=1e-5, help='Learning rate')
parser.add_argument('--weight_decay','-wd', type=float, default=0.01, help='weight decay')
parser.add_argument('--temperature','-temp', type=float, default=0.5, help='Temperature')
parser.add_argument('--top_p','-topp', type=float, default=0.9, help='Top-p sampling parameter')
parser.add_argument('--report_every','-re', type=int, default=400, help='Report frequency')
parser.add_argument('--skip_first','-sf', type=int, default=600, help='Skip first n batches for evaluation')
parser.add_argument('--patience','-pa', type=int, default=5, help='Early stopping patience')
parser.add_argument('--max_new_tokens', type=int, default=500, help='Maximum new tokens to generate')
parser.add_argument('--max_tokens', type=int, default=800, help='Maximum tokens')
parser.add_argument('--use_bf16', action='store_true', help='Use BF16 precision')
parser.add_argument('--hf_token', type=str, default=None, help='HuggingFace access token')
args = parser.parse_args()


# =============================================
# CONFIGURATION
# =============================================
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

# Model settings based on type and size
MODEL_CONFIGS = {
    'qwen': {
        '3B': "Qwen/Qwen2.5-VL-3B-Instruct",
        '7B': "Qwen/Qwen2.5-VL-7B-Instruct",
        '32B': "Qwen/Qwen2.5-VL-32B-Instruct",
        '72B': "Qwen/Qwen2.5-VL-72B-Instruct"
    },
    'llama': {
        '11B': "meta-llama/Llama-3.2-11B-Vision-Instruct",
        '90B': "meta-llama/Llama-3.2-90B-Vision-Instruct"
    },
    'gemma': {
        '4B': "google/gemma-3-4b-it",
        '12B': "google/gemma-3-12b-it",
        '27B': "google/gemma-3-27b-it"
    }
}

# Check if the model type and size combination is valid
if args.model_size not in MODEL_CONFIGS.get(args.model_type, {}):
    raise ValueError(f"Invalid model_size '{args.model_size}' for model_type '{args.model_type}'. "
                     f"Available sizes: {list(MODEL_CONFIGS.get(args.model_type, {}).keys())}")

MODEL_NAME = MODEL_CONFIGS[args.model_type][args.model_size]
MODEL_TYPE = args.model_type

USE_BF16 = args.use_bf16
SYSTEM_DESC = True
ACCESS_TOKEN = args.hf_token

# Training hyperparameters
NUM_EPOCH = args.num_epoch
LR = args.learning_rate
WD = args.weight_decay
BATCH_SIZE = args.batch_size
GRADIENT_ACCUMULATION_STEPS = args.gradient_accumulation
EFFECTIVE_BATCH_SIZE = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS

# Reporting and evaluation settings
REPORT_EVERY = args.report_every
MAX_LEN = args.max_tokens  # Maximum sequence length for input
MAX_LEN_NEW_TOKEN = args.max_new_tokens  # Maximum new tokens for generation
DEV_BATCH_SIZE = BATCH_SIZE
TEST_BATCH_SIZE = BATCH_SIZE

# Evaluation settings
SKIP_INITIAL_BATCHES = args.skip_first  # Skip evaluation for first N batches
SAVE_BEST_ONLY = True  # Only save models that improve on dev loss
EVAL_EVERY = REPORT_EVERY  # Evaluate as often as we report
SAVE_EVERY = REPORT_EVERY * 2  # Save model checkpoint frequency

# Data paths
data_keywords = {"vlm": "round", "simulation": "simulation"}
data_keyword = data_keywords.get(args.data_keyword, args.data_keyword)

train_data_paths = [os.path.join(args.data_path, filename) for filename in os.listdir(args.data_path) if filename.endswith(".csv") and filename.startswith(args.data_version+"--"+data_keyword) and f"-{args.model_size}-" in filename.upper() and args.target_plan+".csv" in filename and f"-max{args.max_refinement_per_round}-" in filename]
test_data_path = args.test_data_path

assert len(train_data_paths) > 0, f"no {args.data_version} training data found in {args.data_path}"
print("train data paths: ", train_data_paths)

permutation_postfix = f"-{args.permutation}" if args.permutation != "" else ""

# Create directory for saving models and results
timestamp = datetime.now().strftime("%m%d_%H%M%S")
model_save_dir = f"{args.output_dir}/{args.data_version}-{args.data_keyword}-{args.model_type}-{args.model_size}-{EFFECTIVE_BATCH_SIZE}-{LR}{permutation_postfix}-{args.target_plan}"
if not SYSTEM_DESC:
    model_save_dir += "-nosystem"
if args.num_sample != -1:
    model_save_dir += f"-{args.num_sample}samples"
else:
    model_save_dir += f"-allsamples"
model_save_dir += f"-{timestamp}"
os.makedirs(model_save_dir, exist_ok=True)
os.makedirs(f"{model_save_dir}/generations", exist_ok=True)
os.makedirs(f"{model_save_dir}/metrics", exist_ok=True)
with open(f'{model_save_dir}/arguments.json', 'w') as f:
    json.dump(vars(args), f, indent=4)
print(f"Saving the trained model & results: {model_save_dir}")

# Set up logging to file
log_file_path = os.path.join(model_save_dir, 'training_log.txt')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# Redirect print to logger
print = lambda *args, **kwargs: logger.info(' '.join(str(a) for a in args))

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# =============================================
# HELPER FUNCTIONS
# =============================================

# Normalize high-level steps function
def normalize_steps(steps_text):
    """Normalize the format of steps for consistent training"""
    if isinstance(steps_text, list):
        return "\n".join([step.strip() for step in steps_text])
    elif isinstance(steps_text, str):
        if "|" in steps_text:
            steps_text = "\n".join([step.strip() for step in steps_text.split("|")])
    if steps_text.count("\n\n") == 1:
        steps_text = steps_text.split("\n\n")[1]
    return steps_text

# Load and preprocess training data
def load_and_preprocess_data(data_path):
    """Load and preprocess the VLM training data"""
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Check if required columns exist
    required_cols = ['Image_Path', 'User_Input', 'High_Level_Plan']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column {col} not found in the dataset")
    
    print(f"Loaded {len(df)} samples")
    
    # Preprocess the data
    df["User_Input"] = df["User_Input"].apply(lambda x: x.strip() if isinstance(x, str) else "")
    df["High_Level_Plan"] = df["High_Level_Plan"].apply(normalize_steps)
         
    # Prepare the input-output formats
    df["input_prompt"] = df.apply(
        lambda row: f"{row['User_Input']}",
        axis=1
    )
    df["output_text"] = df["High_Level_Plan"]
    
    # Keep only necessary columns
    df = df[["Image_Path", "input_prompt", "output_text", "User_Input"]]
    
    # Drop rows with missing values
    df = df.dropna(subset=["Image_Path", "input_prompt", "output_text"])
    return df

# =============================================
# MODEL LOADING FUNCTIONS
# =============================================

def load_model_and_processor(model_type, model_name, use_bf16=True):
    """Load model and processor based on model type."""
    print(f"Loading {model_name} ({model_type})...")
    
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    
    # Define token parameter for HF API
    token_param = {"token": ACCESS_TOKEN} if ACCESS_TOKEN else {}
    
    if model_type == 'qwen':
        processor = AutoProcessor.from_pretrained(model_name, padding_side="left", **token_param)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            **token_param
        )
    elif model_type == 'llama':
        processor = AutoProcessor.from_pretrained(model_name, padding_side="left", **token_param)
        model = MllamaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            **token_param
        )
    elif model_type == 'gemma':
        # Optional: Configure the CUDA backend for Gemma
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

        processor = AutoProcessor.from_pretrained(model_name, padding_side="left", **token_param)
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            attn_implementation="eager",
            **token_param
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model, processor

# =============================================
# DATASET CLASSES
# =============================================

class VLMDataset(Dataset):
    def __init__(self, dataframe, processor, model_type, max_length=512, is_eval=False, system_desc=True):
        self.data = dataframe
        self.processor = processor
        self.model_type = model_type
        self.max_length = max_length
        self.is_eval = is_eval
        self.system_desc = system_desc
        self.eos_token = processor.tokenizer.eos_token if hasattr(processor, 'tokenizer') and processor.tokenizer.eos_token else "<|end_of_sentence|>"
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get the data for this sample
        image_path = self.data.iloc[idx]["Image_Path"]
        input_prompt = self.data.iloc[idx]["input_prompt"]
        output_text = self.data.iloc[idx]["output_text"] + " " + self.eos_token
        
        if self.model_type == 'qwen':
            # The original Qwen implementation
            return self._process_qwen_item(idx, image_path, input_prompt, output_text)
        elif self.model_type == 'llama':
            return self._process_llama_item(idx, image_path, input_prompt, output_text)
        elif self.model_type == 'gemma':
            return self._process_gemma_item(idx, image_path, input_prompt, output_text)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _process_qwen_item(self, idx, image_path, input_prompt, output_text):
        """Process data for Qwen models"""
        messages = []

        if self.system_desc:
            messages.append(
                {
                    "role":"system",
                    "content":[
                        {
                            "type":"text",
                            "text":"You're a helpful assistant and robot whose main role is to plan the steps the robot should take in response to the user's input."
                        }
                    ]
                }
            )

        # Prepare the messages format for Qwen VLM
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": input_prompt},
                ]
            }
        )
        
        # Add assistant's response with the high-level plan
        messages.append({
            "role": "assistant",
            "content": [
                {"type": "text", "text": output_text}
            ]
        })
        
        # Apply chat template to create the full text
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        image_inputs, video_inputs = process_vision_info(messages)
        # Process the inputs for training
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding="max_length",
            max_length=self.max_length+432,   
            padding_side='left',
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        pixel_values = inputs["pixel_values"].squeeze(0)
        image_grid_thw = inputs["image_grid_thw"].squeeze(0)
        
        # Find the position where assistant's response starts to create loss mask
        assistant_msg = self.processor.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        )
        _inputs = self.processor(text=[assistant_msg], images=image_inputs, videos=video_inputs, return_tensors="pt")
        prompt_len = _inputs["input_ids"].shape[1]
        first_non_padding_idx = int(torch.argmax((input_ids != self.processor.tokenizer.pad_token_id).to(dtype=torch.int)))
        labels = input_ids.clone()
        labels[:first_non_padding_idx+prompt_len] = -100  # Don't compute loss on prompt tokens

        if self.is_eval:
            text_only = self.processor.apply_chat_template(
                messages[:-1], tokenize=False, add_generation_prompt=True
            )
            inputs_only = self.processor(
                text=[text_only],
                images=image_inputs,
                videos=video_inputs,
                padding="max_length",
                max_length=self.max_length+432,   
                padding_side='left',
                truncation=True,
                return_tensors="pt"
            )
            input_ids_only = inputs_only["input_ids"].squeeze(0)
            attention_mask_only = inputs_only["attention_mask"].squeeze(0)
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
                "input_ids_only": input_ids_only,
                "attention_mask_only": attention_mask_only,
                "prompt": input_prompt,
                "output_text": self.data.iloc[idx]["output_text"],
                "user_input": self.data.iloc[idx]["User_Input"],
                "image_path": image_path,
                "messages": messages  # Keep the original messages for generation
            }
        else:
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
                "prompt": input_prompt,
                "output_text": self.data.iloc[idx]["output_text"],
                "user_input": self.data.iloc[idx]["User_Input"],
                "image_path": image_path,
                "messages": messages  # Keep the original messages for generation
            }
    
    def _process_llama_item(self, idx, image_path, input_prompt, output_text):
        """Process data for Llama models"""
        # Create messages for Llama format
        image = None
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image = Image.new('RGB', (224, 224), color='black')
        
        system_prompt = "You're a helpful assistant and robot whose main role is to plan the steps the robot should take in response to the user's input." if self.system_desc else ""
        
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            
        # User message with image and input
        messages.append({
            "role": "user", 
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": input_prompt}
            ]
        })
        
        # Assistant's response
        messages.append({
            "role": "assistant",
            "content": output_text
        })
        
        # Process for training
        inputs = self.processor(
            image,
            messages,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            padding_side='left',
            truncation=True
        )
        
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        pixel_values = inputs["pixel_values"].squeeze(0) if "pixel_values" in inputs else None
        
        
        # Find position where assistant's response starts
        assistant_messages = messages[:-1]
        assistant_messages.append({"role": "assistant", "content": ""})
        
        assistant_inputs = self.processor(
            image,
            assistant_messages,
            return_tensors="pt",
            truncation=True
        )
        
        prompt_len = assistant_inputs["input_ids"].shape[1]
        # Account for any padding at the beginning
        first_non_padding_idx = int(torch.argmax((input_ids != self.processor.tokenizer.pad_token_id).to(dtype=torch.int)))
        
        labels = input_ids.clone()
        labels[:first_non_padding_idx+prompt_len] = -100  # Don't compute loss on prompt tokens

        if self.is_eval:
            # For evaluation, prepare inputs without the assistant's response
            eval_messages = messages[:-1]
            eval_inputs = self.processor(
                image,
                eval_messages,
                return_tensors="pt",
                padding="max_length", 
                padding_side='left',
                max_length=self.max_length,
                truncation=True
            )
            
            input_ids_only = eval_inputs["input_ids"].squeeze(0)
            attention_mask_only = eval_inputs["attention_mask"].squeeze(0)
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "pixel_values": pixel_values,
                "input_ids_only": input_ids_only,
                "attention_mask_only": attention_mask_only,
                "prompt": input_prompt,
                "output_text": self.data.iloc[idx]["output_text"],
                "user_input": self.data.iloc[idx]["User_Input"],
                "image_path": image_path,
                "messages": messages
            }
        else:
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "pixel_values": pixel_values,
                "prompt": input_prompt,
                "output_text": self.data.iloc[idx]["output_text"],
                "user_input": self.data.iloc[idx]["User_Input"],
                "image_path": image_path,
                "messages": messages
            }
    
    def _process_gemma_item(self, idx, image_path, input_prompt, output_text):
        """Process data for Gemma models"""
        messages = []

        if self.system_desc:
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": "You're a helpful assistant and robot whose main role is to plan the steps the robot should take in response to the user's input."}]
            })

        # Add user message with image and prompt
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": input_prompt}
            ]
        })
        
        # Add assistant's response
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": output_text}]
        })
        
        # Apply chat template
        inputs = self.processor.apply_chat_template(
            messages, 
            add_generation_prompt=False, 
            tokenize=True,
            return_dict=True, 
            padding="max_length",
            max_length=self.max_length+432,   
            padding_side='left',
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        
        # Create loss mask
        # Apply template to just the prompt to find boundary
        prompt_messages = messages[:-1]
        prompt_inputs = self.processor.apply_chat_template(
            prompt_messages, 
            add_generation_prompt=True, 
            tokenize=True,
            return_dict=True, 
            return_tensors="pt"
        )
        
        prompt_len = prompt_inputs["input_ids"].shape[1]
        
        # Set loss mask for assistant's response
        first_non_padding_idx = 0
        if hasattr(self.processor.tokenizer, "pad_token_id") and self.processor.tokenizer.pad_token_id is not None:
            first_non_padding_idx = int(torch.argmax((input_ids != self.processor.tokenizer.pad_token_id).to(dtype=torch.int)))
        
        labels = input_ids.clone()
        labels[:first_non_padding_idx+prompt_len] = -100  # Don't compute loss on prompt tokens
        
        if self.is_eval:
            eval_inputs = self.processor.apply_chat_template(
                prompt_messages, 
                add_generation_prompt=True, 
                tokenize=True,
                return_dict=True, 
                return_tensors="pt",
                padding="max_length", 
                padding_side='left',
                max_length=self.max_length,
                truncation=True
            )
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "input_ids_only": eval_inputs["input_ids"].squeeze(0),
                "attention_mask_only": eval_inputs["attention_mask"].squeeze(0),
                "prompt": input_prompt,
                "output_text": self.data.iloc[idx]["output_text"],
                "user_input": self.data.iloc[idx]["User_Input"],
                "image_path": image_path,
                "messages": messages
            }
        else:
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "prompt": input_prompt,
                "output_text": self.data.iloc[idx]["output_text"],
                "user_input": self.data.iloc[idx]["User_Input"],
                "image_path": image_path,
                "messages": messages
            }

# =============================================
# GENERATION AND EVALUATION FUNCTIONS
# =============================================

@torch.inference_mode()
def generate_text(model, processor, user_input, image_path, model_type, max_length=MAX_LEN_NEW_TOKEN, temperature=0.5, top_p=0.92):
    """Generate text from the model given an image and prompt."""
    model.eval()
    
    try:
        # Load the image for models that need PIL Image
        if model_type in ['llama', 'gemma']:
            image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        image = Image.new('RGB', (224, 224), color='black')
    
    if model_type == 'qwen':
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
                    {"type": "text", "text": user_input},
                ]
            }
        ]
        
        # Apply chat template
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process inputs
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt"
        ).to(model.device)

        input_length = inputs['input_ids'].shape[1]

        # Generate the output
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_length,
                eos_token_id=processor.tokenizer.eos_token_id,
                top_p=top_p,
                num_return_sequences=1,
                temperature=temperature,
                do_sample=True,
                pad_token_id=processor.tokenizer.eos_token_id if processor.tokenizer.pad_token_id is None else processor.tokenizer.pad_token_id
            )

        # Extract and decode only the new tokens
        new_tokens = output[0][input_length:]
        return processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    elif model_type == 'llama':
        # Create messages for Llama format
        messages = [
            {"role": "system", "content": "You're a helpful assistant and robot whose main role is to plan the steps the robot should take in response to the user's input."},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_input}
            ]}
        ]
        
        # Process inputs
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
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
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
            )
        
        # Decode output
        new_tokens = output[0][input_length:]
        return processor.decode(new_tokens, skip_special_tokens=True)
    
    elif model_type == 'gemma':
        # Configure Gemma-specific settings
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You're a helpful assistant and robot whose main role is to plan the steps the robot should take in response to the user's input."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_input}
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
        ).to(model.device)
        
        # Generate output
        input_length = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            generation = model.generate(
                **inputs, 
                max_new_tokens=max_length,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
            )
            new_tokens = generation[0][input_length:]
        
        # Decode output
        return processor.decode(new_tokens, skip_special_tokens=True)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")

@torch.inference_mode()
def generate_text_batch(model, processor, batch, model_type, max_length=MAX_LEN_NEW_TOKEN, temperature=0.5, top_p=0.92):
    """Generate text from the model for a batch of inputs."""
    model.eval()
    
    input_ids = batch["input_ids_only"].to(model.device)
    attention_mask = batch["attention_mask_only"].to(model.device)
    
    generate_texts = []
    
    if model_type == 'qwen':
        pixel_values = batch["pixel_values"].to(model.device)
        image_grid_thw = batch["image_grid_thw"].to(model.device)
        
        # Generate the output
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                max_new_tokens=max_length,
                eos_token_id=processor.tokenizer.eos_token_id,
                top_p=top_p,
                num_return_sequences=1,
                temperature=temperature,
                do_sample=True,
                pad_token_id=processor.tokenizer.eos_token_id if processor.tokenizer.pad_token_id is None else processor.tokenizer.pad_token_id
            )
        
        # Extract and decode only the new tokens
        for output in outputs:
            last_message_start = torch.argwhere(output == 151644).to(dtype=torch.int)[-1]
            generate_texts.append(processor.tokenizer.decode(output[last_message_start+3:], skip_special_tokens=True)) # Skip assistant and \n
    
    elif model_type == 'llama':
        # For Llama, we need to process one by one with images
        for i in range(len(batch["image_path"])):
            try:
                image = Image.open(batch["image_path"][i]).convert("RGB")
            except Exception as e:
                print(f"Error loading image {batch['image_path'][i]}: {e}")
                image = Image.new('RGB', (224, 224), color='black')
                
            # Get the input prompt
            prompt = batch["prompt"][i]
            # Create messages
            messages = [
                {"role": "system", "content": "You're a helpful assistant and robot whose main role is to plan the steps the robot should take in response to the user's input."},
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]}
            ]
            
            # Process inputs
            input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(model.device)
            
            # Generate output
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                )
            
            # Extract just the generated text
            input_length = inputs['input_ids'].shape[1]
            new_tokens = output[0][input_length:]
            generated_text = processor.decode(new_tokens, skip_special_tokens=True)
            generate_texts.append(generated_text)
    
    elif model_type == 'gemma':
        # For Gemma, batch processing
        for i in range(len(batch["image_path"])):
            try:
                image = Image.open(batch["image_path"][i]).convert("RGB")
            except Exception as e:
                print(f"Error loading image {batch['image_path'][i]}: {e}")
                image = Image.new('RGB', (224, 224), color='black')
            
            # Create messages
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You're a helpful assistant and robot whose main role is to plan the steps the robot should take in response to the user's input."}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": batch["prompt"][i]}
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
            ).to(model.device)
            
            # Generate
            with torch.inference_mode():
                generation = model.generate(
                    **inputs, 
                    max_new_tokens=max_length,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                )
            
            # Decode only new tokens
            input_length = inputs["input_ids"].shape[1]
            new_tokens = generation[0][input_length:]
            generated_text = processor.decode(new_tokens, skip_special_tokens=True)
            generate_texts.append(generated_text)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    return generate_texts

# Function to evaluate the model
def evaluate_model(model, data_loader, processor, model_type, device, epoch=-1, prefix="dev", generate_outputs=False):
    """
    Evaluate model on development or test set
    Args:
        model: The model to evaluate
        data_loader: DataLoader for the dataset
        processor: The processor for tokenization and image processing
        model_type: Type of model being evaluated
        device: Device to run evaluation on
        epoch: Current epoch number (-1 for interim evaluations)
        prefix: Prefix for saved files (dev/test)
        generate_outputs: Whether to generate and save text outputs
    """
    model.eval()
    total_loss = 0.0
    predictions = []
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, total=len(data_loader), leave=False, desc=f"Evaluating {prefix}"):
            # If we're just calculating loss (not generating outputs), run the model
            if not generate_outputs:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device) if "labels" in batch else None
                
                # Forward pass - handle different model types
                if model_type == 'qwen':
                    pixel_values = batch["pixel_values"].to(device)
                    image_grid_thw = batch["image_grid_thw"].to(device)
                    
                    outputs = model(
                        input_ids=input_ids, 
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                        image_grid_thw=image_grid_thw,
                        labels=labels
                    )
                elif model_type in ['llama', 'gemma']:
                    # For Llama and Gemma, pixel_values are part of the inputs
                    pixel_values = batch["pixel_values"].to(device) if "pixel_values" in batch else None
                    
                    if pixel_values is not None:
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            pixel_values=pixel_values,
                            labels=labels
                        )
                    else:
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                else:
                    raise ValueError(f"Unknown model type: {model_type}")

                # Calculate loss with masking
                logits = outputs.logits
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                    # reduction='none'
                )
                
                total_loss += loss.item()
                num_batches += 1
            
            # If we're generating outputs
            if generate_outputs and epoch >= 0:
                generated_texts = generate_text_batch(model, processor, batch, model_type, 
                                                      max_length=MAX_LEN_NEW_TOKEN, 
                                                      temperature=args.temperature, 
                                                      top_p=args.top_p)
                
                predictions.extend([{
                    "prompt": batch["prompt"][i],
                    "generated_text": generated_text,
                    "ground_truth": batch["output_text"][i],
                    "user_input": batch["user_input"][i],
                    "image_path": batch["image_path"][i]
                } for i, generated_text in enumerate(generated_texts)])
                
                # Free up memory
                torch.cuda.empty_cache()
                gc.collect()
    
    # Calculate average loss
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    # Save predictions if epoch is specified and generation was requested
    if epoch >= 0 and generate_outputs and predictions:
        predictions_file = f"{model_save_dir}/generations/{prefix}_predictions_epoch_{epoch}.json"
        with open(predictions_file, 'w') as f:
            json.dump(predictions, f, indent=2)
        print(f"prediction saved to {predictions_file}")
    
    # Free up memory
    torch.cuda.empty_cache()
    
    return avg_loss, predictions if generate_outputs else None

# Function to generate training plots
def generate_training_plots(metrics, save_dir):
    """
    Generate plots showing training progress
    Args:
        metrics: Dictionary of training metrics
        save_dir: Directory to save plots
    """
    # Plot training and validation loss
    plt.figure(figsize=(12, 6))
    plt.plot(metrics['global_batch'], metrics['train_loss'], label='Training Loss')
    plt.plot(metrics['global_batch'], metrics['dev_loss'], label='Validation Loss')
    if 'test_loss' in metrics:
        plt.plot(metrics['global_batch'], metrics['test_loss'], label='Test Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f"{save_dir}/loss_plot.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Plot learning rate if available
    if 'learning_rate' in metrics:
        plt.figure(figsize=(10, 4))
        plt.plot(metrics['global_batch'], metrics['learning_rate'])
        plt.xlabel('Batch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(f"{save_dir}/lr_plot.png", dpi=300, bbox_inches='tight')
        plt.close()

# =============================================
# MAIN EXECUTION
# =============================================

# Load and preprocess data
print("Loading and preprocessing data...")
# Load all training data first
all_train_df = pd.concat([load_and_preprocess_data(train_data_path) for train_data_path in train_data_paths])
num_total_samples = len(all_train_df)
print(f"Loaded total {num_total_samples} samples from training data")

# Load test data (previously dev data)
test_df = load_and_preprocess_data(test_data_path)
print(f"Loaded {len(test_df)} samples for test set")

# Set the dev set size
dev_set_size = args.dev_size
print(f"Setting aside {dev_set_size} samples as dev set")

# Shuffle the data and split into dev and train
shuffled_train_df = all_train_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
dev_df = shuffled_train_df[:dev_set_size].reset_index(drop=True)
remaining_train_df = shuffled_train_df[dev_set_size:].reset_index(drop=True)

# Apply subsampling to training data if specified
if args.num_sample != -1 and len(remaining_train_df) > args.num_sample:
    print(f"Subsampling training data to {args.num_sample} examples")
    train_df = remaining_train_df[:args.num_sample].reset_index(drop=True)
else:
    train_df = remaining_train_df

print(f"Final training set size: {len(train_df)} samples")
print(f"Dev set size: {len(dev_df)} samples")
print(f"Test set size: {len(test_df)} samples")

# Load model and processor
print(f"Loading {MODEL_NAME} ({MODEL_TYPE})...")
model, processor = load_model_and_processor(MODEL_TYPE, MODEL_NAME, USE_BF16)

# Create datasets and dataloaders
train_dataset = VLMDataset(train_df, processor, MODEL_TYPE, max_length=MAX_LEN)
dev_dataset = VLMDataset(dev_df, processor, MODEL_TYPE, max_length=MAX_LEN, is_eval=True)
test_dataset = VLMDataset(test_df, processor, MODEL_TYPE, max_length=MAX_LEN, is_eval=True)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=DEV_BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)

# Determine number of batches
num_batch = len(train_dataloader)
print(f"Training on {num_batch} batches per epoch")

# Tracking metrics
best_dev_loss = float('inf')
best_epoch = -1
best_batch = -1
save_count = 0

# Set up optimizer
total_steps = len(train_dataloader) * NUM_EPOCH
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WD)
scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=total_steps * 0.1,  # 10% of steps for warmup
    num_training_steps=total_steps
)

# Tracking metrics for plotting later
train_metrics = {
    'epoch': [],
    'batch': [],
    'global_batch': [],
    'train_loss': [],
    'dev_loss': [],
    'test_loss': [],
    'learning_rate': []
}

global_batch_count = 0
num_decreased = 0

# Start training
print(f"Starting training for {NUM_EPOCH} epochs...")
for epoch in range(NUM_EPOCH):
    model.train()
    total_loss = 0
    batch_count = 0
    epoch_losses = []
    
    optimizer.zero_grad()
    
    for batch in tqdm(train_dataloader, total=num_batch, desc=f"Epoch {epoch+1}/{NUM_EPOCH}"):
        batch_count += 1
        global_batch_count += 1
        
        # Move inputs to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        # Forward pass - handle different model types
        if MODEL_TYPE == 'qwen':
            pixel_values = batch["pixel_values"].to(device)
            image_grid_thw = batch["image_grid_thw"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                labels=labels
            )
        elif MODEL_TYPE in ['llama', 'gemma']:
            # For Llama and Gemma, pixel_values are part of the inputs
            pixel_values = batch["pixel_values"].to(device) if "pixel_values" in batch else None
            
            if pixel_values is not None:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    labels=labels
                )
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
        else:
            raise ValueError(f"Unknown model type: {MODEL_TYPE}")
        
        # Calculate loss with masking
        logits = outputs.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        
        
        # Scale loss for gradient accumulation
        final_loss = loss / GRADIENT_ACCUMULATION_STEPS
        final_loss.backward()
        
        if batch_count % GRADIENT_ACCUMULATION_STEPS == 0 or batch_count == len(train_dataloader):
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # Track loss
        total_loss += loss.item()
        epoch_losses.append(loss.item())
        
        # Report training progress periodically
        if global_batch_count % REPORT_EVERY == 0:
            avg_loss = sum(epoch_losses[-min(REPORT_EVERY, len(epoch_losses)):]) / len(epoch_losses[-min(REPORT_EVERY, len(epoch_losses)):])
            
            # Quick evaluation on dev set (no generation, just loss)
            dev_loss, _ = evaluate_model(model, dev_dataloader, processor, MODEL_TYPE, device, epoch=-1, generate_outputs=False)
            test_loss, _ = evaluate_model(model, test_dataloader, processor, MODEL_TYPE, device, epoch=-1, generate_outputs=False)
            current_lr = scheduler.get_last_lr()[0]
            
            print(f"Epoch {epoch+1}, Batch {batch_count}/{num_batch} ({global_batch_count}): Train Loss = {avg_loss:.6f}, Dev Loss = {dev_loss:.6f}, Test Loss = {test_loss:.6f}, LR = {current_lr:.8f}")
            
            # Save metrics for later plotting
            train_metrics['epoch'].append(epoch)
            train_metrics['batch'].append(batch_count)
            train_metrics['global_batch'].append(global_batch_count)
            train_metrics['train_loss'].append(avg_loss)
            train_metrics['dev_loss'].append(dev_loss)
            train_metrics['test_loss'].append(test_loss)
            train_metrics['learning_rate'].append(current_lr)
            
            if dev_loss > best_dev_loss and global_batch_count >= SKIP_INITIAL_BATCHES:
                num_decreased += 1
            
            # Check if this is the best model so far and if we should save
            if dev_loss < best_dev_loss and global_batch_count >= SKIP_INITIAL_BATCHES:
                best_dev_loss = dev_loss
                best_epoch = epoch + 1
                best_batch = batch_count
                num_decreased = 0
                
                print(f"New best model found! Epoch {best_epoch}, Batch {best_batch}, Dev Loss: {best_dev_loss:.6f}")
                
                if global_batch_count % SAVE_EVERY == 0:
                    save_count += 1
                    # Generate full evaluation on test set for analysis
                    _, test_predictions = evaluate_model(model, test_dataloader, processor, MODEL_TYPE, device, 
                                          epoch=epoch+1, prefix=f"test_epoch{epoch+1}_batch{batch_count}", 
                                          generate_outputs=True)
                    torch.cuda.empty_cache()
                    gc.collect() 
                    
                    # Save the best model
                    best_model_path = f"{model_save_dir}/best"
                    model.save_pretrained(best_model_path)
                    processor.save_pretrained(f"{best_model_path}")
                    print(f"Model saved to {model_save_dir}/best")
                
                    # Save a snapshot of the current model
                    if not SAVE_BEST_ONLY:
                        model_path = f"{model_save_dir}/epoch{epoch+1}_batch{batch_count}"
                        model.save_pretrained(model_path)
                        processor.save_pretrained(f"{model_path}")
                
                    # Save metrics
                    metrics_file = f"{model_save_dir}/metrics/model_{save_count}_metrics.json"
                    model_metrics = {
                        'save_id': save_count,
                        'epoch': epoch + 1,
                        'batch': batch_count,
                        'global_batch': global_batch_count,
                        'train_loss': avg_loss,
                        'dev_loss': dev_loss,
                        'test_loss': test_loss,
                        'is_best': True,
                        'learning_rate': current_lr
                    }
                    with open(metrics_file, 'w') as f:
                        json.dump(model_metrics, f, indent=2)
            model.train()
            if num_decreased > args.patience:
                print(f"Dev loss has not improved for {args.patience} evaluations. Stopping training.")
                break
    
    # End of epoch summary 
    avg_loss = total_loss / len(train_dataloader)
    
    print(f"\nEpoch {epoch+1} Complete - Avg Loss: {avg_loss:.6f}")
    print(f"Best model so far: Epoch {best_epoch}, Batch {best_batch}, Dev Loss: {best_dev_loss:.6f}")
    
    # Save epoch summary
    metrics_file = f"{model_save_dir}/metrics/epoch_{epoch+1}_summary.json"
    epoch_metrics = {
        'epoch': epoch + 1,
        'train_loss': avg_loss,
        'global_batch': global_batch_count,
        'best_dev_loss': best_dev_loss,
        'best_epoch': best_epoch,
        'best_batch': best_batch
    }
    with open(metrics_file, 'w') as f:
        json.dump(epoch_metrics, f, indent=2)
    
    # Check for early stopping
    if num_decreased > args.patience:
        print(f"Training stopped early at epoch {epoch+1} due to no improvement in dev loss.")
        break

# Don't save final model separately if we're only saving best models
if not SAVE_BEST_ONLY:
    model.save_pretrained(f"{model_save_dir}/final")
    processor.save_pretrained(f"{model_save_dir}/final-processor")

# Save complete training metrics 
with open(f"{model_save_dir}/metrics/training_metrics.json", 'w') as f:
    json.dump(train_metrics, f, indent=2)

# Save epoch summary
epoch_metrics = {
    'best_dev_loss': best_dev_loss,
    'best_epoch': best_epoch,
    'best_batch': best_batch
}
with open(f"{model_save_dir}/metrics/best_summary.json", 'w') as f:
    json.dump(epoch_metrics, f, indent=2)

# Print summary of best model
print(f"\nTraining complete. Total batches: {global_batch_count}")
print(f"Best model was from epoch {best_epoch}, batch {best_batch} with dev loss {best_dev_loss:.6f}")
print(f"Saved {save_count} models in total")

# Generate training progress plots
print("\nGenerating training progress plots...")
generate_training_plots(train_metrics, f"{model_save_dir}/metrics")

# Generate samples using the best model
print("\n=== Sample Generations from Best Model ===\n")

# Load the best model for final evaluation
best_model_path = f"{model_save_dir}/best"
print(f"Loading the best model from {best_model_path}")

model, processor = load_model_and_processor(MODEL_TYPE, best_model_path, USE_BF16)

# Run final evaluation on test set
print("\nRunning final evaluation on test set...")
test_loss, test_predictions = evaluate_model(model, test_dataloader, processor, MODEL_TYPE, device, 
                                          epoch=best_epoch, prefix=f"best_final_test", 
                                          generate_outputs=True)

print(f"Final Test Loss: {test_loss:.6f}")
print("\nTraining complete!")