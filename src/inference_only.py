import os
import argparse
import json
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from transformers import get_linear_schedule_with_warmup
from PIL import Image
import gc
import matplotlib.pyplot as plt
from qwen_vl_utils import process_vision_info

# =============================================
# ARGUMENT PARSING
# =============================================
parser = argparse.ArgumentParser(description='Run SFT with self-distilled data')
parser.add_argument('--model_path','-m', type=str, required=True, help='Path to input data dir')
parser.add_argument('--data_path','-d', type=str, required=True, help='Path to dev data')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument('--data_version','-v', type=str, default='v2', help='version of the training data')
parser.add_argument('--permutation','-p', type=str, default="", help='set if specific permutation')
parser.add_argument('--temperature','-temp', type=float, default=0.5, help='Temprature')
parser.add_argument('--top_p','-topp', type=float, default=0.9, help='Top-p sampling parameter')
parser.add_argument('--batch_size','-bs', type=int, default=1, help='Batch size')
parser.add_argument('--patience','-pa', type=int, default=5, help='Early stopping patience')
parser.add_argument('--max_new_tokens', type=int, default=500, help='Maximum new tokens to generate')
parser.add_argument('--max_tokens', type=int, default=700, help='Maximum new tokens to generate')
parser.add_argument('--use_bf16', action='store_true', help='Use BF16 precision')
args = parser.parse_args()


# =============================================
# CONFIGURATION
# =============================================
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

# Model settings
USE_BF16 = args.use_bf16
SYSTEM_DESC = True

# Reporting and evaluation settings
MAX_LEN = args.max_tokens  # Maximum sequence length for input
MAX_LEN_NEW_TOKEN = args.max_new_tokens  # Maximum new tokens for generation
TEST_BATCH_SIZE = args.batch_size

# Data paths
model_path = args.model_path
test_data_path = args.data_path

# Create directory for saving models and results
timestamp = datetime.now().strftime("%m%d_%H%M%S")
output_save_dir = os.path.join(args.model_path.replace("/best/","/generations/").replace("/best","/generations"), os.path.basename(test_data_path).replace(".csv",".json"))
os.makedirs(os.path.dirname(output_save_dir), exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

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
    required_cols = ['Image_Path', 'User_Input']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column {col} not found in the dataset")
    
    print(f"Loaded {len(df)} samples")
    
    # Preprocess the data
    df["User_Input"] = df["User_Input"].apply(lambda x: x.strip() if isinstance(x, str) else "")
         
    # Prepare the input-output formats
    df["input_prompt"] = df.apply(
        lambda row: f"{row['User_Input']}",
        axis=1
    )
    
    # Keep only necessary columns
    df = df[["Image_Path", "input_prompt", "User_Input"]]
    
    # Drop rows with missing values
    df = df.dropna(subset=["Image_Path", "input_prompt"])
    return df


class VLMDataset(Dataset):
    def __init__(self, dataframe, processor, max_length=512, is_eval=False, system_desc=SYSTEM_DESC):
        self.data = dataframe
        self.processor = processor
        self.max_length = max_length
        self.is_eval = is_eval
        self.system_desc = system_desc
        self.eos_token = processor.tokenizer.eos_token if processor.tokenizer.eos_token else "<|end_of_sentence|>"
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get the data for this sample
        image_path = self.data.iloc[idx]["Image_Path"]
        input_prompt = self.data.iloc[idx]["input_prompt"]
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

        image_inputs, video_inputs = process_vision_info(messages)
        text_only = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs_only = self.processor(
            text=[text_only],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt"
        )
        input_ids_only = inputs_only["input_ids"].squeeze(0)
        attention_mask_only = inputs_only["attention_mask"].squeeze(0)
        pixel_values = inputs_only["pixel_values"].squeeze(0)
        image_grid_thw = inputs_only["image_grid_thw"].squeeze(0)
        return {
            "input_ids_only": input_ids_only,
            "attention_mask_only":attention_mask_only,
            "pixel_values": pixel_values,
            "image_grid_thw":image_grid_thw,
            "prompt": input_prompt,
            "user_input": self.data.iloc[idx]["User_Input"],
            "image_path": image_path,
            "messages": messages  # Keep the original messages for generation
        }

# Function to generate text from the model
@torch.inference_mode()
def generate_text(model, processor, user_input, image_path, max_length=MAX_LEN_NEW_TOKEN):
    """Generate text from the model given an image and prompt"""
    model.eval()
    
    try:
        # Load the image
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        image = Image.new('RGB', (224, 224), color='black')
    
    # Create messages in the correct format for Qwen VLM
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
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
    inputs = processor(
        text=[text],
        return_tensors="pt"
    ).to(device)
    
    input_length = inputs['input_ids'].shape[1]
    
    # Generate the output
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_length,
            eos_token_id=processor.tokenizer.eos_token_id,
            num_return_sequences=1,
            do_sample=True,
            pad_token_id=processor.tokenizer.eos_token_id if processor.tokenizer.pad_token_id is None else processor.tokenizer.pad_token_id
        )
    
    # Extract and decode only the new tokens
    new_tokens = output[0][input_length:]
    return processor.tokenizer.decode(new_tokens, skip_special_tokens=True)

@torch.inference_mode()
def generate_text_batch(model, processor, batch, max_length=MAX_LEN_NEW_TOKEN):
    """Generate text from the model given an image and prompt"""
    model.eval()

    input_ids = batch["input_ids_only"].to(device)
    attention_mask = batch["attention_mask_only"].to(device)
    pixel_values = batch["pixel_values"].to(device)
    image_grid_thw = batch["image_grid_thw"].to(device)
    
    # Generate the output
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            max_new_tokens=max_length,
            eos_token_id=processor.tokenizer.eos_token_id,
            # top_p=0.92,
            num_return_sequences=1,
            # temperature=0.5,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id if processor.tokenizer.pad_token_id is None else processor.tokenizer.pad_token_id
        )
    
    # print(outputs)
    # Extract and decode only the new tokens
    generate_texts = []
    for output in outputs:
        # first_non_padding_idx = int(torch.argmax((output != processor.tokenizer.pad_token_id).to(dtype=torch.int)))
        # new_tokens = output[first_non_padding_idx+:]
        last_message_start=torch.argwhere(output == 151644).to(dtype=torch.int)[-1]
        generate_texts.append(processor.tokenizer.decode(output[last_message_start+3:], skip_special_tokens=True)) # skip assistant and \n
    return generate_texts


# Function to evaluate the model
def evaluate_model(model, data_loader, processor, device, path_output, prefix="dev", generate_outputs=False):
    """
    Evaluate model on development or test set
    Args:
        model: The model to evaluate
        data_loader: DataLoader for the dataset
        processor: The processor for tokenization and image processing
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
                pixel_values = batch["pixel_values"].to(device)
                image_grid_thw = batch["image_grid_thw"].to(device)
                loss_mask = batch["loss_mask"].to(device)
                labels = batch["labels"].to(device) if "labels" in batch else None
                
                # Forward pass
                outputs = model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    labels=labels
                )

                # Calculate loss with masking
                logits = outputs.logits
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                shift_loss_mask = loss_mask[:, 1:].contiguous()
                
                loss_per_token = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction='none'
                )
                
                loss_per_token = loss_per_token.view(shift_labels.size(0), -1)
                masked_loss = loss_per_token * shift_loss_mask
                mask_sum = shift_loss_mask.sum(dim=1) + 1e-8
                loss_per_sample = (masked_loss.sum(dim=1) / mask_sum)
                loss = loss_per_sample.mean()
                total_loss += loss.item()
                num_batches += 1
            
            # If we're generating outputs
            if generate_outputs:
                generated_texts = generate_text_batch(model, processor, batch)
                predictions.extend([{
                    "prompt": batch["prompt"][i],
                    "generated_text": generated_text,
                    # "ground_truth": batch["output_text"][i],
                    "user_input": batch["user_input"][i],
                    "image_path": batch["image_path"][i]
                } for i, generated_text in enumerate(generated_texts)])
                torch.cuda.empty_cache()
                gc.collect()
    
    # Calculate average loss
    with open(path_output, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    # Free up memory
    torch.cuda.empty_cache()
    
    return 0, predictions if generate_outputs else None


# # Main execution

# Load test data (previously dev data)
test_df = load_and_preprocess_data(test_data_path)
print(f"Loaded {len(test_df)} samples for test set")


# Generate samples using the best model
print("\n=== Sample Generations from Best Model ===\n")

# Load the best model for final evaluation
best_model_path = args.model_path
print(f"Loading the best model from {best_model_path}")

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    best_model_path,
    torch_dtype=torch.bfloat16 if USE_BF16 else torch.float32,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(best_model_path)

# Create datasets and dataloaders
test_dataset = VLMDataset(test_df, processor, max_length=MAX_LEN, is_eval=True)
test_dataloader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)


# Run final evaluation on dev set
print("\nRunning final evaluation on dev set...")
test_loss, test_predictions = evaluate_model(model, test_dataloader, processor, device, output_save_dir, 
                                        prefix=f"best_final_test", generate_outputs=True)

print("Generation complete!")