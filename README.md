# SelfReVision

**SelfReVision** is a lightweight, self-supervised framework for improving vision-language models (VLMs) in procedural planning tasks through iterative self-critique and refinement.

This codebase supports experiments from our paper:

> _"Making VLMs More Robot-Friendly: Self-Critical Distillation of Low-Level Procedural Reasoning"_ 

---

## Overview
<p align="center">
  <img src="https://i.imgur.com/jCLTIV0.png" alt="SelfReVision Overview" width="600"/>
</p>

SelfReVision enables models to iteratively:
- **Critique** their own outputs
- **Revise** based on self-generated feedback
- **Verify** improved versions

We evaluate on image-grounded planning tasks using custom datasets and scripts.

---

## Code Structure

src/  
├── image_filtering.py # Preprocessing utilities for evaluation images  
├── inference_only.py # Inference without self-revision  
├── llm_judge_eval.py # LLM-as-a-judge evaluation scripts  
├── main_blocks.py # Run block-based evaluation  
├── main_hamster.py # Run hamster-style evaluation  
├── main_selfrevision.py # Main file for SelfReVision: Critique–Revise–Verify  
├── main_sft.py # Supervised fine-tuning code after generating training data with SelfReVision  

validation-data/  
├── block_images/ # Block evaluation images generated with the [Ravens simulator](https://github.com/google-research/ravens)  
├── hamster_eval_images/ # Hamster task images  
├── block_eval_final_final.csv # Metadata and instructions for block tasks  
├── hamster_eval.csv # Metadata and instructions for hamster tasks  
├── vlm_dev_100.csv # Validation set for VLM evaluation (Places data)  

### Training Data
A larger subset of the Places dataset with GPT-4o-generated plans is available at: [https://huggingface.co/datasets/jrfish/SelfReVision](https://huggingface.co/datasets/jrfish/SelfReVision)

---

## Citation

If you use this code or dataset, please cite: [Paper link to be added]
