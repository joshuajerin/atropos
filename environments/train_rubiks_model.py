#!/usr/bin/env python3
"""
Train a model to solve Rubik's cube using reinforcement learning on collected data.
Based on the example GRPO trainer with modifications for pre-collected data.
"""

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import transformers
import wandb
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainerConfig:
    """Configuration for the trainer."""
    # Model configuration
    model_name: str
    learning_rate: float
    batch_size: int
    gradient_accumulation_steps: int
    sequence_length: int
    warmup_steps: int
    
    # Training configuration
    total_steps: int
    eval_every: int
    save_every: int
    checkpoint_dir: str
    use_wandb: bool
    wandb_project: str
    wandb_run_name: str
    
    # Data configuration
    train_file: str
    validation_size: float
    prefer_higher_scores: bool
    max_samples: int
    
    # RL configuration
    method: str
    temperature: float
    top_p: float
    beta: float
    reference_model: Optional[str] = None


def load_config(config_path: str) -> TrainerConfig:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    
    # The config is already flat, so we use it directly
    return TrainerConfig(**config_dict)


def load_jsonl_data(file_path: str, max_samples: int = -1) -> List[Dict]:
    """Load data from JSONL file."""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
            if max_samples > 0 and len(data) >= max_samples:
                break
    return data


def split_train_val(data: List[Dict], val_size: float) -> Tuple[List[Dict], List[Dict]]:
    """Split data into training and validation sets."""
    val_count = int(len(data) * val_size)
    return data[val_count:], data[:val_count]


def prepare_training_batch(
    data_batch: List[Dict], 
    tokenizer, 
    prefer_higher_scores: bool = True,
    device: str = "cuda"
) -> Dict[str, torch.Tensor]:
    """
    Prepare a batch for training.
    
    Args:
        data_batch: List of data points from JSONL
        tokenizer: Tokenizer for the model
        prefer_higher_scores: If True, higher scores are better
        device: Device to put tensors on
        
    Returns:
        Dict with input_ids, attention_mask, and scores
    """
    batch_tokens = []
    batch_masks = []
    batch_scores = []
    
    for item in data_batch:
        # For each group, select best and worst sequences based on scores
        scores = item["scores"]
        tokens = item["tokens"]
        masks = item["masks"]
        
        if prefer_higher_scores:
            best_idx = max(range(len(scores)), key=lambda i: scores[i])
            worst_idx = min(range(len(scores)), key=lambda i: scores[i])
        else:
            best_idx = min(range(len(scores)), key=lambda i: scores[i])
            worst_idx = max(range(len(scores)), key=lambda i: scores[i])
        
        batch_tokens.extend([tokens[best_idx], tokens[worst_idx]])
        batch_masks.extend([masks[best_idx], masks[worst_idx]])
        batch_scores.extend([scores[best_idx], scores[worst_idx]])
    
    # Convert to tensors
    input_ids = torch.tensor(batch_tokens, dtype=torch.long).to(device)
    attention_mask = torch.tensor(batch_masks, dtype=torch.long).to(device)
    scores = torch.tensor(batch_scores, dtype=torch.float).to(device)
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "scores": scores,
    }


def compute_grpo_loss(
    logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    scores: torch.Tensor,
    beta: float
) -> torch.Tensor:
    """
    Compute the Group Relative Policy Optimization loss.
    
    Args:
        logprobs: Log probabilities from the model (batch_size, seq_len)
        ref_logprobs: Log probabilities from the reference model (batch_size, seq_len)
        scores: Scores for each sequence (batch_size,)
        beta: KL penalty coefficient
        
    Returns:
        Loss tensor
    """
    batch_size = logprobs.shape[0]
    assert batch_size % 2 == 0, "Batch size must be even"
    
    # Reshape to (batch_size/2, 2, seq_len)
    logprobs = logprobs.view(batch_size // 2, 2, -1)
    ref_logprobs = ref_logprobs.view(batch_size // 2, 2, -1)
    scores = scores.view(batch_size // 2, 2)
    
    # Calculate policy gradient loss
    pg_loss = 0
    for i in range(batch_size // 2):
        # Policy gradient - weight by the score difference
        score_diff = scores[i, 0] - scores[i, 1]
        log_ratio_chosen = logprobs[i, 0].sum() - ref_logprobs[i, 0].sum()
        log_ratio_rejected = logprobs[i, 1].sum() - ref_logprobs[i, 1].sum()
        
        # KL penalty
        kl_chosen = (ref_logprobs[i, 0] - logprobs[i, 0]).sum()
        kl_rejected = (ref_logprobs[i, 1] - logprobs[i, 1]).sum()
        
        # Final loss - maximize score difference, minimize KL divergence
        pg_loss += -score_diff * (log_ratio_chosen - log_ratio_rejected)
        pg_loss += beta * (kl_chosen + kl_rejected)
    
    return pg_loss / (batch_size // 2)


def main():
    parser = argparse.ArgumentParser(description="Train a model on Rubik's cube data")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Initialize wandb if specified
    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=vars(config)
        )
    
    # Create checkpoint directory
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # Load tokenizer and model
    logger.info(f"Loading model {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name, 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    model.train()
    
    # Load reference model if specified
    ref_model = None
    if config.reference_model:
        logger.info(f"Loading reference model {config.reference_model}")
        ref_model = AutoModelForCausalLM.from_pretrained(
            config.reference_model,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)
        ref_model.eval()
    
    # Set up optimizer and lr scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.total_steps
    )
    
    # Load and split data
    logger.info(f"Loading data from {config.train_file}")
    all_data = load_jsonl_data(config.train_file, config.max_samples)
    train_data, val_data = split_train_val(all_data, config.validation_size)
    logger.info(f"Loaded {len(train_data)} training and {len(val_data)} validation samples")
    
    # Training loop
    global_step = 0
    best_val_loss = float('inf')
    
    logger.info("Starting training")
    try:
        for epoch in range(100):  # Large number, will break when steps reached
            # Shuffle training data
            import random
            random.shuffle(train_data)
            
            for i in range(0, len(train_data), config.batch_size // 2):
                batch_data = train_data[i:i + config.batch_size // 2]
                if len(batch_data) < config.batch_size // 2:
                    continue  # Skip incomplete batches
                
                # Prepare batch
                batch = prepare_training_batch(
                    batch_data, 
                    tokenizer,
                    prefer_higher_scores=config.prefer_higher_scores,
                    device=device
                )
                
                # Forward pass
                with torch.cuda.amp.autocast(enabled=device == "cuda"):
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        return_dict=True
                    )
                    
                    # Compute log probabilities
                    logits = outputs.logits[:, :-1]
                    logprobs = F.log_softmax(logits, dim=-1)
                    target_ids = batch["input_ids"][:, 1:]
                    masks = batch["attention_mask"][:, 1:]
                    
                    # Get log probs for the chosen tokens
                    chosen_logprobs = torch.gather(
                        logprobs, 
                        dim=2, 
                        index=target_ids.unsqueeze(-1)
                    ).squeeze(-1)
                    
                    # Apply mask
                    chosen_logprobs = chosen_logprobs * masks
                    
                    # Get reference log probs if using a reference model
                    if ref_model:
                        with torch.no_grad():
                            ref_outputs = ref_model(
                                input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"],
                                return_dict=True
                            )
                            ref_logits = ref_outputs.logits[:, :-1]
                            ref_logprobs = F.log_softmax(ref_logits, dim=-1)
                            
                            ref_chosen_logprobs = torch.gather(
                                ref_logprobs, 
                                dim=2, 
                                index=target_ids.unsqueeze(-1)
                            ).squeeze(-1)
                            
                            # Apply mask
                            ref_chosen_logprobs = ref_chosen_logprobs * masks
                    else:
                        # If no reference model, use the current model's initial state
                        ref_chosen_logprobs = chosen_logprobs.detach()
                    
                    # Compute loss
                    loss = compute_grpo_loss(
                        chosen_logprobs, 
                        ref_chosen_logprobs, 
                        batch["scores"],
                        config.beta
                    )
                
                # Backward pass
                loss = loss / config.gradient_accumulation_steps
                loss.backward()
                
                # Update weights if gradient accumulation steps reached
                if (global_step + 1) % config.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                # Log progress
                if global_step % 10 == 0:
                    logger.info(f"Step {global_step}: loss = {loss.item() * config.gradient_accumulation_steps:.4f}")
                    if config.use_wandb:
                        wandb.log({
                            "train/loss": loss.item() * config.gradient_accumulation_steps,
                            "train/learning_rate": scheduler.get_last_lr()[0],
                            "train/step": global_step,
                        })
                
                # Evaluate on validation set
                if global_step % config.eval_every == 0:
                    model.eval()
                    val_losses = []
                    
                    with torch.no_grad():
                        for j in range(0, min(len(val_data), 100), config.batch_size // 2):
                            val_batch_data = val_data[j:j + config.batch_size // 2]
                            if len(val_batch_data) < config.batch_size // 2:
                                continue
                            
                            val_batch = prepare_training_batch(
                                val_batch_data, 
                                tokenizer,
                                prefer_higher_scores=config.prefer_higher_scores,
                                device=device
                            )
                            
                            # Forward pass
                            val_outputs = model(
                                input_ids=val_batch["input_ids"],
                                attention_mask=val_batch["attention_mask"],
                                return_dict=True
                            )
                            
                            # Compute log probabilities
                            val_logits = val_outputs.logits[:, :-1]
                            val_logprobs = F.log_softmax(val_logits, dim=-1)
                            val_target_ids = val_batch["input_ids"][:, 1:]
                            val_masks = val_batch["attention_mask"][:, 1:]
                            
                            # Get log probs for the chosen tokens
                            val_chosen_logprobs = torch.gather(
                                val_logprobs, 
                                dim=2, 
                                index=val_target_ids.unsqueeze(-1)
                            ).squeeze(-1)
                            
                            # Apply mask
                            val_chosen_logprobs = val_chosen_logprobs * val_masks
                            
                            # Get reference log probs
                            if ref_model:
                                ref_val_outputs = ref_model(
                                    input_ids=val_batch["input_ids"],
                                    attention_mask=val_batch["attention_mask"],
                                    return_dict=True
                                )
                                ref_val_logits = ref_val_outputs.logits[:, :-1]
                                ref_val_logprobs = F.log_softmax(ref_val_logits, dim=-1)
                                
                                ref_val_chosen_logprobs = torch.gather(
                                    ref_val_logprobs, 
                                    dim=2, 
                                    index=val_target_ids.unsqueeze(-1)
                                ).squeeze(-1)
                                
                                # Apply mask
                                ref_val_chosen_logprobs = ref_val_chosen_logprobs * val_masks
                            else:
                                # If no reference model, use detached current model outputs
                                ref_val_chosen_logprobs = val_chosen_logprobs.detach()
                            
                            # Compute loss
                            val_loss = compute_grpo_loss(
                                val_chosen_logprobs, 
                                ref_val_chosen_logprobs, 
                                val_batch["scores"],
                                config.beta
                            )
                            val_losses.append(val_loss.item())
                    
                    avg_val_loss = sum(val_losses) / len(val_losses)
                    logger.info(f"Validation loss: {avg_val_loss:.4f}")
                    
                    if config.use_wandb:
                        wandb.log({
                            "val/loss": avg_val_loss,
                            "val/step": global_step,
                        })
                    
                    # Save best model
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        logger.info(f"New best validation loss: {best_val_loss:.4f}")
                        # Save model
                        output_dir = os.path.join(config.checkpoint_dir, f"best_model")
                        model.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)
                    
                    model.train()
                
                # Save checkpoint
                if global_step % config.save_every == 0 and global_step > 0:
                    output_dir = os.path.join(config.checkpoint_dir, f"checkpoint-{global_step}")
                    model.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                
                # Increment step
                global_step += 1
                
                # Exit if reached total steps
                if global_step >= config.total_steps:
                    break
            
            # Exit if reached total steps
            if global_step >= config.total_steps:
                break
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    
    # Save final model
    logger.info("Saving final model")
    output_dir = os.path.join(config.checkpoint_dir, "final_model")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    if config.use_wandb:
        wandb.finish()
    
    logger.info("Training complete")


if __name__ == "__main__":
    main()