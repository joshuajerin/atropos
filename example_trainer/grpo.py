import atexit
import json
import math
import os
import random
import shutil
import string
import subprocess
import time
from typing import List, Optional, Tuple

import numpy as np
import requests
import torch
import torch.nn.functional as F
import wandb  # Added for logging
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

# Global variable to keep track of the vLLM process
vllm_process = None


def cleanup_vllm():
    global vllm_process
    if vllm_process:
        print("\nTerminating vLLM process...")
        vllm_process.terminate()
        try:
            vllm_process.wait(timeout=5)  # Wait a bit for graceful shutdown
            print("vLLM process terminated.")
        except subprocess.TimeoutExpired:
            print("vLLM process did not terminate gracefully, killing.")
            vllm_process.kill()
            vllm_process.wait()
            print("vLLM process killed.")
        vllm_process = None


# Register the cleanup function to be called on script exit
atexit.register(cleanup_vllm)


class TrainingConfig(BaseModel):
    """
    Training details, model, etc
    """

    model_name: str = Field(..., description="Name of the base model to train")
    lr: float = Field(1e-5, description="Learning rate for the optimizer")
    training_steps: int = Field(
        10, description="Number of training steps"
    )  # Renamed from epochs
    batch_size: int = Field(
        1, description="Batch size for training (will be handled by get_data)"
    )
    seq_len: int = Field(1024, description="Sequence length for training")
    gradient_accumulation_steps: int = Field(
        8, description="Number of gradient accumulation steps"
    )
    device: str = Field(
        "cuda" if torch.cuda.is_available() else "cpu", description="Device to train on"
    )
    save_path: str = Field(
        "trained_model_checkpoints", description="Base path to save model checkpoints"
    )
    vllm_restart_interval: int = Field(
        3, description="Restart vLLM every N training steps"
    )
    vllm_port: int = Field(9001, description="Port for the vLLM server")

    # Wandb configuration
    use_wandb: bool = Field(
        False, description="Whether to use Weights & Biases for logging"
    )
    wandb_project: Optional[str] = Field(None, description="Wandb project name")
    wandb_group: Optional[str] = Field(None, description="Wandb group name")


def pad_data_to_good_offset(data, batch_size: int):
    max_token_len = max(
        [max([len(x) for x in item["tokens"]]) for item in data["batch"]]
    )
    # usually 64 is a good choice to ensure nonweird scaling behavior on GPUS
    # so we pad to the nearest multiple of 64
    good_multiple = 64
    if (max_token_len - 1) % (good_multiple) != 0:
        max_token_len = math.ceil((max_token_len - 1) / (good_multiple)) * good_multiple
        token_setup_len = (
            max_token_len + 1
        )  # add 1 so we can make it causal at the proper length
    else:
        token_setup_len = max_token_len
        max_token_len = (
            max_token_len - 1
        )  # since it's causal we need to remove the last bit...
    # pad all tokens to max_token_len and add to lists
    input_ids = list()
    labels = list()
    advantages = list()
    lengths = list()
    for item in data["batch"]:
        # Store original scores directly, normalization will happen globally later
        current_item_scores = np.array(item["scores"], dtype=np.float32)
        print(f"DEBUG: Item original scores (before override): {current_item_scores}")

        if item["overrides"] is not None:
            for i in range(len(item["overrides"])):
                if item["overrides"][i].get("set_advantage_to_zero", False):
                    current_item_scores[i] = 0
            print(f"DEBUG: Item scores after override: {current_item_scores}")

        for i in range(len(item["tokens"])):
            lengths.append(
                math.ceil((len(item["tokens"][i]) - 1) / (good_multiple))
                * good_multiple
            )
            label_item = np.concatenate(
                [
                    np.array(item["masks"][i]),
                    np.full(
                        max(0, token_setup_len - len(item["tokens"][i])),
                        -100,
                        dtype=np.int32,
                    ),
                ]
            )
            item["tokens"][i] = np.concatenate(
                [
                    np.array(item["tokens"][i]),
                    np.zeros(
                        max(0, token_setup_len - len(item["tokens"][i])), dtype=np.int32
                    ),
                ]
            )
            input_ids.append(item["tokens"][i][:-1])
            labels.append(label_item[1:])
            advantages.append(current_item_scores[i]) # Append original/overridden score

    # Global normalization of advantages
    if advantages: # Check if advantages list is not empty
        advantages_np = np.array(advantages, dtype=np.float32)
        print(f"DEBUG: All collected advantages before global normalization: count={len(advantages_np)}, mean={advantages_np.mean():.4f}, std={advantages_np.std():.4f}, min={advantages_np.min():.4f}, max={advantages_np.max():.4f}")
        if len(advantages_np) > 1:
            adv_mean = advantages_np.mean()
            adv_std = advantages_np.std()
            advantages_np = (advantages_np - adv_mean) / max(adv_std, 1e-8)
            print(f"DEBUG: All collected advantages after global normalization: mean={advantages_np.mean():.4f}, std={advantages_np.std():.4f}")
        advantages = advantages_np.tolist() # Convert back to list for stacking
    else:
        print("DEBUG: No advantages were collected in pad_data_to_good_offset for global normalization.")

    # combine all lists into tensors
    token_batches = []
    label_batches = []
    advantage_batches = []
    for i in range(len(input_ids) // batch_size):
        token_batches.append(
            torch.tensor(
                np.stack(input_ids[i * batch_size : (i + 1) * batch_size], axis=0)
            )
        )
        label_batches.append(
            torch.tensor(
                np.stack(labels[i * batch_size : (i + 1) * batch_size], axis=0)
            )
        )
        advantage_batches.append(
            torch.tensor(
                np.stack(advantages[i * batch_size : (i + 1) * batch_size], axis=0)
            ).view(-1, 1)
        )
    return token_batches, label_batches, advantage_batches


def get_data(
    batch_size: int, seq_len: int  # seq_len is not used in this modified version but kept for signature compatibility
) -> List[Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]]:
    """
    Getting data from the local 'environments/rubiks_process_results_32.jsonl' file.
    Each line in the file is expected to be a JSON object compatible with RubiksCubeScoredDataGroup.
    """
    data_file_path = "../environments/rubiks_process_results_32.jsonl"  # Relative to example_trainer directory
    
    loaded_batch_items = []
    try:
        with open(data_file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():  # Ensure line is not empty
                    loaded_batch_items.append(json.loads(line))
    except FileNotFoundError:
        print(f"ERROR: Data file not found at {data_file_path}")
        print(f"Current working directory: {os.getcwd()}")
        # If you are running grpo.py from the project root, the path should be:
        # "environments/rubiks_process_results_32.jsonl"
        # If running from example_trainer/, then "../environments/rubiks_process_results_32.jsonl" is correct.
        # Let's try the project root path as a fallback.
        data_file_path_from_root = "environments/rubiks_process_results_32.jsonl"
        try:
            with open(data_file_path_from_root, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        loaded_batch_items.append(json.loads(line))
            print(f"Successfully loaded data from fallback path: {data_file_path_from_root}")
        except FileNotFoundError:
            print(f"ERROR: Fallback data file not found at {data_file_path_from_root} either.")
            return [] # Return empty if file not found

    if not loaded_batch_items:
        print(f"No data loaded from {data_file_path}. Please check the file.")
        return []

    # The trainer expects data in a dictionary with a "batch" key
    # Each item in loaded_batch_items is already a RubiksCubeScoredDataGroup-like object.
    # So, we wrap the list of these items under the "batch" key.
    data_for_padding = {"batch": loaded_batch_items}

    # Save the loaded data to temp.json for inspection, similar to the original get_data
    # This helps verify that the data format is what pad_data_to_good_offset expects.
    # temp.json will contain a dict: {"batch": [list of loaded items]}
    with open("temp.json", "w", encoding="utf-8") as f:
        json.dump(data_for_padding, f, indent=2) # indent for readability
        print(f"Saved the structure of loaded data to temp.json for inspection.")

    # We process the entire file as one large batch of "groups".
    # pad_data_to_good_offset will then further process this into token_batches, label_batches, etc.
    # The original get_data returned a list of batches. Here, we return a list containing one mega-batch.
    # This seems to align with how the training loop consumes it (batches.pop(0)).
    processed_batch = pad_data_to_good_offset(data_for_padding, batch_size)
    
    if not any(processed_batch): # Check if any of the lists (tokens, labels, advantages) are empty
        print("Warning: pad_data_to_good_offset returned empty batches. Check data format and batch_size.")
        return []
        
    return [processed_batch] # Return as a list containing the single processed batch


def train(config: TrainingConfig):
    """
    Setups and runs GRPO training, restarting vLLM periodically, with wandb logging.
    """
    global vllm_process  # Declare intention to modify the global variable

    # --- Wandb Setup ---
    if config.use_wandb:
        if not config.wandb_project:
            print("Warning: wandb_project not set, disabling wandb.")
            config.use_wandb = False
        else:
            if not config.wandb_group:
                # Set group to random 8 character string
                config.wandb_group = "".join(
                    random.choices(string.ascii_letters + string.digits, k=8)
                )
            try:
                wandb.init(
                    project=config.wandb_project,
                    group=config.wandb_group,
                    config=config.dict(),  # Log config parameters
                )
                print(
                    f"Wandb logging enabled. Run: {wandb.run.name} (Project: {config.wandb_project}) "
                )
            except Exception as e:
                print(f"Error initializing wandb: {e}. Disabling wandb.")
                config.use_wandb = False
    # --- End Wandb Setup ---

    # Initialize model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name, torch_dtype=torch.bfloat16
    )

    model.to(config.device)
    model.gradient_checkpointing_enable()
    model.train()

    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=config.lr)

    print(
        f"Starting training for {config.training_steps} steps on device: {config.device}"
    )
    print(
        f"vLLM will be restarted every {config.vllm_restart_interval} steps on port {config.vllm_port}"
    )

    os.makedirs(config.save_path, exist_ok=True)  # Ensure base save directory exists
    # register_trainer(config) # We don't register if loading from file

    # Init vllm
    # The vLLM parts might need to be disabled or re-thought if only training on static data.
    # For now, let's comment out the initial launch and rely on the loop's logic.
    # If vllm_process remains None, the restart logic might try to launch it later based on a checkpoint.
    # This might be okay, or might need adjustment if no checkpoints are intended to be served this way.
    
    # vllm_command = [
    #     "python",
    #     "-m",
    #     "vllm.entrypoints.openai.api_server",
    #     "--model",
    #     config.model_name, # This would be the base model initially
    #     "--port",
    #     str(config.vllm_port),
    #     "--dtype",
    #     "auto",
    #     "--gpu-memory-utilization",
    #     "0.45",
    #     "--disable-log-requests",
    # ]
    # print(f"  Launching vLLM server: {' '.join(vllm_command)}")
    # try:
    #     vllm_process = subprocess.Popen(vllm_command)
    #     print(f"  vLLM server launched with PID: {vllm_process.pid}")
    #     # Check immediate errors
    #     try:
    #         stdout, stderr = vllm_process.communicate(timeout=2)
    #         if vllm_process.returncode is not None and vllm_process.returncode != 0:
    #             print(f"  Error starting vLLM: {stderr.decode()}")
    #             vllm_process = None
    #             print("  WARNING: Failed to start vLLM server.")
    #     except subprocess.TimeoutExpired:
    #         print("  vLLM process started (check logs for details).")
    # except FileNotFoundError:
    #     print(
    #         "\n *** ERROR: 'python -m vllm...' command not found. Make sure vLLM is installed and accessible. ***\n"
    #     )
    #     print("  Disabling further vLLM restarts.")
    #     config.vllm_restart_interval = (
    #         config.training_steps + 1
    #     ) 
    # except Exception as e:
    #     print(f"\n *** ERROR: Failed to launch vLLM: {e} ***\n")
    #     print("  Disabling further vLLM restarts.")
    #     config.vllm_restart_interval = (
    #         config.training_steps + 1
    #     )

    batches = list() # This will now hold all data, not just one step's worth
    print("Loading all training data...")
    all_data_batches = get_data(config.batch_size, config.seq_len)

    if not all_data_batches:
        print("No data returned from get_data. Ending training.")
        if config.use_wandb:
            wandb.finish()
        return

    # Assuming get_data returns a list containing one tuple: (token_batches, label_batches, advantage_batches)
    # If it can return multiple such tuples, the logic here might need to be adjusted
    # For now, we assume it's a list with one element as per the current get_data implementation.
    if len(all_data_batches) > 1:
        print(f"Warning: get_data returned {len(all_data_batches)} sets of batches. Using only the first one.")
    
    token_batches_all, label_batches_all, advantage_batches_all = all_data_batches[0]
    
    if not token_batches_all:
        print("No token batches found in the loaded data. Ending training.")
        if config.use_wandb:
            wandb.finish()
        return

    print(f"Data loaded. Number of micro-batches: {len(token_batches_all)}")

    for step in range(config.training_steps):
        # Shuffle data at the beginning of each epoch if we consider one pass through data as an epoch
        # For now, we'll just iterate. If training_steps > number of micro-batches, data will repeat.
        # This part might need more sophisticated handling if true epochs are desired.
        current_micro_batch_idx = step % len(token_batches_all)
        
        tokens = token_batches_all[current_micro_batch_idx]
        labels = label_batches_all[current_micro_batch_idx]
        advantages = advantage_batches_all[current_micro_batch_idx]
        
        print(f"Step {step+1}/{config.training_steps} (Using micro-batch {current_micro_batch_idx+1}/{len(token_batches_all)})")
        
        # Reset accumulators for each step (which is now one micro-batch pass)
        # total_loss was previously accumulating losses from multiple micro-batches *within* a single get_data call.
        # Now, one "step" processes one micro-batch from the pre-loaded set.
        # The grpo_loss is already calculated per micro-batch and scaled by gradient_accumulation_steps.
        # So, the reported step loss will be this grpo_loss.

        total_pos_logp = 0
        total_neg_logp = 0
        total_logp = 0 # This might need re-evaluation; it was an average over micro-batches previously
        total_pos = 0
        total_neg = 0
        
        # The old loop `if len(batches) == 0:` and `batches.pop(0)` is no longer needed
        # as we are iterating through pre-loaded data.
        # The inner loop `for tokens, labels, advantages in zip(...)` is also changed
        # because we are now processing one micro-batch per training step.

        tokens, labels, advantages = (
            tokens.to(config.device),
            labels.to(config.device),
            advantages.to(config.device),
        )
        print(f"DEBUG: Micro-batch advantages: mean={advantages.mean().item():.4f}, std={advantages.std().item():.4f}, min={advantages.min().item():.4f}, max={advantages.max().item():.4f}")

        # Forward pass
        outputs = model(tokens)
        logits = outputs.logits

        # Calculate GRPO loss
        logp_per_token = -F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view(labels.shape)

        mask = (labels != -100).float()
        with torch.no_grad():
            pos_mask = (advantages > 0).float() # Renamed from pos to avoid conflict
            neg_mask = (advantages <= 0).float() # Renamed from neg
            
            # Calculate stats for this micro-batch
            # Ensure mask.sum(-1) is not zero to avoid division by zero
            # Sum logp_per_token only where mask is active, then divide by count of active tokens
            sum_masked_logp = (logp_per_token * mask).sum(-1)
            count_masked_tokens = mask.sum(-1)
            
            # Handle cases where count_masked_tokens might be zero for some items in the batch
            avg_logp_per_item = torch.zeros_like(sum_masked_logp)
            valid_mask_items = count_masked_tokens > 0
            avg_logp_per_item[valid_mask_items] = sum_masked_logp[valid_mask_items] / count_masked_tokens[valid_mask_items]
            
            # For logging, we might want the average over the items in the micro-batch
            current_avg_logp = avg_logp_per_item.mean().item() # Average of per-item average logps
            
            # For pos/neg logp, we average over tokens that are positive/negative AND masked
            # This interpretation aligns with how DPO often calculates these metrics.
            # Only consider tokens where labels are not -100
            valid_pos_tokens = pos_mask * mask
            valid_neg_tokens = neg_mask * mask

            if valid_pos_tokens.sum() > 0:
                current_pos_logp = (logp_per_token * valid_pos_tokens).sum() / valid_pos_tokens.sum()
                total_pos_logp = current_pos_logp.item() # Store for logging
            else:
                total_pos_logp = 0 # Or np.nan or some other indicator

            if valid_neg_tokens.sum() > 0:
                current_neg_logp = (logp_per_token * valid_neg_tokens).sum() / valid_neg_tokens.sum()
                total_neg_logp = current_neg_logp.item() # Store for logging
            else:
                total_neg_logp = 0 # Or np.nan

            total_logp = current_avg_logp # Store for logging; this is per micro-batch average
            total_pos = valid_pos_tokens.sum().item() # Count of positive advantage tokens in this micro-batch
            total_neg = valid_neg_tokens.sum().item() # Count of negative advantage tokens


        grpo_loss_term = torch.exp(logp_per_token - logp_per_token.detach())
        
        # Calculate mean loss per sequence, then mean over batch, then scale by grad accum steps
        # Ensure mask.sum(-1) is not zero before division
        masked_sum_loss_term = (-grpo_loss_term * mask).sum(-1)
        
        # Initialize per_item_loss to zeros
        per_item_loss = torch.zeros_like(masked_sum_loss_term)
        # Only calculate for items where there are valid tokens
        per_item_loss[valid_mask_items] = masked_sum_loss_term[valid_mask_items] / count_masked_tokens[valid_mask_items]
        
        # Multiply by advantages and then take the mean over the batch
        # advantages has shape (batch_size, 1), per_item_loss has (batch_size)
        # We need to ensure they are compatible for element-wise multiplication.
        # Reshape advantages to (batch_size) if it's (batch_size, 1)
        final_loss_per_item = per_item_loss * advantages.view(-1).to(logp_per_token.device)
        
        grpo_loss = final_loss_per_item.mean() / config.gradient_accumulation_steps
        
        grpo_loss.backward()
        # total_loss is now just the grpo_loss for this step, as we process one micro-batch per step.
        # The accumulation of total_loss over multiple micro-batches within a step is removed.
        step_loss = grpo_loss.item() * config.gradient_accumulation_steps # Get unscaled loss for logging

        # Gradient accumulation logic:
        # The optimizer step and zero_grad should only happen after 'gradient_accumulation_steps'
        # This seems to be missing. The current code does optimizer.step() every micro-batch.
        # Let's assume for now the user intends to update every micro-batch and gradient_accumulation_steps
        # is just a scaling factor for the loss. If true accumulation is needed, this needs a rewrite.
        # The prompt talks about "constant step loss" and "Step Loss: 0.1253" which implies one loss per step.
        # The division by config.gradient_accumulation_steps and then backward() happens per micro-batch.
        # optimizer.step() also happens per micro-batch. This means gradient_accumulation_steps is only
        # scaling the loss, not actually accumulating gradients over multiple batches.

        # If true gradient accumulation is intended, the following should be outside this loop
        # and happen every `config.gradient_accumulation_steps` micro-batches.
        # For now, I will keep the existing structure where optimizer.step() is called every micro-batch
        # as changing that is a more significant alteration of the training loop.
        # The primary goal here is to fix the "constant loss" by fixing data loading.

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        
        # Logging for this step (micro-batch)
        if config.use_wandb:
            wandb.log(
                {
                    "train/loss": step_loss, # Log the unaccumulated loss for this micro-batch
                    "train/learning_rate": optimizer.param_groups[0]["lr"],
                    "train/grad_norm": grad_norm.item(),
                    "train/pos_logp_micro": total_pos_logp, # Log per micro-batch
                    "train/neg_logp_micro": total_neg_logp, # Log per micro-batch
                    "train/avg_logp_micro": total_logp,     # Log per micro-batch
                    "train/pos_samples_micro": total_pos,
                    "train/neg_samples_micro": total_neg,
                },
                step=step + 1,
            )
        # --- End Wandb Logging ---

        print(f"  Step Loss: {step_loss:.4f}") # Print unscaled loss

        # --- vLLM Restart Logic ---
        # This logic might need to be adjusted if training_steps means epochs vs micro-batches
        # Current assumption: training_steps refers to optimizer updates.
        if (
            step + 1
        ) % config.vllm_restart_interval == 0 or step == config.training_steps - 1:
            checkpoint_path = os.path.join(
                config.save_path, f"step_{step+1}"
            )  # Save as step+1 since it's after step completion
            print(f"  Saving checkpoint to {checkpoint_path}...")
            # Ensure fresh directory for saving
            if os.path.exists(checkpoint_path):
                shutil.rmtree(checkpoint_path)  # Remove old checkpoint if it exists
            os.makedirs(checkpoint_path, exist_ok=True)
            model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)
            print("  Checkpoint saved.")

            # Terminate existing vLLM process if running
            if vllm_process:
                print("  Terminating existing vLLM process...")
                vllm_process.terminate()
                try:
                    vllm_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print(
                        "  Existing vLLM process did not terminate gracefully, killing."
                    )
                    vllm_process.kill()
                    vllm_process.wait()
                vllm_process = None

            # Launch new vLLM process (only if not the very last step, maybe? depends on use case)
            # Let's still launch it on the last step for consistency, cleanup will handle it.
            vllm_command = [
                "python",
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                os.path.join(config.save_path, f"step_{step+1}"),
                "--port",
                str(config.vllm_port),
                "--dtype",
                "auto",
                "--gpu-memory-utilization",
                "0.45",
                "--disable-log-requests",
                "--served-model-name",
                config.model_name,
            ]
            print(f"  Launching vLLM server: {' '.join(vllm_command)}")
            torch.cuda.empty_cache()
            try:
                vllm_process = subprocess.Popen(vllm_command)
                print(f"  vLLM server launched with PID: {vllm_process.pid}")
                # Check immediate errors
                try:
                    stdout, stderr = vllm_process.communicate(timeout=2)
                    if (
                        vllm_process.returncode is not None
                        and vllm_process.returncode != 0
                    ):
                        print(f"  Error starting vLLM: {stderr.decode()}")
                        vllm_process = None
                        # Maybe raise error or just warn?
                        print(
                            "  WARNING: Failed to start vLLM server after checkpoint."
                        )
                except subprocess.TimeoutExpired:
                    print("  vLLM process started (check logs for details).")
            except FileNotFoundError:
                print(
                    "\n *** ERROR: 'python -m vllm...' command not found. ",
                    "Make sure vLLM is installed and accessible. ***\n",
                )
                # Potentially stop training or just disable further vLLM restarts
                print("  Disabling further vLLM restarts.")
                config.vllm_restart_interval = (
                    config.training_steps + 1
                )  # Prevent further restarts
            except Exception as e:
                print(f"\n *** ERROR: Failed to launch vLLM: {e} ***\n")
                print("  Disabling further vLLM restarts.")
                config.vllm_restart_interval = (
                    config.training_steps + 1
                )  # Prevent further restarts
        # --- End vLLM Restart Logic ---

        # Basic check if vLLM process terminated unexpectedly (outside interval check)
        if vllm_process and vllm_process.poll() is not None:
            print(
                f"\n *** WARNING: vLLM process terminated unexpectedly (return code: {vllm_process.returncode}). ",
                "Check vLLM logs. ***\n",
            )
            stderr_output = (
                vllm_process.stderr.read().decode()
                if vllm_process.stderr
                else "No stderr"
            )
            print(f"vLLM stderr: {stderr_output}")
            vllm_process = None  # Reset so it relaunches next interval

    print("Training finished.")
    # --- Wandb Finish ---
    if config.use_wandb:
        wandb.finish()
    # --- End Wandb Finish ---
    # Final cleanup (vLLM termination) is handled by atexit

    # --- Placeholder for final model save ---
    final_save_path = os.path.join(config.save_path, "final_model")
    print(f"Saving final model to {final_save_path}")
    if os.path.exists(final_save_path):
        shutil.rmtree(final_save_path)
    os.makedirs(final_save_path, exist_ok=True)
    model.save_pretrained(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    print("Final model saved.")


# Example usage (optional, can be run from another script)
if __name__ == "__main__":
    # Example: Create a config and run training
    # Replace "gpt2" with your desired model
    training_config = TrainingConfig(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        training_steps=20,  # Use steps
        lr=1e-4, # Increased learning rate
        batch_size=4,  # Reduced from 2
        gradient_accumulation_steps=32, 
        vllm_restart_interval=3,  # Example interval
        use_wandb=True,  # Set to True to enable logging
        wandb_project="grpo-trainer-example",  # Replace with your project name
    )

    # --- End Mock ---

    train(training_config)
