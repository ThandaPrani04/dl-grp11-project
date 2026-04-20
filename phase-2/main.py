"""
Phase 2: ToolAlpaca SFT & ReAct Execution Loop (Qwen 7B)

This script implements:
1. Supervised Fine-Tuning (SFT): Fine-tune Qwen 7B on ToolAlpaca-style trajectories
2. ReAct Execution Engine: Interactive agentic loop with tool execution

Parts:
    A: Qwen 7B SFT (LoRA)
    B: ReAct Execution Engine
    C: DPO (optional post-SFT alignment)
"""

# Imports & Authentication

import ast
import argparse
import inspect
import io
import json
import math
import os
import random
import re
import sys
import time
from contextlib import redirect_stdout
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from datasets import Dataset
from huggingface_hub import login
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed,
)
from transformers.generation import StoppingCriteria, StoppingCriteriaList
from trl import SFTTrainer

try:
    from trl import DPOConfig, DPOTrainer
except Exception:
    DPOConfig = None
    try:
        from trl import DPOTrainer
    except Exception:
        DPOTrainer = None

# Hugging Face login
hf_token = os.environ.get("HF_TOKEN")
local_rank_env = int(os.environ.get("LOCAL_RANK", "0"))

if hf_token:
    # In distributed runs, avoid concurrent token writes from all ranks.
    if local_rank_env == 0:
        try:
            login(token=hf_token, add_to_git_credential=False)
            print("Configured Hugging Face auth from HF_TOKEN (rank 0)")
        except Exception as e:
            print(f"HF login setup skipped ({e}); relying on HF_TOKEN env auth.")
    else:
        print("HF_TOKEN detected; skipping explicit HF login on non-zero local rank.")
else:
    print("HF_TOKEN not set. Inference from public repos will still work; pushing to Hub will fail.")

#Reproducibility
SEED = 42
set_seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# GPU info
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    print(f"Visible GPUs: {gpu_count}")
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name} | VRAM: {props.total_memory / 1e9:.2f} GB")
    print(f"CUDA Version: {torch.version.cuda}")

print("\nAll imports successful")

# Hub settings
HF_USERNAME = "souhhmm"
HF_REPO_SFT = f"{HF_USERNAME}/phase2-toolalpaca-sft-qwen2_5-7b"
HF_REPO_DPO = f"{HF_USERNAME}/phase2-toolalpaca-dpo-qwen2_5-7b"
print(f"\nModel repository:\n  SFT: {HF_REPO_SFT}\n  DPO: {HF_REPO_DPO}")


def parse_runtime_args() -> argparse.Namespace:
    """Parse runtime flags that control training/inference stages."""
    parser = argparse.ArgumentParser(description="Run Phase 2 training and/or inference")
    parser.add_argument(
        "--train_only",
        action="store_true",
        help="Run only SFT training (skip ReAct inference).",
    )
    parser.add_argument(
        "--infer_only",
        action="store_true",
        help="Run only ReAct inference (skip SFT training).",
    )
    parser.add_argument(
        "--run_dpo",
        action="store_true",
        help="Run DPO after SFT to preference-align tool decisions.",
    )
    parser.add_argument(
        "--dpo_only",
        action="store_true",
        help="Skip SFT retraining and run DPO from an existing SFT adapter.",
    )
    parser.add_argument(
        "--sft_adapter_repo",
        type=str,
        default=os.environ.get("SFT_ADAPTER_REPO", HF_REPO_SFT),
        help="SFT adapter repo/path to load for DPO-only mode.",
    )

    args, _ = parser.parse_known_args()
    if args.train_only and args.infer_only:
        parser.error("Choose only one of --train_only or --infer_only.")
    if args.infer_only and (args.run_dpo or args.dpo_only):
        parser.error("DPO flags cannot be combined with --infer_only.")
    if args.dpo_only:
        args.run_dpo = True
    return args


runtime_args = parse_runtime_args()
RUN_TRAINING = not runtime_args.infer_only
RUN_INFERENCE = not runtime_args.train_only
RUN_DPO = RUN_TRAINING and runtime_args.run_dpo
RUN_DPO_ONLY = RUN_DPO and runtime_args.dpo_only
RUN_SFT = RUN_TRAINING and not RUN_DPO_ONLY
SFT_ADAPTER_SOURCE_REPO = runtime_args.sft_adapter_repo
print(
    f"Runtime mode: training={RUN_TRAINING}, sft={RUN_SFT}, dpo={RUN_DPO}, "
    f"dpo_only={RUN_DPO_ONLY}, inference={RUN_INFERENCE}"
)


# Part A: Data Loading and Preparation

DATA_PATH = "./agent_trajectories_2k_new.json"
AUGMENT_BEFORE_TRAIN = os.environ.get("AUGMENT_BEFORE_TRAIN", "1").strip().lower() not in {"0", "false", "no"}
AUGMENT_TARGET_COUNT = int(os.environ.get("AUGMENT_TARGET_COUNT", "10000"))
AUGMENT_SEED = int(os.environ.get("AUGMENT_SEED", str(SEED)))
AUGMENT_SAVE_PATH = os.environ.get("AUGMENT_SAVE_PATH")


def load_trajectories(path: str) -> List[Dict]:
    """Load agent trajectories for training."""
    with open(path, "r") as f:
        trajectories = json.load(f)
    print(f"Loaded {len(trajectories)} trajectories")
    return trajectories


def _load_augment_module():
    """Import augmentation helpers from augment_queries.py."""
    try:
        import augment_queries as augment_module

        return augment_module
    except ModuleNotFoundError:
        script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
        if script_dir not in sys.path:
            sys.path.append(script_dir)
        import augment_queries as augment_module

        return augment_module


def augment_trajectories_for_training(
    originals: List[Dict[str, Any]],
    target_count: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Expand trajectories with synthetic query/action pairs before SFT."""
    if target_count <= len(originals):
        print(
            f"Augmentation skipped: target_count ({target_count}) <= original count ({len(originals)})."
        )
        return originals

    random_state = random.getstate()

    try:
        augment_module = _load_augment_module()
    except Exception as e:
        random.setstate(random_state)
        print(f"Augmentation skipped: unable to import augment_queries.py ({e}).")
        return originals

    n_to_generate = max(0, target_count - len(originals))
    print(f"Generating approximately {n_to_generate} augmented samples before SFT...")

    random.seed(seed)

    failures = 0
    generated: List[Dict[str, Any]] = []
    registry = augment_module.build_pattern_registry(n_to_generate)
    for pattern_key, template_key, sampler_fn, n_samples in registry:
        for _ in range(n_samples):
            try:
                generated.append(augment_module.generate_sample(pattern_key, template_key, sampler_fn))
            except Exception:
                failures += 1

    random.shuffle(generated)
    combined = augment_module.deduplicate(originals + generated)

    # Top up if deduplication removes too many generated samples.
    topup_round = 0
    while len(combined) < target_count and topup_round < 3:
        topup_round += 1
        remaining = target_count - len(combined)
        topup_registry = augment_module.build_pattern_registry(remaining)
        topup_samples: List[Dict[str, Any]] = []
        for pattern_key, template_key, sampler_fn, n_samples in topup_registry:
            for _ in range(n_samples):
                try:
                    topup_samples.append(
                        augment_module.generate_sample(pattern_key, template_key, sampler_fn)
                    )
                except Exception:
                    failures += 1
        random.shuffle(topup_samples)
        combined = augment_module.deduplicate(combined + topup_samples)

    random.shuffle(combined)
    random.setstate(random_state)

    added = len(combined) - len(originals)
    print(f"Augmentation complete: +{added} unique samples (failed generations: {failures}).")
    if len(combined) < target_count:
        print(
            f"Warning: target_count={target_count}, but final deduplicated count is {len(combined)}."
        )

    if AUGMENT_SAVE_PATH:
        with open(AUGMENT_SAVE_PATH, "w") as f:
            json.dump(combined, f, indent=2)
        print(f"Saved augmented trajectories to {AUGMENT_SAVE_PATH}")

    return combined


if RUN_TRAINING:
    trajectories = load_trajectories(DATA_PATH)
    if AUGMENT_BEFORE_TRAIN:
        trajectories = augment_trajectories_for_training(
            trajectories,
            target_count=AUGMENT_TARGET_COUNT,
            seed=AUGMENT_SEED,
        )
    else:
        print("Augmentation disabled via AUGMENT_BEFORE_TRAIN.")

    print(f"Trajectories available for training: {len(trajectories)}")

    print("\n--- Sample Trajectory ---")
    if trajectories:
        sample = trajectories[0]
        print(json.dumps(sample, indent=2)[:500])
else:
    trajectories = []
    print("Training stage skipped: not loading or augmenting trajectories.")


# Strict Turn-Level SFT Dataset

TOOL_SCHEMA = {
    "filter_data": ["column", "value"],
    "group_by": ["column", "aggregate_column", "aggregate_function"],
    "aggregate": ["column", "aggregate_function"],
    "sort_by": ["column", "order"],
    "topk": ["column", "k", "order"],
}


def parse_action_call(action_str: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Parse action string like tool(a=1, b='x') into (tool, args_dict)."""
    if "(" not in action_str or ")" not in action_str:
        return None

    tool_name = action_str.split("(", 1)[0].strip()
    args_str = action_str[action_str.index("(") + 1 : action_str.rindex(")")].strip()

    if not tool_name:
        return None
    if not args_str:
        return tool_name, {}

    try:
        expr = ast.parse(f"f({args_str})", mode="eval").body
        if not isinstance(expr, ast.Call):
            return None
        args: Dict[str, Any] = {}
        for kw in expr.keywords:
            if kw.arg is None:
                continue
            args[kw.arg] = ast.literal_eval(kw.value)
        return tool_name, args
    except Exception:
        return None


def normalize_tool_and_args(
    tool_name: str, args: Dict[str, Any]
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Normalize aliases and argument names to canonical tool schema."""
    tool_alias = {
        "top_k": "topk",
        "top-k": "topk",
        "aggregate_sum": "aggregate",
        "aggregate_mean": "aggregate",
        "aggregate_avg": "aggregate",
        "aggregate_count": "aggregate",
        "aggregate_max": "aggregate",
        "aggregate_min": "aggregate",
    }
    canonical_tool = tool_alias.get(tool_name, tool_name)

    if canonical_tool not in TOOL_SCHEMA:
        return None

    normalized = dict(args)

    if canonical_tool == "aggregate":
        if "aggregation" in normalized and "aggregate_function" not in normalized:
            normalized["aggregate_function"] = normalized.pop("aggregation")
        alias_to_agg = {
            "aggregate_sum": "sum",
            "aggregate_mean": "mean",
            "aggregate_avg": "mean",
            "aggregate_count": "count",
            "aggregate_max": "max",
            "aggregate_min": "min",
        }
        if tool_name in alias_to_agg and "aggregate_function" not in normalized:
            normalized["aggregate_function"] = alias_to_agg[tool_name]

    if canonical_tool == "group_by":
        if "function" in normalized and "aggregate_function" not in normalized:
            normalized["aggregate_function"] = normalized.pop("function")

    allowed = set(TOOL_SCHEMA[canonical_tool])
    normalized = {k: v for k, v in normalized.items() if k in allowed}

    if canonical_tool == "aggregate" and "aggregate_function" in normalized:
        normalized["aggregation"] = normalized.pop("aggregate_function")

    return canonical_tool, normalized


def format_state_summary(prev_actions: List[Dict[str, Any]]) -> str:
    """Compact state summary for turn-level supervision."""
    if not prev_actions:
        return "No tools executed yet."
    recent = prev_actions[-2:]
    lines = [
        f"{idx}. action={step['tool']} args={json.dumps(step['args'], ensure_ascii=True)}"
        for idx, step in enumerate(recent, 1)
    ]
    return "Recent steps: " + " | ".join(lines)


def make_prompt(query: str, state_summary: str) -> str:
    return f"""You are a tool-using data assistant.

User Query: {query}
Current State: {state_summary}

You must output exactly one next tool call.
Rules:
- Choose ONE tool from: filter_data, group_by, aggregate, sort_by, topk
- Output strict JSON in Action Input
- Do not output Final Answer in this training phase

Format:
Thought: <brief reasoning>
Action: <tool name>
Action Input: <valid JSON object>
"""


def make_completion(tool: str, args: Dict[str, Any]) -> str:
    return (
        "Thought: I will execute the next best tool step based on the current state.\n"
        f"Action: {tool}\n"
        f"Action Input: {json.dumps(args, ensure_ascii=True)}"
    )


def create_strict_turn_sft_dataset(
    trajectories: List[Dict[str, Any]],
    train_split: float = 0.9,
    max_turns_per_query: int = 6,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Create strict one-action-per-example SFT dataset.
    Each trajectory with N actions yields up to N training examples.
    """
    dataset: List[Dict[str, str]] = []

    for traj in trajectories:
        query = traj.get("query", "").strip()
        raw_actions = traj.get("actions", [])
        if not query or not raw_actions:
            continue

        parsed_steps: List[Dict[str, Any]] = []
        for action_str in raw_actions:
            parsed = parse_action_call(action_str)
            if not parsed:
                continue
            tool_name, args = parsed
            normalized = normalize_tool_and_args(tool_name, args)
            if not normalized:
                continue
            tool, clean_args = normalized
            parsed_steps.append({"tool": tool, "args": clean_args})

        if not parsed_steps:
            continue

        for t, step in enumerate(parsed_steps[:max_turns_per_query]):
            state_summary = format_state_summary(parsed_steps[:t])
            prompt = make_prompt(query, state_summary)
            completion = make_completion(step["tool"], step["args"])
            dataset.append(
                {
                    "prompt": prompt,
                    "completion": completion,
                    "text": prompt + completion,
                }
            )

    split_idx = int(len(dataset) * train_split)
    train_data = dataset[:split_idx]
    val_data = dataset[split_idx:]

    print(f"Strict SFT examples: {len(dataset)}")
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")

    return train_data, val_data


def _default_arg_value(arg_name: str) -> Any:
    """Fallback values used to synthesize low-quality negative actions for DPO."""
    defaults = {
        "column": "invalid_column",
        "value": "invalid_value",
        "aggregate_column": "invalid_metric",
        "aggregate_function": "mean",
        "aggregation": "mean",
        "order": "asc",
        "k": 1,
    }
    return defaults.get(arg_name, "invalid")


def make_rejected_completion(tool: str, args: Dict[str, Any]) -> str:
    """Create a less-preferred completion to train DPO against errors and weak logic."""
    strategy = random.choice(["wrong_tool", "missing_arg", "bad_value", "invalid_json"])
    bad_tool = random.choice([t for t in TOOL_SCHEMA.keys() if t != tool])

    if strategy == "invalid_json":
        return (
            "Thought: I can answer directly without proper tool reasoning.\n"
            f"Action: {tool}\n"
            "Action Input: {"  # intentionally malformed JSON
        )

    if strategy == "wrong_tool":
        bad_args = {name: _default_arg_value(name) for name in TOOL_SCHEMA[bad_tool]}
        return (
            "Thought: I will use a quick shortcut even if it may be incorrect.\n"
            f"Action: {bad_tool}\n"
            f"Action Input: {json.dumps(bad_args, ensure_ascii=True)}"
        )

    if strategy == "missing_arg":
        bad_args = dict(args)
        if bad_args:
            dropped = random.choice(list(bad_args.keys()))
            bad_args.pop(dropped, None)
        if not bad_args:
            bad_args = {"column": "invalid_column"}
        return (
            "Thought: I can skip some parameters and still proceed.\n"
            f"Action: {tool}\n"
            f"Action Input: {json.dumps(bad_args, ensure_ascii=True)}"
        )

    bad_args = dict(args)
    if not bad_args:
        bad_args = {"column": "invalid_column"}
    first_key = next(iter(bad_args))
    if isinstance(bad_args[first_key], (int, float)):
        bad_args[first_key] = 999999
    elif isinstance(bad_args[first_key], str):
        bad_args[first_key] = "unknown"
    else:
        bad_args[first_key] = _default_arg_value(first_key)

    return (
        "Thought: I will try a likely value without checking prior state.\n"
        f"Action: {tool}\n"
        f"Action Input: {json.dumps(bad_args, ensure_ascii=True)}"
    )


def create_dpo_preference_dataset(
    trajectories: List[Dict[str, Any]],
    train_split: float = 0.95,
    max_turns_per_query: int = 6,
    max_pairs: int = 30000,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Create (prompt, chosen, rejected) preference pairs for DPO."""
    pairs: List[Dict[str, str]] = []

    for traj in trajectories:
        query = traj.get("query", "").strip()
        raw_actions = traj.get("actions", [])
        if not query or not raw_actions:
            continue

        parsed_steps: List[Dict[str, Any]] = []
        for action_str in raw_actions:
            parsed = parse_action_call(action_str)
            if not parsed:
                continue
            tool_name, args = parsed
            normalized = normalize_tool_and_args(tool_name, args)
            if not normalized:
                continue
            tool, clean_args = normalized
            parsed_steps.append({"tool": tool, "args": clean_args})

        if not parsed_steps:
            continue

        for t, step in enumerate(parsed_steps[:max_turns_per_query]):
            prompt = make_prompt(query, format_state_summary(parsed_steps[:t]))
            chosen = make_completion(step["tool"], step["args"])
            rejected = make_rejected_completion(step["tool"], step["args"])
            if chosen.strip() == rejected.strip():
                continue
            pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

    random.shuffle(pairs)
    if max_pairs > 0 and len(pairs) > max_pairs:
        pairs = pairs[:max_pairs]

    split_idx = int(len(pairs) * train_split)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    print(f"DPO preference pairs: {len(pairs)}")
    print(f"DPO train pairs: {len(train_pairs)}")
    print(f"DPO validation pairs: {len(val_pairs)}")

    return train_pairs, val_pairs


if RUN_SFT:
    train_data, val_data = create_strict_turn_sft_dataset(trajectories)

    print("\n--- Sample Strict Training Example ---")
    if train_data:
        print(train_data[0]["text"][:900])
else:
    train_data, val_data = [], []
    print("SFT stage skipped: strict SFT dataset creation disabled.")


# Model Configuration and Setup (Qwen 7B)

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR = "./checkpoints/phase2-sft-qwen2_5-7b"
ADAPTER_OUTPUT_DIR = "./adapters/phase2-sft-qwen2_5-7b"
MAX_SEQ_LENGTH = 1024

DPO_OUTPUT_DIR = "./checkpoints/phase2-dpo-qwen2_5-7b"
DPO_ADAPTER_OUTPUT_DIR = "./adapters/phase2-dpo-qwen2_5-7b"
DPO_BETA = float(os.environ.get("DPO_BETA", "0.1"))
DPO_EPOCHS = float(os.environ.get("DPO_EPOCHS", "1.0"))
DPO_MAX_PAIRS = int(os.environ.get("DPO_MAX_PAIRS", "30000"))


def estimate_warmup_steps(
    dataset_size: int,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    world_size: int,
    num_train_epochs: float,
    warmup_ratio: float = 0.03,
) -> int:
    """Approximate warmup steps from a ratio without using deprecated warmup_ratio."""
    global_batch = max(1, per_device_batch_size * gradient_accumulation_steps * max(1, world_size))
    steps_per_epoch = max(1, (max(1, dataset_size) + global_batch - 1) // global_batch)
    total_steps = max(1, int(math.ceil(steps_per_epoch * max(0.0, num_train_epochs))))
    return max(1, int(math.ceil(total_steps * max(0.0, warmup_ratio))))

if RUN_TRAINING:
    GPU_COUNT = torch.cuda.device_count() if torch.cuda.is_available() else 0
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
    IS_DISTRIBUTED = WORLD_SIZE > 1
    effective_world_size = max(1, WORLD_SIZE)

    supports_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if supports_bf16 else torch.float16
    base_per_device_train_batch_size = 8 if supports_bf16 else 4
    base_gradient_accumulation_steps = 2 if effective_world_size >= 4 else 4

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(ADAPTER_OUTPUT_DIR, exist_ok=True)

    print(f"Loading model: {MODEL_NAME}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"GPUs visible: {GPU_COUNT} | WORLD_SIZE: {WORLD_SIZE} | distributed: {IS_DISTRIBUTED}")
    print(f"Using compute dtype: {compute_dtype}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    device_map = {"": LOCAL_RANK} if IS_DISTRIBUTED else "auto"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        torch_dtype=compute_dtype,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        device_map=device_map,
        trust_remote_code=True,
    )

    model = prepare_model_for_kbit_training(model)
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False

    if RUN_DPO_ONLY:
        print(f"Loading existing SFT adapter for DPO-only mode: {SFT_ADAPTER_SOURCE_REPO}")
        try:
            model = PeftModel.from_pretrained(
                model,
                SFT_ADAPTER_SOURCE_REPO,
                is_trainable=True,
                device_map=device_map,
            )
        except TypeError:
            model = PeftModel.from_pretrained(
                model,
                SFT_ADAPTER_SOURCE_REPO,
                device_map=device_map,
            )
        model.train()
        model.config.use_cache = False
        print("SFT adapter loaded in trainable mode for DPO.")
    else:
        print("Qwen model loaded and prepared for QLoRA")
else:
    print("Training stage skipped: model setup, SFT, and adapter export are disabled.")


# LoRA Configuration

if RUN_SFT:
    candidate_targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    present = set()
    for module_name, _ in model.named_modules():
        leaf = module_name.split(".")[-1]
        if leaf in candidate_targets:
            present.add(leaf)

    target_modules = [m for m in candidate_targets if m in present]
    if not target_modules:
        raise ValueError(
            "No compatible LoRA target modules found. "
            "Inspect model.named_modules() and update candidate_targets."
        )

    print(f"LoRA target modules: {target_modules}")

    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print("\nLoRA configuration applied")


# Training Configuration

if RUN_SFT:
    train_dataset = Dataset.from_dict({"text": [d["text"] for d in train_data]})
    val_dataset = Dataset.from_dict({"text": [d["text"] for d in val_data]})

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")

    per_device_train_batch_size = base_per_device_train_batch_size
    per_device_eval_batch_size = per_device_train_batch_size
    gradient_accumulation_steps = base_gradient_accumulation_steps
    optim_name = "adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch"
    sft_num_epochs = 3
    sft_warmup_steps = estimate_warmup_steps(
        dataset_size=len(train_dataset),
        per_device_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        world_size=effective_world_size,
        num_train_epochs=float(sft_num_epochs),
        warmup_ratio=0.03,
    )

    training_kwargs = {
        "output_dir": OUTPUT_DIR,
        "num_train_epochs": sft_num_epochs,
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": per_device_eval_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": 2e-4,
        "weight_decay": 0.0,
        "warmup_steps": sft_warmup_steps,
        "logging_steps": 5,
        "save_steps": 100,
        "eval_steps": 100,
        "save_total_limit": 2,
        "save_strategy": "steps",
        "logging_strategy": "steps",
        "fp16": torch.cuda.is_available() and not supports_bf16,
        "bf16": supports_bf16,
        "tf32": True,
        "gradient_checkpointing": True,
        "optim": optim_name,
        "max_grad_norm": 0.3,
        "group_by_length": True,
        "lr_scheduler_type": "cosine",
        "ddp_find_unused_parameters": False,
        "dataloader_num_workers": 8,
        "dataloader_pin_memory": True,
        "report_to": "none",
        "save_safetensors": True,
        "remove_unused_columns": False,
        "push_to_hub": True,
        "hub_model_id": HF_REPO_SFT,
        "hub_strategy": "every_save",
        "hub_private_repo": False,
        "hub_token": hf_token,
    }

    supported = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    if "eval_strategy" in supported:
        training_kwargs["eval_strategy"] = "steps"
    elif "evaluation_strategy" in supported:
        training_kwargs["evaluation_strategy"] = "steps"

    if "gradient_checkpointing_kwargs" in supported:
        training_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    if "dataloader_prefetch_factor" in supported:
        training_kwargs["dataloader_prefetch_factor"] = 4
    if "ddp_bucket_cap_mb" in supported:
        training_kwargs["ddp_bucket_cap_mb"] = 200

    training_args = TrainingArguments(**{k: v for k, v in training_kwargs.items() if k in supported})

    global_batch = per_device_train_batch_size * gradient_accumulation_steps * effective_world_size
    print("\nTraining configuration prepared")
    print(f"Will push checkpoints to: {HF_REPO_SFT}")
    print(f"Precision mode: bf16={supports_bf16}, fp16={not supports_bf16 and torch.cuda.is_available()}")
    print(f"Effective global batch size: {global_batch}")
    print(f"Optimizer: {optim_name}")


# Runtime check
if RUN_TRAINING:
    print(f"GPU_COUNT={GPU_COUNT}, WORLD_SIZE={WORLD_SIZE}, LOCAL_RANK={LOCAL_RANK}")
    if GPU_COUNT >= 4 and WORLD_SIZE < 4:
        print("\n4 GPUs are visible but training is not in 4-process distributed mode.")
        print("For maximum throughput, launch as a distributed job (WORLD_SIZE=4),")
        print("or run through an Accelerate/torchrun workflow with 4 processes.")
    else:
        print("\nRuntime is ready for high-throughput multi-GPU training.")


# SFT Training


if RUN_SFT:
    sft_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "dataset_text_field": "text",
        "max_seq_length": MAX_SEQ_LENGTH,
        "packing": True,
        "tokenizer": tokenizer,
    }

    sft_supported = set(inspect.signature(SFTTrainer.__init__).parameters.keys())
    if "dataset_num_proc" in sft_supported:
        sft_kwargs["dataset_num_proc"] = max(1, min(8, os.cpu_count() or 1))
    if "dataset_batch_size" in sft_supported:
        sft_kwargs["dataset_batch_size"] = 1000
    if not isinstance(model, PeftModel):
        sft_kwargs["peft_config"] = lora_config

    trainer = SFTTrainer(**{k: v for k, v in sft_kwargs.items() if k in sft_supported})

    print("Starting Qwen 7B SFT training...")
    print("If using 4 GPUs, launch with distributed runtime (WORLD_SIZE=4) for max throughput.\n")

    start_time = time.time()
    try:
        train_result = trainer.train()
        elapsed_min = (time.time() - start_time) / 60
        print("\nTraining completed successfully")
        print(f"Final training loss: {train_result.training_loss:.4f}")
        print(f"Elapsed time: {elapsed_min:.2f} minutes")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining error: {e}")


# Save adapter
if RUN_SFT:
    model.save_pretrained(ADAPTER_OUTPUT_DIR)
    tokenizer.save_pretrained(ADAPTER_OUTPUT_DIR)

    print(f"Adapter saved locally to {ADAPTER_OUTPUT_DIR}")
    print("\nAdapter files:")
    for file in os.listdir(ADAPTER_OUTPUT_DIR):
        print(f"  - {file}")

    print(f"\nAdapter also pushed to Hugging Face Hub during training:")
    print(f"  Repository: {HF_REPO_SFT}")
    print(f"  Access: https://huggingface.co/{HF_REPO_SFT}")
    print("\nYou can load the adapter later with:")
    print(f"  model = PeftModel.from_pretrained(base_model, '{HF_REPO_SFT}')")


# Part C: DPO (Optional post-SFT alignment)

if RUN_DPO:
    if DPOTrainer is None:
        print("DPO requested, but DPOTrainer is unavailable in this TRL version. Skipping DPO stage.")
    elif not trajectories:
        print("DPO requested, but no trajectories are loaded. Skipping DPO stage.")
    else:
        os.makedirs(DPO_OUTPUT_DIR, exist_ok=True)
        os.makedirs(DPO_ADAPTER_OUTPUT_DIR, exist_ok=True)

        dpo_train_pairs, dpo_val_pairs = create_dpo_preference_dataset(
            trajectories,
            train_split=0.95,
            max_turns_per_query=6,
            max_pairs=DPO_MAX_PAIRS,
        )

        if not dpo_train_pairs:
            print("DPO skipped: no preference pairs generated.")
        else:
            dpo_train_dataset = Dataset.from_list(dpo_train_pairs)
            dpo_val_dataset = Dataset.from_list(dpo_val_pairs) if dpo_val_pairs else None

            dpo_per_device_batch = max(1, base_per_device_train_batch_size // 2)
            dpo_grad_accum = base_gradient_accumulation_steps
            dpo_optim_name = "adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch"
            dpo_warmup_steps = estimate_warmup_steps(
                dataset_size=len(dpo_train_dataset),
                per_device_batch_size=dpo_per_device_batch,
                gradient_accumulation_steps=dpo_grad_accum,
                world_size=effective_world_size,
                num_train_epochs=float(DPO_EPOCHS),
                warmup_ratio=0.03,
            )

            dpo_training_kwargs = {
                "output_dir": DPO_OUTPUT_DIR,
                "num_train_epochs": DPO_EPOCHS,
                "per_device_train_batch_size": dpo_per_device_batch,
                "per_device_eval_batch_size": dpo_per_device_batch,
                "gradient_accumulation_steps": dpo_grad_accum,
                "learning_rate": 5e-6,
                "weight_decay": 0.0,
                "warmup_steps": dpo_warmup_steps,
                "logging_steps": 10,
                "save_steps": 100,
                "eval_steps": 100,
                "save_total_limit": 2,
                "save_strategy": "steps",
                "logging_strategy": "steps",
                "fp16": torch.cuda.is_available() and not supports_bf16,
                "bf16": supports_bf16,
                "tf32": True,
                "gradient_checkpointing": True,
                "optim": dpo_optim_name,
                "max_grad_norm": 0.3,
                "lr_scheduler_type": "cosine",
                "ddp_find_unused_parameters": False,
                "dataloader_num_workers": 8,
                "dataloader_pin_memory": True,
                "report_to": "none",
                "save_safetensors": True,
                "remove_unused_columns": False,
                "push_to_hub": True,
                "hub_model_id": HF_REPO_DPO,
                "hub_strategy": "every_save",
                "hub_private_repo": False,
                "hub_token": hf_token,
                "beta": DPO_BETA,
                "loss_type": "sigmoid",
                "max_length": MAX_SEQ_LENGTH,
                "max_prompt_length": min(768, MAX_SEQ_LENGTH - 256),
                "max_completion_length": 256,
                "max_target_length": 256,
                "truncation_mode": "keep_start",
            }

            dpo_args_cls = DPOConfig if DPOConfig is not None else TrainingArguments
            dpo_supported_args = set(inspect.signature(dpo_args_cls.__init__).parameters.keys())
            if dpo_val_dataset is not None:
                if "eval_strategy" in dpo_supported_args:
                    dpo_training_kwargs["eval_strategy"] = "steps"
                elif "evaluation_strategy" in dpo_supported_args:
                    dpo_training_kwargs["evaluation_strategy"] = "steps"

            if "gradient_checkpointing_kwargs" in dpo_supported_args:
                dpo_training_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
            if "dataloader_prefetch_factor" in dpo_supported_args:
                dpo_training_kwargs["dataloader_prefetch_factor"] = 4
            if "ddp_bucket_cap_mb" in dpo_supported_args:
                dpo_training_kwargs["ddp_bucket_cap_mb"] = 200

            dpo_training_args = dpo_args_cls(
                **{k: v for k, v in dpo_training_kwargs.items() if k in dpo_supported_args}
            )

            dpo_trainer_kwargs: Dict[str, Any] = {
                "model": model,
                "args": dpo_training_args,
                "train_dataset": dpo_train_dataset,
            }
            if dpo_val_dataset is not None:
                dpo_trainer_kwargs["eval_dataset"] = dpo_val_dataset

            dpo_supported = set(inspect.signature(DPOTrainer.__init__).parameters.keys())
            if "ref_model" in dpo_supported:
                dpo_trainer_kwargs["ref_model"] = None
            if "tokenizer" in dpo_supported:
                dpo_trainer_kwargs["tokenizer"] = tokenizer
            if "processing_class" in dpo_supported:
                dpo_trainer_kwargs["processing_class"] = tokenizer
            if "beta" in dpo_supported:
                dpo_trainer_kwargs["beta"] = DPO_BETA
            if "max_length" in dpo_supported:
                dpo_trainer_kwargs["max_length"] = MAX_SEQ_LENGTH
            if "max_prompt_length" in dpo_supported:
                dpo_trainer_kwargs["max_prompt_length"] = min(768, MAX_SEQ_LENGTH - 256)
            if "max_target_length" in dpo_supported:
                dpo_trainer_kwargs["max_target_length"] = 256
            if "loss_type" in dpo_supported:
                dpo_trainer_kwargs["loss_type"] = "sigmoid"

            print("\nStarting DPO alignment...")
            print(f"DPO beta: {DPO_BETA} | max pairs: {DPO_MAX_PAIRS} | epochs: {DPO_EPOCHS}")

            dpo_trainer = DPOTrainer(
                **{k: v for k, v in dpo_trainer_kwargs.items() if k in dpo_supported}
            )

            dpo_start = time.time()
            try:
                dpo_result = dpo_trainer.train()
                dpo_elapsed_min = (time.time() - dpo_start) / 60
                print("DPO training completed successfully")
                print(f"Final DPO loss: {dpo_result.training_loss:.4f}")
                print(f"DPO elapsed time: {dpo_elapsed_min:.2f} minutes")
            except KeyboardInterrupt:
                print("DPO training interrupted by user")
            except Exception as e:
                print(f"DPO training error: {e}")

            model.save_pretrained(DPO_ADAPTER_OUTPUT_DIR)
            tokenizer.save_pretrained(DPO_ADAPTER_OUTPUT_DIR)
            print(f"DPO adapter saved locally to {DPO_ADAPTER_OUTPUT_DIR}")

            if hf_token:
                try:
                    model.push_to_hub(HF_REPO_DPO, token=hf_token)
                    tokenizer.push_to_hub(HF_REPO_DPO, token=hf_token)
                    print(f"DPO adapter pushed to Hugging Face Hub: https://huggingface.co/{HF_REPO_DPO}")
                except Exception as e:
                    print(f"DPO push_to_hub warning: {e}")
else:
    print("DPO stage skipped (pass --run_dpo to enable).")


# ReAct Execution Engine

class StopOnObservationCriteria(StoppingCriteria):
    """Stop generation when the model predicts 'Observation:' token sequence."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.observation_tokens = tokenizer.encode("Observation:", add_special_tokens=False)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if len(input_ids[0]) >= len(self.observation_tokens):
            last_tokens = input_ids[0][-len(self.observation_tokens) :].tolist()
            if last_tokens == self.observation_tokens:
                return True
        return False


class StopOnFinalAnswerCriteria(StoppingCriteria):
    """Stop generation when 'Final Answer:' is detected."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.final_answer_tokens = tokenizer.encode("Final Answer:", add_special_tokens=False)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if len(input_ids[0]) >= len(self.final_answer_tokens):
            last_tokens = input_ids[0][-len(self.final_answer_tokens) :].tolist()
            if last_tokens == self.final_answer_tokens:
                return True
        return False


print("Stopping criteria classes defined")


# Parsing Utilities

def sanitize_json_string(s: str) -> str:
    """Clean up malformed JSON before parsing."""
    s = re.sub(r"```json\n?", "", s)
    s = re.sub(r"```\n?", "", s)
    s = re.sub(r",\s*}", "}", s)
    s = re.sub(r",\s*]", "]", s)
    return s.strip()


def extract_action_name(text: str) -> Optional[str]:
    match = re.search(r"Action:\s*([a-zA-Z_][a-zA-Z0-9_]*)", text)
    return match.group(1) if match else None


def extract_action_input(text: str) -> Optional[Dict[str, Any]]:
    """Extract and parse Action Input JSON or shorthand args from text."""
    match = re.search(r"Action\s+Input:\s*({.*?})", text, re.DOTALL)
    if match:
        json_str = match.group(1)
        brace_count = 0
        for i, char in enumerate(json_str):
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    json_str = json_str[: i + 1]
                    break
        json_str = sanitize_json_string(json_str)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(json_str)
            except Exception:
                return None

    shorthand = re.search(r"Action:\s*[a-zA-Z_][a-zA-Z0-9_]*\((.*?)\)", text, re.DOTALL)
    if shorthand:
        arg_text = shorthand.group(1).strip()
        if not arg_text:
            return {}
        parsed: Dict[str, Any] = {}
        for piece in re.split(r"\s*,\s*", arg_text):
            if "=" not in piece:
                continue
            key, value = piece.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if value.isdigit():
                parsed[key] = int(value)
            else:
                try:
                    parsed[key] = float(value)
                except ValueError:
                    parsed[key] = value
        return parsed

    return None


def extract_thought(text: str) -> Optional[str]:
    match = re.search(r"Thought:\s*(.+?)(?=\n|Action:|$)", text, re.DOTALL)
    return match.group(1).strip() if match else None


print("Robust parsing utilities defined")


# Tool Executor

class ToolExecutor:
    """Tool executor that performs real data operations on a sales CSV."""

    def __init__(self, csv_path: str = "./sales_data.csv"):
        try:
            self.base_df = pd.read_csv(csv_path)
            print(f"Loaded sales data from {csv_path}")
            print(f"Shape: {self.base_df.shape}, Columns: {list(self.base_df.columns)}")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Could not find sales_data.csv at: {csv_path}. Please update the csv_path parameter."
            )
        self.current_df = self.base_df.copy()
        self.execution_history: List[Dict[str, Any]] = []
        self.last_result: Optional[Dict[str, Any]] = None
        self.last_sort_column: Optional[str] = None

    def available_columns(self) -> List[str]:
        return list(self.base_df.columns)

    def is_grouped_view(self, group_col: Optional[str]) -> bool:
        if not group_col or group_col not in self.current_df.columns:
            return False
        return self.current_df[group_col].nunique(dropna=False) == len(self.current_df)

    def reset(self) -> Dict[str, Any]:
        self.current_df = self.base_df.copy()
        self.last_sort_column = None
        result = {"operation": "reset", "row_count": len(self.current_df), "status": "success"}
        self.last_result = result
        return result

    def filter_data(self, column: str = None, value: Any = None, **kwargs) -> Dict[str, Any]:
        try:
            if column is None or value is None:
                return {"error": "column and value required", "status": "failed"}
            if column not in self.current_df.columns:
                return {"error": f"Unknown column: {column}", "status": "failed"}
            self.current_df = self.current_df[self.current_df[column] == value].copy()
            result = {
                "operation": "filter",
                "parameters": {"column": column, "value": value},
                "row_count": len(self.current_df),
                "sample_rows": self.current_df.head(3).to_dict("records"),
                "status": "success",
            }
            self.execution_history.append(result)
            self.last_result = result
            return result
        except Exception as e:
            return {"error": str(e), "status": "failed", "operation": "filter_data"}

    def group_by(
        self,
        column: str = None,
        aggregate_column: str = None,
        aggregate_function: str = "sum",
        **kwargs,
    ) -> Dict[str, Any]:
        try:
            if column is None:
                return {"error": "column required", "status": "failed"}
            if column not in self.current_df.columns:
                return {"error": f"Unknown column: {column}", "status": "failed"}

            agg_fn = kwargs.get("aggregation", aggregate_function)
            if agg_fn in {"avg", "average"}:
                agg_fn = "mean"

            if aggregate_column:
                if aggregate_column not in self.current_df.columns:
                    return {"error": f"Unknown aggregate column: {aggregate_column}", "status": "failed"}
                if aggregate_column == column:
                    return {"error": "aggregate_column must be different from group column", "status": "failed", "operation": "group_by"}
                grouped = self.current_df.groupby(column)[aggregate_column].agg(agg_fn)
                grouped_df = grouped.reset_index(name=aggregate_column)
                self.current_df = grouped_df.copy()
                self.last_sort_column = aggregate_column
                result = {
                    "operation": "group_by",
                    "parameters": {"column": column, "aggregate_column": aggregate_column, "function": agg_fn},
                    "results": grouped.head(50).to_dict(),
                    "groups_count": len(grouped),
                    "status": "success",
                }
            else:
                grouped = self.current_df.groupby(column).size()
                result = {
                    "operation": "group_by",
                    "parameters": {"column": column},
                    "group_counts": grouped.to_dict(),
                    "groups_count": len(grouped),
                    "status": "success",
                }

            self.execution_history.append(result)
            self.last_result = result
            return result
        except Exception as e:
            return {"error": str(e), "status": "failed", "operation": "group_by"}

    def aggregate(self, column: str = None, aggregate_function: str = "sum", **kwargs) -> Dict[str, Any]:
        try:
            if column is None:
                return {"error": "column required", "status": "failed"}
            if column not in self.current_df.columns:
                return {"error": f"Unknown column: {column}", "status": "failed"}

            agg_fn = kwargs.get("aggregation", aggregate_function)
            if agg_fn in {"avg", "average"}:
                agg_fn = "mean"

            fn_map = {
                "sum": lambda df, col: df[col].sum(),
                "mean": lambda df, col: df[col].mean(),
                "count": lambda df, col: df[col].count(),
                "max": lambda df, col: df[col].max(),
                "min": lambda df, col: df[col].min(),
            }
            if agg_fn not in fn_map:
                return {"error": f"Unknown function: {agg_fn}", "status": "failed"}

            result_value = fn_map[agg_fn](self.current_df, column)
            result = {
                "operation": "aggregate",
                "parameters": {"column": column, "function": agg_fn},
                "result": float(result_value),
                "row_count": len(self.current_df),
                "status": "success",
            }
            self.execution_history.append(result)
            self.last_result = result
            return result
        except Exception as e:
            return {"error": str(e), "status": "failed", "operation": "aggregate"}

    def sort_by(self, column: str = None, order: str = "asc", **kwargs) -> Dict[str, Any]:
        try:
            if column is None:
                return {"error": "column required", "status": "failed"}
            if column not in self.current_df.columns:
                return {"error": f"Unknown column: {column}", "status": "failed"}
            ascending = order.lower() != "desc"
            self.current_df = self.current_df.sort_values(by=column, ascending=ascending).copy()
            self.last_sort_column = column
            result = {
                "operation": "sort_by",
                "parameters": {"column": column, "order": order},
                "sample_rows": self.current_df.head(5).to_dict("records"),
                "sorted_count": len(self.current_df),
                "status": "success",
            }
            self.execution_history.append(result)
            self.last_result = result
            return result
        except Exception as e:
            return {"error": str(e), "status": "failed", "operation": "sort_by"}

    def topk(self, column: str = None, k: int = 5, order: str = "desc", **kwargs) -> Dict[str, Any]:
        try:
            if column is None:
                column = kwargs.get("sort_by") or self.last_sort_column
            if column is None:
                return {"error": "column required", "status": "failed"}
            if column not in self.current_df.columns:
                return {"error": f"Unknown column: {column}", "status": "failed"}
            try:
                k = int(k)
            except Exception:
                k = 5
            k = max(1, k)
            ascending = order.lower() != "desc"
            self.current_df = self.current_df.nsmallest(k, column) if ascending else self.current_df.nlargest(k, column)
            result = {
                "operation": "topk",
                "parameters": {"column": column, "k": k, "order": order},
                "results": self.current_df.to_dict("records"),
                "count": len(self.current_df),
                "status": "success",
            }
            self.execution_history.append(result)
            self.last_result = result
            return result
        except Exception as e:
            return {"error": str(e), "status": "failed", "operation": "topk"}

    def top_k(self, **kwargs) -> Dict[str, Any]:
        return self.topk(**kwargs)

    def execute(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        alias_map = {
            "topk": ("topk", {}),
            "top_k": ("topk", {}),
            "top-k": ("topk", {}),
            "aggregate_sum": ("aggregate", {"aggregation": "sum"}),
            "aggregate_mean": ("aggregate", {"aggregation": "mean"}),
            "aggregate_avg": ("aggregate", {"aggregation": "mean"}),
            "aggregate_count": ("aggregate", {"aggregation": "count"}),
            "aggregate_max": ("aggregate", {"aggregation": "max"}),
            "aggregate_min": ("aggregate", {"aggregation": "min"}),
        }

        tool_name = (tool_name or "").strip().lower()
        mapped = alias_map.get(tool_name)
        if mapped is not None:
            tool_name = mapped[0]
            merged_kwargs = mapped[1].copy()
            merged_kwargs.update(kwargs)
            kwargs = merged_kwargs

        try:
            if hasattr(self, tool_name):
                return getattr(self, tool_name)(**kwargs)
            return {"error": f"Unknown tool: {tool_name}", "status": "failed"}
        except Exception as e:
            return {"error": str(e), "status": "failed", "tool": tool_name}


print("Tool executor defined with real CSV data")


# ReAct Execution Engine

class ReActExecutionEngine:
    """Main execution engine for the ReAct paradigm."""

    def __init__(
        self,
        model,
        tokenizer,
        tool_executor: ToolExecutor,
        max_turns: int = 5,
        max_observation_length: int = 500,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.tool_executor = tool_executor
        self.max_turns = max_turns
        self.max_observation_length = max_observation_length
        self.conversation_history: List[Dict[str, Any]] = []
        self.error_count = 0
        self.max_errors = 3

    def truncate_observation(self, observation: str) -> str:
        if len(observation) > self.max_observation_length:
            return observation[: self.max_observation_length] + "...[truncated]"
        return observation

    def format_system_prompt(self) -> str:
        return """You are an AI agent that solves data analysis tasks using available tools.

Available tools:
- filter_data
- group_by
- aggregate
- sort_by
- topk

Rules:
- Use tools only when needed.
- Use prior observations to decide the next action.
- Emit Final Answer only after enough evidence is available.

Response format:
Thought: [reasoning]
Action: [tool_name]
Action Input: {"param": value}

When done:
Final Answer: [natural-language answer]"""

    def _normalize_action_name(self, action_name: Optional[str]) -> Optional[str]:
        if not action_name:
            return None
        raw = action_name.strip()
        lowered = raw.lower().replace("-", "_").replace(" ", "_")
        if lowered in {"final_answer", "finalanswer"}:
            return "FINAL_ANSWER"
        alias_names = {
            "topk": "topk",
            "top_k": "topk",
            "aggregate_sum": "aggregate_sum",
            "aggregate_mean": "aggregate_mean",
            "aggregate_avg": "aggregate_avg",
            "aggregate_count": "aggregate_count",
            "aggregate_max": "aggregate_max",
            "aggregate_min": "aggregate_min",
            "group_by": "group_by",
            "filter_data": "filter_data",
            "sort_by": "sort_by",
            "aggregate": "aggregate",
        }
        return alias_names.get(lowered, lowered)

    def _normalize_token(self, token: Optional[str]) -> Optional[str]:
        if not token:
            return None
        t = token.strip().lower()
        if t.endswith("ies"):
            return t[:-3] + "y"
        if t.endswith("s") and not t.endswith("ss"):
            return t[:-1]
        return t

    def _column_by_token(self, token: Optional[str]) -> Optional[str]:
        if not token:
            return None
        norm = self._normalize_token(token)
        for c in self.tool_executor.available_columns():
            if c.lower() == norm:
                return c
        return None

    def _infer_metric_column(self, user_query: str) -> Optional[str]:
        q = user_query.lower()
        top_by = re.search(r"\btop\s+\d+\s+([a-zA-Z_]+)\s+by\s+([a-zA-Z_]+)", q)
        if top_by:
            col = self._column_by_token(top_by.group(2))
            if col:
                return col
        by_match = re.search(r"\bby\s+([a-zA-Z_]+)", q)
        if by_match:
            candidate = self._column_by_token(by_match.group(1))
            if candidate and candidate.lower() in {"revenue", "profit", "cost", "units_sold"}:
                return candidate
        for c in self.tool_executor.available_columns():
            if c.lower() in q:
                return c
        metric_hints = ["revenue", "profit", "cost", "units_sold", "units", "sales", "amount"]
        for hint in metric_hints:
            if hint in q:
                for c in self.tool_executor.available_columns():
                    lc = c.lower()
                    if hint == lc or hint in lc or lc in hint:
                        return c
        return None

    def _infer_group_column(self, user_query: str) -> Optional[str]:
        q = user_query.lower()
        top_by = re.search(r"\btop\s+\d+\s+([a-zA-Z_]+)\s+by\s+([a-zA-Z_]+)", q)
        if top_by:
            col = self._column_by_token(top_by.group(1))
            if col:
                return col
        by_match = re.search(r"\bby\s+([a-zA-Z_]+)", q)
        if by_match:
            col = self._column_by_token(by_match.group(1))
            if col:
                return col
        for c in self.tool_executor.available_columns():
            if c.lower() in {"city", "region", "category", "product", "year", "month"} and c.lower() in q:
                return c
        return None

    def _extract_requested_k(self, user_query: str, default_k: int = 5) -> int:
        m = re.search(r"\btop\s+(\d+)\b", user_query.lower())
        return max(1, int(m.group(1))) if m else default_k

    def _infer_agg_function(self, user_query: str, default_fn: str = "sum") -> str:
        q = user_query.lower()
        if "average" in q or "mean" in q or "avg" in q:
            return "mean"
        if "count" in q:
            return "count"
        if "maximum" in q or "max" in q:
            return "max"
        if "minimum" in q or "min" in q:
            return "min"
        if "total" in q or "sum" in q:
            return "sum"
        return default_fn

    def _is_aggregate_query(self, user_query: str) -> bool:
        q = user_query.lower()
        return any(k in q for k in ["total", "sum", "average", "mean", "count", "max", "min"])

    def _is_topk_query(self, user_query: str) -> bool:
        q = user_query.lower()
        return any(k in q for k in ["top", "highest", "lowest", "rank", "best", "worst"])

    def _is_grouped_aggregate_query(self, user_query: str) -> bool:
        return self._is_aggregate_query(user_query) and (" by " in user_query.lower())

    def _repair_action(
        self,
        user_query: str,
        action_name: str,
        action_input: Optional[Dict[str, Any]],
    ) -> Tuple[str, Dict[str, Any]]:
        """Repair incomplete/misaligned action arguments with intent constraints."""
        action_name = self._normalize_action_name(action_name)
        action_input = action_input or {}
        if action_name == "FINAL_ANSWER":
            return action_name, action_input

        by_col = self._infer_group_column(user_query)
        metric_col = self._infer_metric_column(user_query)
        agg_fn = self._infer_agg_function(user_query)
        last_turn = self.conversation_history[-1] if self.conversation_history else None

        alias_names = {
            "aggregate_sum": ("aggregate", "sum"),
            "aggregate_mean": ("aggregate", "mean"),
            "aggregate_avg": ("aggregate", "mean"),
            "aggregate_count": ("aggregate", "count"),
            "aggregate_max": ("aggregate", "max"),
            "aggregate_min": ("aggregate", "min"),
        }
        if action_name in alias_names:
            action_name, inferred_fn = alias_names[action_name]
            action_input.setdefault("aggregation", inferred_fn)

        if action_name == "group_by" and self._is_aggregate_query(user_query) and not self._is_grouped_aggregate_query(user_query):
            return "aggregate", {"column": metric_col, "aggregation": agg_fn}

        if action_name == "filter_data":
            if "column" in action_input and "value" not in action_input and by_col and action_input.get("column") == by_col:
                return "group_by", {"column": by_col}
            if self._is_topk_query(user_query) and by_col and metric_col and last_turn:
                same_filter = (
                    last_turn.get("action") == "filter_data"
                    and last_turn.get("action_input", {}).get("column") == action_input.get("column")
                    and last_turn.get("action_input", {}).get("value") == action_input.get("value")
                )
                if same_filter:
                    if not self.tool_executor.is_grouped_view(by_col):
                        return "group_by", {"column": by_col, "aggregate_column": metric_col, "aggregate_function": "sum"}
                    return "topk", {"column": metric_col, "k": self._extract_requested_k(user_query, default_k=5), "order": "desc"}

        if action_name == "group_by":
            if by_col:
                action_input["column"] = by_col
            if self._is_grouped_aggregate_query(user_query):
                action_input.setdefault("aggregate_column", metric_col)
                action_input.setdefault("aggregate_function", agg_fn)
            if self._is_topk_query(user_query) and by_col and metric_col:
                action_input.setdefault("aggregate_column", metric_col)
                action_input.setdefault("aggregate_function", "sum")

        if action_name == "aggregate":
            action_input.setdefault("column", metric_col)
            action_input.setdefault("aggregation", agg_fn)
            if self._is_grouped_aggregate_query(user_query) and by_col:
                return "group_by", {"column": by_col, "aggregate_column": action_input.get("column", metric_col), "aggregate_function": action_input.get("aggregation", agg_fn)}
            if self._is_topk_query(user_query) and by_col and metric_col:
                if self.tool_executor.is_grouped_view(by_col):
                    return "topk", {"column": action_input.get("column", metric_col), "k": self._extract_requested_k(user_query, default_k=5), "order": "desc"}
                return "group_by", {"column": by_col, "aggregate_column": metric_col, "aggregate_function": "sum"}

        if action_name in {"topk", "top_k"}:
            action_name = "topk"
            action_input.setdefault("k", self._extract_requested_k(user_query, default_k=5))
            action_input.setdefault("order", "desc")
            action_input.setdefault("column", metric_col or getattr(self.tool_executor, "last_sort_column", None))
            if self._is_topk_query(user_query) and by_col and metric_col:
                if not self.tool_executor.is_grouped_view(by_col):
                    return "group_by", {"column": by_col, "aggregate_column": metric_col, "aggregate_function": "sum"}

        if action_name == "sort_by":
            action_input.setdefault("column", metric_col)
            action_input.setdefault("order", "desc")
            if self._is_topk_query(user_query) and by_col and metric_col:
                if not self.tool_executor.is_grouped_view(by_col):
                    return "group_by", {"column": by_col, "aggregate_column": metric_col, "aggregate_function": "sum"}

        action_input = {k: v for k, v in action_input.items() if v is not None}
        return action_name, action_input

    def _format_aggregate_answer(self, aggregate_result: Dict[str, Any]) -> str:
        value = aggregate_result.get("result")
        params = aggregate_result.get("parameters", {})
        column = params.get("column", "value")
        fn = params.get("function", "sum")
        value_text = f"{value:,.2f}" if isinstance(value, float) else str(value)
        return f"Based on the analysis, the {fn} of {column} is {value_text}."

    def _format_topk_answer(self, user_query: str, topk_result: Dict[str, Any]) -> str:
        rows = topk_result.get("results", [])
        params = topk_result.get("parameters", {})
        metric_col = params.get("column", self._infer_metric_column(user_query) or "value")
        requested_k = self._extract_requested_k(user_query, default_k=params.get("k", 5))

        if not rows:
            return "I could not find matching results for this query."

        if isinstance(rows, dict):
            group_col = params.get("column", "group")
            items = sorted(rows.items(), key=lambda x: x[1], reverse=True)
            rows = [{group_col: k, metric_col: v} for k, v in items]

        rows = rows[:requested_k]
        lines = []
        for i, row in enumerate(rows, 1):
            entity = (
                row.get("city") or row.get("region") or row.get("product")
                or row.get("category") or row.get("year") or row.get("month")
            )
            metric = row.get(metric_col, "N/A")
            if isinstance(metric, float):
                metric = f"{metric:,.2f}"
            lines.append(f"{i}. {entity} ({metric_col}: {metric})" if entity else f"{i}. {metric_col}: {metric}")
        return f"Top {requested_k} results by {metric_col}:\n" + "\n".join(lines)

    def _format_groupby_answer(self, groupby_result: Dict[str, Any]) -> str:
        params = groupby_result.get("parameters", {})
        group_col = params.get("column", "group")

        if "results" in groupby_result:
            items = sorted(groupby_result["results"].items(), key=lambda x: x[1], reverse=True)
            lines = []
            for k, v in items[:10]:
                if isinstance(v, float):
                    v = f"{v:,.2f}"
                lines.append(f"- {k}: {v}")
            return f"Grouped summary by {group_col}:\n" + "\n".join(lines)

        if "group_counts" in groupby_result:
            items = sorted(groupby_result["group_counts"].items(), key=lambda x: x[1], reverse=True)
            lines = [f"- {k}: {v} rows" for k, v in items[:10]]
            return f"Row distribution by {group_col}:\n" + "\n".join(lines)

        return "I could not find grouped results to summarize."

    def _summarize_observation(self, result: Dict[str, Any]) -> str:
        if not isinstance(result, dict):
            return "Tool returned an unstructured response."
        if result.get("status") == "failed":
            op = result.get("operation") or result.get("tool") or "tool"
            return f"{op} failed: {result.get('error', 'unknown error')}."
        op = result.get("operation", "tool")
        if op == "filter":
            p = result.get("parameters", {})
            return f"Filtered {p.get('column')} = {p.get('value')}; {result.get('row_count')} rows remain."
        if op == "aggregate":
            p = result.get("parameters", {})
            val = result.get("result")
            if isinstance(val, float):
                val = f"{val:,.2f}"
            return f"{p.get('function')}({p.get('column')}) = {val}."
        if op == "group_by":
            p = result.get("parameters", {})
            if "aggregate_column" in p:
                return f"Grouped by {p.get('column')} and computed {p.get('function')}({p.get('aggregate_column')})."
            return f"Grouped by {p.get('column')} into {result.get('groups_count')} groups."
        if op == "sort_by":
            p = result.get("parameters", {})
            return f"Sorted by {p.get('column')} in {p.get('order')} order."
        if op == "topk":
            p = result.get("parameters", {})
            return f"Selected top {p.get('k')} by {p.get('column')}."
        return f"{op} completed successfully."

    def _format_final_answer_from_result(self, user_query: str, result: Dict[str, Any]) -> Optional[str]:
        if not isinstance(result, dict) or result.get("status") != "success":
            return None
        op = result.get("operation")
        if op == "aggregate":
            return None if self._is_topk_query(user_query) else self._format_aggregate_answer(result)
        if op == "group_by":
            if self._is_topk_query(user_query) and "results" in result:
                params = result.get("parameters", {})
                group_col = params.get("column", "group")
                metric_col = params.get("aggregate_column", self._infer_metric_column(user_query) or "value")
                grouped_items = sorted(result["results"].items(), key=lambda x: x[1], reverse=True)
                rows = [{group_col: k, metric_col: v} for k, v in grouped_items]
                synthetic = {
                    "operation": "topk",
                    "parameters": {"column": metric_col, "k": self._extract_requested_k(user_query, default_k=5)},
                    "results": rows,
                    "status": "success",
                }
                return self._format_topk_answer(user_query, synthetic)
            return self._format_groupby_answer(result)
        if op == "topk":
            return self._format_topk_answer(user_query, result)
        if op == "sort_by":
            sample_rows = result.get("sample_rows", [])
            synthetic = {
                "operation": "topk",
                "parameters": {"column": result.get("parameters", {}).get("column", self._infer_metric_column(user_query) or "value"), "k": self._extract_requested_k(user_query, default_k=5)},
                "results": sample_rows,
                "status": "success",
            }
            return self._format_topk_answer(user_query, synthetic)
        return None

    def generate_action(self, user_input: str, system_prompt: str) -> Tuple[str, Optional[str], Optional[Dict]]:
        full_prompt = f"""{system_prompt}\n\nUser Query: {user_input}\n\nThought:"""
        for turn in self.conversation_history:
            full_prompt += (
                f"\n{turn['thought']}\nAction: {turn['action']}\n"
                f"Action Input: {json.dumps(turn['action_input'])}\n"
                f"Observation: {turn['observation']}\n\nThought:"
            )

        inputs = self.tokenizer(full_prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        stopping_criteria = StoppingCriteriaList(
            [StopOnObservationCriteria(self.tokenizer), StopOnFinalAnswerCriteria(self.tokenizer)]
        )

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=160,
                do_sample=False,
                use_cache=True,
                stopping_criteria=stopping_criteria,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        if "Final Answer:" in generated_text:
            final_answer = generated_text.split("Final Answer:", 1)[1].strip()
            return generated_text, "FINAL_ANSWER", {"answer": final_answer}

        action_name = extract_action_name(generated_text)
        action_input = extract_action_input(generated_text)
        action_name = self._normalize_action_name(action_name)
        if action_name == "FINAL_ANSWER":
            return generated_text, "FINAL_ANSWER", action_input or {"answer": ""}
        return generated_text, action_name, action_input

    def execute_action(self, action_name: str, action_input: Dict) -> Dict[str, Any]:
        if not action_name:
            result = {"status": "failed", "error": "Invalid action format", "tool": "unknown"}
            return {"observation": self._summarize_observation(result), "result": result}
        if action_input is None:
            result = {"status": "failed", "error": "Invalid action input format", "tool": action_name}
            return {"observation": self._summarize_observation(result), "result": result}
        try:
            result = self.tool_executor.execute(action_name, **action_input)
            observation_text = self.truncate_observation(self._summarize_observation(result))
            return {"observation": observation_text, "result": result}
        except Exception as e:
            result = {"status": "failed", "error": str(e), "tool": action_name}
            return {"observation": self._summarize_observation(result), "result": result}

    def run(self, user_query: str) -> Dict[str, Any]:
        self.conversation_history = []
        self.error_count = 0
        if hasattr(self.tool_executor, "reset"):
            self.tool_executor.reset()

        system_prompt = self.format_system_prompt()

        for turn_number in range(self.max_turns):
            print(f"\n--- Turn {turn_number + 1}/{self.max_turns} ---")
            generated_text, action_name, action_input = self.generate_action(user_query, system_prompt)
            print(f"Generated: {generated_text[:200]}...")

            if action_name == "FINAL_ANSWER":
                last_result = getattr(self.tool_executor, "last_result", None)
                formatted = self._format_final_answer_from_result(user_query, last_result)
                if formatted:
                    print(f"Final Answer (formatted): {formatted}")
                    return {"status": "success", "final_answer": formatted, "turns": len(self.conversation_history), "actions": [h["action"] for h in self.conversation_history]}

            if not action_name:
                self.error_count += 1
                print("Failed to parse action. Retrying.")
                if self.error_count >= self.max_errors:
                    break
                continue

            action_name, action_input = self._repair_action(user_query, action_name, action_input)
            if action_name == "FINAL_ANSWER":
                last_result = getattr(self.tool_executor, "last_result", None)
                formatted = self._format_final_answer_from_result(user_query, last_result)
                if formatted:
                    return {"status": "success", "final_answer": formatted, "turns": len(self.conversation_history), "actions": [h["action"] for h in self.conversation_history]}
                continue

            execution = self.execute_action(action_name, action_input)
            observation = execution["observation"]
            raw_result = execution["result"]

            print(f"Action: {action_name}")
            print(f"Observation: {observation[:200]}...")

            self.conversation_history.append({
                "thought": extract_thought(generated_text) or "[thought]",
                "action": action_name,
                "action_input": action_input or {},
                "observation": observation,
                "raw_result": raw_result,
            })

            last_result = getattr(self.tool_executor, "last_result", None)
            if isinstance(last_result, dict) and last_result.get("status") == "success":
                op = last_result.get("operation")

                if self._is_grouped_aggregate_query(user_query):
                    if op == "group_by" and "aggregate_column" in last_result.get("parameters", {}):
                        final_answer = self._format_groupby_answer(last_result)
                        print(f"Final Answer (auto): {final_answer}")
                        return {"status": "success", "final_answer": final_answer, "turns": len(self.conversation_history), "actions": [h["action"] for h in self.conversation_history]}
                elif self._is_aggregate_query(user_query):
                    if op == "aggregate":
                        final_answer = self._format_aggregate_answer(last_result)
                        print(f"Final Answer (auto): {final_answer}")
                        return {"status": "success", "final_answer": final_answer, "turns": len(self.conversation_history), "actions": [h["action"] for h in self.conversation_history]}

                if self._is_topk_query(user_query) and op in {"topk", "group_by"}:
                    final_answer = (
                        self._format_final_answer_from_result(user_query, last_result)
                        if op == "group_by"
                        else self._format_topk_answer(user_query, last_result)
                    )
                    print(f"Final Answer (auto): {final_answer}")
                    return {"status": "success", "final_answer": final_answer, "turns": len(self.conversation_history), "actions": [h["action"] for h in self.conversation_history]}

        last_result = getattr(self.tool_executor, "last_result", None)
        fallback = self._format_final_answer_from_result(user_query, last_result)
        if fallback:
            return {"status": "success", "final_answer": fallback, "turns": len(self.conversation_history), "actions": [h["action"] for h in self.conversation_history]}

        return {
            "status": "completed",
            "final_answer": None,
            "turns": len(self.conversation_history),
            "actions": [h["action"] for h in self.conversation_history],
            "error": "Max turns reached without final answer",
        }


print("ReAct Execution Engine defined")

# Load Fine-Tuned Model & Run Inference

def load_finetuned_model(hf_adapter_repo: str):
    """Load Qwen base model + fine-tuned LoRA adapter from Hugging Face Hub."""
    print(f"Loading base model: {MODEL_NAME}")

    supports_bf16_local = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype_local = torch.bfloat16 if supports_bf16_local else torch.float16

    world_size_local = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank_local = int(os.environ.get("LOCAL_RANK", "0"))
    distributed_local = world_size_local > 1
    device_map_local = {"": local_rank_local} if distributed_local else "auto"

    bnb_config_local = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype_local,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config_local,
        torch_dtype=compute_dtype_local,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        device_map=device_map_local,
        trust_remote_code=True,
    )

    print(f"Loading adapter from: {hf_adapter_repo}")
    ft_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if ft_tokenizer.pad_token is None:
        ft_tokenizer.pad_token = ft_tokenizer.eos_token
    ft_tokenizer.padding_side = "left"

    ft_model = PeftModel.from_pretrained(base_model, hf_adapter_repo, device_map=device_map_local)
    ft_model.eval()
    ft_model.config.use_cache = True
    print("Adapter loaded successfully from Hugging Face Hub")

    return ft_model, ft_tokenizer


if RUN_INFERENCE:
    print("Loading trained model from Hugging Face Hub...\n")
    default_infer_repo = HF_REPO_DPO if RUN_DPO else HF_REPO_SFT
    HF_ADAPTER_REPO = os.environ.get("HF_ADAPTER_REPO", default_infer_repo)
    print(f"Using HF repo: {HF_ADAPTER_REPO}\n")

    try:
        ft_model, ft_tokenizer = load_finetuned_model(HF_ADAPTER_REPO)

        CSV_PATH = "./sales_data.csv"
        tool_executor = ToolExecutor(csv_path=CSV_PATH)

        engine = ReActExecutionEngine(
            model=ft_model,
            tokenizer=ft_tokenizer,
            tool_executor=tool_executor,
            max_turns=5,
            max_observation_length=600,
        )
        print("\nExecution engine initialized and ready")

    except Exception as e:
        print(f"Error loading model: {e}")
        print("Continuing with demonstration...")
        engine = None


    # Test Queries

    test_queries = [
        "What is the total revenue in 2022?",
        "What is the average profit by region?",
        "Top 3 cities by revenue in 2022",
    ]

    all_results: List[Dict[str, Any]] = []

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = "./phase2_inference_results.json"
    log_file = f"./phase2_execution_log_{timestamp}.txt"

    if engine is not None:
        log_buffer = io.StringIO()

        with redirect_stdout(log_buffer):
            print("Testing ReAct Execution Engine\n")
            print("=" * 60)
            print(f"Run timestamp: {timestamp}")
            print("=" * 60)

            for i, query in enumerate(test_queries):
                print(f"\n\nQuery {i+1}: {query}")
                print("-" * 60)

                result = engine.run(query)
                result["query"] = query
                result["turn_history"] = engine.conversation_history
                all_results.append(result)

                print("\n" + "=" * 60)
                print("\nFinal Result:")
                print(json.dumps(result, indent=2))

            print("\n" + "=" * 60)
            print("Run complete")

        with open(log_file, "w") as f:
            f.write(log_buffer.getvalue())

        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2)

        print(log_buffer.getvalue())
        print(f"\nStructured results saved to {output_file}")
        print(f"Full execution log saved to {log_file}")
        print("  Download these files to view complete trajectories.")
    else:
        print("\nExecution engine not initialized.")
        print("Please ensure the model loading step completed successfully.")
else:
    print("Inference stage skipped: ReAct execution is disabled.")