import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from tool_executor import ToolExecutor


SYSTEM_PROMPT = (
    "You are a data-analysis tool-planning agent. "
    "Return ONLY valid JSON with keys: actions, answer. "
    "actions must be a list of tool calls as strings, and answer must be null."
)


def extract_first_json_object(text):
    """
    Extract the first valid JSON object from noisy model output.
    Useful when the model emits extra text before/after JSON.
    """
    decoder = json.JSONDecoder()
    i = 0
    while i < len(text):
        if text[i] != "{":
            i += 1
            continue
        try:
            obj, _ = decoder.raw_decode(text[i:])
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        i += 1
    raise ValueError("No valid JSON object found in model output.")


def _get_base_model_name(adapter_dir):
    cfg_path = Path(adapter_dir) / "adapter_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"adapter_config.json not found in {adapter_dir}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    base_name = cfg.get("base_model_name_or_path")
    if not base_name:
        raise ValueError("adapter_config.json missing 'base_model_name_or_path'")
    return base_name


def load_planner_model(adapter_dir):
    """
    Load base model + LoRA adapter similar to app.py.
    Returns: (model, tokenizer, device)
    """
    adapter_dir = Path(adapter_dir)
    base_model_name = _get_base_model_name(adapter_dir)

    tokenizer = None
    # Try base model tokenizer first, then adapter tokenizer.
    for source in (base_model_name, str(adapter_dir)):
        for use_fast in (True, False):
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    source,
                    use_fast=use_fast,
                    trust_remote_code=True,
                )
                break
            except Exception:
                continue
        if tokenizer is not None:
            break

    if tokenizer is None:
        raise RuntimeError("Tokenizer could not be loaded from base model or adapter directory.")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, str(adapter_dir), is_trainable=False)
        device = torch.device("cuda:0")
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        model = PeftModel.from_pretrained(base_model, str(adapter_dir), is_trainable=False)
        device = torch.device("cpu")

    model.eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model.config.use_cache = True

    return model, tokenizer, device


def generate_model_output(model, tokenizer, device, query, max_new_tokens=220):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Query: {query}\nOutput JSON:"},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(device)

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen_only = out_ids[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gen_only, skip_special_tokens=True).strip()


def parse_model_output(model_text):
    """
    Parse model text and return JSON object with actions.
    """
    obj = extract_first_json_object(model_text)
    actions = obj.get("actions", [])
    if not isinstance(actions, list) or not all(isinstance(a, str) for a in actions):
        raise ValueError("Parsed JSON must contain 'actions' as list[str].")
    return obj

def parse_agent_action(action_str):
    """
    Parses a string-based agent action into the dictionary format
    required by the ToolExecutor.
    """
    if action_str.startswith("filter_data"):
        # Supports values like 2022 or 'Bengaluru'.
        match = re.search(r"column='([^']+)'\s*,\s*value=(.+)\)$", action_str)
        if match:
            raw_value = match.group(2)
            try:
                value = ast.literal_eval(raw_value)
            except Exception:
                value = raw_value.strip("'")
            return {
                "tool": "filter", 
                "args": {"column": match.group(1), "op": "==", "value": value}
            }

    elif action_str.startswith("group_by"):
        match = re.search(r"column='([^']+)'", action_str)
        if match:
            return {"tool": "groupby", "args": {"column": match.group(1)}}

    elif action_str.startswith("aggregate_sum"):
        match = re.search(r"column='([^']+)'", action_str)
        if match:
            return {"tool": "aggregate", "args": {"column": match.group(1), "agg": "sum"}}

    elif action_str.startswith("aggregate_mean"):
        match = re.search(r"column='([^']+)'", action_str)
        if match:
            return {"tool": "aggregate", "args": {"column": match.group(1), "agg": "mean"}}

    elif action_str.startswith("aggregate_count"):
        match = re.search(r"column='([^']+)'", action_str)
        if match:
            return {"tool": "aggregate", "args": {"column": match.group(1), "agg": "count"}}

    elif action_str.startswith("sort_by"):
        match = re.search(r"column='([^']+)', order='([^']+)'", action_str)
        if match:
            # Convert 'desc' to ascending=False, otherwise True
            is_ascending = False if match.group(2) == 'desc' else True
            return {"tool": "sort", "args": {"column": match.group(1), "ascending": is_ascending}}

    elif action_str.startswith("top_k"):
        match = re.search(r"k=(\d+)", action_str)
        if match:
            return {"tool": "topk", "args": {"k": int(match.group(1))}}

    raise ValueError(f"Could not parse action: {action_str}")


def execute_actions_from_list(df, raw_actions):
    """
    Convert string actions to executor actions, run ToolExecutor,
    and return a JSON-serializable answer.
    """
    parsed_actions = [parse_agent_action(a) for a in raw_actions]

    # Validate referenced columns before execution for clear errors.
    valid_cols = set(df.columns.astype(str).tolist())
    invalid_cols = []
    for step in parsed_actions:
        col = step.get("args", {}).get("column")
        if col is not None and col not in valid_cols:
            invalid_cols.append(col)
    if invalid_cols:
        raise ValueError(
            f"Unsupported columns in actions: {sorted(set(invalid_cols))}. "
            f"Available columns: {sorted(valid_cols)}"
        )

    executor = ToolExecutor(df)
    result_df = executor.execute(parsed_actions)

    if isinstance(result_df, pd.DataFrame):
        if len(result_df) == 1 and len(result_df.columns) == 1:
            return result_df.iloc[0, 0]
        return result_df.to_dict(orient="records")
    return result_df


def run_from_model_json(csv_path, model_obj):
    """
    model_obj format:
    {
      "actions": ["filter_data(...)", ...],
      "answer": null
    }
    Returns:
    {
      "actions": [...],
      "answer": <computed value>
    }
    """
    if not isinstance(model_obj, dict):
        raise ValueError("model_obj must be a dict.")

    raw_actions = model_obj.get("actions", [])
    if not isinstance(raw_actions, list) or not all(isinstance(a, str) for a in raw_actions):
        raise ValueError("model_obj['actions'] must be a list of strings.")

    df = pd.read_csv(csv_path)
    answer = execute_actions_from_list(df, raw_actions)
    return {"actions": raw_actions, "answer": answer}


def run_from_model_text(csv_path, model_text):
    """
    Parse first JSON object from raw model text and execute its actions.
    """
    model_obj = extract_first_json_object(model_text)
    return run_from_model_json(csv_path, model_obj)


def run_query_with_model(adapter_dir, csv_path, query, max_new_tokens=220):
        """
        End-to-end single-query run:
        1) load model+adapter
        2) generate output text
        3) parse JSON actions
        4) execute actions on csv

        Returns:
        {
            "raw_output": <str>,
            "parsed_output": <dict>,
            "final_output": {"actions": [...], "answer": ...}
        }
        """
        model, tokenizer, device = load_planner_model(adapter_dir)
        raw_text = generate_model_output(model, tokenizer, device, query, max_new_tokens=max_new_tokens)
        parsed_obj = parse_model_output(raw_text)
        final_obj = run_from_model_json(csv_path, parsed_obj)

        return {
                "raw_output": raw_text,
                "parsed_output": parsed_obj,
                "final_output": final_obj,
        }

def run_automated_pipeline(csv_path, json_path, output_path="pipeline_results.txt"):
    # 1. Load the data
    df = pd.read_csv(csv_path)

    # 2. Load the trajectories (queries and string actions)
    with open(json_path, 'r') as file:
        trajectories = json.load(file)

    lines = []

    # 3. Iterate through each query and its associated actions
    for entry in trajectories:
        query = entry["query"]
        raw_actions = entry["actions"]

        lines.append(f"Query: {query}")
        lines.append("-" * 40)

        # 4. Parse strings into dictionary format
        # 5. Execute actions and collect answer
        try:
            answer = execute_actions_from_list(df, raw_actions)
            lines.append("Answer:")
            if isinstance(answer, list):
                lines.append(pd.DataFrame(answer).to_string(index=False))
            else:
                lines.append(str(answer))
        except Exception as e:
            lines.append(f"Error: {e}")

        lines.append("")

    # 6. Write all results to a text file
    with open(output_path, 'w') as out_file:
        out_file.write("\n".join(lines))

    print(f"Results written to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["batch", "single"], default="single")
    parser.add_argument("--csv_path", type=str, default="sales_data.csv")
    parser.add_argument("--json_path", type=str, default="sample_trajectories.json")
    parser.add_argument("--output_path", type=str, default="pipeline_results.txt")
    parser.add_argument("--adapter_dir", type=str, default=None)
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=220)
    args = parser.parse_args()

    if args.mode == "batch":
        run_automated_pipeline(args.csv_path, args.json_path, args.output_path)
    else:
        if not args.adapter_dir or not args.query:
            raise ValueError("For --mode single, provide both --adapter_dir and --query")
        result = run_query_with_model(
            adapter_dir=args.adapter_dir,
            csv_path=args.csv_path,
            query=args.query,
            max_new_tokens=args.max_new_tokens,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))