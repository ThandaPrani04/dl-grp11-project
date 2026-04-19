---
title: Qwen Tool Planner Demo
emoji: ""
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# Qwen Tool Planner Demo

This Space serves your LoRA adapter trained for query-to-action planning.

## What it does

- Single query inference: Natural language query -> JSON with `actions` and `answer`.
- Test-set evaluation: Runs over `test_trajectory_2k.json` and reports exact action-match accuracy.

## Expected repository contents

- `app.py`
- `requirements.txt`
- `test_trajectory_2k.json`
- `training_runs/<run_name>/adapter/*`

The app auto-detects the most recently modified adapter under:

`training_runs/*/adapter/adapter_model.safetensors`

## Recommended Space hardware

- GPU is strongly recommended (T4 or better).
- CPU-only can work but will be much slower.
