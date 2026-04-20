# DL Project - Group 11

This repository contains our two-phase Deep Learning course project on building an autonomous AI agent for tool use.

## Phase 1 [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/thandaprani/dl-query-test-phase1)


Phase 1 builds a data-analysis agent over sales CSV data. The agent interprets natural-language queries, chooses and executes predefined tools (filter/group/aggregate/sort/top-k/plot), and returns a strict JSON output with `actions` and `answer`.

Trained model: [souhhmm/phase-1-qwen2_5-1.5b](https://huggingface.co/souhhmm/phase-1-qwen2_5-1.5b)

Testing notebook: [dl-query-test-phase1](https://www.kaggle.com/code/thandaprani/dl-query-test-phase1)
## Phase 2 [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/sohamkalburgi/dl-query-test-phase-2)

Phase 2 focuses on ToolAlpaca-based instruction tuning and an online ReAct execution loop:

- SFT for a 7B model to generate strict `Thought -> Action -> Action Input` tool-call format.
- A pure-Python bounded ReAct loop with stopping criteria, robust parsing, observation truncation, error recovery, and `Final Answer` termination.
- Alignment with DPO to improve reliability.

Trained models:

- [garam-icecream/phase2-toolalpaca-sft-phi2](https://huggingface.co/garam-icecream/phase2-toolalpaca-sft-phi2)
- [souhhmm/phase2-toolalpaca-dpo-qwen2_5-7b](https://huggingface.co/souhhmm/phase2-toolalpaca-dpo-qwen2_5-7b)
- [souhhmm/phase2-toolalpaca-sft-qwen2_5-7b](https://huggingface.co/souhhmm/phase2-toolalpaca-sft-qwen2_5-7b)

Testing notebook: [dl-query-test-phase-2](https://www.kaggle.com/code/sohamkalburgi/dl-query-test-phase-2)
