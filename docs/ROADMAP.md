# Pipis 6.7 roadmap

## Phase 0 — Foundation
- [x] Create repository
- [x] Add project documentation
- [x] Choose a base model for the first experiment: Qwen/Qwen2.5-1.5B-Instruct
- [ ] Verify model license and redistribution terms before publishing weights

## Phase 1 — Training pipeline
- [x] Choose an accessible GPU environment for the first experiment: Google Colab Tesla T4
- [x] Create a reproducible QLoRA notebook
- [x] Add Transformers + TRL + PEFT/QLoRA
- [x] Complete a first training run and save a LoRA adapter
- [x] Prepare an isolated second-experiment configuration that does not overwrite Experiment 001
- [ ] Run Experiment 002 in Colab and inspect its held-out outputs
- [ ] Add automated checks for the final saved adapter and inference setup

## Phase 2 — Pipis dataset
- [x] Define JSONL dataset format
- [x] Separate training and evaluation data
- [x] Add Russian-language dialogue and technical examples
- [x] Expand training examples across identity, technical accuracy, coding, evaluation, uncertainty, and response style
- [x] Add explicit contrast examples for LoRA versus LSTM and adapter versus full model
- [x] Preserve the existing evaluation file without adding its examples to training
- [x] Check 121 train rows and 8 eval rows for JSONL validity, exact duplicates, and exact train/eval overlap
- [ ] Review examples for factual accuracy, style consistency, and accidental contradictions

## Phase 3 — Pipis 0.x
- [x] Train first adapter
- [x] Compare Base and Adapter on initial prompts
- [x] Record known failures in docs/EXPERIMENT_001.md
- [x] Define the held-out Base-vs-Adapter comparison flow for Experiment 002
- [ ] Build a repeatable evaluation rubric and score outputs
- [ ] Run the second experiment and record results in docs/EXPERIMENT_002.md

## Phase 4 — Pipis 1.x+
- [ ] Improve dataset quality based on measured evaluation failures
- [ ] Increase model size only after the small-model pipeline is reliable
- [ ] Add tool-use experiments
- [ ] Consider memory/retrieval outside the model

## Phase 5 — Pipis 6.7
- [ ] Stable model recipe
- [ ] Reproducible evaluation
- [ ] Packaging/inference setup
- [ ] Prepare a usable Pipis application

## Future
- Pipis 6.9 is reserved for a later major iteration.
