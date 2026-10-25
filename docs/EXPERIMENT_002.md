# Experiment 002 — expanded dataset QLoRA run

## Status

Prepared in GitHub; **not run yet**. The notebook now points to the current training and held-out evaluation JSONL files, checks that the training set has at least 100 rows, validates exact duplicates and train/eval overlap, and writes its outputs to separate experiment-002 paths.

## Dataset

- Training file: `data/pipis_train.jsonl`
- Training rows at preparation: 121
- Evaluation file: `data/pipis_eval.jsonl`
- Evaluation rows: 8
- Exact duplicate training rows: 0
- Exact duplicate evaluation rows: 0
- Exact train/eval overlap: 0

The evaluation set is kept unchanged from Experiment 001. The notebook uses these held-out prompts for Base-vs-Adapter generation; do not train on them.

## Planned differences from Experiment 001

- One full epoch over the expanded training dataset, instead of stopping after 10 steps.
- Learning rate reduced from `2e-4` to `1e-4`.
- Evaluate during training every 10 steps.
- Output directory: `/content/pipis-6.7-exp002-output`.
- Adapter directory: `/content/pipis-6.7-exp002-adapter`.
- Download name: `pipis-6.7-exp002-adapter.zip`.

These are experimental settings, not a guarantee of better results. The first adapter ZIP should remain untouched.

## Run procedure

1. Open the notebook in a fresh Colab session and select a GPU runtime.
2. Run all cells from top to bottom.
3. Confirm dataset validation passes and reports 121 train rows and 8 evaluation rows.
4. Review the Base and Adapter outputs for all eight held-out prompts.
5. Download and preserve `pipis-6.7-exp002-adapter.zip` separately from the first ZIP.
6. Record whether the new adapter improves factual accuracy, identity consistency, concise stopping, and uncertainty handling. Do not judge from loss alone.

## Limitations

- Eight evaluation examples are too few for strong generalization claims.
- Dataset validation catches formatting, exact duplicates, and exact train/eval overlap; it does not prove every training answer is high quality.
- Base model license and redistribution terms still need review before distributing model weights.
