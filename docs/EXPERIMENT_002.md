# Experiment 002 — expanded dataset QLoRA run

## Status

**Training completed by the user in Colab and the Experiment 002 adapter ZIP was downloaded.** Some notebook cell outputs later disappeared, so the full eight-prompt comparison is not available for review. The notebook was updated afterward to save the evaluation into downloadable JSON and Markdown files. The new persistence change has not yet been run by the user.

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
4. Preserve the already downloaded `pipis-6.7-exp002-adapter.zip` separately from the first ZIP.
5. To recover a comparable evaluation without retraining, run the updated notebook's model-load/evaluation portion or use `scripts/evaluate_adapter.py` with the saved adapter and the held-out evaluation file.
6. Download and preserve the resulting `results.json` and `results.md`. Record factual accuracy, identity consistency, concise stopping, and uncertainty handling. Do not judge from loss alone.

## Limitations

- Eight evaluation examples are too few for strong generalization claims.
- Dataset validation catches formatting, exact duplicates, and exact train/eval overlap; it does not prove every training answer is high quality.
- Base model license and redistribution terms still need review before distributing model weights.


## Persistent evaluation files

- Script: `scripts/evaluate_adapter.py`
- The notebook evaluation cell now writes `/content/pipis-6.7-exp002-evaluation.json` and `/content/pipis-6.7-exp002-evaluation.md`, then offers both for download.
- This fix prevents the only copy of the comparison from existing solely in transient Colab cell output. It does not recover output that was already lost.
