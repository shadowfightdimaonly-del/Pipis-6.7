# Re-test Experiment 002 without retraining

The Experiment 002 adapter has already been trained. This script loads the saved adapter and reruns the eight held-out prompts; it does **not** train anything.

## In Google Colab

1. Open the repository notebook or a new Colab notebook with **T4 GPU**.
2. Upload and extract `pipis-6.7-exp002-adapter.zip` into a folder, for example `/content/pipis-6.7-exp002-adapter`.
3. Clone/download this repository and make sure `data/pipis_eval.jsonl` and `scripts/compare_adapter.py` are available.
4. Install dependencies if needed:
   ```bash
   pip install -U transformers peft accelerate bitsandbytes
   ```
5. Run:
   ```bash
   python scripts/compare_adapter.py \
     --adapter /content/pipis-6.7-exp002-adapter \
     --eval data/pipis_eval.jsonl \
     --output-dir /content/pipis-exp002-results
   ```
6. Download both `comparison.txt` and `comparison.jsonl` from `/content/pipis-exp002-results`.

If the ZIP contains the adapter files at its root, extract them directly into the adapter folder. The folder should contain `adapter_config.json` and `adapter_model.safetensors`.

## Outputs

- `comparison.txt`: readable Base / Adapter / reference answers.
- `comparison.jsonl`: one JSON object per prompt, with empty review fields for factual accuracy, instruction following, and repetition.

Keep the original ZIP and Experiment 001 adapter untouched. Save the results files somewhere persistent, because Colab runtime output can disappear.

## Notes

- Use the same eight examples from `data/pipis_eval.jsonl`; do not add them to training data.
- The script uses deterministic greedy decoding (`do_sample=False`) and a 120-token response cap by default.
- A 4-bit base is loaded on CUDA to fit a T4. Model outputs may still vary with library versions.
- This is a small manual evaluation, not proof of general performance. Review factual accuracy rather than judging only fluency.
