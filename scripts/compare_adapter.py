#!/usr/bin/env python3
"""Compare a base causal LM with a saved PEFT/LoRA adapter and persist every result.

Designed for Colab with a CUDA GPU. The script never trains; it only loads models
and writes JSONL + Markdown-friendly text output so notebook output loss is harmless.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
        text = row.get("text") if isinstance(row, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{path}:{line_no}: expected object with non-empty string 'text'")
        if "\nПипис:" not in text:
            raise ValueError(f"{path}:{line_no}: expected 'Пользователь: ...\\nПипис: ...' format")
        prompt, reference = text.split("\nПипис:", 1)
        rows.append({
            "prompt": prompt.strip() + "\nПипис:",
            "reference": reference.strip(),
        })
    if not rows:
        raise ValueError(f"No evaluation rows found in {path}")
    return rows


def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True, type=Path,
                        help="Path to extracted Experiment 002 adapter folder")
    parser.add_argument("--eval", default=Path("data/pipis_eval.jsonl"), type=Path,
                        help="Held-out evaluation JSONL (default: data/pipis_eval.jsonl)")
    parser.add_argument("--output-dir", default=Path("results/exp002"), type=Path,
                        help="Where to save comparison.jsonl and comparison.txt")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max-new-tokens", default=120, type=int)
    parser.add_argument("--no-4bit", action="store_true",
                        help="Load base in fp16 instead of 4-bit; needs more VRAM")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this script. In Colab select T4 GPU.")
    if not args.adapter.is_dir():
        raise FileNotFoundError(
            f"Adapter directory not found: {args.adapter}. Extract the experiment ZIP first."
        )
    rows = load_rows(args.eval)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Evaluation prompts: {len(rows)}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Adapter: {args.adapter}")
    print(f"Output: {args.output_dir.resolve()}")

    tokenizer = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {"device_map": "auto", "torch_dtype": torch.float16}
    if args.no_4bit:
        load_kwargs["torch_dtype"] = torch.float16
    else:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    base = AutoModelForCausalLM.from_pretrained(args.base_model, **load_kwargs)
    base.eval()
    tuned = PeftModel.from_pretrained(base, str(args.adapter))
    tuned.eval()

    results = []
    text_parts = [
        "# Pipis 6.7 — Base vs Adapter",
        "",
        f"- Date (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"- Base model: {args.base_model}",
        f"- Adapter: {args.adapter}",
        f"- Eval file: {args.eval}",
        f"- Prompt count: {len(rows)}",
        "",
        "Reference answers are shown for comparison; they are not used as model outputs.",
        "",
    ]

    for index, row in enumerate(rows, 1):
        prompt = row["prompt"]
        with tuned.disable_adapter():
            base_answer = generate(tuned, tokenizer, prompt, args.max_new_tokens)
        adapter_answer = generate(tuned, tokenizer, prompt, args.max_new_tokens)
        item = {
            "index": index,
            "prompt": prompt,
            "reference_answer": row["reference"],
            "base_answer": base_answer,
            "adapter_answer": adapter_answer,
            "review": {
                "factual_accuracy_0_to_2": None,
                "instruction_following_0_to_2": None,
                "repetition_0_to_2": None,
                "notes": "",
            },
        }
        results.append(item)
        text_parts.extend([
            f"## {index}. {prompt.splitlines()[0]}",
            "",
            f"**Reference:** {row['reference']}",
            "",
            f"**BASE:** {base_answer or '[empty response]'}",
            "",
            f"**ADAPTER:** {adapter_answer or '[empty response]'}",
            "",
            "**Review:** factual accuracy __/2; instruction following __/2; repetition __/2.",
            "",
            "---",
            "",
        ])
        print(f"[{index}/{len(rows)}] {prompt.splitlines()[0]}")

    json_path = args.output_dir / "comparison.jsonl"
    txt_path = args.output_dir / "comparison.txt"
    json_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in results),
        encoding="utf-8",
    )
    txt_path.write_text("\n".join(text_parts), encoding="utf-8")
    print("\nSaved results:")
    print(json_path)
    print(txt_path)
    print("Download these files from the output directory; no training was performed.")


if __name__ == "__main__":
    main()
