#!/usr/bin/env python3
"""Compare a base causal LM with a local PEFT adapter on a held-out JSONL set.

Example (Colab/Linux):
  python scripts/evaluate_adapter.py \
    --adapter /content/pipis-6.7-exp002-adapter \
    --eval data/pipis_eval.jsonl \
    --output-dir /content/pipis-6.7-exp002-evaluation

The script writes results.json and results.md so the evaluation survives a
lost notebook output. It does not train or modify the adapter.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict) or not isinstance(row.get("text"), str):
            raise ValueError(f"{path}:{line_number}: expected an object with string field 'text'")
        text = row["text"]
        if "\\nПипис:" not in text and "\nПипис:" not in text:
            raise ValueError(f"{path}:{line_number}: expected 'Пользователь: ...\\nПипис: ...' format")
        rows.append(row)
    if not rows:
        raise ValueError(f"Evaluation dataset is empty: {path}")
    return rows


def prompt_from_text(text: str) -> tuple[str, str]:
    marker = "\nПипис:"
    if marker not in text:
        raise ValueError("Dataset example has no newline followed by 'Пипис:'")
    user_part = text.split(marker, 1)[0]
    user_prompt = user_part + marker
    expected = text.split(marker, 1)[1].strip()
    return user_prompt, expected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True, type=Path, help="Directory containing adapter_config.json and adapter_model.safetensors")
    parser.add_argument("--eval", dest="eval_path", type=Path, default=Path("data/pipis_eval.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation-results"))
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--no-4bit", action="store_true", help="Load model without bitsandbytes 4-bit quantization")
    args = parser.parse_args()

    if not (args.adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"adapter_config.json not found in {args.adapter}")
    if not (args.adapter / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"adapter_model.safetensors not found in {args.adapter}")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    rows = load_rows(args.eval_path)
    prompts_and_expected = [prompt_from_text(row["text"]) for row in rows]

    model_kwargs: dict[str, Any] = {"device_map": "auto"}
    if not args.no_4bit:
        if not torch.cuda.is_available():
            raise RuntimeError("4-bit mode requires CUDA. Run in Colab with T4 GPU, or use --no-4bit if memory permits.")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model_kwargs["torch_dtype"] = torch.float16
    else:
        model_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"Base model: {args.base_model}")
    print(f"Adapter: {args.adapter}")
    print(f"Held-out prompts: {len(prompts_and_expected)}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    model = PeftModel.from_pretrained(base_model, str(args.adapter))
    model.eval()

    def generate(prompt: str) -> str:
        encoded = tokenizer(prompt, return_tensors="pt")
        input_device = model.get_input_embeddings().weight.device
        encoded = {key: value.to(input_device) for key, value in encoded.items()}
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(generated[0][encoded["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    results: list[dict[str, str | int]] = []
    for index, (prompt, expected) in enumerate(prompts_and_expected, 1):
        print(f"\n[{index}/{len(prompts_and_expected)}] {prompt.splitlines()[0]}")
        with model.disable_adapter():
            base_answer = generate(prompt)
        adapter_answer = generate(prompt)
        result = {
            "index": index,
            "prompt": prompt,
            "reference_answer": expected,
            "base_answer": base_answer,
            "adapter_answer": adapter_answer,
        }
        results.append(result)
        print("BASE:", base_answer)
        print("ADAPTER:", adapter_answer)
        print("REFERENCE:", expected)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "results.json"
    markdown_path = args.output_dir / "results.md"
    json_path.write_text(json.dumps({
        "base_model": args.base_model,
        "adapter_path": str(args.adapter),
        "evaluation_file": str(args.eval_path),
        "count": len(results),
        "results": results,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Pipis 6.7 — Base vs Adapter evaluation",
        "",
        f"- Base model: `{args.base_model}`",
        f"- Adapter: `{args.adapter}`",
        f"- Evaluation file: `{args.eval_path}`",
        f"- Held-out prompts: {len(results)}",
        "",
        "> Outputs are model generations, not automatic quality scores. Review factual correctness, relevance, instruction-following, repetition, and format manually.",
        "",
    ]
    for item in results:
        lines += [
            f"## {item['index']}. {str(item['prompt']).splitlines()[0]}",
            "",
            "**BASE**",
            str(item["base_answer"]) or "(empty output)",
            "",
            "**ADAPTER**",
            str(item["adapter_answer"]) or "(empty output)",
            "",
            "**REFERENCE**",
            str(item["reference_answer"]) or "(empty reference)",
            "",
        ]
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved JSON: {json_path}")
    print(f"Saved Markdown: {markdown_path}")
    print("Evaluation complete. No training was performed.")


if __name__ == "__main__":
    main()
