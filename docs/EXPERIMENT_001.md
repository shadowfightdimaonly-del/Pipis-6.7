# Experiment 001 — first Pipis 6.7 QLoRA adapter

## Status

Completed technically in Google Colab on a Tesla T4 GPU. The notebook completed without a reported error, saved an adapter ZIP, and ran Base-vs-Adapter generation.

## Setup

- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- Method: QLoRA / PEFT LoRA adapter
- Dataset at experiment time: 30 training examples and 8 evaluation examples
- Training limit: `max_steps=10`
- Downloaded artifact: `pipis-6.7-adapter.zip`, approximately 38 MB
- Files observed in ZIP: `adapter_config.json`, `adapter_model.safetensors`, tokenizer files, and README

The ZIP is an adapter artifact, not the full base model.

## Manual Base-vs-Adapter observations

The notebook generated answers for three prompts.

### 1. “Кто ты?”

- Base output described itself as generic code.
- Adapter output claimed it was created by Anthropic.

Assessment: the adapter did not reliably learn Pipis identity and produced a false creator claim.

### 2. “Объясни LoRA простыми словами.”

- Base output was confused and included an incorrect expansion of LoRA as “Long Short-Term Memory.”
- Adapter output was more fluent but incorrectly described LoRA as an AI trained on huge datasets.

Assessment: fluency improved in this example, but factual correctness did not.

### 3. “Что ты сделаешь, если не уверен в ответе?”

- Base output was rambling and generic.
- Adapter output moved toward asking the user for clarification, but continued into repetitive dialogue-like text.

Assessment: some desired behavior appeared, but response control and stopping behavior were weak.

## Conclusion

This is a successful pipeline smoke test, not evidence that Pipis is already a useful model. The adapter changes generation, but the three examples are a tiny manual sample and do not establish overall quality or generalization.

## Next experiment requirements

1. Review and expand the training dataset while preserving the existing evaluation split.
2. Keep identity statements consistent: Pipis is a fine-tuned Qwen-based experimental model, not created by Anthropic, OpenAI, or another unrelated provider.
3. Correct technical definitions, especially LoRA versus LSTM.
4. Add examples that demonstrate concise answers, no unsupported identity claims, uncertainty handling, and non-repetitive stopping.
5. Run the same held-out evaluation prompts for Base and Adapter, score them with a fixed rubric, and record the outputs.
6. Do not merge or distribute a final model until the base-model license and redistribution terms have been reviewed.
