# Pipis 6.7 evaluation rubric

Use this rubric for the held-out Base-vs-Adapter evaluation. It is intentionally small and manual: the current eval set has only 8 prompts, so the score is diagnostic, not a benchmark.

## Per-prompt score

Give each model 0 or 1 point for each criterion.

1. **Factual accuracy** — The answer is materially correct and does not introduce a false technical claim.
2. **Task relevance** — The answer directly addresses the user's request instead of drifting into an unrelated topic.
3. **Instruction/style fit** — The answer follows the requested simplicity, language, and format when applicable.
4. **Stopping/conciseness** — The model stops after a useful answer and does not repeat itself or continue as a fake dialogue.
5. **Uncertainty handling** — When the prompt involves uncertainty, the answer distinguishes known facts from guesses and avoids pretending to know what it does not know.

Maximum: 5 points per prompt, 40 points per model across 8 prompts.

## Comparison

Record Base and Adapter scores separately. Also record a short note for each prompt describing the most important difference. Do not treat a higher score on one prompt as proof of overall superiority.

Useful labels for notes:

- `improved` — Adapter is clearly better on the requested task.
- `regressed` — Adapter is clearly worse.
- `mixed` — Adapter improves one important property but introduces another problem.
- `tie` — No meaningful difference.

## What counts as a serious regression

Flag an Adapter answer even when it is fluent if it:

- invents the creator or identity of Pipis;
- confuses LoRA with LSTM or quantum computing;
- claims to know an uncertain fact without evidence;
- ignores the user's actual question;
- repeats the prompt or produces a long continuation instead of answering;
- reveals or requests secrets such as API keys.

## Aggregate reporting

Report:

- Base total: `/40`
- Adapter total: `/40`
- Number of prompts improved by Adapter
- Number regressed
- Number mixed
- Number tied

Do not convert the eight-prompt result into a percentage and present it as a general model benchmark. The sample is too small for that claim.
