# Kaggle T4 Qwen3 GSM8K Pilot

## Scope and status

This notebook is a separate local, open-weight experiment. It does not replace the Phase 3A provider benchmark, call LiteLLM, use credentials, alter the paid-provider gate, or generate records accepted by the Phase 3A scientific exporter. Local records carry `provider=local_transformers` and `scientific_eligible=false`.

The current deliverable status is **NOTEBOOK_READY_FOR_KAGGLE**. No Kaggle inference has been run or measured as part of preparing this repository change.

Notebook: [`experiments/notebooks/kaggle_t4_qwen3_gsm8k.ipynb`](../../experiments/notebooks/kaggle_t4_qwen3_gsm8k.ipynb)

## Frozen experiment

The notebook consumes the checked-in `configs/phase3a_pilot.collection-plan.json` and the exact processed dataset/manifest created by Phase 3A. It validates the complete dataset checksum, plan hash, validation split, selected count, ID hash, missing/duplicate IDs, record provenance, and exclusion of official-test IDs before inference. It will not download another GSM8K snapshot or substitute samples.

Expected frozen identity:

| Item | Value |
|---|---|
| Dataset revision | `3101c7d5072418e28b9008a6636bde82a006892c` |
| Processed dataset content SHA-256 | `aa75c686332c09e12ce6412571b32b7efae779dfc8696d3448b29a45e5542e4b` |
| Collection plan SHA-256 | `1d5065acc16b50c1521422ec2a912ed28a10bd483d905054fbdd24aefd642493` |
| Ordered sample IDs SHA-256 | `077275a61e532506eab2d10c09d36960976253969380ca256af7b95cbcced5ed` |
| Split and size | validation, 10 samples |

The processed data files are not bundled in Git. Prepare a Kaggle Dataset containing the following paths (the directory root may be the dataset root):

```text
data/processed/gsm8k/train.jsonl
data/processed/gsm8k/validation.jsonl
data/processed/gsm8k/test.jsonl
data/manifests/gsm8k.json
```

Use the exact Phase 3A frozen artifacts. For example, from a workspace containing those files, stage the four files while retaining the directory structure and create a Kaggle Dataset through Kaggle's dataset UI. Name/attach it as `gsm8k-frozen-pilot`, so the notebook sees `/kaggle/input/gsm8k-frozen-pilot/data/...`. Do not regenerate the split or use Kaggle's unrelated public GSM8K dataset.

## Models and protocol

| Role | Hugging Face model | Pinned revision | Loading |
|---|---|---|---|
| Small | `Qwen/Qwen3-1.7B` | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | FP16 on GPU 0 |
| Strong | `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` | bitsandbytes NF4 4-bit, FP16 compute on GPU 0 |

The revisions are pinned to immutable commits in the [Qwen3-1.7B model repository](https://huggingface.co/Qwen/Qwen3-1.7B/commit/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e) and [Qwen3-8B model repository](https://huggingface.co/Qwen/Qwen3-8B/commit/b968826d9c46dd6066d109eabc6255188de91218). The official Qwen3 model card shows the Transformers interface and chat-template use; its README requires a Transformers release that recognizes Qwen3 (4.51 or later). See the [Qwen3-1.7B model card](https://huggingface.co/Qwen/Qwen3-1.7B) and [Transformers bitsandbytes guide](https://huggingface.co/docs/transformers/quantization/bitsandbytes). Current bitsandbytes docs list NVIDIA compute capability 6.0+ for supported builds; Kaggle T4 is Turing (compute capability 7.5). Actual Kaggle package/CUDA compatibility and model load are still smoke-test requirements; see the [bitsandbytes install requirements](https://huggingface.co/docs/bitsandbytes/installation).

Prompt instruction is copied from `experiments/prompts.py`, with the user's question in a separate user message and no reference answer in model input. The Qwen tokenizer applies its own chat template with `enable_thinking=False` and `add_generation_prompt=True`; this framing is not identical to LiteLLM. The notebook records both the original prompt hash and effective rendered prompt/template hashes.

Primary generation settings: batch size 1, maximum 1,024 input tokens, maximum 512 new tokens, seed 42, one generation per sample, no retry, no fallback, `do_sample=False`. Greedy decoding is a deterministic experiment choice and differs from Qwen's recommended sampling recipe. The notebook does not set `temperature=0`. Overlength prompts are recorded as not executed and are never truncated.

## Kaggle setup

1. Open Kaggle and create a Notebook.
2. Select GPU T4; select T4 x2 if available. The experiment selects GPU 0 only and never assumes that two cards share/aggregate VRAM.
3. Enable Internet if model repositories need to be downloaded. The pinned repositories are public; no provider credentials are needed.
4. Import/upload `experiments/notebooks/kaggle_t4_qwen3_gsm8k.ipynb`.
5. Attach the exact frozen dataset described above under the input name `gsm8k-frozen-pilot`.
6. Run environment, GPU and dataset checks. The setup cell reports existing package versions first and installs only missing or incompatible inference/plotting packages with `--no-deps`; it intentionally does not replace Kaggle's PyTorch/CUDA build. If required dependencies cannot import afterward, use a compatible Kaggle image rather than upgrading CUDA/PyTorch in the notebook.
7. Set `RUN_SMALL_SMOKE = True` in CELL 7, run through CELL 8, and inspect `outputs/checkpoints/small_smoke.jsonl`. A smoke test is a real model call and will download/load the model.
8. Set `RUN_STRONG_SMOKE = True` (and small smoke false), run through CELL 10, and inspect `outputs/checkpoints/strong_smoke.jsonl`. The 8B model must load successfully under NF4 before proceeding.
9. In CELL 7, set `SMOKE_TESTS_CONFIRMED = True` only after inspecting both smoke outputs, then set `RUN_FULL_PILOT = True`. Run all cells to execute the same ten planned IDs with both models. These flags default to false; running the notebook as committed does not invoke either model.
10. Download `/kaggle/working/outputs.zip` after export. Keep the notebook outputs and run manifest with the archive for provenance.

The 1.7B FP16 weights and 8B NF4 model may fit on a 16 GB T4, but actual activation/KV cache, allocator fragmentation, library version, and available GPU memory affect loading and generation. A load or generation OOM is recorded and does not trigger CPU inference/offload. If resources are insufficient, first restart the session and verify GPU memory is free. A lower `max_new_tokens` or input limit can be explored only as a separately labeled protocol/run; it must not be substituted silently for this primary pilot.

## Resume and outputs

Execution checkpoints live at `/kaggle/working/outputs/checkpoints/{small_full,strong_full}.jsonl`. Each outcome is durably appended before advancing. Existing entries, including failures and overlength non-executions, are not retried on resume. To intentionally reissue a failed sample, archive the checkpoint and document the intervention in the run notes; the notebook does not silently retry. Smoke checkpoints are separate from full-run checkpoints.

The notebook exports:

- `small_results.jsonl`, `strong_results.jsonl` — local records and existing evaluator results;
- `paired_results.csv` — exact planned samples, execution/evaluation statuses and paired outcomes;
- `metrics.json` — denominators, descriptive accuracy, generation-only latency, tokens, VRAM and paired counts;
- `environment.json`, `model_revisions.json`, `run_manifest.json` — runtime and provenance;
- `error_analysis.md` — actual failures, evaluator-incorrect/unknown outputs, if present;
- `figures/model_comparison.png`, `figures/paired_outcomes.png`;
- `/kaggle/working/outputs.zip` — the exported result files and figures, excluding weights, caches and secrets.

Latency means CUDA-synchronized `model.generate` duration only. It excludes download/model loading, prompt rendering/tokenization, any scheduling/queueing, evaluation and export. Do not compare it as end-to-end pipeline latency. With ten paired validation questions, metrics are descriptive; report counts and denominators and avoid strong statistical claims.

## Integrity and interpretation limits

- Outputs are from local `transformers` inference, not the commercial provider path.
- They are explicitly not Phase 3A `scientific_eligible` records and are not accepted by the existing real-provider exporter.
- No inference results are embedded in the notebook or documentation.
- `NOTEBOOK_READY_FOR_KAGGLE` is the repository status. Use `KAGGLE_SMOKE_PASSED` or `KAGGLE_PILOT_COMPLETED` only after actual Kaggle GPU evidence is saved and reviewed.
- GSM8K contains only ten samples in this frozen pilot; error categories require human review, and no statistical generalization is justified.
