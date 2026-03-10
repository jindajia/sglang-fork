# KV Rotation Evaluation Notes

## Repository Overview

- **Branch**: `jinda_rotation` (built on top of `jinda_deploy_int4_8cache`)
- **Evaluation script**: `eval_kv_rotation.sh`
- **Evaluation framework**: `tore-eval/` (git submodule, branch `jisen/kv_rotation_eval`)

---

## SGLang Fork: Key Features

### INT4/INT8 KV Cache (`jinda_deploy_int4_8cache`)
- Adds INT4 and INT8 KV cache quantization to SGLang
- Uses Triton kernels to quantize-and-write directly to cache buffers (no extra global memory write)
- Stores per-layer scale/zero tensors alongside quantized K/V buffers
- Entry point: `MHATokenToKVPool.set_kv_buffer()` in `python/sglang/srt/mem_cache/memory_pool.py`

### Hadamard Rotation (`dac7b6f` — `jinda_rotation`)
Applies randomized Hadamard rotation to K/V before INT4 quantization (QuIP#/QuaRot/SpinQuant technique).
Rotation makes activations more Gaussian/isotropic, reducing INT4 quantization error.

**Environment variables (all default OFF):**
| Variable | Default | Effect |
|----------|---------|--------|
| `HADAMARD=1` | 0 | Rotate K before quantization; counter-rotate Q at decode |
| `ROTATE_V=1` | 0 | Also rotate V; counter-rotate attention output |
| `HADAMARD_ORDER=16` | 16 | Block size for block-Hadamard (must divide head_dim) |

**Write path** (`memory_pool.py`): reshape `(..., head_dim)` → `(..., head_dim/order, order)`, apply `H·x/sqrt(order)`, reshape back, then INT4 quantize.

**Decode path**: `triton_backend.py` and `flashattention_backend.py` apply the same Hadamard to Q before attention (so Q·K^T is invariant), and inverse-rotate output if `ROTATE_V=1`.

**Dependency**: `fast_hadamard_transform` third-party package.

---

## Evaluation Script: `eval_kv_rotation.sh`

### MODEL_CONFIGS Format
```bash
"model_name|tp_size|ep_size|dp_size|gpu_devices|tasks"
```
- `model_name`: full HuggingFace model ID (e.g. `Qwen/Qwen3-8B`) — **must be full name**, short name breaks tokenizer loading
- `tasks`: comma-separated preset names, run **sequentially** per model
- Multiple models run **in parallel**, each on assigned GPUs

### Available Preset Names
| Preset | Framework | Notes |
|--------|-----------|-------|
| `gpqa_think` | simple-evals | GPQA Diamond, n_repeats=4 |
| `humaneval_think` | simple-evals | HumanEval pass@k |
| `customized_livecodebench_think` | livecodebench | release_v6, 2025-01-05 ~ 2025-04-07 |
| `aime24_think` | simple_math | AIME 2024 |
| `math_500_think` | simple_math | MATH-500 |

### Output Paths
```
sglang-fork/
├── eval_results/
│   └── {model_short_name}_{task}/      # per-task results (created by local logger)
│       ├── {scenario}_{n}_{temp}.json          # main results
│       ├── {scenario}_{n}_{temp}_eval.json     # metrics summary
│       ├── {scenario}_{n}_{temp}_eval_all.json # per-problem eval
│       └── {scenario}_{n}_{temp}_stream.jsonl  # streaming write
└── eval_logs/
    ├── inference_logs/   # SGLang server logs
    ├── slurm_logs/       # SLURM output
    └── batch_logs/       # per-model batch logs
```

**Ownership note**: Docker container runs as `--user $(id -u):$(id -g)` to avoid root-owned files.

---

## tore-eval Framework Architecture

### Preset System
`--framework preset --preset_name xxx` loads a YAML config from `src/tore_eval/evaluators/preset/configs/`.

**Critical limitation**: `preset_evaluator._create_target_evaluator()` only copies a fixed list of fields from CLI args to the target evaluator:
```python
["model_name", "model_name_or_path", "tokenizer_name_or_path",
 "provider", "api_key", "base_url", "chat_template",
 "num_workers", "log_file", "num_examples", "loggers"]
```
→ `temperature`, `top_p`, `seed` passed via CLI are **silently ignored** when using `--framework preset`.

→ Temperature and top_p must be set directly in the YAML file.

### Evaluator Frameworks
| Framework | Used by | temperature/top_p |
|-----------|---------|-------------------|
| `simple-evals` | GPQA, HumanEval | `SimpleEvalsArguments` has both |
| `simple_math` | AIME24, MATH_500 | `SimpleMathArguments` has both |
| `livecodebench` | LiveCodeBench | `LiveCodeBenchArguments` has both |

### Streaming Write (custom modifications)
| Evaluator | Mechanism |
|-----------|-----------|
| GPQA | `map_with_progress(output_file=...)` — per-question, thread-safe JSONL append |
| HumanEval | Same as GPQA (added `output_file` param) |
| AIME/MATH_500 | Per-response write in `generate_responses()` as futures complete |
| LiveCodeBench | Per-problem write via `stream_writer` callback in `api_runner.run_batch()` |

**LCB streaming detail**: `api_runner.py` uses `asyncio.gather()` — all problems sent concurrently. Each task is wrapped in `run_with_callback()` which fires `stream_writer(orig_idx, result)` immediately upon individual task completion, before `gather()` returns.

---

## Repeat / Seed / top_k Analysis

### `n_repeats` in tore-eval
- **GPQA**: duplicates the example list (`examples * n_repeats`), relies on API randomness for diversity
- **AIME/MATH_500**: repeats the dataset n times, computes pass@n_repeats as max score
- **LCB**: `n` is the number of completions **per API call** (OpenAI `n` parameter in `choices`), used for pass@k. With `deepseek` mode only `n=1` is effectively used (not passed to API kwargs)
- **None of these set a seed** — diversity comes entirely from temperature-based sampling randomness

### top_k
- **SGLang default**: `top_k = -1` → disabled, equivalent to `TOP_K_ALL = 1<<30` (full vocabulary)
- **tore-eval**: no `top_k` field in any evaluator arguments, never passed to API
- To support: need to add `top_k` to evaluator arguments + runner/sampler chain

### seed
- **SGLang**: supports per-request `sampling_seed` (per-request tensor on GPU, with hash-based per-token seed derivation: `step_seed = (seed * 19349663) ^ (position * 73856093)`)
- **SGLang default**: `sampling_seed = None` → completely random (no determinism)
- **Requires**: `--enable-deterministic-inference` server flag to activate seed logic; without it, `sampling_seed` tensor is `None` and seeds are ignored
- **tore-eval**: no seed field anywhere, never passed to API
- **Conclusion**: Not passing seed is fine. With `temperature > 0` and no `--enable-deterministic-inference`, each repeat naturally produces different results due to GPU sampling randomness. Explicit seed only needed for strict reproducibility.

### Plan B: Bypass Preset, Call Evaluators Directly
Instead of `--framework preset --preset_name xxx`, call the target framework directly:
```bash
# LCB example
python3 -m tore_eval.eval \
    --framework livecodebench \
    --release_version release_v6 \
    --start_date 2025-01-05 --end_date 2025-04-07 \
    --scenario codegeneration \
    --max_tokens 32768 --temperature 0.6 --top_p 0.95 \
    --openai_reason_mode deepseek --n 1 \
    --model_name_or_path "Qwen/Qwen3-8B" \
    --provider custom --base_url "http://localhost:8000/v1" \
    --num_workers 64 --log_file "..." --loggers "..."
```

**Advantages**: temperature/top_p/seed fully controllable from CLI; repeat via shell `for` loop with different `--log_file` suffixes.

**Disadvantage**: shell script must maintain per-framework parameter sets (different args for livecodebench vs simple-evals vs simple_math).

**top_k and seed still require code changes** regardless of Plan A or B — they are not wired through any evaluator argument chain.

---

## Known Issues / Bugs Fixed

1. **Short model name bug**: tore-eval tokenizer loading uses `model_name_or_path`. Must pass full HF ID (`Qwen/Qwen3-8B`), not short name (`Qwen3-8B`).

2. **LCB output path**: `livecodebench_evaluator.py` does `os.chdir()` to its own source directory and writes to relative `livecodebench_results/`. Fixed to use `loggers["local"]["output_dir"]` as base path.

3. **LCB filename**: `{scenario}` used enum `__str__` giving `Scenario.codegeneration`. Fixed to `scenario.value` giving `codegeneration`.

4. **`openai_reason_mode: None`**: `api_runner.py` raises `ValueError` for unrecognized mode. Workaround: set `openai_reason_mode: deepseek` in YAML (standard chat completion with temperature/top_p).

5. **Root-owned files**: Docker container previously ran as root. Fixed with `--user $(id -u):$(id -g)` + `pip install --user`.

6. **api_runner result ordering bug**: original `run_batch()` appended cached results first then new results, breaking order when partial cache hits occurred. Fixed by tracking `orig_idx` and reassembling in order.

---

## LiveCodeBench release_v6 Data

Source: `livecodebench/code_generation_lite` on HuggingFace

| File | Problems | Date Range |
|------|----------|------------|
| test5.jsonl | 167 | 2024-09-22 ~ 2025-01-04 |
| test6.jsonl | 175 | 2025-01-04 ~ 2025-04-06 |

`customized_livecodebench_think.yaml` config:
- `release_version: release_v6`, `start_date: 2025-01-05`, `end_date: 2025-04-07`
- Evaluates only the 175 new problems added in v6
- `temperature: 0.6`, `max_tokens: 32768`, `n: 1`, `openai_reason_mode: deepseek`
