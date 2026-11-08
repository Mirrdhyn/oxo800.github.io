# Serving Laguna S 2.1 NVFP4 on a single NVIDIA DGX Spark (GB10): the complete runbook

> vLLM runbook to serve Laguna S 2.1 NVFP4 (117B, MoE) on a single DGX Spark GB10: avoiding the JIT OOM, 256K context, DFlash speculative decoding, detailed throughput benchmarks.

- Date: 2026-07-23
- Language: en
- URL: https://0x0800.com/blog/laguna-s21-single-spark/
- Tags: dgx-spark, gb10, laguna, vllm, nvfp4, dflash, moe
- Translation (fr): https://0x0800.com/fr/blog/laguna-s21-single-spark/


From the box killed by its own JIT compiler to a working 256K service. A 117.6B-parameter MoE (8.5B active), DFlash speculative decoding, native 256K context, all on a single unified-memory machine — no TP=2, no cross-node, no NCCL. Every command in this runbook is given verbatim.

Laguna fits on a single machine, where DeepSeek-V4 required two nodes and a coordination layer. The hard part is therefore no longer distribution but the **box's memory**, which an unbridled JIT exhausts in a few seconds. This runbook covers the hardware context, the three traps that kill the boot, the complete single-node launch and the benchmark results measured on our hardware.

## Hardware context

- **1× NVIDIA DGX Spark** (GB10, Grace-Blackwell, compute capability **12.1 / sm_121a**, 128 GB of unified memory, aarch64). A single node.
- **Unified** memory: CPU and GPU share the same 128 GB. Direct consequence: `--gpu-memory-utilization` bounds the KV allocation, but not the transient profiling peaks nor JIT compilation. A host-side peak draws on the same reserve as the weights.
- Model: **poolside/Laguna-S-2.1-NVFP4** — MoE (256 routed experts + 1 shared, 8.5B active out of 117.6B), 48 layers (12 global attention + 36 sliding-window, window 512), **FP8 KV cache**, native **256K** context (262,144), native weights up to 1M. ~67 GB on disk. **Fits on a single Spark: no TP=2, no cross-node, no RoCE, no NCCL.** Unlike DeepSeek-V4-Flash, no multi-node machinery is needed.
- Native speculative decoding: **DFlash** drafter (`DFlashLagunaForCausalLM`), supplied separately.

## Why the "naive" launch kills the box

A `vllm serve` of an NVFP4 model on SM121, launched without precautions, does not crash cleanly within the process: it brings down **earlyoom**, which sends a SIGTERM to the biggest consumer. We ran into it three times, at three different places:

| Step | Symptom | Cause |
|---|---|---|
| Profiling / KV sizing | earlyoom SIGTERM around ~116 GB | the profiling forward (batch × 256-expert MoE) adds ~33 GB of transient memory |
| Warmup / torch.compile | earlyoom kills cc1plus | inductor JIT compiling in parallel |
| FP4 kernels | earlyoom kills cc1plus / nvcc | the real culprit — see the box below |

What they have in common: none of these peaks is bounded by `--gpu-memory-utilization`. This flag reserves the KV **after** profiling; it says nothing about the host's transient allocations. earlyoom, for its part, triggers on `MemAvailable` (reclaimable page cache included) and not on `free`. So you have to **cut the peaks at the source**, not just lower the target utilisation.

{{< callout label="Box — MAX_JOBS=4: the nvcc fan-out that empties 128 GB" >}}
**What we observe (fact).** At first boot on a cold `~/.cache/flashinfer` cache, vLLM JIT-compiles the NVFP4 kernels (FlashInfer). nvcc splits the compilation into an **unbounded fan-out** of `cc1plus`. Each `cc1plus` of an FP4 kernel is greedy; launched all together, they saturate the 128 GB of unified memory while the weights (~67 GB) are already resident. earlyoom steps in, SIGTERM, boot dead. No Xid, no CUDA trace: it is not the GPU that goes down but the **host** that runs out of RAM.

**The fix** (from the DGX Spark section of the model's README). Cap the fan-out:

```bash
export MAX_JOBS=4
```

Four concurrent `cc1plus` instead of N. Compilation is a bit slower at first boot, but it **fits within the memory envelope**. Combined with `CUTE_DSL_ARCH=sm_121a` (the arch expected by the FP4 kernels' JIT) and with `nvcc` on the `PATH`, this is what gets the launch through.

**Persistent trap.** Never remove `MAX_JOBS=4` thinking "the cache is warm, it'll be fine". A wiped `~/.cache/flashinfer`, a FlashInfer update or an arch change relaunches the full fan-out. Without the cap, the box goes down again. Keeping it is free once the cache is warm (nothing to recompile); forgetting it costs a round trip to the server room.
{{< /callout >}}

## Prerequisites: venv and models

The **~/.venv-vllm** venv (the jasl fork built for DeepSeek-V4, vLLM `20260721b1.dev0`) loads Laguna without modification: it is the one we use here. The model's README recommends a stock vLLM **0.25.1** in a dedicated venv; both work, we stayed on the fork, already in place and more recent.

Two local repositories, weights **and** drafter:

```text
~/models/lagunas21-nvfp4          # main weights (~67 GB)
~/models/lagunas21-nvfp4-dflash   # DFlash drafter for speculative decoding
```

Stock recipe, for the record — dedicated `vllm025` venv: `uv venv` + `python3.12-dev`, `vllm==0.25.1`, torch cu130, and the nightly FlashInfer trio `flashinfer-python` / `flashinfer-cubin` / `flashinfer-jit-cache==0.6.15.dev20260712`. The README's DGX Spark section gives the exact pins.

## Required environment variables

Three families: the usual Marlin/FP8 tuning, then the **three** Spark variables without which the FP4 JIT breaks or brings the box down. Gathered in the launch script (§5):

```bash
export VLLM_MARLIN_USE_ATOMIC_ADD=1
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1
export VLLM_FLOAT32_MATMUL_PRECISION=high
export VLLM_USE_NCCL_SYMM_MEM=0
export SAFETENSORS_FAST_GPU=1
export VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
# README — DGX Spark section (the three that matter)
export CUTE_DSL_ARCH=sm_121a          # arch of the FP4 kernels' JIT
export PATH=/usr/local/cuda/bin:$PATH # nvcc for the JIT
export MAX_JOBS=4                     # cap on the nvcc fan-out (without it, box OOM)
```

## Single-node launch

Script `~/launch.sh` — a single process, OpenAI API on `:8082`:

```bash
#!/usr/bin/env bash
export VLLM_MARLIN_USE_ATOMIC_ADD=1
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1
export VLLM_FLOAT32_MATMUL_PRECISION=high
export VLLM_USE_NCCL_SYMM_MEM=0
export SAFETENSORS_FAST_GPU=1
export VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export CUTE_DSL_ARCH=sm_121a
export PATH=/usr/local/cuda/bin:$PATH
export MAX_JOBS=4

cd ~
exec ~/.venv-vllm/bin/vllm serve ~/models/lagunas21-nvfp4 \
  --served-model-name laguna-s21 \
  --host 0.0.0.0 --port 8082 \
  --max-model-len 262144 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.85 \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser poolside_v1 \
  --reasoning-parser poolside_v1 \
  --override-generation-config '{"temperature":0.7,"top_p":0.95}' \
  --speculative-config '{"model":"/home/<user>/models/lagunas21-nvfp4-dflash","num_speculative_tokens":7,"method":"dflash"}' \
  --moe-backend auto
```

Three settings hold the memory envelope, in addition to the env variables:

- **--max-num-batched-tokens 4096** (default 8192). The profiling forward runs on this batch; at 8192 × 256-expert MoE, the peak exceeds the reserve and earlyoom steps in. 4096 goes through.
- **--gpu-memory-utilization 0.85** (not 0.92). Leaves a margin for the host for the JIT and the warmup; at 0.92 inductor's `cc1plus` has no room left.
- **--max-num-seqs 8**. DFlash crashes at the default 256; the README recommends 32, we went down to 8 for the 256K context.

Launch in the foreground inside a `tmux` so that the session holds the process and captures the log; the JIT kernels are cached for the following starts:

```bash
tmux new-session -d -s laguna "bash ~/launch.sh > ~/laguna.log 2>&1"
```

**Successful start.** `health 200`, OpenAI API on `:8082`. Cold boot ~9 min (loading the 67 GB + JIT of the FP4 kernels); later boots, with a warm cache, are much shorter. `poolside_v1` tool-calling answers correctly (standard OpenAI `tool_calls`).

## Memory discipline: restarting without getting killed

Unified memory imposes restart hygiene. Two principles.

**earlyoom stays as the safety net, we do not disarm it.** It triggers on `MemAvailable`, reclaimable page cache included. A warm checkpoint in page cache therefore counts as available. But if a real peak exceeds the reserve, earlyoom kills vLLM **instead of freezing the box**: better a dead process than a round trip to the server room.

**Kill THEN drop_caches, never grouped.** After a run, RAM can stay at ~110 GB (weights still resident). If you relaunch without reclaiming, the loader sees too little free RAM for the checkpoint (`Available RAM: 46.27 GiB` < `Checkpoint size: 66.98 GiB`) and crawls. You have to reclaim, but in **separate, reliable steps**, never in a single command: the kill breaks the ssh connection with an exit 255, and the grouped `drop_caches` never runs.

```bash
# 1. kill vLLM
pkill -f 'vllm serve' ; tmux kill-session -t laguna 2>/dev/null

# 2. CONFIRM that the process is really dead (separate step)
pgrep -af 'vllm serve'   # must return nothing

# 3. ONLY THEN, reclaim memory (separate step)
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches

# 4. check free RAM before relaunching
free -g

# 5. relaunch
tmux new-session -d -s laguna "bash ~/launch.sh > ~/laguna.log 2>&1"
```

Never run `drop_caches` while vLLM is running, never glue the kill and the drop in the same command.

## KV cache at 256K

At `--max-model-len 262144` with an **FP8** KV cache and `--gpu-memory-utilization 0.85`, vLLM allocates a **KV cache of 738,410 tokens**, i.e. **~2.82× concurrency** at the full 256K context. Sliding-window attention (36 layers out of 48, window 512) is decisive: only the 12 global layers carry the long KV, which makes the 256K context sustainable on a single box. Unlike DeepSeek-V4, we did **not** have to trim the context to fit the model: native 256K, from the first stable launch.

## Benchmark results

Throughput bench (`bench_laguna.py`, stdlib urllib), config of §5 (**DFlash**, `num_speculative_tokens=7`):

| Condition | Decode throughput |
|---|---|
| code, thinking OFF, 512 tok | 49.3 tok/s (min 44.3 / max 52.9) |
| code, thinking OFF, 1024 tok | 51.0 tok/s |
| code, thinking ON, 512 tok | 52.4 tok/s |
| prose, thinking OFF, 512 tok | 30.7 tok/s |

| Prefill (streaming TTFT) | |
|---|---|
| ~4k ctx (4,490 tok) | TTFT 1.43 s → 3,130 tok/s |
| ~16k ctx (17,820 tok) | TTFT 3.97 s → 4,483 tok/s |

| Concurrency (aggregate) | |
|---|---|
| code OFF 512, 2 connections | 54.3 tok/s aggregate |
| code OFF 512, 4 connections | 88.8 tok/s aggregate |

**Reading.** Decoding depends mostly on the type of content, not on *thinking* (code ON ≈ code OFF, ~49–52 tok/s). The cause is DFlash's **acceptance length**, which we measured step by step by reading the delta of the `vllm:spec_decode_*` counters from `/metrics` around isolated generations:

| Content | Drafts accepted / step | Acceptance length |
|---|---|---|
| code (predictable structure) | 4.36 | 5.36 tokens/step |
| prose | 2.12 | 3.12 tokens/step |

(acceptance length = 1 + drafts accepted/step, i.e. the number of tokens emitted per forward pass.) This ~1.7× factor between code and prose explains the code ~50 / prose ~30 tok/s gap: on code, each pass outputs more than 5 tokens. Prefill stays fast (3,000–4,500 tok/s) and the TTFT on a short prompt drops to **252 ms** (median of 5, min 245 / max 294) — the long context does not collapse the first token.

{{< callout label="Correction" >}}
An earlier version of this note claimed "DFlash accepts 2.9–3.1 tokens/step on code". That was in fact the **prose** figure, mislabelled; the isolated measurement through `/metrics` gives 5.36 tokens/step on code.
{{< /callout >}}

### Business bench — 209 cases, 5 conditions

Harness `bench_guided.py` (from a second Spark to the serving node `:8082`), DFlash speculative decoding active:

| Condition | Correct | Compliant | Hallu |
|---|---|---|---|
| A — free | 181/209 (86.6 %) | 99.5 % | 0 |
| C — guided_choice | 204/209 (97.6 %) | 100 % | 5 |
| D — retrieval + guided | 204/209 (97.6 %) | 100 % | 4 |
| E — strict retrieval + guided | 202/209 (96.7 %) | 100 % | 6 |
| F — retrieval + firewall (LP) | 205/209 (98.1 %) | 100 % | 4 |

**Zero hard errors** across the 5 conditions. Compliance is perfect (100 %) as soon as guidance is in place; in free mode (A), correctness falls back to 86.6 %, the expected gap between free decoding and constrained decoding. The logit-processor firewall (F) offers the best correctness/hallucination trade-off: 98.1 %, 4 hallucinations. The guided conditions hallucinate slightly more than DeepSeek-V4 on the same corpus (4 to 6 versus 1 to 4), without ever producing a hard error.

## Note — num_speculative_tokens: 7 rather than 15

At `num_speculative_tokens=15`, DFlash drafted too far: on low-acceptance content (prose), the majority of speculated tokens were rejected, and the wasted draft compute made throughput **drop**. Direct comparison, same config otherwise:

| Metric | spec=15 | spec=7 |
|---|---|---|
| prose OFF 512 | 26.0 tok/s | 30.7 tok/s (+18 %) |
| code OFF 512 | 49.6 tok/s | 49.3 tok/s (≈) |
| prefill ~4k | 3,004 tok/s | 3,130 tok/s (+4 %) |

spec=7 wins clearly on prose and matches on code: it is the value retained in the §5 script. A speculative drafter is tuned on the **acceptance profile** of the content served, not on the idea that "more tokens = better".

## The figures to remember

{{< keyfigures >}}
[
  {"label": "KV cache at 256K", "value": "738,410 tokens", "detail": "≈2.82× concurrency at full context, thanks to sliding-window (36/48 layers)"},
  {"label": "Decode throughput", "value": "49–52 tok/s", "detail": "code, DFlash speculative decoding active (num_speculative_tokens=7)"},
  {"label": "Short-prompt TTFT", "value": "252 ms", "detail": "median of 5 runs, long context has no effect on the first token"},
  {"label": "Business bench (209 cases)", "value": "98.1 %", "detail": "best condition (retrieval + LP firewall), zero hard errors across the 5 conditions"}
]
{{< /keyfigures >}}

A model that fits on a single box, a native 256K context and a speculative drafter that nearly doubles throughput on code: the hard part is not inference itself, but the memory hygiene of a unified architecture. Three safeguards are enough — `MAX_JOBS=4` at first boot, cautious margins on `--gpu-memory-utilization` and `--max-num-batched-tokens`, and a kill/drop_caches cycle in separate steps at every relaunch.

{{< callout label="We install and tune your DGX Spark." >}}
If this runbook saves you time but you would rather entrust the installation, the memory tuning and the move to production to someone who has already done it, write to us.
{{< /callout >}}

## Also read

- [DeepSeek-V4-Flash-0731 on two DGX Spark: the complete recipe](/blog/dsv4-2x-spark-recette/): 96/100 correct code over 100 draws judged by execution, 188.9 tok/s aggregate at sixteen streams, thermals and RDMA latency measured under load.
- [Running vLLM properly on a DGX Spark](/blog/dgx-spark-vllm-nvfp4/): the three bugs that block the build and the first server on the SM121 architecture, and how to fix them.


## Sources

- Measurements on our DGX Spark (GB10), July 2026: bench_laguna.py (throughput, prefill, concurrency), vllm:spec_decode_* counters from /metrics (acceptance length), bench_guided.py (business bench, 209 cases, 5 conditions)
- poolside/Laguna-S-2.1-NVFP4: model README, DGX Spark section (MAX_JOBS, CUTE_DSL_ARCH, FlashInfer pins, vLLM 0.25.1)

