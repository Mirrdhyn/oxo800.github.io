# DeepSeek-V4-Flash 0731 on 2× DGX Spark: Setup, Measurements, Tuning

> Twelve serving parameters ruled out of sixteen, a +31 s latency cliff with b12x, and 188.9 tok/s at sixteen streams with DeepSeek-V4-Flash on two DGX Spark.

- Date: 2026-08-04
- Language: en
- URL: https://0x0800.com/blog/dsv4-2x-spark-rapport/
- Tags: dgx-spark, deepseek-v4, vllm, tp2, local-inference
- Translation (fr): https://0x0800.com/fr/blog/dsv4-2x-spark-rapport/


{{< keyfigures >}}
[
  {"label": "Retained quality", "value": "96/100", "detail": "judged by execution, before any throughput"},
  {"label": "Single stream", "value": "78.55 tok/s", "detail": "code 512, at the ceiling imposed by k"},
  {"label": "Sixteen streams", "value": "188.9 tok/s", "detail": "the only setting that produced a gain"},
  {"label": "KV pool", "value": "1.79 M", "detail": "tokens, servable context 393,216"},
  {"label": "Parameters ruled out", "value": "12 of 16", "detail": "the most minimal configuration is the optimum"},
  {"label": "b12x cliff", "value": "+31 s", "detail": "of fixed cost from ~1850 tokens of context"}
]
{{< /keyfigures >}}

Reproducible report. It describes how to set up the stack end to end, which numbers you get, what each parameter changes, and which combination gives the best performance/quality trade-off.

All the numbers come from measurements made on the hardware described below. What is estimated or unverified is flagged as such.

## Hardware and Software

Two DGX Spark linked over RoCE, one GPU per node, tensor-parallel 2.

| | spark-1 (head) | spark-2 (worker) |
|---|---|---|
| role | `--node-rank 0`, serves the API | `--node-rank 1 --headless` |
| kernel | 6.17.0-1026-nvidia | same |
| CPU | Cortex-X925, 20 threads | same |
| RAM | 121.6 GiB | same |
| GPU | NVIDIA GB10, driver 595.84, clock ceiling 3003 MHz | same |
| CUDA | 13.2 | same |
| RoCE | 2 HCA × 200 Gb/s | same |

Software stack measured in this report (path A):

| component | version |
|---|---|
| Python | 3.12.13 |
| torch | 2.13.0+cu132 |
| NCCL | 2.29.7 |
| FlashInfer | 0.6.14 |
| vLLM | `jasl/vllm` fork, branch `b12x-nvfp4-mxfp4`, commit `48d274954` |
| checkpoint | `DeepSeek-V4-Flash-0731`, 155.43 GiB, MXFP4 experts, dense/attn FP8 E4M3 |

The GB10 is **sm_121**. Several code paths require datacenter Blackwell (sm_10x) and are therefore unavailable here (see §6).

## Path A: Building the Stack from the Fork

This is the path used for all the measurements in this report.

### 2.1 Why a fork

Upstream vLLM does not run this model on GB10 as is: `hyperconnection.hpp` rejects the architecture. The `jasl/vllm` fork carries sm_121 support and tracks upstream: the `b12x-nvfp4-mxfp4` branch integrates `upstream/main` as of 07/21/2026 and its ongoing fixes. This is the fork used here.

The NCCL that runs is **2.29.7**, the one torch 2.13.0+cu132 is linked against (`torch.cuda.nccl.version()`). The venv also contains an `nvidia-nccl-cu13 2.30.4` package, but torch does not load it: installing a newer version through pip does not change the effective NCCL. Check with the torch call, not with `pip list`.

The stack measured in this report ships **neither DeepGEMM nor an indexer fix**: `import deep_gemm` fails in the measurement venv, and no indexer file is modified. Both items belong to the NVFP4 requantized variant, which is a separate effort.

### 2.2 Steps

```bash
# 1. venv dédié, Python 3.12 (3.12.13 ici)
python3.12 -m venv ~/.venv-dsv4

# 2. torch pour GB10 (cu132) — pas de torchaudio, il casse la résolution
~/.venv-dsv4/bin/pip install torch==2.13.0+cu132 \
  --index-url https://download.pytorch.org/whl/cu132

# 3. fork vLLM
git clone https://github.com/jasl/vllm.git ~/vllm-jasl-dsv4
cd ~/vllm-jasl-dsv4 && git checkout b12x-nvfp4-mxfp4

# 4. LE seul correctif obligatoire : lever l'épinglage torch==2.11.0
git apply ~/models/_runbook/0001-depin-torch-cuda.patch

# 5. build — MAX_JOBS borne la RAM, c'est le réglage anti-OOM
MAX_JOBS=4 ~/.venv-dsv4/bin/pip install -e . --no-build-isolation
```

Step 4 is the one without which nothing starts: the fork pins `torch == 2.11.0` in `pyproject.toml`, `requirements/build/cuda.txt` and `requirements/cuda.txt`, whereas the GB10 needs 2.13.0+cu132. The patch removes these pins and the `torchaudio`/`torchcodec` dependencies that drag the resolution along with them. It touches no line of engine code.

**TP=2 constraint:** both nodes must have **matched** venvs: same source tree (same hash) and same torch version. A one-commit difference is enough to make tensor-parallel fail or drift. Check before each campaign:

```bash
for h in spark-1 spark-2; do
  ssh $h '~/.venv-dsv4/bin/python -c "import vllm,torch;print(vllm.__version__,torch.__version__)"'
done
```

### 2.3 Launch

The worker starts first, the head afterwards. **Always in tmux**: an `ssh + nohup` on a long process gets cut off, and a phantom restart overwrites the log.

```bash
# sur spark-2 (worker)
tmux new-session -d -s dsv4 -c ~ "ENVS ./dsv4_run.sh worker N0731 2>&1 | tee ~/worker.log"

# sur spark-1 (tête), 10 s plus tard
tmux new-session -d -s dsv4 -c ~ "APIHOST=127.0.0.1 APIPORT=8090 ENVS ./dsv4_run.sh head N0731 2>&1 | tee ~/head.log"
```

Effective serving arguments:

```args
--tensor-parallel-size 2 --distributed-executor-backend mp
--max-model-len 393216 --max-num-seqs 2 --max-num-batched-tokens 4096
--kv-cache-dtype fp8_ds_mla --block-size 256 --enable-prefix-caching
--gpu-memory-utilization 0.85
--speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"greedy"}'
--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'
```

Environment variables:

```env
CUTE_DSL_ARCH=sm_121a
VLLM_USE_FLASHINFER_SAMPLER=1
VLLM_USE_BREAKABLE_CUDAGRAPH=0
```

### 2.4 Clean shutdown

A `pkill -f dsv4_run.sh` is not enough and triggers on itself. The correct sequence:

```bash
tmux kill-session -t dsv4                       # sur les deux nœuds
pkill -9 -f "[b]in/vllm"; pkill -9 -f "[V]LLM::" # le traînard Worker survit sinon
sync; sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
```

The `drop_caches` before each launch avoids memory pressure that turns into a stuck startup (see §6.4).

## Path B: The Upstream Repository's Stack (Exact Recipe)

A third-party operator's repository publishes its numbers on a different runtime: the image `ghcr.io/anemll/dspark-vllm-gx10:0.1.1`, whose sources are in `Anemll/dspark-vllm-gx10`. Its `upstream.lock` pins everything:

```lock
VLLM_TAG=v0.25.1              VLLM_COMMIT=752a3a504485790a2e8491cacbb35c137339ad34
FLASHINFER_COMMIT=0472b9b3f2fba11b463f8526f390297d52a8aad7   FLASHINFER_VERSION=0.6.15
B12X_REPOSITORY=https://github.com/lukealonso/b12x.git
B12X_COMMIT=7dc6fb8fcc6446ea093537d1657df81985fa5f43         B12X_VERSION=0.15.3
CUTLASS_DSL_VERSION=4.5.2     CUDA_PYTHON_VERSION=13.3.1     TVM_FFI_VERSION=0.1.9
TORCH_VERSION=2.11.0+cu130    CUDA_VERSION=13.0.2            PYTHON_VERSION=3.12
```

**Important point: vLLM is not compiled there.** Their image starts from the official wheel `vllm/vllm-openai:v0.25.1` and applies an overlay of 12 Python files. Transposed to a native venv, without Docker:

```bash
python3.12 -m venv ~/.venv-anemll
~/.venv-anemll/bin/pip install "vllm==0.25.1"

~/.venv-anemll/bin/pip install \
  "cuda-python==13.3.1" \
  "nvidia-cutlass-dsl==4.5.2" \
  "nvidia-cutlass-dsl-libs-base==4.5.2" \
  "nvidia-cutlass-dsl-libs-cu13==4.5.2" \
  "apache-tvm-ffi==0.1.9"

~/.venv-anemll/bin/pip install --no-deps --no-build-isolation \
  "https://github.com/lukealonso/b12x/archive/7dc6fb8fcc6446ea093537d1657df81985fa5f43.tar.gz"

~/.venv-anemll/bin/pip install --no-deps --no-build-isolation --force-reinstall \
  "https://github.com/flashinfer-ai/flashinfer/archive/0472b9b3f2fba11b463f8526f390297d52a8aad7.tar.gz"

# overlay : 12 fichiers par-dessus le paquet installé
git clone https://github.com/Anemll/dspark-vllm-gx10.git ~/anemll
cp -r ~/anemll/overlay/vllm/. ~/.venv-anemll/lib/python3.12/site-packages/vllm/

export FLASHINFER_DISABLE_VERSION_CHECK=1
```

Exhaustive list of the overlay (12 files):

```files
vllm/envs.py
vllm/utils/torch_utils.py
vllm/config/vllm.py
vllm/config/cache.py
vllm/v1/kv_cache_interface.py
vllm/models/deepseek_v4/attention.py
vllm/models/deepseek_v4/sparse_mla.py
vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py
vllm/model_executor/layers/fused_moe/experts/b12x_mxfp4_moe.py
vllm/model_executor/layers/fused_moe/oracle/mxfp4.py
vllm/v1/attention/backends/mla/sparse_swa.py
vllm/v1/attention/backends/mla/flashmla_sparse.py
```

Code gap with path A, measured by `diff`:

| file | Anemll | jasl fork | diverging lines |
|---|---:|---:|---:|
| `b12x_mxfp4_moe.py` | 877 | 785 | 83 |
| `flashmla_sparse.py` | 895 | 876 | 201 |
| `models/deepseek_v4/attention.py` | 825 | 921 | 141 |
| `models/deepseek_v4/sparse_mla.py` | 408 | 470 | 80 |
| `envs.py` | 2,179 | 2,328 | 210 |
| `config/cache.py` | 294 | 293 | 26 |

{{< hbars title="Diverging lines between the Anemll overlay and the jasl fork" caption="The two most divergent files (b12x_mxfp4_moe.py, flashmla_sparse.py) are the ones that carry the behaviors measured in §5.2 and §5.3." minwidth=600 barheight=16 marginleft=230 >}}
{"bars": [{"label": "envs.py", "value": 210, "text": "210 lines", "color": "accent"}, {"label": "flashmla_sparse.py", "value": 201, "text": "201 lines", "color": "accent"}, {"label": "models/deepseek_v4/attention.py", "value": 141, "text": "141 lines", "color": "accent"}, {"label": "b12x_mxfp4_moe.py", "value": 83, "text": "83 lines", "color": "accent"}, {"label": "models/deepseek_v4/sparse_mla.py", "value": 80, "text": "80 lines", "color": "accent"}, {"label": "config/cache.py", "value": 26, "text": "26 lines", "color": "accent"}]}
{{< /hbars >}}

The two most divergent files (`b12x_mxfp4_moe.py`, `flashmla_sparse.py`) are the ones that carry the behaviors measured in §5.2 and §5.3.

### 3.1 Points verified at setup

**torch 2.11.0+cu130 works on GB10.** The wheel does not ship an sm_121 cubin:

```python
>>> torch.cuda.get_arch_list()
['sm_80', 'sm_90', 'sm_100', 'sm_110', 'sm_120']
>>> torch.cuda.get_device_capability(0)
(12, 1)
```

The GB10 is indeed sm_121, and support goes through sm_120 binary compatibility: a bfloat16 `matmul` on GPU gives the right result. No home-built torch is needed for path B, unlike path A.

Remaining caveat: `b12x` and `flashinfer` are installed with `--no-build-isolation` and therefore compile locally. They pin CUDA 13.0.2; on a machine running 13.2, compilation may diverge.

### 3.2 Upstream serving profile

Reproduced from their `docker-compose.dspark.yml`, to be put as is on a native `vllm serve`:

```args
--tensor-parallel-size 2 --distributed-executor-backend mp --nnodes 2
--kv-cache-dtype nvfp4_ds_mla --block-size 256
--max-model-len 1048576 --max-num-seqs 6 --max-num-batched-tokens 8192
--max-cudagraph-capture-size 24        # = max_num_seqs × (MTP+1)
--gpu-memory-utilization 0.85
--moe-backend flashinfer_b12x
--async-scheduling --enable-chunked-prefill
--generation-config vllm
--speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}'
```

Two details that matter:

- `MTP_NUM_TOKENS` is 3 by default in their compose, but their own README states that the checkpoint's `dspark_block_size` is 5 and that **k must be ≥ 5**. Use 5.
- `VLLM_USE_BREAKABLE_CUDAGRAPH=0` is **mandatory** here, otherwise their code switches by itself to the slow path (see §5.5).

### 3.3 Status

**Setup verified up to import**: `vllm 0.25.1`, `b12x 0.15.3`, `flashinfer-python 0.6.15`, overlay applied (11 files overwritten, 1 new, `b12x_mxfp4_moe.py`), imports of `b12x_mxfp4_moe` and `models.deepseek_v4.sparse_mla` successful.

{{< callout label="Actual status of path B" >}}
**As of August 3, 2026, path B has never been started or measured.** Replication on the second node, necessary for TP=2, has not been done. **All the numbers in this report are those of path A**, without exception. This section documents a setup verified up to import, nothing more. We publish it because the exact recipe is hard to reconstruct, not because it carries a result.
{{< /callout >}}

Reproducibility tip: back up every overwritten file before laying down the overlay, which lets you return to stock 0.25.1 without reinstalling.

```bash
cd overlay/vllm && find . -name '*.py' | while read f; do
  cp -n "$SP/$f" "$SP/$f.orig" 2>/dev/null
done
cp -r overlay/vllm/. "$SP/"
```

## Measurement Protocol

Three instruments, in this order. Throughput is never published without quality.

### 4.1 Quality gate: code benchmark

20 tasks (10 Python, 10 Go) × 5 draws, budget 16,000 tokens. Each output is compiled, vetted and tested. The score is the number of **correct** draws out of 100.

```bash
URL=http://127.0.0.1:8090/v1 MODEL=deepseek-v4-flash TAG=<bras> \
  python3 codebench/run.py --out resultats-<bras>.json
```

We also record the number of **unstable tasks** (neither 0/5 nor 5/5), which detects a degradation that the overall score masks.

### 4.2 Decoding throughput

Streaming probe, unique prefix per request to defeat the prefix cache, 5 long warm-up generations, then concurrency levels × repetitions.

Formula, which excludes prefill:

```formule
decode_tok_s = (tokens_out - 1) / (durée_totale - TTFT)
```

### 4.3 Time to first token

This is the instrument missing from most benchmarks, including the upstream repository's: both measure decoding **starting from** the first token. A prefill penalty is structurally invisible there.

The probe sends a prompt sized to the exact token via `/tokenize`, `max_tokens=16`, and times it up to the first delta, counting `content` **and** `reasoning_content`, without which the model reasons without ever emitting any `content` and the measurement returns nothing.

### 4.4 Machine telemetry

Sampling every 2 s: GPU power, SM clock, temperature, utilization, free RAM, RoCE throughput (counters in 4-byte words).

GB10 limitations to know about: `power.draw` only covers the GPU domain; `power.limit`, `memory.used` and `memory.total` return `[N/A]` (unified memory); no module power sensor in `/sys/class/hwmon`; no `dcgmi`.

## Results

### 5.1 Three combinations, quality and throughput

Same server, same checkpoint, one variable at a time.

| arm | MoE | KV cache | quality /100 | Python | Go | unstable | C1 tok/s | C2 tok/s | aggregate C2 | KV pool |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **A — baseline** | **MARLIN** | **`fp8_ds_mla`** | **98** | **50/50** | **48/50** | **1** | **47.86** | **35.80** | **68.71** | **1,859,337** |
| **B — b12x** | B12X_MXFP4 | `fp8_ds_mla` | **96** | 50/50 | 46/50 | 3 | **38.77** | 34.50 | 65.21 | 1,793,221 |
| **C — stacked** | B12X_MXFP4 | `nvfp4_ds_mla` | **92** | 48/50 | 44/50 | 6 | — | — | — | 1,811,142 |

Repetitions of arm B, to show that the gap is not noise: C1 = 38.74 / 38.77 / 39.53 (2% spread); C2 = 34.39 / 34.57 / 34.50.

{{< hbars title="Quality per arm, over 100 draws" caption="Quality first, measured by executing the code over 100 draws. Stacking b12x and nvfp4 costs arm C six points, and takes the unstable tasks from 1 to 6." minwidth=620 barheight=18 marginleft=160 max=100 marginright=170 >}}
{"bars": [{"label": "A — baseline", "value": 98, "text": "98/100 · 1 unstable", "color": "signal"}, {"label": "B — b12x", "value": 96, "text": "96/100 · 3 unstable", "color": "signal"}, {"label": "C — stacked", "value": 92, "text": "92/100 · 6 unstable", "color": "alert"}]}
{{< /hbars >}}

{{< groupedbars title="Throughput per arm, one and two streams" caption="Throughput of the two arms measured. b12x costs 19% at one stream; the gap narrows at two streams (−3.6%), which is consistent with a kernel built for large batches." minwidth=620 marginleft=150 >}}
{"series": [{"label": "C1", "color": "accent"}, {"label": "C2", "color": "signal"}, {"label": "aggregate C2", "color": "accent-deep"}], "groups": [{"label": "A — baseline", "values": [47.86, 35.8, 68.71], "texts": ["47.86", "35.80", "68.71"]}, {"label": "B — b12x", "values": [38.77, 34.5, 65.21], "texts": ["38.77", "34.50", "65.21"]}]}
{{< /groupedbars >}}

Readings:

- **b12x costs 19% of throughput at C1** in this regime. The gap narrows with concurrency (−19% at C1, −3.6% at C2), which is consistent with a kernel built for large batches. Hypothesis not tested beyond C2.
- **`nvfp4_ds_mla` costs quality without giving back memory.** The KV pool varies by 0.5% between B and C, within the allocation variability between two startups. The boot log says `padded nvfp4_ds_mla`: this is the 584-byte page, not the compact 416-byte layout. The score goes from 96 to 92 and Python drops to 48/50 for the first time.

### 5.2 Time-to-first-token latency cliff

Same probe, prompts sized to the exact token, 2 draws per point.

| actual tokens | A — MARLIN + fp8 | B — b12x + fp8 | C — b12x + nvfp4 |
|---:|---:|---:|---:|
| 1,760 | 1.15 / 1.14 | 1.17 / 1.19 | 0.95 / 1.07 |
| 1,793 | 1.05 / 1.03 | 1.07 / 1.06 | 1.08 / 1.09 |
| 1,826 | 1.05 / 1.05 | 1.07 / 1.08 | 1.10 / 1.10 |
| **1,856** | **1.06 / 1.06** | **31.85 / 31.90** | **32.43 / 32.43** |
| 1,889 | 1.07 / 1.07 | 32.27 / 32.61 | 33.13 / 32.73 |
| 1,922 | 1.08 / 1.08 | 32.60 / 32.88 | 33.00 / 33.04 |

{{< linechart title="Time to first token by prompt length" caption="Mean of the two draws of each point. A single variable separates A from B: the MoE backend. Enabling b12x adds ~31 seconds of fixed cost per request from ~1850 tokens of context, deterministically (spread under 1%)." minwidth=620 xlabel="actual prompt tokens" ymin=0 ymax=36 ydec=0 yunit=" s" ylabel="TTFT" >}}
{"xs": [1760, 1793, 1826, 1856, 1889, 1922], "xticks": [1760, 1793, 1826, 1856, 1889, 1922], "yticks": [0, 9, 18, 27, 36], "series": [{"label": "A — MARLIN + fp8", "points": [1.145, 1.04, 1.05, 1.06, 1.07, 1.08], "color": "signal"}, {"label": "B — b12x + fp8", "points": [1.18, 1.065, 1.075, 31.875, 32.44, 32.74], "color": "alert"}, {"label": "C — b12x + nvfp4", "points": [1.01, 1.085, 1.1, 32.43, 32.93, 33.02], "color": "accent-deep"}]}
{{< /linechart >}}

**A single variable separates A from B: the MoE backend.** Enabling b12x adds **~31 seconds of fixed cost per request** from ~1850 tokens of context, deterministically (spread under 1%). The KV cache has nothing to do with it: `nvfp4_ds_mla` gives the same result as `fp8_ds_mla` once b12x is active.

Beyond the step, the slope resumes: 1939 → 33.46 s, 2131 → 35.61 s, 2323 → 37.96 s, i.e. ~87 tok/s of marginal prefill.

Practical consequence: a file of a few thousand tokens trivially exceeds this threshold. With b12x, each request pays half a minute before the first word.

### 5.3 Effect of the measurement protocol

On the **same server, in the same minute, same stack** (arm B):

| benchmark | tok/s |
|---|---:|
| code benchmark (executable tasks) | 38.77 |
| **upstream repository script** | **55.35** |

{{< hbars title="Throughput by benchmark, identical stack" caption="+43% for a mere change of workload, on the same server in the same minute. A numbered enumeration is the best possible case for speculation; executable code is among the worst." minwidth=520 barheight=20 marginleft=250 >}}
{"bars":[
  {"label":"code benchmark (executable)","value":38.77,"text":"38.77 tok/s","color":"accent"},
  {"label":"upstream repository script","value":55.35,"text":"55.35 tok/s","color":"signal"}]}
{{< /hbars >}}

+43% for a mere change of workload. The upstream script asks for *"exactly 128 numbered lowercase English words"* over a repeated filler context: a numbered enumeration, which MTP speculation predicts very well.

Two characteristics of this workload explain the gap:

- The prompt starts with `unique request {nonce}`, which **deliberately defeats the prefix cache**. Each request pays its prefill.
- The output metric counts `reasoning_content`. The actual lengths observed range from 592 to 7,871 tokens for the same instruction: the inter-request variance is high.

**A numbered enumeration is the best possible case for speculation.** Executable code is among the worst. Both numbers are honest; they simply do not measure the same thing. This is why the rest of this report publishes its workloads along with its throughputs.

### 5.4 Two acceptance metrics not to be confused

The server exposes two quantities with similar names, in the same log block:

| metric | observed range | what it counts |
|---|---:|---|
| `acceptance rate` | 0.718 – 0.771 | acceptance length relative to `k` |
| `Draft acceptance rate` | 40.9 – 42.3% | draft tokens accepted out of emitted |

They do not have the same denominator and differ by a factor close to 1.8. Reading one while believing you are reading the other is enough to fabricate a gap that does not exist. The tables in this report use the second, and say so.

### 5.5 Breakable CUDA graphs

The repository's table is read **"classic graphs" column**, "breakable" column. Classic graphs are the fastest, and their README confirms it: Anemll automatically enables the slower breakable path when the variable is not set.

{{< callout label="Note of August 3, 2026" >}}
The values we quoted here (95.9 and 74.55 tok/s) come from an upstream publication that has since been replaced. The methodology note of the August 3 publication explicitly disowns them: "*Historical C1≈95.9 on older images is not this freeze*", and gives 79.2 tok/s for the current freeze. We therefore remove these two numbers from the body of the text: the reasoning of this section is about the **mechanism** of switching to the breakable path, which is verified in the code below and remains valid.
{{< /callout >}}

The difference is in the code, and it is clear-cut:

- **Path B (Anemll overlay)**: `config/vllm.py:1113-1131` auto-enables the breakable path when the architecture is `DeepseekV4ForCausalLM` or `DeepSeekV4MTPModel`, *unless* the variable is explicitly set. Without `VLLM_USE_BREAKABLE_CUDAGRAPH=0`, you get the slow configuration by construction. Hence the warning in their README.
- **Path A (jasl fork)**: same `False` default in `envs.py`, but the auto-enable list (`config/vllm.py:79-86`) only contains Inkling and MiniMax. DeepSeek V4 is not in it: **the fast configuration is already the default**, there is nothing to enable.

Worth noting if you test it anyway: enabling it **disables the torch.compile pipeline** (`vllm.py:1218`, equivalent to `-cc.mode=none`), and reduces the KV pool from 1,816,971 to 1,718,754 tokens (−5%), with captured graphs going from 0.12 to 0.34 GiB.

### 5.6 Telemetry under load

Decoding probe of arm B, 88 samples:

| quantity | median | min | max |
|---|---:|---:|---:|
| GPU power | 43.12 W | — | 45.70 W |
| SM clock | 2522 MHz | 2411 | 2541 |
| temperature | 68 °C | — | — |
| utilization | 95% | — | — |
| free RAM | 9.54 GiB | — | — |
| RoCE | 0.70 Gb/s | — | — |

No throttling (3003 MHz ceiling never approached but stable clock), no interconnect saturation (0.70 Gb/s on 200 Gb/s links). The b12x deficit is in the kernel, not in the platform.

Useful landmark: earlyoom's SIGTERM threshold is at ~4.86 GiB of free RAM, the observed margin is 9.54 GiB.

### 5.7 One-knob sweep

After establishing that `torch.compile` is not the lever (see 5.8), we take the throughput question back up through serving settings. Method: one knob at a time, same checkpoint, same server restarted for each arm, quality gate before any number.

Comparison instrument between arms: `~/bench_mono.py`, single stream, streaming, three fixed workloads. It applies the formula of 4.2. The numbers in this section compare **with each other**; they do not compare with the C1 columns of 5.1, which relate to the code benchmark workload.

Reference baseline of the sweep:

```bash
CGMODE=FULL_AND_PIECEWISE NOEP=1 SPECK=5 GPUUTIL=0.85 ./dsv4_run.sh <head|worker> N0731
```

| arm | quality | code 512 | code 1024 | prose 512 | C2 index | KV pool |
|---|---:|---:|---:|---:|---:|---:|
| **B0 — base** | **96/100 (k=5)** | **78.55** | **75.33** | **47.16** | **77.3** | **1,796,092** |
| **B1 — --async-scheduling** | 40/40 (k=2) | 78.30 | 76.64 | 48.19 | 76.9 | 1,830,803 |
| **B2 — SPECK=3** | refused at startup | — | — | — | — | — |
| **B3 — CGCAPTURE=12** | 39/40 (k=2) | 77.52 | 75.96 | 49.46 | 74.6 | 1,803,747 |
| **B4 — BATCHTOK=8192** | 40/40 (k=2) | 77.18 | 76.92 | 48.05 | 76.7 | 1,029,078 |
| **B5 — REJSAMPLE=block** | 38/40 (k=2) | 76.40 | 73.97 | 48.87 | 77.4 | 1,790,611 |

{{< groupedbars title="One-knob sweep, three workloads and concurrency index" caption="The five arms of the sweep, all measured the same way on the same burst shape. They compare with each other, not with the C1 columns of §5.1. B2 (SPECK=3) is refused at startup and has no figure." minwidth=660 marginleft=185 max=85 >}}
{"series": [{"label": "code 512", "color": "accent"}, {"label": "code 1024", "color": "accent-deep"}, {"label": "prose 512", "color": "signal"}, {"label": "C2 index", "color": "faint"}], "groups": [{"label": "B0 — base", "values": [78.55, 75.33, 47.16, 77.3], "texts": ["78.55", "75.33", "47.16", "77.3"]}, {"label": "B1 — --async-scheduling", "values": [78.3, 76.64, 48.19, 76.9], "texts": ["78.30", "76.64", "48.19", "76.9"]}, {"label": "B2 — SPECK=3", "values": [null, null, null, null], "texts": ["refused at startup", " ", " ", " "]}, {"label": "B3 — CGCAPTURE=12", "values": [77.52, 75.96, 49.46, 74.6], "texts": ["77.52", "75.96", "49.46", "74.6"]}, {"label": "B4 — BATCHTOK=8192", "values": [77.18, 76.92, 48.05, 76.7], "texts": ["77.18", "76.92", "48.05", "76.7"]}, {"label": "B5 — REJSAMPLE=block", "values": [76.4, 73.97, 48.87, 77.4], "texts": ["76.40", "73.97", "48.87", "77.4"]}]}
{{< /groupedbars >}}

{{< callout label="What the “C2 index” is, and is not" >}}
**The “C2 index” is not a two-stream throughput.** It is the mean of the `Avg generation throughput` lines of the server log filtered on `Running: 2`, over short bursts. Now this log line divides a number of tokens by a ten-second window, whatever happened inside it: a window where the two streams only ran half the time counts like the others and pulls the average down. On a sustained five-minute measurement, the same server gives 100.4 tok/s and the client counts the same to within 0.4% (§7.1).

The column remains useful for what it serves here, comparing arms with each other, all measured the same way on the same burst shape. It must not be read as the machine's capacity. The general trap: **a metric averaged over a fixed window only measures throughput if the load fills the window.**
{{< /callout >}}

The sweep arms pass the gate at k=2: here it serves to detect that a knob breaks the model, not to measure a spread. The retained arm goes back to k=5 before publication of its number.

B0, detail: Python 50/50, Go 46/50, zero truncation, inter-task standard deviation 0.08. The four unstable tasks are those already identified in 5.1 (`ParseMontant`, `MergeIntervals`, `MapConcurrent`, `RetryContexte`); the two-draw gap with arm A's 98 falls within the non-reproducibility documented at zero temperature.

Two lessons from the baseline arm alone:

**The code/prose gap is 1.67×** on the same machine in the same minute (78.55 versus 47.16). It is the clearest confirmation of 5.3: the workload weighs more than any setting we are going to test next. Any throughput announcement without a description of the workload is uninterpretable.

**TTFT weighs almost nothing at these lengths.** Gap between the formula excluding TTFT and a division by total duration: 5% (78.55 versus 74.69). The two protocols diverge, but the gap remains small compared with that of the workload.

Under load, the server log also gives the speculation metrics, which cannot be read anywhere else:

| quantity | observed range |
|---|---|
| mean acceptance length (out of k=5) | 3.10 → 3.87 |
| mean acceptance rate | 42.0% → 57.4% |
| drafted throughput | 75 → 113 tok/s |
| acceptance per position | 0.82 / 0.63 / 0.47 / 0.33 / 0.21 |

{{< hbars title="Acceptance per draft position" caption="Acceptance per draft position, at k=5. The fifth position is accepted only one time in five — but the decline cannot be exploited downward: num_speculative_tokens is bounded by the DSpark block size." minwidth=520 barheight=18 marginleft=150 max=100 dec=2 >}}
{"bars": [{"label": "position 1", "value": 82, "text": "0.82", "color": "accent"}, {"label": "position 2", "value": 63, "text": "0.63", "color": "accent"}, {"label": "position 3", "value": 47, "text": "0.47", "color": "accent"}, {"label": "position 4", "value": 33, "text": "0.33", "color": "accent"}, {"label": "position 5", "value": 21, "text": "0.21", "color": "accent"}]}
{{< /hbars >}}

The decline per position is striking: the fifth position is accepted only one time in five. But **it cannot be exploited downward**: `num_speculative_tokens` is bounded by the DSpark block size. At `SPECK=3`, the server refuses to start:

```log
DSpark requires num_speculative_tokens >= dspark_block_size (5); got 3.
Smaller values produce incorrect output. Use num_speculative_tokens=5 or larger.
```

The safeguard is in the code and it fails at opening rather than producing plausible and wrong text. `k=5` is therefore the **floor**. That it is also the optimum cannot be deduced from this decline: it had to be measured upward, which §7.4 does.

#### B1, `--async-scheduling`: the flag is already the default

Checked on the log of an arm launched *without* the flag, the line `Asynchronous scheduling is enabled.` is present anyway. Same for `enable_chunked_prefill=True`. **These two flags are not knobs in this version: enabling them explicitly changes nothing because they are already active.** Check in the startup log before wasting an arm on them.

B1 is therefore a duplicate of B0, and that is what makes it useful: two successive startups of the same configuration give the **reproducibility of the protocol**.

| workload | B0 | B1 | gap |
|---|---:|---:|---:|
| code 512 | 78.55 | 78.30 | 0.3% |
| code 1024 | 75.33 | 76.64 | 1.7% |
| prose 512 | 47.16 | 48.19 | 2.2% |
| C2 index | 77.3 | 76.9 | 0.5% |
| mean acceptance | 50.3% | 51.1% | 0.8 pt |

**Error bar between two startups: ±2%.** No arm can be declared a winner below this threshold. The KV pool sizes confirm it on another axis: 1,796,092 / 1,830,803 / 1,803,747 for configurations none of which touches memory.

#### B3, `CGCAPTURE=12`: single stream sees nothing, concurrency loses 3.5%

First arm where the error bar really serves to decide, and in both directions.

The `max_cudagraph_capture_size` knob **is not honored as is**: requested at 12, the engine brings it back to 8 (it only accepts sizes from its own grid). The list of captured sizes goes from `[1, 2, 4, 8, 16, 24]` to `[1, 2, 4, 8]`. Check in the startup log: the requested value is not the one applied.

| workload | B0 | B3 | gap |
|---|---:|---:|---:|
| code 512 | 78.55 | 77.52 | −1.3% |
| code 1024 | 75.33 | 75.96 | +0.8% |
| prose 512 | 47.16 | 49.46 | +4.9% |
| **C2 index** | **77.3** | **74.6** | **−3.5%** |

The three single-stream workloads are within the noise, including prose at +4.9%, which is only +2.6% from B1 and whose interval overlaps that of B0. **Without the B0/B1 duplicate, this +4.9% would have been announced as a gain.** That is the exact use of an error bar.

The aggregate at 2 streams, for its part, falls out of the noise downward, and the mechanism is legible: one stream in speculative decoding consumes 1 + 5 = 6 slots, so two streams require 12. With a capture ceiling at 8, no captured graph covers this batch anymore; with the original grid, size 16 absorbed it. Single stream (batch of 6) remains covered by size 8 and loses nothing.

**What this teaches about sizing the parameter:** the largest captured size must cover `max_num_seqs × (1 + num_speculative_tokens)`, i.e. 12 here. Going below saves capture memory and is paid for exactly where the machine is supposed to serve, under concurrency. The default grid is already correct; this knob is not a throughput lever, it is a memory lever.

B3 quality: 39/40, a single task failing, `compile+vet` at 40/40, zero truncation. No regression.

#### B4, `BATCHTOK=8192`: nothing in throughput, 43% less KV pool

This arm was launched on the strength of a warning from the engine itself:

```log
max_num_scheduled_tokens is set to 4088 based on the speculative decoding
settings. This may lead to suboptimal performance. Consider increasing
max_num_batched_tokens to accommodate the additional draft token slots.
```

**The warning does not close.** At 8192, it reappears word for word with `set to 8184`. It does not signal a bad sizing: it notes that the slots reserved for the draft (`max_num_seqs × num_speculative_tokens`) make what is schedulable lower than what was requested, which is true at any value. It is an unquenchable warning, not a recommendation. **General rule: check that a warning closes before treating it as a knob.**

| workload | B0 | B4 | gap |
|---|---:|---:|---:|
| code 512 | 78.55 | 77.18 | −1.7% |
| code 1024 | 75.33 | 76.92 | +2.1% |
| prose 512 | 47.16 | 48.05 | +1.9% |
| C2 index | 77.3 | 76.7 | −0.8% |
| mean acceptance | 50.3% | 49.4% | −0.9 pt |
| **KV pool** | 1,796,092 | **1,029,078** | **−43%** |

The three single-stream workloads are within the bar, with opposite signs: the very profile of a null effect. Quality 40/40, no regression.

The cost, for its part, is massive and legible: the prefill buffer is drawn from the same memory envelope as the KV cache, at constant `GPUUTIL`. Doubling the former cuts the latter by 43%, i.e. about 767,000 tokens of servable context traded for nothing.

**This parameter is not a throughput lever on this profile.** The reasoning says so a posteriori: at `max_num_seqs=2`, prefill never reaches 4096 tokens on the workloads tested, so the ceiling was not binding. It would only become so with many streams or very long prompts, and it is then the amputated KV pool that becomes the limiting factor.

#### B5, `rejection_sample_method=block`: no effect at zero temperature

Last knob of the sweep, and the only one to target the quantity still far from its ceiling: the acceptance rate, stuck at 50% for an acceptance length of 3.1 out of 5. The `block` mode verifies the draft tokens jointly (Sun et al.) instead of one by one. It was not exposed in `dsv4_run.sh`; it now is through `REJSAMPLE`, alongside `SPECK` and `DRAFTMETH`.

| workload | B0 | B5 | gap |
|---|---:|---:|---:|
| code 512 | 78.55 | 76.40 | −2.7% |
| code 1024 | 75.33 | 73.97 | −1.8% |
| prose 512 | 47.16 | 48.87 | +3.6% |
| C2 index | 77.3 | 77.4 | +0.1% |
| mean acceptance | 50.3% | 51.1% | +0.8 pt |

Gaps of opposite signs around the bar, aggregate identical to the tenth, acceptance identical. **The mode has no measurable effect on this configuration**, and above all it does not raise acceptance, which was its only reason for being here.

Mechanical hypothesis, **unverified**: the benchmark runs at zero temperature, where token-by-token verification and joint verification retain the same set of tokens. `block` would show a difference at non-zero temperature. Not tested.

The flag was indeed applied: `rejection_sample_method': 'block'` appears in the effective configuration at startup, and not in B0. The check was made before concluding, following the precedent of `CGCAPTURE`.

Quality 38/40, the lowest recorded in the sweep, on two compilation failures in Go. **It is not usable as a regression**: `ParseMontant` is one of the four tasks already unstable in 5.1, and a two-draw gap at k=2 falls within the non-reproducibility at zero temperature. Settling it would have required a pass at k=5, not spent for lack of a throughput gain to defend.

{{< callout label="Reading trap to know about" >}}
The same parameter accepts the value `synthetic`, which **accepts draft tokens according to an acceptance rate set by the user, without real verification**:

`'synthetic' accepts draft tokens with a decaying probability calibrated to synthetic_acceptance_rate.`

It is a scheduling simulation tool. It mechanically inflates the displayed throughput and the logged acceptance length, at the expense of the model's output. No measurement in this report uses it; all run in `standard`, except B5 in `block`. A throughput figure in speculative decoding is only interpretable accompanied by the verification mode, just like the workload.
{{< /callout >}}

{{< deviationbars title="Deviations from the baseline configuration, all arms and all workloads" caption="All the deviations from B0 published by the four tables above, on a single scale. The band covers the ±2% error bar established by the B0/B1 duplicate. The report only retains B3 under concurrency and B5 on code 512 as significant; positive excursions are discussed arm by arm in the text." minwidth=640 bound=2 max=5.5 >}}
{"rows": [{"label": "B1 · code 512", "value": -0.3}, {"label": "B1 · code 1024", "value": 1.7}, {"label": "B1 · prose 512", "value": 2.2}, {"label": "B1 · C2 index", "value": -0.5}, {"label": "B3 · code 512", "value": -1.3}, {"label": "B3 · code 1024", "value": 0.8}, {"label": "B3 · prose 512", "value": 4.9}, {"label": "B3 · C2 index", "value": -3.5}, {"label": "B4 · code 512", "value": -1.7}, {"label": "B4 · code 1024", "value": 2.1}, {"label": "B4 · prose 512", "value": 1.9}, {"label": "B4 · C2 index", "value": -0.8}, {"label": "B5 · code 512", "value": -2.7}, {"label": "B5 · code 1024", "value": -1.8}, {"label": "B5 · prose 512", "value": 3.6}, {"label": "B5 · C2 index", "value": 0.1}]}
{{< /deviationbars >}}

**Sweep summary: no knob produced a gain.** Five arms, two results outside the error bar and both negative (B3 under concurrency, B5 on code 512), three null results. Two knobs out of five did not do what their name announced: `--async-scheduling` already active by default, `CGCAPTURE` silently brought back to the engine's grid. And a third was launched on the strength of an engine warning that never closes.

**The baseline configuration is the serving optimum.** This result is consistent with the telemetry: the two GB10 hold 2463 MHz out of 3003, 95-96% utilization, 36 and 46 W. Decoding is limited by memory bandwidth, not by scheduling. No serving setting moves this ceiling; only the quantizer, the drafter and the hardware can.

### 5.8 torch.compile on DeepSeek-V4 NVIDIA: dead end, and why

`torch.compile` is the throughput lever you think of first on a vLLM stack. We tried to enable it on our platform. It does not enable, and the reason lies in the model: **DeepSeek-V4 is not wired for compilation on NVIDIA.**

- The official vLLM recipe launches DSv4 with `--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'` and without `--enforce-eager`. `cudagraph_mode` drives the capture of CUDA graphs, not Dynamo compilation. The model does not carry `@support_torch_compile`.
- The decorator exists in the tree, but on the **XPU** variant of the same model. The NVIDIA port did not take it over.

On the 0.25 line, DSv4 ships the CuTe DSL indexer (18 `.compile()` calls at runtime) and a multi-stream orchestration (8 sites in attention). Both are untraceable by Dynamo. The model traded compilation for hand-fused kernels: it is an implementation choice, not an oversight.

**The hard point, to know before trying:** `fullgraph=True` in `vllm/compilation/wrapper.py`. Under this setting, a Dynamo graph break is not a performance cost, it is a fatal error. The whole family "disable tracing on the offending function" (`torch.compiler.disable`, `dynamo.disable`) is therefore ineffective: Dynamo answers `Skip inlining torch.compiler.disable()d function` and stops.

Five patches, each unblocking one layer and revealing the next:

| # | patch | precedent in the tree |
|---|---|---|
| 1 | `@support_torch_compile` on `DeepseekV4Model` | yes (XPU variant) |
| 2 | six MHC sites switched to `torch.ops.vllm.*` | yes (`layers/mhc.py`) |
| 3 | registration of `mhc_pre_broadcast_tilelang` | no |
| 4 | workspace reservation at layer init | partial |
| 5 | memoization of `get_device_capability` by dict | no |

Blockers encountered in order, all of the same kind, Python code that inspects its own source or manipulates a C object at call time: `torch.accelerator.empty_cache()` in the workspace growth; `functools._lru_cache_wrapper` on `get_device_capability`; the CuTe DSL JIT of the indexer (`IndexerQFp8Kernel.compile`), reached via the multi-stream orchestration; then `inspect.getsourcelines` called by TileLang to decide its JIT mode, which falls back on `_patched_getfile`, explicitly marked untraceable by Dynamo.

The generic remedy for the third: make attention **opaque** by registering it as a custom op, on the upstream pattern `unified_attention_with_output`: `@eager_break_during_capture`, empty dummy implementation, `mutates_args=["out"]`, layer retrieved through its prefix in `static_forward_context` since a custom op cannot receive an `nn.Module`. It works: tracing goes through attention and dies further on, on `hc_head_fused_kernel_tilelang`: a seventh TileLang kernel to register.

Stopped there. The pattern is bottomless: every hand-fused kernel demands its own registered op. Bringing compilation back amounts to undoing, one by one, the fusions that make the fork's performance.

**Useful conclusion:** `torch.compile` is not the source of throughput on modern DSv4. Upstream does without it, and the recipe that keeps it pays a pinned version for a result of the same order. Throughput is to be found in the serving settings.

### 5.9 Context length and throughput: a curve you cannot measure naively

We had first swept context length the obvious way: have increasingly long texts summarized, output capped at 256 tokens, and read the decoding throughput. The series obtained went down from 52 tok/s at 16 k to 35 tok/s at 131 k. Thirty percent loss, and the break fell just above 65,536, exactly the checkpoint's YaRN threshold. The explanation wrote itself.

It was wrong. Two checks demolished it.

**The code first.** YaRN scaling is a cos/sin cache built once at initialization (`yarn_scaling_rope.py::_compute_cos_sin_cache`), over the whole range of positions. At runtime, it is a table lookup, with no branch depending on position. The per-token cost is identical at 1 k and at 131 k. A YaRN threshold cannot produce a length-dependent throughput drop, not because it is unlikely, but because there is no code path by which it would happen.

**The intermediate points next.** By adding 70 k, 80 k, 100 k and 115 k, the series stops being monotonic: 47.0 / 54.8 / 41.7 / 55.1 tok/s. And the 131 k point gives 35.1 then 39.6 on two runs. The gap between draws of the same point reaches 27.8%. The noise exceeds the effect sought: there was no curve.

**The fault is in the protocol.** Varying the context length by changing the text to summarize varies something else at the same time: the predictability of the output. Now the output is produced by speculation, and the throughput of speculative decoding is proportional to the number of tokens accepted per step. On this sweep, acceptance ranged from 34.0% to 63.6%, a factor of 1.87, which largely suffices to explain the factor of 1.32 observed on throughput. We were measuring the predictability of the text, believing we were measuring context length.

**The correct protocol holds the output constant.** The prompt is filler of variable length followed by "ignore the text above, copy the following passage word for word", with a fixed passage, at temperature 0. The output is then strictly identical at all lengths, verified at each point, a single distinct string over three draws. Acceptance is read from `/metrics` before and after each point, hence measured exactly over the interval.

| input | TTFT | prefill | decoding | acceptance | accept. length | steps/s |
|---:|---:|---:|---:|---:|---:|---:|
| 4,240 | 2.30 s | 1,846 tok/s | 76.32 | 88.9% | 4.44 | **14.02** |
| 16,520 | 8.67 | 1,905 | 71.05 | 82.1% | 4.10 | **13.92** |
| 32,880 | 18.12 | 1,814 | 68.27 | 77.8% | 3.89 | **13.97** |
| 65,680 | 40.16 | 1,636 | 61.23 | 69.5% | 3.47 | **13.69** |
| 80,120 | 49.69 | 1,612 | 79.09 | 93.2% | 4.66 | **13.97** |
| 100,120 | 63.60 | 1,574 | 76.26 | 91.1% | 4.56 | **13.73** |
| 131,200 | 87.28 | 1,503 | 84.03 | 100.0% | 5.00 | **14.00** |

{{< linechart title="Decoding steps per second by input length" caption="The last column of the table: the number of decoding steps per second, i.e. throughput divided by the speculation multiplier. Constant from 13.69 to 14.02, a 2.4% spread, below the ±2% error bar between two restarts." minwidth=620 xscale="log" xdiv="k" xdec=0 xlabel="input tokens" ymin=0 ymax=16 ylabel="steps/s" height=266 >}}
{"xs": [4240, 16520, 32880, 65680, 80120, 100120, 131200], "xticks": [4240, 16520, 32880, 65680, 131200], "yticks": [0, 4, 8, 12, 16], "band": {"from": 13.69, "to": 14.02, "label": "amplitude of the seven points: 2.4%"}, "series": [{"label": "steps per second", "points": [14.02, 13.92, 13.97, 13.69, 13.97, 13.73, 14.0], "color": "accent"}]}
{{< /linechart >}}

The last column is the number of decoding steps per second, i.e. throughput divided by the speculation multiplier `1 + acceptance length`.

**It is constant: 13.69 to 14.02, a 2.4% spread**, below the ±2% error bar between two startups. The result is simply stated:

> **Context length has no measurable effect on the speed of a decoding step between 4 k and 131 k tokens.** All the variation in throughput in tokens per second comes from the speculation acceptance.

{{< linechart title="Decoding throughput and acceptance by input length" caption="Throughput follows acceptance, curve for curve. The longest point is in fact the fastest: 84.0 tok/s at 131,200 input tokens, with 0.8% spread between the three draws, because the copy is accepted at 100% there." minwidth=620 xscale="log" xdiv="k" xdec=0 xlabel="input tokens" ymin=50 ymax=95 ylabel="decoding, tok/s" rmin=60 rmax=105 rdec=0 runit="%" rlabel="acceptance" >}}
{"xs": [4240, 16520, 32880, 65680, 80120, 100120, 131200], "xticks": [4240, 16520, 32880, 65680, 131200], "yticks": [50, 60, 70, 80, 90], "rticks": [60, 70, 80, 90, 100], "series": [{"label": "decoding, tok/s", "points": [76.32, 71.05, 68.27, 61.23, 79.09, 76.26, 84.03], "color": "accent"}, {"label": "acceptance, %", "points": [88.9, 82.1, 77.8, 69.5, 93.2, 91.1, 100.0], "color": "signal", "axis": "right"}]}
{{< /linechart >}}

The longest point is in fact the fastest: 84.0 tok/s at 131,200 input tokens, with 0.8% spread between the three draws, because the copy is accepted at 100% there. Any reading that makes 131 k a degraded regime is mistaken.

What really depends on length is prefill: 1,846 tok/s at 4 k versus 1,503 at 131 k, **−19%**, and above all 87 seconds of waiting before the first token. This is where a long-context agent is sized, and it is what the prefix cache avoids paying again (§5.6).

**Why it is constant: attention reads a fixed budget.** This constancy is not luck, it is the expected signature of DeepSeek-V4's sparse attention. The checkpoint declares `index_topk: 512`. At decoding time, the `_fp8_decode` path of `vllm/v1/attention/backends/mla/flashmla_sparse.py` passes the kernel a `topk_indices` tensor of shape `(num_decodes, seq_len, 512)`: whatever the context length, **attention only reads 512 KV cache entries**, the ones the indexer selected. The startup log confirms the active path:

```log
DeepSeek V4: using official FlashInfer SM120 packed sparse-MLA decode
DSA indexer decode path: use_flattening=True (next_n=6, ...)
```

A decoding step therefore costs reading the weights plus a bounded attention workload, independent of length. On dense attention, the same sweep would give a clear decline. This is a point to remember before transposing these numbers to another model: **the result holds for sparse attention with a fixed budget, not for any architecture.**

**A one-parameter law.** The seven points boil down to:

> `throughput ≈ 13.90 × (1 + tokens accepted per step)`

{{< linechart title="Throughput predicted by the law versus measured throughput" caption="Prediction of the law versus measurement, over the seven lengths. Maximum deviation 1.5%. A step lasts 72 ms." minwidth=620 xscale="log" xdiv="k" xdec=0 xlabel="input tokens" ymin=55 ymax=90 ylabel="tok/s" height=286 >}}
{"xs": [4240, 16520, 32880, 65680, 80120, 100120, 131200], "xticks": [4240, 16520, 32880, 65680, 131200], "yticks": [55, 65, 75, 85], "series": [{"label": "measured", "points": [76.32, 71.05, 68.27, 61.23, 79.09, 76.26, 84.03], "color": "accent"}, {"label": "predicted by the law", "points": [75.66, 70.93, 67.92, 62.17, 78.65, 77.2, 83.38], "color": "alert"}]}
{{< /linechart >}}

Prediction versus measurement, over the seven lengths: 75.66 / 70.93 / 67.92 / 62.17 / 78.65 / 77.20 / 83.38 versus 76.32 / 71.05 / 68.27 / 61.23 / 79.09 / 76.26 / 84.03. **Maximum deviation 1.5%.** A step lasts 72 ms.

Only one thing moves this coefficient, and it is no scheduling setting: `k`. At `k=7` the step goes to 82.5 ms and the law becomes `12.12 × (1 + accepted)`, remaining just as constant over the context range (§7.4). The term measures the verification cost of a step, nothing else.

**Direct consequence: the single-stream ceiling is set by `k`.** With `num_speculative_tokens = 5`, throughput cannot exceed `13.90 × 6 = 83.4` tok/s. The 131 k point, accepted at 100%, measures 84.03: **we are at the ceiling, not near it.** This retrospectively explains the sweep result (§5.7): no scheduling setting could win, because none touches this wall. The only dimension that moves it is `k` itself, and it moves it downward (§7.4).

**What to take away beyond this machine.** On a speculative decoding engine, any sweep that lets the output content vary measures acceptance before anything else. A "throughput as a function of X" curve is only interpretable if X is the only thing that varies, and publishing acceptance next to throughput is the minimum for a reader to be able to check. We had written the wrong curve and the YaRN explanation that went with it; they are removed from this report, and this section replaces them.

## Pitfalls and Parameters That Do Nothing

### 6.1 Documented but inert environment variables

The profile published upstream lists about fifteen variables. Checked in the installed runtime (`grep` over the complete `site-packages` tree):

- **`VLLM_USE_B12X_MOE`** is declared in `envs.py` (lines 50 and 1021) and read by **no** Python file, neither in the fork nor in the Anemll overlay. The MoE backend is chosen only with `--moe-backend flashinfer_b12x`.
- **`VLLM_USE_B12X_WO_PROJECTION`**, **`VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK`**, **`VLLM_DSPARK_LOCAL_ARGMAX`**, **`VLLM_DSPARK_REPLICATE_MARKOV_W1`**, **`VLLM_DSPARK_FUSED_MARKOV_ARGMAX`**, **`VLLM_DSPARK_CONFIDENCE_SCHEDULER`**, **`VLLM_DSV4_B12X_COMPRESSED_MLA`** are **not even declared** in their own `envs.py`. They have no effect on runtime 0.1.1.
- **`B12X_W4A16_TC_DECODE`** is the only one with a real consumer, and it is outside vLLM: `b12x/integration/tp_moe.py` and `b12x/moe/fused/w4a16/kernel.py`.

Practical conclusion: do not spend time tuning these variables, and check in the boot log that the option actually took effect.

### 6.2 b12x requires `NOEP=1`

`B12xExperts._supports_parallel_config` (`experts/b12x_mxfp4_moe.py:504`) requires `not use_ep and not use_all2all_kernels and not enable_eplb`. With expert parallelism active, **the backend is silently ignored** and MARLIN is used in its place. Without checking the boot log, you believe you are measuring b12x and you are measuring MARLIN.

### 6.3 `max_cudagraph_capture_size` is not a CLI flag

It is a field of `CompilationConfig`. It must go **inside** the `--compilation-config` JSON. The upstream profile sets it to `MAX_NUM_SEQS × (MTP+1)`, i.e. 36 at k=5 and 6 slots. Note: vLLM already derives this value by itself: a boot without the parameter displays the same `FULL=6 (largest=36)`.

### 6.4 Stuck startup after `Profiling CUDA graph memory`

Symptoms: `shm_broadcast.py:705 No available shared memory broadcast block found in 60 seconds` in a loop, GPU at 96% utilization but only 12 W (versus 40–45 W under real load), two threads at ~100% CPU, and **no write to the compilation cache**, which distinguishes an active wait from a legitimate compilation.

Fix: full shutdown of both nodes following the sequence of §2.4, `drop_caches`, restart. In our trials the block did not reproduce after cleanup; the exact cause is not established.

### 6.5 Paths unavailable on GB10

- **`use_fp4_indexer_cache`**: an assertion (`v1/attention/backends/mla/indexer.py:307-314`) requires datacenter Blackwell sm_10x. The GB10 is sm_121: the flag is unusable, regardless of the KV cache chosen.
- **GPUDirect RDMA**: impossible on GB10 (unified memory). Force `NCCL_NET_GDR_LEVEL=0`.
- **FlashInfer in attention**: not supported, force `TRITON_ATTN`, in both places, including for the drafter.

### 6.6 Cost of slots: it is not `max-num-seqs` that costs

An earlier version of this report wrote that `--max-num-seqs 6` made the KV pool fall from ~1.8 M to 1,016,517 tokens. **That was an attribution error, and we correct it here**: the log of this arm (`dsv4_seqs6b_head_20260802.log`) shows that `max_num_batched_tokens` was **8192** at the same time. Two variables changed, only one blamed.

Arm B4 of the sweep (§5.7) had already isolated the real cause: `BATCHTOK=8192` alone, at `max-num-seqs 2`, gives 1,029,078 tokens, the same figure to within 1.2%. The prefill buffer is drawn from the same envelope as the KV cache.

Direct measurement of the isolated parameter, at constant `BATCHTOK=4096`:

| `max-num-seqs` | KV pool | gap |
|---:|---:|---:|
| 2 | 1,793,656 | reference |
| 16 | 1,717,797 | **−4.2%** |

{{< hbars title="KV pool by isolated parameter" caption="The memory cost of slots is marginal (−4.2%), whereas doubling the prefill batch cuts the pool by 43%. It is the opposite of what one spontaneously assumes." minwidth=520 barheight=18 marginleft=230 marginright=110 max=1900000 dec=0 >}}
{"bars":[
  {"label":"max-num-seqs 2","value":1793656,"text":"1,793,656","color":"signal"},
  {"label":"max-num-seqs 16","value":1717797,"text":"1,717,797","color":"signal"},
  {"label":"max-num-batched-tokens 8192","value":1029078,"text":"1,029,078","color":"alert"}]}
{{< /hbars >}}

**Increasing the number of slots is almost free in memory; increasing the prefill batch costs 43%.** It is the opposite of what one spontaneously assumes, and the opposite of what this report claimed.

The real cost of slots is elsewhere: in CUDA graph capture, which the engine extends by itself from 6 to 27 sizes (up to 192) between 2 and 16 slots, and in per-stream throughput (§7.1).

## Recommended Combination

On this platform, with path A, the best performance at high quality is the **most minimal** configuration:

```bash
CGMODE=FULL_AND_PIECEWISE   # graphes CUDA complets + piecewise
NOEP=1                      # pas d'expert parallelism
SPECK=5                     # MTP k=5
GPUUTIL=0.85
KVDTYPE=fp8_ds_mla          # PAS nvfp4_ds_mla
# PAS de --moe-backend : on garde MARLIN
VLLM_USE_BREAKABLE_CUDAGRAPH=0
CUTE_DSL_ARCH=sm_121a
```

### 7.1 What this configuration gives

**Quality**, measured before any throughput figure, on 20 executable tasks × 5 draws, code actually compiled and tested:

| | correct | compile+vet | truncations |
|---|---:|---:|---:|
| Python | 50/50 | 50/50 | 0 |
| Go | 46/50 | 47/50 | 0 |
| **Total** | **96/100** | **97/100** | **0** |

Four unstable Go tasks, always the same ones (`ParseMontant`, `MergeIntervals`, `MapConcurrent`, `RetryContexte`). Inter-task standard deviation 0.08.

**Throughput**, single stream, streaming, excluding TTFT:

| workload | tok/s |
|---|---:|
| **code, 512 tokens** | **78.55** |
| code, 1024 tokens | 75.33 |
| prose, 512 tokens | 47.16 |

**Throughput at two concurrent streams: 100.4 tok/s aggregate**, i.e. ~50.2 per stream. Measured over five minutes of sustained load, 108 requests, 30,558 tokens in 304.4 s of wall time. The server log gives 100.8 over the same windows: the two metrics agree to within 0.4%.

Two streams is the ceiling of this configuration, which serves `--max-num-seqs 2`. Beyond that, requests wait for a slot.

{{< callout label="Correction" >}}
**This ceiling is a parameter, not a property of the machine, and we were wrong about it.** An earlier version of this report concluded, on the strength of a scale measured at `max-num-seqs 2`, that "beyond two streams you do not buy throughput, you buy waiting". That was true of the configuration, and false of the platform. Two protocol defects had made it look unassailable: the only variable available was capped at 2, and the bursts used natural completions of very uneven lengths (`[3342, 640, 5498, 7832]` on one draw), so a slot freed early stayed empty until the end of the batch.
{{< /callout >}}

**Corrected protocol.** Output fixed at exactly 512 tokens (`ignore_eos`), T=0, unique prefix per request, three draws per point: all the slots of a batch free up together, there is no straggler anymore. A single variable changes between the two series, `--max-num-seqs`, everything else being the retained configuration.

| streams | `seqs=2` | median TTFT | `seqs=16` | median TTFT |
|---:|---:|---:|---:|---:|
| 1 | 40.05 | 0.23 s | 43.15 | 0.25 s |
| 2 | 63.44 | 0.39 s | 62.80 | 0.35 s |
| 4 | 64.48 | **7.54 s** | **88.11** | 0.32 s |
| 6 | 62.63 | **15.58 s** | **116.51** | 0.51 s |
| 8 | 59.81 | **24.25 s** | **134.30** | 0.44 s |
| 16 | 60.34 | **56.64 s** | **188.88** | **0.75 s** |

{{< linechart title="Aggregate throughput by number of streams" caption="Aggregate throughput. The two series overlap at C1 and C2, as expected: the two configurations do the same thing there. They diverge exactly where `max-num-seqs 2` saturates." minwidth=620 xlabel="concurrent streams" ymax=205 ylabel="aggregate tok/s" >}}
{"xs": [1, 2, 4, 6, 8, 16], "xticks": [1, 2, 4, 6, 8, 16], "yticks": [0, 50, 100, 150, 200], "series": [{"label": "seqs=16", "points": [43.15, 62.8, 88.11, 116.51, 134.3, 188.88], "color": "accent"}, {"label": "seqs=2", "points": [40.05, 63.44, 64.48, 62.63, 59.81, 60.34], "color": "alert"}]}
{{< /linechart >}}

{{< linechart title="Time to first token by number of streams" caption="Median latency before the first token, logarithmic scale. At seqs=2 it goes from 0.39 s to 56.6 s, and up to 121.6 s at worst. At seqs=16 it stays under 0.8 s at every level." minwidth=620 xlabel="concurrent streams" yscale="log" ymin=0.2 ymax=80 ydec="auto" ylabel="median TTFT, s" >}}
{"xs": [1, 2, 4, 6, 8, 16], "xticks": [1, 2, 4, 6, 8, 16], "yticks": [0.2, 1, 5, 20, 80], "series": [{"label": "seqs=2", "points": [0.23, 0.39, 7.54, 15.58, 24.25, 56.64], "color": "alert"}, {"label": "seqs=16", "points": [0.25, 0.35, 0.32, 0.51, 0.44, 0.75], "color": "accent"}]}
{{< /linechart >}}

The two series overlap at C1 and C2, as expected: the two configurations do the same thing there. They diverge exactly where `max-num-seqs 2` saturates:

- At `seqs=2`, the aggregate is **flat from C2 to C16** (63.4 → 60.3) while the wait before the first token goes from 0.39 s to 56.6 s, and up to 121.6 s at worst. All you buy is indeed waiting, but because the parameter imposes it.
- At `seqs=16`, the aggregate **climbs monotonically up to 188.88 tok/s**, i.e. 3.1× the previous ceiling, and TTFT stays **under 0.8 s at every level**.

The price is per-stream throughput: 44.1 tok/s alone, 13.1 at sixteen streams. You trade per-user latency for total throughput, which is the normal trade-off of a server, and there was no reason to believe it did not exist here. The memory cost is negligible (§6.6): 1,717,797 tokens of KV pool against 1,793,656, i.e. −4.2%.

**What this changes for sizing.** For a single agent, `--max-num-seqs 2` gives the best latency and the best single-stream throughput. To serve several users, raising the parameter is the most profitable lever of this entire report: the only one, in reality, that produced a gain. The rest of our single-stream measurements remains valid as is, since they do not depend on this parameter.

**On the sustained two-stream load.** The 100.4 tok/s quoted above come from a five-minute load in continuous regime, not a burst: the slots are never empty there. This is why they exceed the 63.4 tok/s of the scale above, which measures short bursts. The two do not contradict each other, they measure two regimes, and the benchmark workload is not the same (§5.3).

**Speculation**: acceptance 50.3%, acceptance length 3.10 to 3.87 out of k=5, draft stable around 111 tok/s.

**Capacity**: 1.8 M tokens of KV cache, servable context 393,216 tokens.

**Long context: the cost of the first token**, single stream, output fixed at 256 tokens, three draws per point, unique prefix per draw to neutralize the prefix cache:

| input | TTFT | prefill |
|---:|---:|---:|
| 308 | 0.38 s | 810 tok/s |
| 1,068 | 0.61 s | 1,751 |
| 4,148 | 2.25 s | 1,844 |
| 16,428 | 8.58 s | 1,915 |
| 65,588 | 40.13 s | 1,634 |
| 131,108 | 87.28 s | 1,502 |

{{< linechart title="Prefill and TTFT by input length" caption="Prefill peaks around 1,900 tok/s near 16 k of input then comes back down. Serving 131 k of context costs 87 seconds before the first token: this is the real cost of long context, and it is what the prefix cache avoids paying again." minwidth=620 xscale="log" xdiv="k" xdec=0 xlabel="input tokens" ymin=600 ymax=2000 ydec=0 ylabel="prefill, tok/s" rmin=0 rmax=100 rdec=0 runit=" s" rlabel="TTFT" >}}
{"xs": [308, 1068, 4148, 16428, 65588, 131108], "xticks": [308, 1068, 4148, 16428, 65588, 131108], "yticks": [600, 950, 1300, 1650, 2000], "rticks": [0, 25, 50, 75, 100], "series": [{"label": "prefill, tok/s", "points": [810, 1751, 1844, 1915, 1634, 1502], "color": "accent"}, {"label": "TTFT, s", "points": [0.38, 0.61, 2.25, 8.58, 40.13, 87.28], "color": "alert", "axis": "right"}]}
{{< /linechart >}}

Prefill peaks around 1,900 tok/s near 16 k of input then comes back down. Serving 131 k of context costs 87 seconds before the first token: this is the real cost of long context, and it is what the prefix cache avoids paying again (§5.6).

**Decoding throughput as a function of context length is not measurable with this protocol**, and we therefore do not publish a curve. The detail of this dead end is in §5.9: it is worth reading before trusting a curve of this type, published by us or by anyone else.

**Machine**: the two GB10 hold 2463 MHz out of a ceiling of 3003, 95-96% utilization, 36 and 46 W, 66 and 73 °C.

### 7.2 Complete matrix: what pays and what does not

Each row is an arm measured on this platform, one parameter at a time, server restarted, quality gate before any number.

| parameter tested | effect on throughput | effect on quality | retained |
|---|---|---|---|
| `CGMODE=FULL_AND_PIECEWISE` | reference | 96/100 | ✅ |
| `NOEP=1` (no expert parallelism) | positive in single stream | — | ✅ |
| `SPECK=5` | **optimum demonstrated (see §7.4)** | 96/100 | ✅ |
| `GPUUTIL=0.85` | KV pool 1.8 M | — | ✅ |
| `--moe-backend flashinfer_b12x` | **−19%** at C1, +31 s beyond ~1850 tokens | — | ❌ |
| `--kv-cache-dtype nvfp4_ds_mla` | null | **−6 points** | ❌ |
| `--async-scheduling` | null (already the default) | — | ➖ |
| `--enable-chunked-prefill` | null (already the default) | — | ➖ |
| `SPECK=3` | refused at startup | — | ❌ |
| `SPECK=7` | **−11.5% to −17.1%** | 98/100 | ❌ |
| `CGCAPTURE=12` | **−3.5%** under concurrency | 39/40 | ❌ |
| `BATCHTOK=8192` | null, **−43% of KV pool** | 40/40 | ❌ |
| `REJSAMPLE=block` | null, acceptance unchanged | 38/40 | ❌ |
| `torch.compile` | **cannot be enabled (see 5.8)** | — | ❌ |

Twelve parameters ruled out, four retained, and the four retained are those of the start. **The most minimal configuration is the optimum.**

Single stream is capped by `k` (§5.9, §7.3), and it is the only dimension where a gain remained conceivable. It is measured: `SPECK=7` costs 12.8% of step speed without returning a single extra accepted token (§7.4). `SPECK=5` is therefore not only the floor imposed by `dspark_block_size`: it is the demonstrated optimum.

Two results deserve to be singled out because they are counter-intuitive:

- **`flashinfer_b12x` costs 19% of throughput and a latency cliff.** The MARLIN kernel, which is highlighted nowhere, wins on both tables.
- **`nvfp4_ds_mla` costs 6 points of quality without saving memory.** The expected gain of a more compact format does not materialize here.

### 7.3 Why nothing else pays

The telemetry gives the answer and it is consistent with the entire matrix: at 95-96% utilization, 2463 MHz and 36-46 W, decoding is limited by **memory bandwidth**, not by compute or by scheduling.

It is structural for this type of workload: a decoding token rereads all the active weights to produce a single token. No serving setting moves this ceiling. The only levers that remain are of another nature: the quantization format, the quality of the speculative drafter, and the hardware.

This also explains why speculation is profitable: it is the only mechanism tested that produces several tokens for a single reread of the weights. It returns 3.1 tokens per pass instead of 1.

**And why the sweep was looking in the wrong dimension.** The law established in §5.9 (`throughput ≈ 13.90 × (1 + tokens accepted per step)`, a step at 72 ms) says that single stream is capped at `13.90 × (1 + k)`. At `k = 5`, that makes 83.4 tok/s, and our best point measures 84.03. None of the five scheduling knobs tested touches this term: they could not win.

**The only remaining avenue was `k > 5`. It is measured, and it is negative** (see §7.4).

### 7.4 SPECK=7: the only possible lever, measured, and it costs

The law of §5.9 says that single stream is capped by `k`. The engine requires `num_speculative_tokens >= dspark_block_size` (5 here) and its own error message suggests a higher value:

```log
DSpark requires num_speculative_tokens >= dspark_block_size (5); got 3.
Use num_speculative_tokens=5 or larger (e.g. 7).
```

**What we already knew, and why it was not enough.** An earlier campaign (August 1st, `PIECEWISE + EP` baseline, published in the tutorial) had measured `k=7` and `k=10` and found them losing: 38.92 and 23.90 tok/s at small context against 47.96 at `k=5`. Two caveats prevented concluding anything for the configuration retained here. The baseline was not the same, neither `FULL_AND_PIECEWISE` nor `NOEP=1`. And above all the only explanation put forward was "acceptance collapses", read off a rate, without ever separating the cost of the step from the number of tokens returned. The one-knob sweep of §5.7, for its part, measured no value other than 5: its only speculative arm aimed at `SPECK=3` and never started.

We therefore brought up a `SPECK=7` arm on the retained baseline, a single parameter changed, server restarted, quality gate first, this time to obtain the mechanism, not just the sign.

**The engine adapts graph capture by itself.** No need to touch `CGCAPTURE`: at `k=5` the default gives `cudagraph_capture_sizes [1,2,4,8,16,24]`, at `k=7` it goes to `[1,2,4,8,16,24,32]`. It covers `max_num_seqs × (1 + k)` in both cases. Forcing `CGCAPTURE=12` on the contrary fell back to 8, below the default: that is the explanation of arm B3.

**Quality: intact.** 98/100 (Python 50/50, Go 48/50), 99/100 compile+vet, zero truncation, against 96/100 for the reference. Verification was necessary, because the engine warns that an invalid `k` produces corrupted output rather than degraded acceptance. At `k=7` the output is sound. KV pool 1,788,784 tokens against 1,796,092, i.e. −0.4%.

**Throughput: clearly negative.**

| workload | k=5 | k=7 | gap |
|---|---:|---:|---:|
| code, 512 tokens | 78.55 | 65.14 | **−17.1%** |
| code, 1024 tokens | 75.33 | 65.95 | **−12.4%** |
| prose, 512 tokens | 47.16 | 41.75 | **−11.5%** |

{{< groupedbars title="Throughput at k=5 and k=7, three workloads" caption="Three workloads, a single parameter changed on the retained baseline. k=7 loses 11.5% to 17.1% depending on the workload, with quality intact." minwidth=600 marginleft=170 >}}
{"series": [{"label": "k=5", "color": "signal"}, {"label": "k=7", "color": "alert"}], "groups": [{"label": "code, 512 tokens", "values": [78.55, 65.14], "texts": ["78.55", "65.14"]}, {"label": "code, 1024 tokens", "values": [75.33, 65.95], "texts": ["75.33", "65.95"]}, {"label": "prose, 512 tokens", "values": [47.16, 41.75], "texts": ["47.16", "41.75"]}]}
{{< /groupedbars >}}

**The cause, decomposed by the law of §5.9.** Taking the constant-output protocol back up over the seven lengths:

| | k=5 | k=7 |
|---|---:|---:|
| steps per second | 13.90 | **12.12** |
| duration of a step | 72.0 ms | **82.5 ms** |
| **tokens accepted per step** | **4.30** | **4.30** |

{{< groupedbars title="Throughput breakdown at k=5 and k=7" caption="The step costs 12.8% more at k=7, and the number of tokens accepted per step does not move by a hundredth. You pay more for verification without gaining anything in output: the DSpark drafter saturates around 4.3 accepted tokens." minwidth=560 marginleft=210 max=15 >}}
{"series": [{"label": "k=5", "color": "signal"}, {"label": "k=7", "color": "alert"}], "groups": [{"label": "steps per second", "values": [13.9, 12.12], "texts": ["13.90", "12.12"]}, {"label": "tokens accepted per step", "values": [4.3, 4.3], "texts": ["4.30", "4.30"]}]}
{{< /groupedbars >}}

Two facts, and the second is decisive:

- The step costs 12.8% more: verifying eight positions instead of six adds 10.5 ms. Gap measured over the seven lengths: −10.9% to −15.8%.
- **The number of tokens accepted per step does not move: 4.30 in both cases.** The DSpark drafter saturates around 4.3 accepted tokens. Asking it for seven does not make it return more.

You therefore pay more for verification without gaining anything in output. Acceptance per position (0.82 / 0.63 / 0.47 / 0.33 / 0.21) left hope for a residual value at positions 6 and 7; in practice it is absorbed by the cost of the step.

**And this is where the trap of §5.4 closes.** Saying "acceptance collapses at `k=7`" is accurate if you are talking about the *rate*: 4.30 tokens out of 5 proposed make 86%, the same 4.30 out of 7 make only 61%. But this rate says nothing about what produces throughput, which is the *number* of tokens returned per step. And this number does not move by a hundredth. The rate drops solely because its denominator increases. A conclusion drawn from the rate alone would have given the right sign for the wrong reason, and would not have made it possible to predict that no higher `k` can win.

**Conclusion, and it closes the question.** `SPECK=5` is not only the floor imposed by `dspark_block_size`: it is **the demonstrated optimum**. Since acceptance caps at 4.30 tokens per step (below even `k=5`), no higher value can win: it only slows down the step. The engine's floor happens to be the best setting, and the only unexplored dimension of the sweep is now closed.

The law holds at `k=7` too, incidentally: `steps/s` stays constant there over the whole context range (11.80 to 12.38, a 4.7% spread). Only the coefficient changes, which confirms that it is the verification cost, and not the context length, that drives this term.

## Reproducing the Measurements

```bash
# 1. barrière qualité (~45 min)
URL=http://127.0.0.1:8090/v1 MODEL=deepseek-v4-flash TAG=<bras> \
  python3 codebench/run.py --out resultats-<bras>.json

# 2. débit de décodage
python3 decode_probe.py --base-url http://127.0.0.1:8090/v1 \
  --label <bras> --warmup 5 --repeats 3 --conc 1 2 --out probe-<bras>.json

# 3. latence au premier token (sonde de seuil)
python3 seuil_probe.py     # cible 1760..1920 tokens, 2 tirages

# 4. débit mono-flux, streaming, hors TTFT (instrument du balayage)
~/.venv-dsv4/bin/python bench_mono.py <bras>
```

Two methodology rules that change the numbers:

- **Warm up.** 5 long generations minimum before any measurement. The first cold draw can be 10× slower (10.66 s versus 0.83 s observed).
- **Isolate the server.** Bind it to `127.0.0.1` during campaigns. A test server reachable from the network gets its queue slots consumed by application traffic, and the concurrency numbers become wrong.

---

All the measurements in this report are those of path A, without exception, made from August 1 to 3, 2026 on two DGX Spark (GB10, sm_121). What is estimated or unverified is flagged as such in the body of the text. The step-by-step version, oriented toward commissioning, is in the [complete recipe](/blog/dsv4-2x-spark-recette/).


## Sources

- Public recipe from a third-party operator (DeepSeek-V4-Flash 0731 on two DGX Spark): 74.55 and 95.9 tok/s (C1, warm median decode), cited in §5.5
- Internal measurements, two NVIDIA DGX Spark (GB10, sm_121), TP=2, path A, August 1 to 3, 2026: codebench code benchmark, decode_probe.py, seuil_probe.py, bench_mono.py, machine telemetry
- jasl/vllm, branch b12x-nvfp4-mxfp4, commit 48d274954 (envs.py, config/vllm.py, experts/b12x_mxfp4_moe.py, flashmla_sparse.py, indexer.py, yarn_scaling_rope.py)
- Anemll/dspark-vllm-gx10 (upstream.lock, 12-file overlay, image ghcr.io/anemll/dspark-vllm-gx10:0.1.1) and the third-party operator's recipe (serving profile, docker-compose.dspark.yml, README)
- vLLM v0.25.1 (752a3a504485790a2e8491cacbb35c137339ad34), FlashInfer 0472b9b3f2fba11b463f8526f390297d52a8aad7 (0.6.15), lukealonso/b12x 7dc6fb8fcc6446ea093537d1657df81985fa5f43 (0.15.3)
- DeepSeek-V4-Flash-0731 checkpoint: configuration (index_topk 512, dspark_block_size 5)
- Benchmark startup and serving logs, including dsv4_seqs6b_head_20260802.log

