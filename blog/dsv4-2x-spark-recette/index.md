# Serving DeepSeek-V4-Flash-0731 on Two DGX Spark (TP=2): The Full Recipe

> 96/100 correct code judged by execution, 78.55 tok/s single-stream and 188.9 tok/s at sixteen streams with DeepSeek-V4-Flash-0731 on two DGX Spark.

- Date: 2026-08-04
- Language: en
- URL: https://0x0800.com/blog/dsv4-2x-spark-recette/
- Tags: dgx-spark, deepseek-v4, vllm, tp2, local-inference
- Translation (fr): https://0x0800.com/fr/blog/dsv4-2x-spark-recette/


{{< keyfigures >}}
[
  {"label": "Correct code", "value": "96/100", "detail": "judged by execution, shipped config"},
  {"label": "Single-stream throughput", "value": "78.55 tok/s", "detail": "code, 512 tokens, excluding TTFT"},
  {"label": "Aggregate throughput", "value": "188.9 tok/s", "detail": "at sixteen streams, TTFT under 0.8 s"},
  {"label": "KV cache", "value": "1.79 M", "detail": "tokens, read from the startup log"},
  {"label": "Power", "value": "41 W", "detail": "GPU power per box at peak, no throttling"},
  {"label": "RDMA latency", "value": "1.86 µs", "detail": "typical under load, 8-byte messages"}
]
{{< /keyfigures >}}

This document describes what we did to serve DeepSeek-V4-Flash-0731 in TP=2 on two DGX Spark (GB10, SM121), and what we measured. It includes the pitfalls that cost us time, because those are the ones missing everywhere else.

It is written for anyone who has to decide whether a model of this size can run on site, inside their own network, without sending a single line of code to a third party. Two boxes, about €8,000 of hardware, 41 W of GPU power at peak per box. Everything below can be replayed with the commands given; the measurement scripts are in `codebench/`, next to this file.

Measurement dates: August 1 to 3, 2026. Each table carries its own date when it matters. Machines: `spark-1` and `spark-2`, 121 GB of unified memory each, connected directly by a 200 Gb/s link (`enp1s0f0np0`). RDMA latency measured between the two: **1.86 µs** typical (see below).

## Result

Final configuration, single stream, decoding excluding TTFT, measured client-side on the SSE stream. Measurements of August 2, 2026, three draws per workload:

| workload | decoding |
|---|---:|
| **code, 512 output tokens** | **78.55 tok/s** |
| code, 1024 tokens | 75.33 |
| prose, 512 tokens | 47.16 |

{{< hbars title="Decoding throughput per workload, final configuration" caption="Prose is slower than code, not the other way around: the drafter follows code generation better. This is counter-intuitive and it is reproducible." minwidth=560 barheight=18 marginleft=210 >}}
{"bars":[
  {"label":"code, 512 output tokens","value":78.55,"text":"78.55 tok/s","color":"accent"},
  {"label":"code, 1024 tokens","value":75.33,"text":"75.33 tok/s","color":"accent"},
  {"label":"prose, 512 tokens","value":47.16,"text":"47.16 tok/s","color":"accent"}]}
{{< /hbars >}}

Prose is slower than code, not the other way around: the drafter follows code generation better. This is counter-intuitive and it is reproducible.

**Throughput does not depend on context length.** From 4 k to 131 k input tokens, decoding stays within a 1.5% band in steps per second. Prefill, on the other hand, declines: 1,846 tok/s at 4 k, 1,503 at 131 k. Details and explanation below.

Under concurrency: **100.4 tok/s aggregate at two streams** with the final configuration, and up to **188.9 tok/s at sixteen streams** by raising a single parameter (see "Raising the ceiling").

Quality gate: the generated code is extracted and then **executed** against hidden assertions. Execution, not a keyword test. Score of the final configuration: **96/100** (Python 50/50, Go 46/50), no unstable task over one hundred draws. The full benchmark is below.

The same engine also takes over the agent's small tasks, classification and entity extraction, 25 out of 25. The second model that used to serve them has been retired.

## The Real Checkpoint Format

Worth checking yourself rather than taking on faith. Reading the headers of the 48 safetensors:

- Experts: `I8` packed as 4 bits + `F8_E8M0` scales, block 32 → **MXFP4** (`expert_dtype: fp4`).
- Dense and attention: `F8_E4M3` + `F8_E8M0` scales, `weight_block_size [128,128]`, `scale_fmt: ue8m0`.
- DSpark heads included in the checkpoint: `dspark_target_layer_ids: [40,41,42]`, `dspark_markov_rank: 256`.
- 43 layers, 256 routed experts, `max_position_embeddings: 1048576`, YaRN factor 16 above 65536.

{{< callout label="Vocabulary trap" >}}
It costs an hour: **MXFP4 and NVFP4 are two different axes**. MXFP4 describes the expert weights, NVFP4 shows up in other people's recipes as the dtype of the KV cache. Seeing "MXFP4" in one place and "NVFP4" in another is not a contradiction.
{{< /callout >}}

## The Launch

vLLM from the `jasl/vllm` fork (build `20260721b1.dev0+g48d274954.d20260721.cu132`), installed in a `.venv-dsv4` venv **identical on both nodes**. TP=2 requires matched builds: same tree, same torch. A mismatch shows up as obscure crashes during collective initialization.

Topology:

```bash
--distributed-executor-backend mp --nnodes 2 \
--master-addr 192.168.100.10 --master-port 29519
```

Head: `--node-rank 0 --port 8082`. Worker: `--node-rank 1 --headless`.

Options that matter:

```bash
--tokenizer-mode deepseek_v4 --kv-cache-dtype fp8 --block-size 256 \
--enable-prefix-caching --max-num-seqs 2 --max-num-batched-tokens 4096 \
--gpu-memory-utilization 0.85 --no-enable-flashinfer-autotune \
--reasoning-parser deepseek_v4 --enable-auto-tool-choice \
--tool-call-parser deepseek_v4
```

Speculation:

```json
{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"greedy"}
```

Compilation:

```json
{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}
```

Plus `VLLM_USE_FLASHINFER_SAMPLER=1`, and **not** `--enable-expert-parallel`.

## Reproducing Step by Step

The complete sequence, in order, as we replay it every session.

### 1. The network

The two machines are directly connected, one RoCE address on each side (`192.168.100.10` and `.11`). Check that both HCAs come up: `ls /sys/class/infiniband/`. Two useful ones are expected, `rocep1s0f0` and `roceP2p1s0f0`. Declare **both** to NCCL, with merging:

```env
NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0
NCCL_IB_MERGE_NICS=1
NCCL_PROTO=Simple
NCCL_NET_GDR_LEVEL=0        # GPUDirect RDMA n'existe pas sur GB10
```

One cable, two interfaces to declare: forgetting the second costs 39% of collective bandwidth.

### 2. The venv and the build

To be done **on both nodes**, identically. Allow about thirty minutes of compilation per node.

```bash
# venv dédié, Python 3.12
python3.12 -m venv ~/.venv-dsv4

# torch pour GB10 : cu132, surtout pas torchaudio (il casse la résolution)
~/.venv-dsv4/bin/pip install torch==2.13.0+cu132 \
  --index-url https://download.pytorch.org/whl/cu132

# le fork, sur la branche qui porte sm_121
git clone https://github.com/jasl/vllm.git ~/vllm-jasl-dsv4
cd ~/vllm-jasl-dsv4
git checkout b12x-nvfp4-mxfp4

# le seul correctif obligatoire : lever l'épinglage torch==2.11.0
git apply ~/models/_runbook/0001-depin-torch-cuda.patch

# build. MAX_JOBS=4 est LE réglage anti-OOM sur une boîte 128 Go unifiée :
# sans lui, l'édition de liens se fait tuer.
MAX_JOBS=4 ~/.venv-dsv4/bin/pip install -e . --no-build-isolation

# le NCCL effectif vient de torch, pas de pip : vérifier, ne rien installer
~/.venv-dsv4/bin/python -c "import torch;print(torch.cuda.nccl.version())"
# attendu ici : (2, 29, 7)
```

A pitfall along the way: `pip list` shows `nvidia-nccl-cu13 2.30.4` in this venv, whereas torch is linked against 2.29.7 and loads that one. Installing a newer NCCL through pip does not change the NCCL that runs. The only source of truth is `torch.cuda.nccl.version()`.

The patch from step 4 removes `torch == 2.11.0` from `pyproject.toml`, `requirements/build/cuda.txt` and `requirements/cuda.txt`, along with the `torchaudio`/`torchcodec` dependencies that drag the resolution along with them. **It touches no line of engine code.** That is everything you need to patch to reproduce the numbers in this tutorial: the fork serves this checkpoint as is.

Then check that the two nodes are properly matched: same source tree, same torch. A mismatch shows up as unreadable crashes during collective initialization, never as a clear message:

```bash
for h in spark-1 spark-2; do
  ssh $h '~/.venv-dsv4/bin/python -c "import vllm,torch;print(vllm.__version__,torch.__version__)"'
done
# attendu, deux fois la même ligne :
# 20260721b1.dev0+g48d274954.d20260721.cu132 2.13.0+cu132
```

Versions actually installed here, for comparison: Python 3.12.13, torch 2.13.0+cu132, triton 3.7.1, flashinfer-python 0.6.14 (0.6.13 is not enough), tilelang 0.1.9, nvidia-nccl-cu13 2.30.4, numpy 2.3.5.

What is **not** in this stack, contrary to what you read elsewhere for DeepSeek-V4 on GB10: DeepGEMM and the indexer fix. `import deep_gemm` fails in the venv that produced all the measurements. Both items concern the NVFP4 requantized variant, not the official 0731 checkpoint.

### 3. The checkpoint

48 shards, 235 GB. You fetch it once, then copy it to the second node over the direct link:

```bash
# depuis le nœud qui l'a déjà
rsync -a --info=progress2 --no-compress -e 'ssh -c aes128-gcm@openssh.com' \
  ~/models/dsv4-0731/ spark-2:~/models/dsv4-0731/
```

672 MB/s measured, 3 min 56 s. Then check every safetensors header, **not** file sizes (see below): a truncated shard has the right size if the copy was restarted with `cp -n`.

### 4. Drop the page cache before launching

`sync && echo 3 > /proc/sys/vm/drop_caches` on both nodes. On unified memory, the page cache competes with the model for space.

### 5. Launch

Two tmux sessions, never `ssh + nohup`:

```bash
tmux new-session -d -s dsv4 \
  'CGMODE=FULL_AND_PIECEWISE NOEP=1 SPECK=5 GPUUTIL=0.85 ./dsv4_run.sh head N0731'
# sur le second nœud, la même chose avec 'worker'
```

Wait for `Application startup complete` in the log. Allow about six minutes, most of it spent capturing the CUDA graphs.

### 6. Warm up before measuring anything

At least five long generations. An engine that has just started is not the engine that serves: the first requests pay an overhead that the startup log shows nowhere. Whoever measures cold measures the cold path.

### 7. Measure

The code quality harness is launched like this:

```bash
URL=http://127.0.0.1:8082/v1 MODEL=deepseek-v4-flash \
  python codebench/run.py -k 5 --out resultats.json
```

Before judging a model, run `python codebench/run.py --selftest`: the reference solutions must pass 20/20. If the harness fails on its own answers, it measures nothing.

## What Drives Throughput, Measured

Each row is a complete restart of the model on both nodes, followed by a short measurement. Configurations that do not boot are recorded as such instead of interrupting the sweep.

Campaign of August 1, 2026, measurement protocol of the time: the absolute numbers in this table cannot be compared with those in the header, which come from a stricter benchmark (fixed output, excluding TTFT). What reads here are the **differences between rows**, all measured the same day in the same way.

| Configuration | small | medium | acceptance |
|---|---:|---:|---:|
| no speculation | 17.06 | 16.69 | — |
| baseline: PIECEWISE + EP + k=5 | 36.34 | 38.62 | 47.4% |
| + CUDA graphs FULL_AND_PIECEWISE | 44.08 | 44.84 | 42.1% |
| no EP alone (k=5) | 37.65 | 41.00 | 46.1% |
| CUDA graphs FULL + no EP + k=5 | 49.99 | 45.72 | 45.6% |
| **the same + memory 0.85** | **47.96** | **53.68** | **50.7%** |
| k=7 | 38.92 | 42.39 | 29.4% |
| k=10 | 23.90 | 27.15 | 24.0% |
| MoE backend triton | does not start | | |

{{< groupedbars title="Throughput per configuration, small and medium context" caption="Campaign of August 1, 2026, two context lengths. DSpark speculation does the bulk of the work; beyond k=5, throughput drops back." minwidth=660 marginleft=230 dec=2 >}}
{"series":[{"label":"small context","color":"accent"},{"label":"medium context","color":"signal"}],
 "groups":[
  {"label":"no speculation","values":[17.06,16.69],"texts":["17.06","16.69"]},
  {"label":"baseline: PIECEWISE + EP + k=5","values":[36.34,38.62],"texts":["36.34","38.62"]},
  {"label":"+ CUDA graphs FULL_AND_PIECEWISE","values":[44.08,44.84],"texts":["44.08","44.84"]},
  {"label":"no EP alone (k=5)","values":[37.65,41.0],"texts":["37.65","41.00"]},
  {"label":"CUDA graphs FULL + no EP + k=5","values":[49.99,45.72],"texts":["49.99","45.72"]},
  {"label":"the same + memory 0.85","values":[47.96,53.68],"texts":["47.96","53.68"]},
  {"label":"k=7","values":[38.92,42.39],"texts":["38.92","42.39"]},
  {"label":"k=10","values":[23.9,27.15],"texts":["23.90","27.15"]},
  {"label":"MoE backend triton","values":[null,null],"texts":["does not start"," "]}]}
{{< /groupedbars >}}

The lessons:

1. **DSpark speculation does everything.** 16.7 → 38.6 tok/s, a factor of 2.3. It is the main lever, and it costs nothing: the heads are in the checkpoint.
2. **`FULL_AND_PIECEWISE` is worth +16%.** The default CUDA graph mode leaves performance on the table at batch 1.
3. **Expert-parallel *costs* throughput here.** Counter-intuitive: with 128 local experts per node, the dispatch traffic weighs more than the parallelism gain in single-stream mode. EP is justified under heavy concurrency, not for a single agent.
4. **k=5 is the optimum, and it is also the minimum.** DSpark refuses anything below (`num_speculative_tokens >= dspark_block_size (5)`). Above it, throughput falls: 38.92 at k=7 and 23.90 at k=10 against 49.99 at k=5. Confirmed on August 3 on the final configuration, where k=7 costs 11.5% to 17.1% depending on the load.

   **Be careful not to read the acceptance rate as the cause.** It drops (45.6% → 29.4% → 24.0%) but its denominator grows at the same time. What counts is the **number** of tokens accepted per step, and in the August 3 measurement it is **identical at k=5 and k=7: 4.30**. The drafter does not get worse, it proposes more without anything more being kept, and each step costs more to verify: 72.0 ms at k=5, 82.5 ms at k=7. Throughput pays for the verification, not for the quality of the draft.
5. **The drafter's acceptance varies from 41 to 52% from one trial to the next** on the same prompt. It is the main source of noise between two measurements; do not conclude on a 5% gap between two configurations.

Prefill, for its part, does not move with these levers: 1,972 tok/s at 8K, 1,593 at 82K, 1,425 at 164K. The degradation is gentle, the model holds up on long context.

## Throughput Depends on the Content, Not Only on the Machine

This is what most often distorts published numbers. Same engine, same configuration, same day, five different contents:

| What we ask for | Throughput |
|---|---:|
| Count from 1 to 300 | 85.4 tok/s |
| Produce 60 JSON objects | 77.9 tok/s |
| Multiplication table by 12 | 74.4 tok/s |
| Binary search tree | 64.1 tok/s |
| 500-word short story | 43.8 tok/s |

The first and the last are within a factor of two. The reason is mechanical: speculative decoding proposes several tokens ahead and keeps only those that the large model validates. On a sequence of numbers, it guesses almost everything. On prose, almost nothing. A benchmark built on counting measures the predictability of the text, not the power of the machine.

On our own production content, the range narrows:

| Task | Throughput |
|---|---:|
| Batch JSON extraction | 78.1 tok/s |
| Single JSON extraction | 54.9 tok/s |
| Document processing | 54.2 tok/s |
| Writing Go types | 50.3 tok/s |
| Writing a Go handler | 46.4 tok/s |

{{< hbars title="Throughput by generated content" caption="Both tables combined, sorted by decreasing throughput. In blue the benchmark contents, in green our production contents. The factor of two between counting and prose is entirely due to the predictability of the text." minwidth=660 headroom=22 barheight=16 marginleft=230 >}}
{"legend":[{"label":"benchmark contents","color":"accent"},{"label":"production contents","color":"signal"}],
 "bars":[
  {"label":"Count from 1 to 300","value":85.4,"text":"85.4","color":"accent"},
  {"label":"Batch JSON extraction","value":78.1,"text":"78.1","color":"signal"},
  {"label":"Produce 60 JSON objects","value":77.9,"text":"77.9","color":"accent"},
  {"label":"Multiplication table by 12","value":74.4,"text":"74.4","color":"accent"},
  {"label":"Binary search tree","value":64.1,"text":"64.1","color":"accent"},
  {"label":"Single JSON extraction","value":54.9,"text":"54.9","color":"signal"},
  {"label":"Document processing","value":54.2,"text":"54.2","color":"signal"},
  {"label":"Writing Go types","value":50.3,"text":"50.3","color":"signal"},
  {"label":"Writing a Go handler","value":46.4,"text":"46.4","color":"signal"},
  {"label":"500-word short story","value":43.8,"text":"43.8","color":"accent"}]}
{{< /hbars >}}

Peak 78.1, mean 56.8, floor 46.4. The floor is the only number you can promise a customer. And it falls on a case we did not expect there: Go code generation is slower for us than prose.

## How Many Sessions, How Much Context

The KV cache allocated in the final configuration holds **1,793,656 tokens**, a figure read from the startup log (`GPU KV cache size`, cross-checked by `Maximum concurrency for 393,216 tokens per request: 4.56x`).

{{< callout label="Mind the source" >}}
**The server gives two figures that differ by a factor of two.** On the same server in the same second, `/metrics` reports `kv_cache_size_tokens 877 986` and `kv_cache_max_concurrency 2.23`. Both pairs are consistent with themselves, and yet they do not say the same thing. The code settles it: the log value comes from `get_kv_cache_capacity()`, commented in `kv_cache_utils.py` as "*Sourcing this from the concurrency calculation handles hybrid layouts correctly*", and this model has precisely a hybrid layout. An additional hint that the metric is not up to date: the same `/metrics` block shows `block_size="4"` although we launched with `--block-size 256`. **Quote the startup log, not `/metrics`.** An earlier version of this tutorial published the `/metrics` figure; it was two times too low, and the table below with it.
{{< /callout >}}

| Context per session | Simultaneous sessions in memory |
|---|---:|
| 384,000 tokens | 4.7 |
| 256,000 | 7.0 |
| 128,000 | 14.0 |
| 32,000 | 56.1 |
| 8,000 | 224.2 |

{{< hbars title="Simultaneous sessions by context size" caption="Logarithmic scale. Memory is nonetheless not the constraint: the ceiling on in-flight requests is." minwidth=560 barheight=18 scale="log" max=240 marginright=110 >}}
{"bars":[
  {"label":"384,000 tokens","value":4.7,"text":"4.7 sessions","color":"accent"},
  {"label":"256,000","value":7.0,"text":"7.0 sessions","color":"accent"},
  {"label":"128,000","value":14.0,"text":"14.0 sessions","color":"accent"},
  {"label":"32,000","value":56.1,"text":"56.1 sessions","color":"accent"},
  {"label":"8,000","value":224.2,"text":"224.2 sessions","color":"accent"}]}
{{< /hbars >}}

But memory is not the constraint: `--max-num-seqs 2` is. Beyond two in-flight requests, the following ones wait in a queue. A measurement at "concurrency 6" on this configuration does not measure six streams, it measures a waiting queue. **This ceiling is a choice, not a limit of the machine**, and it can be raised almost for free.

At two truly simultaneous streams, we sustain **100.4 tok/s aggregate**, i.e. ~50.2 per stream: almost double the single-stream figure, without losing anything per stream. Reference measurement, five minutes of sustained load, 108 requests, 30,558 tokens in 304.4 s.

### Raising the ceiling: the only measurement that produced a gain

Two scales, a single variable that changes (`--max-num-seqs`), same harness: output fixed at exactly 512 tokens (`ignore_eos`), T=0, a unique prefix per request to neutralize the prefix cache, three draws per point. Fixed output is the key point: all the slots of a batch free up together, so none stays empty waiting for a straggler. This bias is what made us underestimate capacity in our previous campaigns.

| streams | `seqs=2` | median TTFT | `seqs=16` | median TTFT |
|---:|---:|---:|---:|---:|
| 1 | 40.05 | 0.23 s | 43.15 | 0.25 s |
| 2 | 63.44 | 0.39 s | 62.80 | 0.35 s |
| 4 | 64.48 | **7.54 s** | **88.11** | 0.32 s |
| 6 | 62.63 | **15.58 s** | **116.51** | 0.51 s |
| 8 | 59.81 | **24.25 s** | **134.30** | 0.44 s |
| 16 | 60.34 | **56.64 s** | **188.88** | **0.75 s** |

{{< linechart title="Aggregate throughput by number of streams" caption="Aggregate throughput. The two series coincide at one and two streams, which validates the harness, then diverge exactly where `--max-num-seqs 2` saturates. At sixteen streams, 188.88 versus 60.34 tok/s: a factor of 3.1." minwidth=620 xlabel="concurrent streams" ymax=205 ylabel="aggregate tok/s" >}}
{"xs":[1,2,4,6,8,16],"xticks":[1,2,4,6,8,16],"yticks":[0,50,100,150,200],
 "series":[
  {"label":"seqs=16","points":[43.15,62.8,88.11,116.51,134.3,188.88],"color":"accent"},
  {"label":"seqs=2","points":[40.05,63.44,64.48,62.63,59.81,60.34],"color":"alert"}]}
{{< /linechart >}}

{{< linechart title="Time to first token by number of streams" caption="Median latency before the first token, logarithmic scale. At seqs=2, the wait goes from 0.39 s to 56.6 s while aggregate throughput stays flat: all you buy is waiting. At seqs=16, it stays under 0.8 s everywhere." minwidth=620 xlabel="concurrent streams" yscale="log" ymin=0.2 ymax=80 ydec="auto" ylabel="median TTFT, s" >}}
{"xs":[1,2,4,6,8,16],"xticks":[1,2,4,6,8,16],"yticks":[0.2,1,5,20,80],
 "series":[
  {"label":"seqs=2","points":[0.23,0.39,7.54,15.58,24.25,56.64],"color":"alert"},
  {"label":"seqs=16","points":[0.25,0.35,0.32,0.51,0.44,0.75],"color":"accent"}]}
{{< /linechart >}}

The two series coincide at one and two streams: the two configurations do the same thing there, which validates the harness. They diverge exactly where `max-num-seqs 2` saturates: on the left the aggregate is flat from C2 to C16 while the wait before the first token goes from 0.39 s to 56.6 s; on the right it climbs up to **188.88 tok/s**, i.e. 3.1×, with a TTFT that stays under 0.8 s everywhere.

The price is the per-stream throughput: 44.1 tok/s alone, 13.1 at sixteen streams. This is the normal trade-off of a server. The memory cost, for its part, is marginal: the KV pool goes from 1,793,656 to 1,717,797 tokens, **−4.2%**. It is `--max-num-batched-tokens` that costs memory, not `--max-num-seqs`.

**To choose:** a single agent, `--max-num-seqs 2`, best latency and best single-stream throughput. Several users, raise the parameter: it is the most profitable lever of this entire campaign, and the only one that produced a net gain.

{{< callout label="What we published before" >}}
An earlier version of this tutorial concluded that beyond two streams all you buy is waiting. That was true of the configuration measured, false of the machine, because the only variable available was capped at 2, and the heterogeneous burst protocol hid the rest.
{{< /callout >}}

**Measuring this figure is trickier than it looks.** On short bursts, the same server gives between 77 and 116 tok/s depending on how you count. The `Avg generation throughput` line of the log divides a number of tokens by a ten-second window, whatever happened inside it: during a long prefill it displays 0.1 tok/s while the machine is saturated, and on a four-second burst it mixes the useful window with the ramp-up and ramp-down windows. Over five sustained minutes, log and client converge to within 0.4%. **A metric averaged over a fixed window only measures throughput if the load fills the window.**

### Long context: what it really costs

Output held **strictly identical** at all lengths: the prompt is filler of variable length followed by "ignore the text above, copy the following passage word for word", with a fixed passage, at temperature 0. Three draws per point, a unique prefix so that the prefix cache does not make the following draws free, acceptance read from `/metrics` before and after each point.

| input | TTFT | prefill | decoding | acceptance | steps/s |
|---:|---:|---:|---:|---:|---:|
| 4,240 | 2.30 s | 1,846 tok/s | 76.3 | 88.9% | **14.02** |
| 16,520 | 8.67 s | 1,905 | 71.1 | 82.1% | **13.92** |
| 32,880 | 18.12 s | 1,814 | 68.3 | 77.8% | **13.97** |
| 65,680 | 40.16 s | 1,636 | 61.2 | 69.5% | **13.69** |
| 80,120 | 49.69 s | 1,612 | 79.1 | 93.2% | **13.97** |
| 100,120 | 63.60 s | 1,574 | 76.3 | 91.1% | **13.73** |
| 131,200 | 87.28 s | 1,503 | 84.0 | 100.0% | **14.00** |

{{< linechart title="Decoding throughput and acceptance by input length" caption="Throughput in tokens per second follows the speculation acceptance, curve for curve — and not the context length. The longest point is the fastest, 84.0 tok/s at 131 k, because the copy is accepted at 100% there." minwidth=620 xscale="log" xdiv="k" xdec=0 xlabel="input tokens" ymin=50 ymax=95 ylabel="decoding, tok/s" rmin=60 rmax=105 rdec=0 runit="%" rlabel="acceptance" >}}
{"xs":[4240,16520,32880,65680,80120,100120,131200],"xticks":[4240,16520,32880,65680,131200],
 "yticks":[50,60,70,80,90],"rticks":[60,70,80,90,100],
 "series":[
  {"label":"decoding, tok/s","points":[76.3,71.1,68.3,61.2,79.1,76.3,84.0],"color":"accent"},
  {"label":"acceptance, %","points":[88.9,82.1,77.8,69.5,93.2,91.1,100.0],"color":"signal","axis":"right"}]}
{{< /linechart >}}

{{< linechart title="Decoding steps per second by input length" caption="The same sweep in decoding steps per second: 13.69 to 14.02, a 2.4% spread, below the ±2% error bar between two restarts. The band covers the full amplitude of the seven points." minwidth=620 xscale="log" xdiv="k" xdec=0 xlabel="input tokens" ymin=0 ymax=16 ylabel="steps/s" height=266 >}}
{"xs":[4240,16520,32880,65680,80120,100120,131200],"xticks":[4240,16520,32880,65680,131200],
 "yticks":[0,4,8,12,16],
 "band":{"from":13.69,"to":14.02,"label":"amplitude of the seven points: 2.4%"},
 "series":[{"label":"steps per second","points":[14.02,13.92,13.97,13.69,13.97,13.73,14.0],"color":"accent"}]}
{{< /linechart >}}

The `steps/s` column is the number of decoding steps per second: throughput divided by the speculation multiplier `1 + acceptance length`. **It is constant to within 2.4% over the whole range.**

The result that matters for sizing an agent: **context length has no measurable effect on the speed of a decoding step between 4 k and 131 k.** This is the wait you do not have with dense attention. Throughput in tokens per second moves, but it follows the speculation acceptance, not the length, and the longest point is even the fastest, 84.0 tok/s at 131 k, because the copy is accepted at 100% there.

**Why**, and this is what lets you know when to transpose this result: the checkpoint declares `index_topk: 512`, and at decoding time the kernel receives `topk_indices` of shape `(num_decodes, seq_len, 512)`. Attention reads a **fixed** budget of 512 cache entries, whatever the context length. The startup log confirms it (`FlashInfer SM120 packed sparse-MLA decode`, `DSA indexer decode path`). On dense attention, the same sweep would give a clear decline: this result holds for sparse attention with a fixed budget, not for any architecture.

The seven points fit a one-parameter formula, accurate to within 1.5%:

> **throughput ≈ 13.90 × (1 + accepted tokens per step)**, a step lasts 72 ms.

It has an immediate practical consequence: with `SPECK=5`, single-stream cannot exceed `13.90 × 6 = 83.4` tok/s, and our best measured point is 84.03. **We are at the ceiling, not near the ceiling.** No use looking for throughput in the scheduling settings: none of them touches this term.

The only lever is `k`, and the engine accepts 7. **We measured it, and it costs**: the step goes from 72.0 to 82.5 ms (−12.8% of steps per second) while the tokens accepted per step stay identical, at 4.30 in both cases. The DSpark drafter saturates below 5: asking it for seven proposals does not make it return more, you just pay more for verification. In practice `SPECK=7` loses 11.5% to 17.1% of throughput depending on the load, with quality intact (98/100). **`SPECK=5` is not only the floor imposed by the engine, it is the best setting.**

{{< linechart title="Prefill and TTFT by input length" caption="Prefill and wait before the first token. This is the only term that really depends on length: −19% of prefill throughput between 4 k and 131 k, and 87 seconds before the first word." minwidth=620 xscale="log" xdiv="k" xdec=0 xlabel="input tokens" ymin=1400 ymax=2000 ydec=0 ylabel="prefill, tok/s" rmin=0 rmax=100 rdec=0 runit=" s" rlabel="TTFT" >}}
{"xs":[4240,16520,32880,65680,80120,100120,131200],"xticks":[4240,16520,32880,65680,131200],
 "yticks":[1400,1550,1700,1850,2000],"rticks":[0,25,50,75,100],
 "series":[
  {"label":"prefill, tok/s","points":[1846,1905,1814,1636,1612,1574,1503],"color":"accent"},
  {"label":"TTFT, s","points":[2.3,8.67,18.12,40.16,49.69,63.6,87.28],"color":"alert","axis":"right"}]}
{{< /linechart >}}

What really costs is the first token: 87 seconds to swallow 131 k, and a prefill that loses 19% between 4 k and 131 k. Hence the value of the prefix cache, next section.

{{< callout label="Measurement trap, valid for any engine with speculative decoding" >}}
Our first sweep varied the length by changing the text to summarize. It gave a 30% drop at 131 k, which fell right above the checkpoint's YaRN threshold (65,536). A very convincing explanation, and a wrong one. By changing the text, we changed the predictability of the output, hence the acceptance of the draft: it ranged from 34% to 64%, and throughput followed. Adding intermediate points was enough to break the monotony of the curve. **A "throughput as a function of X" curve is only interpretable if X is the only thing that varies; publish the acceptance next to the throughput so that people can check.**
{{< /callout >}}

### The prefix cache survives requests

A question that decides the architecture of an agent: is the injected context thrown away at the end of each turn? No. Measurement, a 32,818-token file placed at the head of the prompt, the question varying at each turn:

| Turn | Prefill | Tokens found in cache |
|---|---|---|
| 1, new prefix | 18.1 s | 0 out of 32,818 |
| **2, same file, other question** | **1.0 s** | **32,512 out of 32,819** |
| **3, question from turn 1** | **0.9 s** | **32,512 out of 32,818** |

{{< hbars title="Prefill duration per turn" caption="Factor 18 on prefill from the second turn. Sharing also holds between two distinct agents that receive the same preamble." minwidth=520 barheight=18 marginleft=230 >}}
{"bars":[
  {"label":"1, new prefix","value":18.1,"text":"18.1 s","color":"alert"},
  {"label":"2, same file, other question","value":1.0,"text":"1.0 s","color":"signal"},
  {"label":"3, question from turn 1","value":0.9,"text":"0.9 s","color":"signal"}]}
{{< /hbars >}}

KV blocks survive the request that created them and are reused until LRU eviction takes them back. Factor 18 on prefill, and sharing also holds between two distinct agents that receive the same preamble.

Two conditions: the file must be **at the start** of the prompt, before anything that varies, because matching breaks at the first differing token; and it must not change between turns. This is also what makes the size of the KV pool concrete: it does not limit what a turn can receive (`--max-model-len` takes care of that), it decides how much history survives before eviction when several agents alternate.

## What the Machines Do During a Run

Recorded over 90 seconds, two streams under load, one sample per second on each node:

| | first node | second node |
|---|---|---|
| SM clock min / med. / max | 2,411 / 2,457 / 2,496 MHz | 2,411 / 2,483 / 2,509 MHz |
| Power | 33 W median, 35 W max | 40 W median, 41 W max |
| Temperature | 56 °C median, 61 max | 62 °C median, 67 max |
| GPU utilization | 96% | 95% |
| **Hardware throttling** | **`0x0` over 90 readings** | **`0x0` over 90 readings** |

{{< thermalchart title="Median and maximum temperatures per node" caption="No thermal or electrical throttling over 180 readings. At 67 °C at worst and 41 W of GPU power per box, an office is enough, no air-conditioned room needed." minwidth=560 >}}
{"rows":[["First node",56,61],["Second node",62,67]]}
{{< /thermalchart >}}

No thermal or electrical throttling over 180 readings. Clocks hold around 2.45 GHz out of a theoretical maximum of 3.0 GHz; the gap comes from the workload, not from a limitation. At 67 °C at worst and 41 W of GPU power, an office is enough, no air-conditioned room needed.

On the network side, over the same window: 5,272 MiB sent on the first interface, 3,240 MiB on the second, i.e. about 112 MiB/s during the real load. That is **less than 1% of a 200 Gb/s link**. The RDMA latency measured under load (`ib_write_lat`, 8-byte messages, 2,000 iterations) gives 1.86 µs typical, 2.67 µs at the 99th percentile, 3.15 µs at worst.

For anyone sizing a setup, communication between the two boxes is therefore **bound by latency, not by bandwidth**. Buying a faster link would bring nothing. The ICMP RTT, for its part, comes out at 0.42 ms on average: that is the kernel's network stack, not the RDMA path, and quoting it as interconnect latency would be wrong by three orders of magnitude.

## The Quality of the Code Produced

Speed is worth nothing if the code is wrong. The benchmark therefore does not judge style, it executes.

**The method.** Twenty exercises, ten in Python and ten in Go, drawn from problems we actually run into: amount parsing, VAT breakdown, end-of-month due dates, interval merging, reconciliation of two sets of entries, concurrent counters, LRU cache. Each prompt goes to the model, the generated code block is extracted, dropped into a throwaway module, and judged by the machine:

- in Go, `go vet` then `go test`; two exercises additionally go through the race detector;
- in Python, an isolated import first, then **hidden** assertions from the model, executed in a subprocess.

The verdict is mechanical and graded: truncated answer, no usable code, does not compile, compiles but fails the tests, correct. No human jury enters anywhere.

**Each exercise is drawn five times.** At zero temperature, the same prompt does not always give the same code: speculative decoding and routing between experts make the model non-reproducible. A single-pass score is a draw, not a measurement. One hundred draws, 43 minutes.

**The result.**

| | Success | Compiles | Truncated | Inter-task standard deviation |
|---|---|---|---|---|
| Python | **50/50 (100%)** | 50/50 | 0 | 0.00 |
| Go | **45/50 (90%)** | 45/50 | 0 | 0.10 |
| **Total** | **95/100 (95%)** | **95/100** | **0** | **0.09** |

{{< qualitygrid title="Success per exercise, five draws each" caption="Success over five draws, per exercise, campaign of August 1, 2026. No exercise fails systematically: all twenty are solved at least once." minwidth=640 >}}
{"rows":[
  ["parse_montant","py",5],["ventiler_tva","py",5],["echeance_fin_de_mois","py",5],["merge_intervals","py",5],
  ["cle_tva_fr","py",5],["solde_progressif","py",5],["top_k_deterministe","py",5],["parse_ligne_csv","py",5],
  ["rapprochement","py",5],["cache_lru","py",5],
  ["ParseMontant","go",4],["VentilerTVA","go",4],["Echeance","go",5],["MergeIntervals","go",4],
  ["CompteurConcurrent","go",5],["MapConcurrent","go",5],["TopK","go",5],["ParseLigneCSV","go",5],
  ["RetryContexte","go",4],["Rapprocher","go",4]]}
{{< /qualitygrid >}}

Campaign of August 1, 2026. The same benchmark replayed on August 2 on the final configuration gives **96/100** (Python 50/50, Go 46/50), and 98/100 on the `k=7` arm discarded elsewhere. Three points of difference over one hundred draws, between close configurations: **that is noise, not a hierarchy.** We publish all three rather than keep the most flattering one. The figure to remember for the shipped configuration is 96/100.

No exercise fails systematically: all twenty are solved at least once. The five failures are concentrated on five Go tasks that sometimes pass and sometimes do not.

**The nature of the failures matters more than their number.** All five are compilation errors, never a wrong business test. Four out of five come down to a missing import (`fmt`, `strconv`, `context`) on otherwise correct code. The fifth mixes the `int64` and `int` types on the same line. The model therefore knows how to solve the problem and sometimes misses the declarative mechanics of Go. A standard tooling pass before compilation would fix the first four; we have not replayed it on the faulty outputs, which were not kept, and the argument remains to be verified.

On two desktop boxes, this model therefore produces executable, correct code in 95% of cases, with residual defects that the build chain catches.

#### Duration per exercise

Median over five draws, budget 16,000 tokens:

| Exercise | Language | Success | Median |
|---|---|---|---:|
| parse_montant | Python | 5/5 | 158.6 s |
| ventiler_tva | Python | 5/5 | 68.9 s |
| echeance_fin_de_mois | Python | 5/5 | 23.8 s |
| merge_intervals | Python | 5/5 | 15.3 s |
| cle_tva_fr | Python | 5/5 | 9.8 s |
| solde_progressif | Python | 5/5 | 11.2 s |
| top_k_deterministe | Python | 5/5 | 10.3 s |
| parse_ligne_csv | Python | 5/5 | 96.9 s |
| rapprochement | Python | 5/5 | 43.8 s |
| cache_lru | Python | 5/5 | 14.7 s |
| ParseMontant | Go | 4/5 | 127.4 s |
| VentilerTVA | Go | 4/5 | 93.2 s |
| Echeance | Go | 5/5 | 23.1 s |
| MergeIntervals | Go | 4/5 | 32.5 s |
| CompteurConcurrent | Go | 5/5 | 12.9 s |
| MapConcurrent | Go | 5/5 | 43.9 s |
| TopK | Go | 5/5 | 14.4 s |
| ParseLigneCSV | Go | 5/5 | 88.6 s |
| RetryContexte | Go | 4/5 | 18.6 s |
| Rapprocher | Go | 4/5 | 67.6 s |

{{< hbars title="Median duration per exercise" caption="Time follows the length of the reasoning, not the apparent difficulty. In blue Python, in green Go." minwidth=640 headroom=6 >}}
{"bars":[
  {"label":"parse_montant","value":158.6,"text":"158.6 s","color":"accent"},
  {"label":"ParseMontant","value":127.4,"text":"127.4 s","color":"signal"},
  {"label":"parse_ligne_csv","value":96.9,"text":"96.9 s","color":"accent"},
  {"label":"VentilerTVA","value":93.2,"text":"93.2 s","color":"signal"},
  {"label":"ParseLigneCSV","value":88.6,"text":"88.6 s","color":"signal"},
  {"label":"ventiler_tva","value":68.9,"text":"68.9 s","color":"accent"},
  {"label":"Rapprocher","value":67.6,"text":"67.6 s","color":"signal"},
  {"label":"MapConcurrent","value":43.9,"text":"43.9 s","color":"signal"},
  {"label":"rapprochement","value":43.8,"text":"43.8 s","color":"accent"},
  {"label":"MergeIntervals","value":32.5,"text":"32.5 s","color":"signal"},
  {"label":"echeance_fin_de_mois","value":23.8,"text":"23.8 s","color":"accent"},
  {"label":"Echeance","value":23.1,"text":"23.1 s","color":"signal"},
  {"label":"RetryContexte","value":18.6,"text":"18.6 s","color":"signal"},
  {"label":"merge_intervals","value":15.3,"text":"15.3 s","color":"accent"},
  {"label":"cache_lru","value":14.7,"text":"14.7 s","color":"accent"},
  {"label":"TopK","value":14.4,"text":"14.4 s","color":"signal"},
  {"label":"CompteurConcurrent","value":12.9,"text":"12.9 s","color":"signal"},
  {"label":"solde_progressif","value":11.2,"text":"11.2 s","color":"accent"},
  {"label":"top_k_deterministe","value":10.3,"text":"10.3 s","color":"accent"},
  {"label":"cle_tva_fr","value":9.8,"text":"9.8 s","color":"accent"}]}
{{< /hbars >}}

{{< callout label="The token budget trap" >}}
DeepSeek-V4-Flash reasons before answering, and this trace does not count in the rendered content. Our first version of the harness allowed 1,600 tokens: the simplest exercise of the suite returned **empty** content with 5,327 characters of reasoning. On the first Go exercise, 23,510 characters of reasoning before the first line of code. A budget that is too short does not measure a model unable to write; it measures a harness that cuts off before the answer. We are at 16,000, and zero truncation over one hundred draws.
{{< /callout >}}

## One Model for the Small Tasks

An agent does not only call its big model. It classifies documents into a taxonomy, extracts entities, tags, summarizes. These passes used to run here on a dedicated small model, one billion parameters served alongside, on the second machine. The question put to the benchmark: can DeepSeek-V4-Flash take them all over?

The protocol reuses the prompts, the sampling parameters, the token budgets and the response parsing code exactly as they run in production, without adapting them to the new model. Changing even one of them would no longer test the envisaged replacement. Synthetic documents, no real data.

| Pass | Budget | Result | Latency | Tokens emitted |
|---|---|---|---|---|
| Classification into a closed taxonomy of 20 labels | 64 tokens | 20/20 | 0.4 to 0.5 s | 2 to 7 |
| Entity extraction constrained by a JSON schema | 2,048 tokens | 5/5 | 1.2 to 2.1 s | 56 to 87 |

The expected breaking point was the 64-token budget. A model that reasons before answering can spend all of them on thinking and return empty content. The template parameter `chat_template_kwargs: {"enable_thinking": false}` is honored by this model, and the classification answers are four tokens long. We did not measure the same pass without this parameter. What we know is that the failure would look like a model unable to classify when it simply would not have had room to answer.

The only formal discrepancy across the twenty documents comes from a commercial lease, filed under both real estate and legal. Both labels are defensible, and the code keeps two.

**What fits alongside the model.** Removing the small model frees much more than its size, because `--gpu-memory-utilization` reserves a fraction of the total in advance, unrelated to the weight actually loaded. A vLLM launched at `0.1` for a one-billion-parameter model claims 12.2 GiB. On a machine where the main model already holds 0.85, it does not fit.

The embeddings server, for its part, allocates what it consumes. Measurement of available memory before and after startup: **941 MiB** for a 379 MiB model in Q6_K, context 8,192. Reservation on one side, allocation on the other: this difference decides what can coexist, and no datasheet gives it.

One last point for anyone stacking services on unified memory: keep a margin below the OOM killer's threshold. Configured with a preference for Python processes, it will designate the vLLM worker as the victim, and its death breaks TP=2 on both machines at once.

## The Pitfalls That Cost an Hour Each

**The prefix cache falsifies prefill.** A warm-up call before the measurement primes the cache, and you read 127,000 tok/s of prefill. That is physically impossible and has nothing to do with the model. To measure honestly: **unique** prompts (a nanosecond timestamp is enough) and `max_tokens=1`.

**`pkill -f` kills the shell that launches it.** The remote SSH command line contains the pattern. Bracket the first letter: `pkill -9 -f "[b]in/vllm"`.

**`pkill -f "my_script.sh"` destroys the entire tmux server.** The tmux server inherits the argv of the command that started it. If this script is called `dsv4_run.sh`, the pattern matches the server itself and all sessions die at once. Close the session with `tmux kill-session`, then target only the binaries.

**Never launch a long process through `ssh + nohup`.** Always `tmux new-session -d` then `capture-pane`. And beware of the straggler `VLLM::Worker` that survives the server's shutdown.

**In bash `set -e`, a `[ cond ] && VAR=x` as the last statement of a `case` branch returns 1 when the condition is false and aborts the script.** Write `if ... fi`.

**`cp -n file file.bak` after editing the file produces a contaminated backup.** The `-n` protects against overwriting an existing backup, it does not go back in time. A `.bak` with exactly the same size as the current file is the symptom. Back up before editing, and check the backup with a `diff` or a `grep` of the added pattern, not by its mere presence.

**A `tmux new-session -d` session does not inherit the virtual environment.** It starts a clean login shell: the `python` of the `.venv` activated in the current terminal does not exist there, and the benchmark dies in one line (`python: command not found`) while the server itself is running. Call the interpreter by absolute path, `~/.venv-dsv4/bin/python`, rather than counting on an inherited activation.

**Not all engine warnings are knobs.** At startup, vLLM emits `max_num_scheduled_tokens is set to N ... Consider increasing max_num_batched_tokens`. The message fires again identically after doubling `max_num_batched_tokens`: it compares what is schedulable to the slots reserved for the draft, a gap that exists at any value. Check that a warning closes before treating it as a recommendation.

## Verifying a Checkpoint Transfer

48 shards, 235 GB, by rsync over the direct link with `aes128-gcm` and no compression: 672 MB/s, 3 min 56 s. Verification does not consist of comparing sizes: you reread the header of every safetensors file and check that `8 + header_size + max_offset == file_size` exactly, and that no tensor of the index is missing. 48 shards, 72,317 tensors, zero discrepancy.

## torch.compile on DeepSeek-V4 NVIDIA: Why It Is Not There

A competing recipe announces higher throughputs than ours while keeping `torch.compile` active. We set out to find whether this path was missing from our setup. Short answer: **DeepSeek-V4 is not wired for `torch.compile` on NVIDIA, and that is a choice, not an oversight.**

The verifiable facts:

- The official vLLM recipe launches DSv4 with `--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'` and without `--enforce-eager`. `cudagraph_mode` drives the capture of CUDA graphs, not Dynamo compilation. The upstream model does not carry `@support_torch_compile`.
- The decorator does exist in the tree, on the XPU variant of the same model. The NVIDIA port did not take it over.
- The recipe that keeps compilation remains pinned to vLLM 0.21.1, and says so explicitly: the B12X MoE kernels and the `torch.compile` path live on that version. This is not concealment, it is an acknowledged version debt.

Between 0.21 and 0.25, DSv4 gained the CuTe DSL indexer (18 `.compile()` calls at runtime, spread over three ops files) and a multi-stream orchestration (8 sites in attention). Both are untraceable. **The fork traded compilation for hand-fused kernels.**

### What restarting it costs, measured

We tried anyway, because a black box is not auditable. Five patches, applied one after the other, each unblocking one layer and revealing the next:

| # | patch | precedent in the tree |
|---|---|---|
| 1 | `@support_torch_compile` on `DeepseekV4Model` | yes (XPU variant) |
| 2 | six MHC sites switched to `torch.ops.vllm.*` | yes (`layers/mhc.py`) |
| 3 | registration of `mhc_pre_broadcast_tilelang` | no |
| 4 | workspace reservation at layer init | partial |
| 5 | memoization of `get_device_capability` by dict | no |

The hard point to know before starting: **`fullgraph=True`** (`vllm/compilation/wrapper.py`). Under this setting, a Dynamo graph break is not a performance cost, it is a **fatal error**. The whole family of solutions "disable tracing on the offending function" (`torch.compiler.disable`, `dynamo.disable`) is therefore ineffective: Dynamo answers `Skip inlining torch.compiler.disable()d function` and stops. We lost one boot verifying that.

The blockers encountered, in order, all of the same kind, Python code that inspects its own source or manipulates a C object at call time:

- `torch.accelerator.empty_cache()` in the workspace growth;
- `functools._lru_cache_wrapper` on `get_device_capability`;
- the CuTe DSL JIT of the indexer (`IndexerQFp8Kernel.compile`), reached via the multi-stream orchestration;
- `inspect.getsourcelines` called by TileLang to decide its JIT mode, which falls back on `_patched_getfile`, a function that Dynamo explicitly marks as untraceable.

The generic remedy for the third point: make attention **opaque** by registering it as a custom op, on the upstream pattern `unified_attention_with_output` (`@eager_break_during_capture`, empty dummy implementation, `mutates_args=["out"]`, layer retrieved through its prefix in `static_forward_context` since a custom op cannot receive an `nn.Module`). It works: tracing goes through attention without flinching and dies further on, on `hc_head_fused_kernel_tilelang`: a seventh TileLang kernel to register.

That is where we stopped. The pattern is clear and bottomless: every hand-fused kernel needs its own registered op. Bringing compilation back amounts to undoing, one by one, the fusions that make the fork's performance.

### The useful conclusion

`torch.compile` is not the source of throughput on modern DSv4. Upstream does without it, and the recipe that keeps it pays a pinned version for a result that stays of the same order. If you are looking for throughput on this stack, look for it in the serving settings, not in compilation.

## What We Have Not Done Yet

Published recipes announce ~83 tok/s at short context with the B12X MoE and an NVFP4 KV cache. Our analysis of the code: FlashInfer 0.6.14 already ships the complete W4A16 SM120 path: **nothing to compile**. Two things are missing on our side:

- `oracle/mxfp4.py` has no B12X entry (zero occurrences); the routing exists only in `oracle/nvfp4.py`.
- `FlashInferB12xExperts` enforces `quant_dtype == "nvfp4"` by assertion and refuses expert-parallel.

{{< callout label="Important caveat" >}}
`prepare_w4a16_packed_weights` expects a `source_format` of `modelopt` or `compressed_tensors`, whereas our checkpoint is in native DeepSeek MXFP4. A weight layout error does not crash, it produces plausible and wrong text. Any B12X result must therefore pass the executable quality gate before we quote a single number from it.
{{< /callout >}}

## Measuring Without Lying to Yourself

Two rules that served throughout:

- The quality gate comes **before** the performance number, never after. And it executes the generated code; a keyword test does not detect a subtly broken model.
- Throughput is measured excluding TTFT: `(output_tokens - 1) / (total_duration - TTFT)`, in streaming. Otherwise you mix prefill and decoding and compare things that cannot be compared.

---

Measurements made from August 1 to 3, 2026 on two DGX Spark (GB10, SM121), 121 GB of unified memory each, directly linked by a 200 Gb/s link. Measurement harness and reproducible exercise sets: `codebench/`. The complete setup of the two software paths, the one-button sweep and the matrix of parameters that do not pay off are in the [detailed report](/blog/dsv4-2x-spark-rapport/).


## Sources

- Internal measurements, two NVIDIA DGX Spark (GB10, SM121), TP=2, August 1 to 3, 2026: throughput, capacity, thermals, RDMA (ib_write_lat), codebench code benchmark (selftest 20/20)
- jasl/vllm, branch b12x-nvfp4-mxfp4, build 20260721b1.dev0+g48d274954.d20260721.cu132 (startup log, kv_cache_utils.py, oracle/mxfp4.py, oracle/nvfp4.py)
- DeepSeek-V4-Flash-0731 checkpoint: headers of the 48 safetensors files and configuration (index_topk 512, dspark_target_layer_ids, expert_dtype fp4)
- Official vLLM recipe for DeepSeek-V4 (cudagraph_mode FULL_AND_PIECEWISE, custom_ops all, no --enforce-eager)
- FlashInfer 0.6.14: W4A16 SM120 path, prepare_w4a16_packed_weights

