# TensorFold on Intel Arc Pro B70: from a Python port to a Zig engine

> How we ported the TensorFold inference engine to the Intel Arc Pro B70 (Xe2), first in Python, then in Zig on Level Zero: Qwen3.8-27B at 122.9 tok/s, measured against vLLM and llama.cpp.

- Date: 2026-10-23
- Language: en
- URL: https://0x0800.com/blog/tensorfold-xe2-b70/
- Tags: b70, tensorfold, xpu, zig, level-zero, vllm, llama-cpp
- Translation (fr): https://0x0800.com/fr/blog/tensorfold-xe2-b70/


TensorFold only knew Macs and NVIDIA GPUs. We ported it to the Intel Arc Pro B70 card (Xe2 architecture, 32 GB), first in Python through PyTorch, then in Zig directly on Level Zero, with no PyTorch and no Python at run time. The second port has been part of the official project since TensorFold 1.0.12. This text gives the path, the pitfalls and the measurements, including the ones that are less favourable to us.

{{< keyfigures >}}
[
  {"label": "One stream, drafted", "value": "122.9 tok/s", "detail": "Zig engine, code, greedy; 99.3 tok/s in conversation"},
  {"label": "Concurrent == alone", "value": "64 / 64", "detail": "cells bit-identical, 1 to 8 streams, greedy and sampled"},
  {"label": "Status", "value": "TensorFold 1.0.12", "detail": "backend merged into the official project (PR #802), preview you build yourself"}
]
{{< /keyfigures >}}

## A 32 GB card the fast engines ignored

[TensorFold](https://github.com/ashhart/TensorFold) is an open-source inference engine (Apache-2.0) written by Ash Hart. It serves language models behind an OpenAI-compatible API, on Apple Silicon and on NVIDIA GPUs. Its particularity: each model family has its own kernels and its own draft verification, and a draft is accepted only if it equals the token the engine would have produced alone. On Qwen3.8-27B, it relies on DFlash2, a small model that proposes several tokens ahead, organised as a tree, which the large model verifies in one pass.

The target card is the Intel Arc Pro B70: Battlemage G31 chip, Xe2 architecture, 32 GB of memory, a measured read bandwidth of 577 GB/s and a power limit set to 230 W by the firmware. The model is Qwen3.8-27B quantised to 4 bits (symmetric GPTQ, groups of 128), which fits on the card with its cache.

TensorFold's CUDA path for this model relies on hand-written kernels tuned for NVIDIA. None of it runs as is on an Intel card. At the start, the reference on the B70 was vLLM-XPU: **33.8 tok/s** without drafts, **45 to 61 tok/s** with its own prediction mechanism (MTP).

## Three equalities to hold, bit for bit

A backend enters TensorFold only if it holds three properties, checked token by token, in greedy as well as sampled mode:

- **drafted == serial**: an answer produced with drafts is identical to the same answer produced one token at a time;
- **resumed == fresh**: a prompt resumed from the cache gives the same continuation as a prompt computed from scratch;
- **concurrent == alone**: a request served alongside others gives the same answer as the same request served alone.

The Intel card resists on this point. The 4-bit matrix product of the vllm-xpu-kernels library (`int4_gemm_w4a16`) reaches the bandwidth ceiling, but its bits depend on the number of rows in the call, and sometimes on the position of a row within its block. At 192 or 256 rows, it is not even repeatable from one call to the next. On the same stack, `F.linear` is not deterministic either, and the compiler fuses multiply-adds differently depending on the kernel. Any one of these three sources of divergence is enough to break the drafted == serial equality.

{{< callout label="What vLLM does with this problem" >}}
vLLM does not promise this property. On the same card, we compared each vLLM v0.31 answer served four at a time with the same request served alone: **27 answers out of 48** identical with MTP4, **16 out of 48** without drafts. Two successive solo runs of the same request give the same answer in only **11 cases out of 16**. In code, they agree; in conversation, they diverge.
{{< /callout >}}

## Python and torch.xpu: keep the engine, replace the kernels

The first port keeps all of TensorFold's Python engine and adds a `--backend xpu` backend. A compatibility module answers `torch.cuda` calls with `torch.xpu` and replaces the CUDA kernels: the gated delta rule (GDN), tree attention, prompt attention, dense projections, the drafter. A pleasant surprise: TensorFold's Triton kernels run almost unmodified on the B70. The first token came out on 5 October, at 14 tok/s, exact from the start (12 answers out of 12 in drafted == serial, 12 out of 12 in resumed == fresh).

### Holding exactness with a GEMM that changes its bits

The remedy: never let the number of rows vary where it matters. Decoding and verification calls are padded to exactly 16 rows (16 × N with N streams, including for a single stream served with `--parallel`). Prompt calls use block sizes whose bits are identical for the whole family (512 and 1,024 rows), and the `down` projection uses a grid anchored on absolute positions, multiples of 1,024. The Triton kernels that must agree run without multiply-add fusion, and the dense projections go through a Triton kernel with a fixed summation order.

### The pitfalls that cost the most

- **The first `torch.topk` compiles for 38 seconds** on this stack, for slices of 4,096 elements and more. The first request took 50 s. Replaced by a two-stage top-k.
- **Register spills**: the tree attention kernels spilled (256 registers per thread, about 30,000 spills). Spreading the output columns over four programs and computing the scores in slices made these kernels 5 to 14 times faster, with the same bits.
- **A mask on a load inside the loop** costs a factor of 5 to 6 on this compiler. It is better to bound the index and mask only the weights.
- **The bf16 / fp16 conversions** around every 4-bit call: about 740 launches per step. Keeping the activations in fp16 end to end removed them.

Result of the Python port, published on 9 October: serial decoding at 38 ms per token (26 tok/s), drafted decoding at **76 to 86 tok/s** in short context, prefill at **1,870 tok/s** at 2,000 tokens, and up to four exact streams in parallel.

## The comparison, with the line that is less favourable to us

Our first measurements pitted TensorFold against vLLM v0.31 with MTP4, its default prediction mechanism. But vLLM v0.31 can also use the DFlash2 drafter, provided it runs in bf16 (in fp16, its acceptance rate drops to zero without an error). Comparing without this line would have inflated the gap. Here it is, on the same card, the same GPTQ checkpoint and the same bench (192 tokens per stream, code and conversation prompts, temperature 0 and 1).

| Throughput (tok/s) | TensorFold Python | vLLM + DFlash2 | vLLM + MTP4 | vLLM alone |
|---|---:|---:|---:|---:|
| **1 stream, code** | 116-119 | 87-99 (7 drafts) | 62-63 | 33 |
| **1 stream, conversation** | 78-83 | 53-78 (7 drafts) | 55-58 | 33 |
| **4 streams, aggregate** | 154-220 | 161-217 (3 drafts) | 137-202 | 119-121 |

{{< hbars title="Throughput of a single stream (tok/s)" caption="One stream, code prompt, greedy, same card and same 4-bit GPTQ checkpoint. The Zig and llama.cpp rows come from the PR #802 measurement (13-14 October), the others from the 6 October bench." barheight=16 marginleft=250 >}}
{"bars":[
  {"label":"TensorFold Zig, DFlash2","value":122.9,"text":"122.9","color":"accent"},
  {"label":"TensorFold Python, DFlash2","value":119.2,"text":"119.2","color":"accent"},
  {"label":"vLLM v0.31, DFlash2 (7 drafts)","value":86.9,"text":"86.9","color":"signal"},
  {"label":"vLLM v0.31, MTP4","value":61.8,"text":"61.8","color":"signal"},
  {"label":"TensorFold Zig, no drafts","value":33.5,"text":"33.5","color":"faint"},
  {"label":"vLLM v0.31, no drafts","value":33.4,"text":"33.4","color":"faint"},
  {"label":"llama.cpp SYCL, Q4_0","value":27.7,"text":"27.7","color":"faint"}
]}
{{< /hbars >}}

Read without rounding in our favour, these figures say three things:

- **For a single user, TensorFold stays ahead**, even against vLLM with the same drafter: 116-119 versus 87-99 tok/s in code. The gap comes from the draft trees (4 to 5 tokens accepted per step), which vLLM does not do.
- **At four streams, it is a tie**: 154-220 versus 161-217 tok/s. And vLLM has to choose: with 7 drafts it is good alone but admits only two requests at a time (89-160 tok/s at four), with 3 drafts it holds the load but loses on a single stream (66-69 tok/s in code).
- **The difference that is not measured in tok/s**: each TensorFold cell also checks that each stream equals its solo and serial answer. No deviation. vLLM does not hold this property (see above).

The raw TensorFold and vLLM + MTP4 files are published in `tools/xpu-bench/refs/` of the repository. The vLLM + DFlash2 measurements date from the same day (6 October) and had not been published yet.

## Zig on Level Zero: no more PyTorch, no more Python

TensorFold also has a native engine written in Zig, first for Metal. The second port adds an `xpu` backend that drives the card directly through **Level Zero**, Intel's low-level interface: no PyTorch, no SYCL, no Python at run time. The `tensorfold-native` server is a single executable that embeds all its kernels.

- The kernels are written in **OpenCL C** and compiled offline by `ocloc`: 4-bit decoding GEMM, chunked and serial GDN, attention, norms, drafter.
- The prompt GEMMs are generated as ISA at build time by **nGEN** (the oneDNN generator), on the Xe2 DPAS matrix units.
- Prompt attention is a FlashAttention-2 written for DPAS, in oneDNN's orientation (one query per lane, softmax within the lane). First slower than the Triton kernel of the Python port (275 ms versus 185 at 6,700 tokens), it came down to 155 ms, with the same tokens.
- Multi-stream serving shares the verification steps between slots, each stream keeping its DFlash2 context; a stream left alone goes back to a tree.

Correctness is checked against a reference: the greedy tokens equal those of an fp32 PyTorch reference from the Python port, in MLX and GPTQ formats. During the review, we added an FP64 reference for the 4-bit GEMMs.

### The Zig engine measurements

Validation of 13 October, B70 at 230 W, server `--parallel 8 --context 16384`, 4-bit GPTQ format with groups of 128:

| | GPTQ g128 |
|---|---:|
| Prefill 1k / 4k / 8k tokens | 1,975 / 1,965 / 1,930 tok/s |
| One stream, DFlash2 tree, code / conversation | 133 / 90 tok/s |
| 2 / 4 / 8 streams, code (aggregate) | 235 / 310 / 270 tok/s |
| 2 / 4 / 8 streams, conversation (aggregate) | 148 / 215 / 208 tok/s |
| bench_openai decoding, 1 / 2 / 4 / 8 requests | 165 / 300 / 337 / 439 tok/s |

The validation covers four configurations (MLX, GPTQ, GPTQ with the prompt in W4A8, weights rotated in W4A4), 1, 2, 4 and 8 streams, greedy and sampled: **64 cells, all equal** to their solo and serial run, with no errors in the logs. Prefill is power-bound: at 230 W, the card runs around 2,180 MHz during a prompt.

### Against llama.cpp, on the same card

Measurement requested during the review: llama.cpp (commit 8c2134a, `llama-bench`, Q4_0, flash attention) against TensorFold on a cold prompt, no cache.

{{< groupedbars title="Prefill (tok/s)" caption="Cold prefill, Qwen3.8-27B, Arc Pro B70 at 230 W; TensorFold in 4-bit GPTQ with groups of 128, llama.cpp in Q4_0. llama.cpp Vulkan stops at 32,000 tokens (barrier wait)." >}}
{"series":[
  {"label":"TensorFold Zig","color":"accent"},
  {"label":"llama.cpp SYCL","color":"signal"},
  {"label":"llama.cpp Vulkan","color":"faint"}
],
"groups":[
  {"label":"2,000 tokens","values":[2051,788,645],"texts":["2,051","788","645"]},
  {"label":"8,000 tokens","values":[1985,787,484],"texts":["1,985","787","484"]},
  {"label":"32,000 tokens","values":[1719,696,null],"texts":["1,719","696",null]},
  {"label":"64,000 tokens","values":[1473,600,null],"texts":["1,473","600",null]}
]}
{{< /groupedbars >}}

In decoding without drafts, TensorFold does **33.5 tok/s** against 27.7 for llama.cpp SYCL. With DFlash2: **122.9 tok/s** in code and **99.3** in conversation.

## From the Hygur Labs repository to the official project

On 13 October, we proposed the backend to Ash Hart (issue [#801](https://github.com/ashhart/TensorFold/issues/801)) with pull request [#802](https://github.com/ashhart/TensorFold/pull/802): only new files, plus two lines of `build.zig`. The review asked, among other things, to remove the old Triton kernel captured as SPIR-V, to size the number of streams on the card's actual free memory (read through Sysman), and to bound every input parameter. It also prompted the FP64 reference for the GEMMs.

The backend ships in **TensorFold 1.0.12** (16 October) and the pull request was merged on 18 October. It is a preview you build yourself: the release archives do not contain it. Ash has no Intel card; the integration rests on our measurements on the B70 and on his own host-side checks (build, tests, code analysis).

The Python port remains available in [hygurlabs/tensorfold-xpu](https://github.com/hygurlabs/tensorfold-xpu) (`main` branch, Apache-2.0), with its measurement scripts and raw results. It was not proposed to the official project: only the Zig engine was.

## What this backend does not do

- **One model for our runner**: the Qwen3.8-27B dimensions are constants, and some shapes are written into the kernels. Since then, another contributor has added Nemotron 3.5 Lightning to the same XPU server, as an experimental version, on the same Level Zero interface.
- **Not in the archives**: the backend has been in TensorFold's code since 1.0.12, but the archives published with each release (macOS and Linux) do not contain it. You have to build it with `zig build xpu-native`.
- **4-bit formats only**: MLX with groups of 64 and symmetric GPTQ with groups of 128. The W4A8 prompt requires GPTQ.
- **A single card** was used for the measurements. A B70 from another manufacturer, with another power limit, will give other prefill figures.
- **Drafted throughput depends on the text**: code is predicted better than conversation, hence the ranges.

## Building and running the Zig engine

Prerequisites: Zig 0.17, the `xe` driver and the Intel compute runtime (Level Zero), `ocloc`, a C++17 compiler and the oneDNN nGEN headers.

```bash
zig build xpu-native -Doptimize=ReleaseFast -Dngen-include=<path to ngen>
zig build xpu-qwen27 -Doptimize=ReleaseFast -Dngen-include=<path to ngen>

tensorfold-native serve <Qwen3.8-27B GPTQ> --drafter <DFlash2> \
  --parallel 8 --context 16384 --host 127.0.0.1 --port 8081
```

The benches (`bench_openai.py` for prefill and decoding, `bench_concurrent.py --alone --serial` for exactness under concurrency) are in `tools/`. If you have an Arc Pro B70, replay them and send us your figures, even if they differ from ours.

{{< callout label="A B70 at your place?" >}}
We install and tune on-site inference servers, including on Intel cards: sizing, commissioning, connection to your sources, training of your teams.
{{< /callout >}}

## Also read

[A private AI on a €1,900 card: what the Intel Arc Pro B70 changes for regulated professions](/blog/ia-locale-b70-metiers/): the same measurements, read for an accounting firm, a notary's office or an HR department: cost, response time at the office, and answers an auditor can replay.


## Sources

- hygurlabs/tensorfold-xpu, main branch: docs/recipes/xpu.md and tools/xpu-bench/refs/*.json (Python port)
- hygurlabs/tensorfold-xpu, zig branch: docs/xpu/README.md (Zig engine, validation of 2026-10-13)
- ashhart/TensorFold#802: llama.cpp measurement comment (13-14 October 2026), XPU backend merged on 18 October, TensorFold 1.0.12
- ashhart/TensorFold#801: backend proposal
- Internal measurement, vLLM v0.31 + DFlash2 and vLLM exactness: raw bench files (vllm_dflash_*.json, exact_vllm_*.json), 6 October 2026

