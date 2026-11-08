# Running vLLM on an NVIDIA DGX Spark (GB10): from git clone to NVFP4 inference

> Full vLLM install on a DGX Spark GB10 (SM121): the three bugs that block the build and the launch of an NVFP4 model, and how to fix them.

- Date: 2026-07-20
- Language: en
- URL: https://0x0800.com/blog/dgx-spark-vllm-nvfp4/
- Tags: dgx-spark, gb10, vllm, nvfp4, cuda, sm121
- Translation (fr): https://0x0800.com/fr/blog/dgx-spark-vllm-nvfp4/


The DGX Spark ships a GB10 GPU (Blackwell, SM121 compute architecture) on an aarch64 board. An architecture recent enough that several libraries of the CUDA ecosystem do not yet have a precompiled kernel for SM121. Three fixes that do not jump out of the official documentation, explained here from cloning the repository to launching an NVFP4-quantised model.

This article covers the full installation, from cloning the repository to the successful launch of an NVFP4-quantised model (Mistral-Small 119B), explaining why each step is necessary. All the commands are tested on our hardware; adapt the paths (`~/models/...`, model names) to yours.

## What the Spark provides out of the box

Out of the box, the DGX Spark runs DGX OS (Ubuntu) with the NVIDIA driver and a CUDA toolkit already installed. Two commands are enough to know your starting point:

```bash
nvidia-smi -q | grep -i "CUDA Version"   # the driver's JIT compilation ceiling
nvcc --version                            # CUDA toolkit used to compile
```

On our hardware: driver `595.71.05`, JIT ceiling at **CUDA 13.2**, system toolkit installed at **13.3**. This one-notch gap is the source of the first bug detailed below. Keep these two numbers handy, they matter.

Another GB10 particularity to know before going further: GPU memory is system memory (unified architecture). `nvidia-smi --query-gpu=memory.used,memory.free` returns `[N/A]`: to monitor available memory, use `free -h`. And `--gpu-memory-utilization` in vLLM is computed on the machine's total RAM, not on a dedicated VRAM. On a 128 GiB machine, `0.85` therefore asks for about 103 GiB, and the OS or desktop environment nibbles away part of the margin before you have launched anything.

## Python environment and first vLLM build

We use `uv` to manage the virtual environment, faster than classic `venv`+`pip` for a project of this size:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv ~/.venv --python 3.12
source ~/.venv/bin/activate
```

Clone vLLM and install it in editable mode:

```bash
git clone https://github.com/vllm-project/vllm.git ~/vllm-020
cd ~/vllm-020
export MAX_JOBS=18   # adjust to the number of available cores minus a margin
~/.venv/bin/pip install --no-build-isolation -e .
```

**Important point**: `pip` must be called through its full path in the venv (`~/.venv/bin/pip`), never a bare `pip` in a non-interactive script. On Debian/Ubuntu, a bare `pip` resolves to the system pip, protected by PEP 668 ("externally-managed-environment"), and refuses the installation.

Two prerequisites to check before launching this build, otherwise it will fail midway through compilation:

- **ninja installed at system level**, not only in the venv: `sudo apt install -y ninja-build`. If `ninja` is only visible through the activated venv, a later call to the build (for example through a script that does not activate the venv) will make CMake fall back to the "Unix Makefiles" generator, in conflict with a CMake cache already generated with Ninja: blocking error ("Does not match the generator used previously").
- **$CUDA_HOME/bin on the PATH**, not only `$CUDA_HOME/lib64` on `LD_LIBRARY_PATH`. Without it, `nvcc` is invisible in a normal login shell even though a previous build may have worked (through a hand-modified PATH). Add to `~/.bashrc`, right after the export of `CUDA_HOME`:

```bash
export CUDA_HOME="/usr/local/cuda"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
```

The full build takes a good while (allow several tens of minutes depending on the number of cores). Once finished, check the installed version:

```bash
~/.venv/bin/python -c "import vllm; print(vllm.__version__)"
```

**One last first-build trap**: if your installation bumps `flashinfer-python` along the way (as is the case on a recent `main`), check that `flashinfer-cubin` (the precompiled kernel cache) follows the same version. Since version 0.6.14, this package is no longer published on PyPI but on a separate index:

```bash
~/.venv/bin/pip install flashinfer-cubin --extra-index-url https://flashinfer.ai/whl/
```

Without this explicit index, `pip` stays stuck on the latest PyPI version (0.6.13) and vLLM starts with a kernel cache that no longer matches `flashinfer-python`: a version error at launch, not at install time.

## First control launch

A small model is enough to validate that the whole chain works before moving on to serious things:

```bash
vllm serve <model_path> --served-model-name my-model \
  --max-model-len 8192 --enforce-eager --gpu-memory-utilization 0.1 \
  --attention-backend FLASHINFER
```

If this launch goes through and answers a `/v1/chat/completions` request correctly, the base is sound. If you run into one of the three following bugs, this is the time to deal with them.

## Bug 1 — cudaErrorUnsupportedPtxVersion

**Symptom**: the server starts, loads the weights, then crashes on the first real matrix computation (often visible on a model with a Mamba block, but the mechanism potentially affects any GEMM not precompiled for SM121):

```text
torch.AcceleratorError: CUDA error: the provided PTX was compiled with an unsupported toolchain.
```

**Cause**: on an architecture as recent as SM121, several kernels have no native binary and are compiled to PTX at runtime (JIT), directly by the driver. If this PTX was generated by a CUDA toolkit newer than what the driver can interpret in JIT (reminder: ceiling 595.71.05 → CUDA 13.2), the driver refuses to load it. The 13.3 toolkit used to compile vLLM is one notch above this ceiling, hence the crash.

{{< callout label="Fix" >}}
Recompile vLLM with a CUDA toolkit aligned on the driver's JIT ceiling (13.2 in our case), not with the latest available version.

```bash
# Install the 13.2 toolkit next to the existing 13.3 (same NVIDIA apt repository)
sudo apt install cuda-toolkit-13-2

# Switch the active alternative
sudo update-alternatives --config cuda
# → choose /usr/local/cuda-13.2

# Check
nvcc --version   # must report release 13.2

# Recompile vLLM from this new environment
cd ~/vllm-020
~/.venv/bin/pip install --no-build-isolation -e .
```

`update-alternatives` manages the `/usr/local/cuda` symlink: the two toolkits coexist without conflict, and switching from one to the other does not require touching `.bashrc` each time, as long as `CUDA_HOME` points to this symlink.

Rerun the control repro above: the `cudaErrorUnsupportedPtxVersion` error must have disappeared.
{{< /callout >}}

## Bug 2 — FlashAttention-2 failing silently on SM121

**Symptom**: the server runs, answers requests, but the log prints in a loop a message that interrupts nothing:

```text
FATAL: FlashAttention requires building with sm version sm80-sm90, but was built for < 8.0!
```

This is worse than a merely noisy message: the FlashAttention-2 kernels vendored in vLLM are guarded by a compilation macro (`__CUDA_ARCH__ >= 800`). For any kernel variant that does not natively match SM121, the compiled code boils down to printing this message and **returning without computing the attention**, without raising an exception. The risk is not a crash, it is a silently wrong result if vLLM's automatic selection oracle routes even a part of the computation to FA2.

{{< callout label="Fix" >}}
Explicitly force another attention backend at launch, to exclude FA2 from the execution path:

```bash
vllm serve <model_path> ... --attention-backend FLASHINFER
```

(or `TRITON_MLA` depending on the model's architecture: both avoid FA2). This is not a fix of the vLLM code, it is a launch-time workaround, but it is sufficient: no more occurrences of the FATAL message once the flag is set, consistent answers verified on a real request.
{{< /callout >}}

## Bug 3 — NVFP4/Marlin missing from the build on SM121 with a CUDA toolkit ≥ 13.0

This one only shows when launching an NVFP4-quantised model (typically Mistral checkpoints or similar shipped as `-NVFP4`).

**Symptom**:

```text
WARNING: Your GPU does not have native support for FP4 computation...
NotImplementedError: Could not run '_C::gptq_marlin_repack' with arguments from the 'CUDA' backend
```

This message is misleading: the GB10 natively supports NVFP4. The problem is in the build, not in the hardware.

**Cause**: in vLLM's `CMakeLists.txt`, the list of candidate architectures tested to enable the NVFP4 kernels (CUTLASS SM120 path) and Marlin changes depending on the CUDA compiler version:

```cmake
# With CUDA toolkit >= 13.0, only the "family-compat" candidate is tested:
cuda_archs_loose_intersection(MARLIN_ARCHS "8.0+PTX;12.0f" "${CUDA_ARCHS}")
# With CUDA toolkit < 13.0, the explicit 12.1a arch is included:
cuda_archs_loose_intersection(MARLIN_ARCHS "8.0+PTX;12.0a;12.1a" "${CUDA_ARCHS}")
```

With a 13.2 toolkit or later (our case after the fix for bug 1), only `12.0f` is tested, not `12.1a`. If the auto-detection of target architectures does not match the GB10 to this candidate, both kernels (CUTLASS NVFP4 and Marlin) are silently excluded from the build.

{{< callout label="Fix" >}}
Explicitly force the target architecture instead of relying on auto-detection, then recompile:

```bash
export TORCH_CUDA_ARCH_LIST="12.1a"
export MAX_JOBS=18
export CUTE_DSL_ARCH=sm_121a

cd ~/vllm-020
~/.venv/bin/pip install --no-build-isolation -e .
```

**Verification**: more reliable than a deduction from the symptoms, it queries the PyTorch dispatcher directly to confirm that the CUDA kernel is indeed registered:

```python
import vllm
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import cutlass_fp4_supported
print(cutlass_fp4_supported())  # must return True

import torch._C as C
print(C._dispatch_dump("_C::gptq_marlin_repack"))  # must list a CUDA registration
```
{{< /callout >}}

## Launching an NVFP4 model for real

With the three fixes in place, a realistic launch looks like this (example with a Mistral 119B model):

```bash
vllm serve <model_path> --served-model-name my-nvfp4-model \
  --tokenizer-mode mistral --config-format mistral --load-format mistral \
  --host 0.0.0.0 --port 8086 \
  --max-model-len 65536 --gpu-memory-utilization 0.85 \
  --enforce-eager --attention-backend TRITON_MLA --moe_backend flashinfer_b12x \
  --enable-auto-tool-choice --tool-call-parser mistral --reasoning-parser mistral
```

Two settings to keep in mind for a first launch on a model of this size:

- On unified memory, stay cautious with `--gpu-memory-utilization` (0.85 rather than 0.92) and with `--max-model-len`: the real margin available is often smaller than what a naive calculation on the total RAM suggests.
- If the model uses an FP4 kernel with no precompiled cubin for your architecture, `flashinfer` compiles it in JIT at first launch. Export `MAX_JOBS` (for example `4`) before the `vllm serve` command itself to bound the number of parallel compilations: a value that is too high can saturate memory while the model loads in parallel.

Once the server has started ("Application startup complete" in the logs), a simple test confirms that everything works:

```bash
curl -s http://localhost:8086/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "my-nvfp4-model",
  "messages": [{"role": "user", "content": "In one sentence, what is the capital of France?"}],
  "max_tokens": 50
}'
```

A coherent answer, with no trace of `NotImplementedError`, FP4 warning, or `FATAL` message in the logs, confirms that the native NVFP4 path is used end to end.

## The three bugs at a glance

| Bug | Symptom | Fix |
|---|---|---|
| CUDA toolkit too recent | cudaErrorUnsupportedPtxVersion at the first GEMM | Align the toolkit on the driver's JIT ceiling (nvidia-smi -q \| grep CUDA) via update-alternatives, then rebuild |
| FlashAttention-2 not supported on SM121 | FATAL: FlashAttention requires... in a loop, no crash | --attention-backend FLASHINFER (or TRITON_MLA) at launch |
| NVFP4/Marlin excluded from the build (toolkit ≥ 13.0) | NotImplementedError on gptq_marlin_repack, misleading FP4 warning | Rebuild with TORCH_CUDA_ARCH_LIST="12.1a", CUTE_DSL_ARCH=sm_121a |

These three problems share the same root: SM121 is too recent an architecture for the default auto-detection and compatibility paths of the CUDA/vLLM ecosystem to cover it correctly. In all three cases, the workaround consists in no longer trusting auto-detection (of the toolkit, the attention backend, or the architecture list) and explicitly forcing the right value.

{{< callout label="We install and tune your DGX Spark." >}}
If these fixes save you time but you would rather entrust the installation, the backend tuning and the move to production to someone who has already done it, write to us.
{{< /callout >}}

## Also read

- [DeepSeek-V4-Flash-0731 on two DGX Spark: the complete recipe](/blog/dsv4-2x-spark-recette/): 96/100 correct code over 100 draws judged by execution, 188.9 tok/s aggregate at sixteen streams, thermals and RDMA latency measured under load.
- [Serving Laguna S 2.1 NVFP4 on a single DGX Spark](/blog/laguna-s21-single-spark/): a 117.6B-parameter MoE on a single box: memory hygiene, native 256K context, DFlash speculative decoding.


## Sources

- Installation and checks on our DGX Spark (GB10), July 2026: driver 595.71.05, CUDA JIT ceiling 13.2, system toolkit 13.3 (every command in the article was tested there)
- vllm-project/vllm: CMakeLists.txt, NVFP4 (CUTLASS SM120) and Marlin architecture lists depending on the CUDA toolkit version
- flashinfer-cubin: published on the flashinfer.ai/whl index since version 0.6.14 (PyPI stops at 0.6.13)

