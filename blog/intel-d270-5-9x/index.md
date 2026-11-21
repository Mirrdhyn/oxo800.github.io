# Intel D270 vs RTX PRO 6000: the 5.9x is real, on very narrow ground

> Intel's 5.9x comes from FP64 option pricing. On inference the gap shrinks to 12 %, at a little over half the power, and the D270's 160 to 480 GB of LPDDR5X matter more for local deployment.

- Date: 2026-11-21
- Language: en
- URL: https://0x0800.com/blog/intel-d270-5-9x/
- Tags: intel, d270, nvidia, fp64, local-inference
- Translation (fr): https://0x0800.com/fr/blog/intel-d270-5-9x/


Intel says its D270 (350 W) delivers up to 5.9 times the throughput of Nvidia's RTX PRO 6000 Blackwell Server Edition (600 W). I read the report and its notes: the figure is real, but it covers very narrow ground.

The 5.9x comes from double-precision (FP64) compute applied to option pricing: European Monte Carlo (5.9x), European Black-Scholes (5.6x), binomial tree (4.6x). On the American option the gap falls to 1.92x, and the four tests average 4.5x. Nvidia limits FP64 on its workstation cards to a 1:64 ratio, while the D270, built on the Xe3P architecture, runs it at full rate, with 64 FP64 FMAs per core. Intel is attacking where Nvidia left a gap.

A few reservations about the method. The measurements come from Intel, on pre-production silicon. According to Intel, each test used a single card on each side, but the software stacks differ: oneAPI on the Intel side, a CUDA 12.8 container with DPC++ and oneMath on the Nvidia side.

On inference, the gap narrows. On Qwen3-30B-A3B in FP8, the D270 delivers 12 % more throughput than the RTX PRO 6000, at 350 W against 600 W, and without torch.compile, which the Nvidia card used. On a decision model (Drex v2.0), Intel reports 29 % lower latency at concurrency 1. For both tests, the Nvidia card ran in an AWS VM. The 5.9x is gone, but a high-end Blackwell card is beaten on a 30-billion-parameter MoE with a little over half the power.

For local deployment, memory matters most. The RTX PRO 6000 tops out at 96 GB of GDDR7. Intel skips HBM and stacks LPDDR5X on an air-cooled PCIe card: 160 GB on the reference design (1.5 TB/s), up to 480 GB from partners. Models whose weights used to need a rack of cards fit on a single 350 W card. At SC26, Intel and SambaNova are showing a hybrid architecture: the D270 handles the prefill of very large contexts, the SN50 generates the tokens.

Price and availability are still unknown. But if Nvidia keeps limiting the memory on its cards, high-capacity inference could drift toward these LPDDR5X cards.


## Sources

- Intel Newsroom, "The case for Intel GPU D270 in high-performance computing", 16 November 2026, and its configuration notes
- Average of the four financial tests: computed from Intel's figures

