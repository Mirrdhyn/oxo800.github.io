# A private AI on a €1,900 card: what the Intel Arc Pro B70 changes for regulated professions

> A 27-billion-parameter model on a €1,900 (ex-VAT) Intel card, with no document leaving the premises: speed against vLLM, office cases and answers an auditor can replay.

- Date: 2026-10-23
- Language: en
- URL: https://0x0800.com/blog/ia-locale-b70-metiers/
- Tags: b70, tensorfold, local-ai, regulated-professions
- Translation (fr): https://0x0800.com/fr/blog/ia-locale-b70-metiers/


On a €1,900 (ex-VAT) graphics card, a 27-billion-parameter model answers at more than 100 tokens per second, without any document leaving the company. What makes it possible: our port of the TensorFold engine to the Intel Arc Pro B70 card, now merged into the official project.

{{< keyfigures >}}
[
  {"label": "Card price", "value": "€1,900 ex-VAT", "detail": "Intel Arc Pro B70, 32 GB, against €5,700 ex-VAT for an RTX 5090 with the same amount of memory"},
  {"label": "One person", "value": "90 to 133 tok/s", "detail": "Qwen3.8-27B, depending on the text produced (conversation or code)"},
  {"label": "Replayable answers", "value": "100 %", "detail": "an answer served alongside others equals the same answer served alone, on every try"}
]
{{< /keyfigures >}}

## The real obstacle is hosting

In an accounting firm, a notary's office, a mutual insurer or an HR department, AI first runs into a simple question: where do the documents go? Client files, payslips or medical reports do not go to an external API, because professional secrecy and the GDPR forbid it.

There is another way: run the model on site, within the company's walls. Projects then often stall on the price of the hardware.

## 32 GB of memory for a third of the price

A useful model, such as Qwen3.8-27B in 4 bits, needs about 32 GB of memory on the graphics card. NVIDIA's RTX 5090 has them, for €5,700 ex-VAT at the cheapest, in stock, at the time of writing. The Intel Arc Pro B70 has the same 32 GB for **€1,900 ex-VAT**.

Until now, the price gap was of little use: most fast inference engines could only work with CUDA, hence only with NVIDIA. We ported TensorFold, an open-source engine written by Ash Hart, to the B70. The port has been part of the official project since version 1.0.12, under the Apache-2.0 licence. The technical detail is in [the article that tells the story of the port](/blog/tensorfold-xe2-b70/).

## What the user experiences

We compared TensorFold with vLLM v0.31, the most widespread engine, on the same card and with the same model. vLLM was tested with its own acceleration mechanism (MTP4) and with the same small look-ahead model as TensorFold (DFlash2), so that the comparison is fair.

| Throughput (tokens/s) | TensorFold | vLLM, best setting |
|---|---:|---:|
| One person, writing code | 116 to 133 | 87 to 99 |
| One person, conversation | 78 to 99 | 53 to 78 |
| Four people at the same time, total | 154 to 220 | 161 to 217 |

For a single person, TensorFold answers markedly faster. With four people at the same time, the two engines are on a par.

{{< callout label="Where these ranges come from" >}}
The gap between the low and the high end of each range comes from the text produced: code is predicted better than conversation. TensorFold's high figures for one person come from the most recent version of the engine; those for four people, from the version measured against vLLM on 6 October.
{{< /callout >}}

## A twenty-page file, four colleagues

We simulated an office day: a fictitious accounting file of about twenty pages (roughly 18,700 tokens), queried at the same time by four colleagues on different points (VAT deadlines, the amount of an invoice, etc.). Each answer is checked against the facts of the file.

| | TensorFold | vLLM + MTP4 |
|---|---:|---:|
| The same file, already open, four questions at once | 7.8 s | 11.7 s |
| Four different files opened at the same time | 54.9 s | 54.5 s |

When the file is already loaded, everyone has their answer in 7.8 seconds instead of 11.7. When four colleagues open four different files at the same time, the two engines are tied: reading the documents dominates, and neither speeds it up. This scenario was not measured with vLLM + DFlash2.

## An answer an auditor can replay

In a regulated sector, another result weighs as much as speed. With TensorFold, an answer computed alone and the same answer computed alongside three others are **identical, token for token**. We checked it in every scenario, without a single deviation. An auditor can replay an answer and get the same text back, whatever the server load was when it was produced.

vLLM does not hold this property. On the same card, it returns under load the same answer as alone in only 27 cases out of 48 with its acceleration, and in 16 out of 48 without. Even alone, two successive tries of the same question in conversation do not always give the same text.

## Who it is for and how far it goes

- **A small team**: an office machine and a €1,900 card, with no subscription and no data travelling.
- **Dozens of continuous users**: you will need to measure the real load before buying. Our measurements go up to eight simultaneous streams on one card.
- **Two models** for now on this card: Qwen3.8-27B in 4 bits, the one in our measurements, and Nemotron 3.5 Lightning, added since by another contributor as an experimental version.
- **An installation you build yourself**: the Intel backend is part of TensorFold's code, but not yet of the ready-to-use archives published with each version.

The code, the measurement scripts and the raw results are public, so that anyone can rerun the tests. If you have an Arc Pro B70 at hand, replay our measurements and tell us what you get.

{{< callout label="Your documents stay with you." >}}
We size, install and tune the server on site, connect it to your sources and train your teams. After a day of scoping on site, we tell you whether a B70 covers your use.
{{< /callout >}}


## Sources

- hygurlabs/tensorfold-xpu: tools/xpu-bench/workflow_sim.py and raw results wf_*.json (office case)
- ashhart/TensorFold#802: Intel Arc Pro B70 backend, merged since version 1.0.12 (Apache-2.0)
- Internal measurement, TensorFold against vLLM v0.31 (MTP4, DFlash2) on Intel Arc Pro B70, bench of 6 October 2026
- RTX 5090 price (€5,700): LDLC listing at the time of writing, October 2026

