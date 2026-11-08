# Servir Laguna S 2.1 NVFP4 sur un seul DGX Spark (GB10) : le runbook complet

> Runbook vLLM pour servir Laguna S 2.1 NVFP4 (117 Md, MoE) sur un seul DGX Spark GB10 : éviter l'OOM au JIT, contexte 256K, décodage spéculatif DFlash, benchmarks de débit détaillés.

- Date: 2026-07-23
- Language: fr
- URL: https://0x0800.com/fr/blog/laguna-s21-single-spark/
- Tags: dgx-spark, gb10, laguna, vllm, nvfp4, dflash, moe
- Translation (en): https://0x0800.com/blog/laguna-s21-single-spark/


De la box tuée par son propre compilateur JIT au service 256K fonctionnel. Un MoE de 117,6 Md de paramètres (8,5 Md actifs), décodage spéculatif DFlash, contexte natif 256K, le tout sur une seule machine à mémoire unifiée — sans TP=2, sans cross-node, sans NCCL. Chaque commande de ce runbook est donnée verbatim.

Laguna tient sur une seule machine, là où DeepSeek-V4 imposait deux nœuds et une couche de coordination. Le point dur n'est donc plus la distribution mais la **mémoire de la box**, qu'un JIT non bridé épuise en quelques secondes. Ce runbook couvre le contexte matériel, les trois pièges qui tuent le boot, le lancement mono-nœud complet et les résultats de bench mesurés sur notre matériel.

## Contexte matériel

- **1× NVIDIA DGX Spark** (GB10, Grace-Blackwell, compute capability **12.1 / sm_121a**, 128 Go de mémoire unifiée, aarch64). Un seul nœud.
- Mémoire **unifiée** : CPU et GPU partagent les mêmes 128 Go. Conséquence directe : `--gpu-memory-utilization` borne l'allocation KV, mais pas les pics transitoires de profiling ni la compilation JIT. Un pic côté hôte prend sur la même réserve que les poids.
- Modèle : **poolside/Laguna-S-2.1-NVFP4** — MoE (256 experts routés + 1 partagé, 8,5 Md actifs sur 117,6 Md), 48 couches (12 attention globale + 36 sliding-window, fenêtre 512), **KV cache FP8**, contexte natif **256K** (262 144), poids natifs jusqu'à 1M. ~67 Go sur disque. **Tient sur un seul Spark : pas de TP=2, pas de cross-node, pas de RoCE, pas de NCCL.** Contrairement à DeepSeek-V4-Flash, aucune mécanique multi-nœud n'est nécessaire.
- Décodage spéculatif natif : drafter **DFlash** (`DFlashLagunaForCausalLM`), fourni séparément.

## Pourquoi le lancement « naïf » tue la box

Un `vllm serve` NVFP4 sur SM121, lancé sans précaution, ne plante pas proprement dans le process : il fait tomber **earlyoom**, qui envoie un SIGTERM au plus gros consommateur. Nous l'avons rencontré à trois reprises, à trois endroits différents :

| Étape | Symptôme | Cause |
|---|---|---|
| Profiling / dimensionnement KV | earlyoom SIGTERM vers ~116 Go | le forward de profiling (batch × MoE 256 experts) ajoute ~33 Go transitoires |
| Warmup / torch.compile | earlyoom tue cc1plus | JIT inductor qui compile en parallèle |
| Kernels FP4 | earlyoom tue cc1plus / nvcc | le vrai coupable — voir l’encart ci-dessous |

Le point commun : aucun de ces pics n'est borné par `--gpu-memory-utilization`. Ce flag réserve le KV **après** le profiling ; il ne dit rien des allocations transitoires de l'hôte. earlyoom, lui, déclenche sur `MemAvailable` (page cache réclamable incluse) et non sur `free`. Il faut donc **réduire les pics à la source**, pas seulement baisser l'utilisation cible.

{{< callout label="Encart — MAX_JOBS=4 : le fan-out nvcc qui vide 128 Go" >}}
**Ce qu'on observe (fait).** Au premier boot sur cache `~/.cache/flashinfer` froid, vLLM compile en JIT les kernels NVFP4 (FlashInfer). nvcc éclate la compilation en un **fan-out non borné** de `cc1plus`. Chaque `cc1plus` d'un kernel FP4 est glouton ; lancés tous ensemble, ils saturent les 128 Go de mémoire unifiée alors que les poids (~67 Go) sont déjà résidents. earlyoom tranche, SIGTERM, boot mort. Aucun Xid, aucune trace CUDA : ce n'est pas le GPU qui tombe mais l'**hôte** qui manque de RAM.

**Le correctif** (issu de la section DGX Spark du README du modèle). Plafonner le fan-out :

```bash
export MAX_JOBS=4
```

Quatre `cc1plus` concurrents au lieu de N. La compilation est un peu plus lente au premier boot, mais elle **tient dans l'enveloppe mémoire**. Combiné à `CUTE_DSL_ARCH=sm_121a` (l'arch attendue par le JIT des kernels FP4) et à `nvcc` dans le `PATH`, c'est ce qui fait passer le lancement.

**Piège persistant.** Ne jamais retirer `MAX_JOBS=4` en pensant « le cache est chaud, ça ira ». Un `~/.cache/flashinfer` vidé, une mise à jour de FlashInfer ou un changement d'arch relance le fan-out complet. Sans le plafond, la box retombe. Le garder est gratuit une fois le cache chaud (rien à recompiler) ; l'oublier coûte un aller-retour en salle serveur.
{{< /callout >}}

## Prérequis : venv et modèles

Le venv **~/.venv-vllm** (le fork jasl construit pour DeepSeek-V4, vLLM `20260721b1.dev0`) charge Laguna sans modification : c'est celui que nous utilisons ici. Le README du modèle recommande un vLLM stock **0.25.1** dans un venv dédié ; les deux marchent, nous sommes restés sur le fork, déjà en place et plus récent.

Deux dépôts locaux, poids **et** drafter :

```text
~/models/lagunas21-nvfp4          # poids principaux (~67 Go)
~/models/lagunas21-nvfp4-dflash   # drafter DFlash pour le spéculatif
```

Recette stock, pour mémoire — venv dédié `vllm025` : `uv venv` + `python3.12-dev`, `vllm==0.25.1`, torch cu130, et le trio FlashInfer nightly `flashinfer-python` / `flashinfer-cubin` / `flashinfer-jit-cache==0.6.15.dev20260712`. La section DGX Spark du README donne les pins exacts.

## Variables d'environnement requises

Trois familles : le tuning Marlin/FP8 usuel, puis les **trois** variables Spark sans lesquelles le JIT FP4 casse ou fait tomber la box. Réunies dans le script de lancement (§5) :

```bash
export VLLM_MARLIN_USE_ATOMIC_ADD=1
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1
export VLLM_FLOAT32_MATMUL_PRECISION=high
export VLLM_USE_NCCL_SYMM_MEM=0
export SAFETENSORS_FAST_GPU=1
export VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
# README — section DGX Spark (les trois qui comptent)
export CUTE_DSL_ARCH=sm_121a          # arch du JIT des kernels FP4
export PATH=/usr/local/cuda/bin:$PATH # nvcc pour le JIT
export MAX_JOBS=4                     # plafond du fan-out nvcc (sans ça, OOM de la box)
```

## Lancement mono-nœud

Script `~/launch.sh` — un seul process, API OpenAI sur `:8082` :

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

Trois réglages tiennent l'enveloppe mémoire, en plus des variables d'env :

- **--max-num-batched-tokens 4096** (défaut 8192). Le forward de profiling se fait sur ce batch ; à 8192 × MoE 256 experts, le pic dépasse la réserve et earlyoom tranche. 4096 passe.
- **--gpu-memory-utilization 0.85** (pas 0.92). Laisse une marge à l'hôte pour le JIT et le warmup ; à 0.92 le `cc1plus` d'inductor n'a plus de place.
- **--max-num-seqs 8**. DFlash crashe au défaut 256 ; le README recommande 32, nous sommes descendus à 8 pour le contexte 256K.

Lancer en avant-plan dans un `tmux` pour que la session tienne le process et capture le log ; les kernels JIT sont mis en cache pour les démarrages suivants :

```bash
tmux new-session -d -s laguna "bash ~/launch.sh > ~/laguna.log 2>&1"
```

**Démarrage abouti.** `health 200`, API OpenAI sur `:8082`. Boot à froid ~9 min (chargement des 67 Go + JIT des kernels FP4) ; les boots suivants, cache chaud, sont bien plus courts. Le tool-calling `poolside_v1` répond correctement (`tool_calls` standard OpenAI).

## Discipline mémoire : relancer sans se faire tuer

La mémoire unifiée impose une hygiène de redémarrage. Deux principes.

**earlyoom reste le filet, on ne le désarme pas.** Il déclenche sur `MemAvailable`, page cache réclamable comprise. Un checkpoint chaud en page cache compte donc comme disponible. Mais si un pic réel dépasse la réserve, earlyoom tue vLLM **au lieu de figer la box** : mieux vaut un process mort qu'un aller-retour en salle serveur.

**Kill PUIS drop_caches, jamais groupés.** Après un run, la RAM peut rester à ~110 Go (poids encore résidents). Si on relance sans réclamer, le loader voit trop peu de RAM libre pour le checkpoint (`Available RAM: 46.27 GiB` < `Checkpoint size: 66.98 GiB`) et rame. Il faut réclamer, mais en **étapes séparées et fiables**, jamais dans une commande unique : le kill casse la connexion ssh avec un exit 255, et le `drop_caches` groupé ne part jamais.

```bash
# 1. tuer vLLM
pkill -f 'vllm serve' ; tmux kill-session -t laguna 2>/dev/null

# 2. CONFIRMER que le process est bien mort (étape à part)
pgrep -af 'vllm serve'   # doit ne rien renvoyer

# 3. SEULEMENT alors, réclamer la mémoire (étape à part)
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches

# 4. vérifier la RAM libre avant de relancer
free -g

# 5. relancer
tmux new-session -d -s laguna "bash ~/launch.sh > ~/laguna.log 2>&1"
```

Ne jamais lancer `drop_caches` tant que vLLM tourne, ne jamais coller le kill et le drop dans la même commande.

## KV cache à 256K

À `--max-model-len 262144` avec KV cache **FP8** et `--gpu-memory-utilization 0.85`, vLLM alloue un **KV cache de 738 410 tokens**, soit **~2,82× de concurrence** au contexte plein 256K. L'attention sliding-window (36 couches sur 48, fenêtre 512) est décisive : seules les 12 couches globales portent le KV long, ce qui rend le contexte 256K tenable sur une seule box. Contrairement à DeepSeek-V4, il n'a **pas** fallu rogner le contexte pour caser le modèle : 256K natif, dès le premier lancement stable.

## Résultats du bench

Bench de débit (`bench_laguna.py`, stdlib urllib), config du §5 (**DFlash**, `num_speculative_tokens=7`) :

| Condition | Débit décode |
|---|---|
| code, thinking OFF, 512 tok | 49,3 tok/s (min 44,3 / max 52,9) |
| code, thinking OFF, 1024 tok | 51,0 tok/s |
| code, thinking ON, 512 tok | 52,4 tok/s |
| prose, thinking OFF, 512 tok | 30,7 tok/s |

| Prefill (TTFT streaming) | |
|---|---|
| ~4k ctx (4 490 tok) | TTFT 1,43 s → 3 130 tok/s |
| ~16k ctx (17 820 tok) | TTFT 3,97 s → 4 483 tok/s |

| Concurrence (agrégé) | |
|---|---|
| code OFF 512, 2 connexions | 54,3 tok/s agrégé |
| code OFF 512, 4 connexions | 88,8 tok/s agrégé |

**Lecture.** Le décode dépend surtout du type de contenu, pas du *thinking* (code ON ≈ code OFF, ~49–52 tok/s). La cause est la **longueur d'acceptation** de DFlash, qu'on a mesurée poste par poste en lisant le delta des compteurs `vllm:spec_decode_*` de `/metrics` autour de générations isolées :

| Contenu | Drafts acceptés / pas | Longueur d'acceptation |
|---|---|---|
| code (structure prévisible) | 4,36 | 5,36 tokens/pas |
| prose | 2,12 | 3,12 tokens/pas |

(longueur d'acceptation = 1 + drafts acceptés/pas, soit le nombre de tokens émis par passe forward.) Ce facteur ~1,7× entre code et prose explique l'écart code ~50 / prose ~30 tok/s : sur du code, chaque passe sort plus de 5 tokens. Le prefill reste rapide (3 000–4 500 tok/s) et le TTFT sur prompt court tombe à **252 ms** (médiane sur 5, min 245 / max 294) — le contexte long n'écroule pas le premier token.

{{< callout label="Correctif" >}}
Une version antérieure de cette note affirmait « DFlash accepte 2,9–3,1 tokens/pas sur du code ». C'était en fait le chiffre de la **prose**, mal étiqueté ; la mesure isolée par `/metrics` donne 5,36 tokens/pas sur du code.
{{< /callout >}}

### Bench métier — 209 cas, 5 conditions

Harnais `bench_guided.py` (depuis un second Spark vers le nœud de service `:8082`), spéculatif DFlash actif :

| Condition | Juste | Docile | Hallu |
|---|---|---|---|
| A — libre | 181/209 (86,6 %) | 99,5 % | 0 |
| C — guided_choice | 204/209 (97,6 %) | 100 % | 5 |
| D — retrieval + guided | 204/209 (97,6 %) | 100 % | 4 |
| E — retrieval strict + guided | 202/209 (96,7 %) | 100 % | 6 |
| F — retrieval + firewall (LP) | 205/209 (98,1 %) | 100 % | 4 |

**Zéro erreur dure** sur les 5 conditions. La docilité est parfaite (100 %) dès qu'un guidage est en place ; en mode libre (A), la justesse retombe à 86,6 %, l'écart attendu entre décodage libre et décodage contraint. Le firewall par logit processor (F) offre le meilleur compromis justesse/hallucination : 98,1 %, 4 hallucinations. Les conditions guidées hallucinent un peu plus que DeepSeek-V4 sur le même corpus (4 à 6 contre 1 à 4), sans jamais produire d'erreur dure.

## Note — num_speculative_tokens : 7 plutôt que 15

À `num_speculative_tokens=15`, DFlash draftait trop loin : sur du contenu à faible acceptation (prose), la majorité des tokens spéculés étaient rejetés, et le compute de draft gaspillé faisait **chuter** le débit. Comparaison directe, même config par ailleurs :

| Métrique | spec=15 | spec=7 |
|---|---|---|
| prose OFF 512 | 26,0 tok/s | 30,7 tok/s (+18 %) |
| code OFF 512 | 49,6 tok/s | 49,3 tok/s (≈) |
| prefill ~4k | 3 004 tok/s | 3 130 tok/s (+4 %) |

spec=7 gagne nettement sur la prose et égale le code : c'est la valeur retenue dans le script du §5. Un drafter spéculatif se règle sur le **profil d'acceptation** du contenu servi, pas sur l'idée que « plus de tokens = mieux ».

## Les chiffres à retenir

{{< keyfigures >}}
[
  {"label": "KV cache à 256K", "value": "738 410 tokens", "detail": "≈2,82× de concurrence au contexte plein, grâce au sliding-window (36/48 couches)"},
  {"label": "Débit décode", "value": "49–52 tok/s", "detail": "code, spéculatif DFlash actif (num_speculative_tokens=7)"},
  {"label": "TTFT prompt court", "value": "252 ms", "detail": "médiane sur 5 essais, contexte long sans effet sur le premier token"},
  {"label": "Bench métier (209 cas)", "value": "98,1 %", "detail": "meilleure condition (retrieval + firewall LP), zéro erreur dure sur les 5 conditions"}
]
{{< /keyfigures >}}

Un modèle qui tient sur une seule box, un contexte 256K natif et un drafter spéculatif qui double presque le débit sur du code : le point dur n'est pas l'inférence elle-même, mais l'hygiène mémoire d'une architecture unifiée. Trois garde-fous suffisent — `MAX_JOBS=4` au premier boot, des marges prudentes sur `--gpu-memory-utilization` et `--max-num-batched-tokens`, et un cycle kill/drop_caches en étapes séparées à chaque relance.

{{< callout label="Nous installons et réglons votre DGX Spark." >}}
Si ce runbook vous fait gagner du temps mais que vous préférez confier l'installation, le réglage mémoire et la mise en production à quelqu'un qui l'a déjà fait, écrivez-nous.
{{< /callout >}}

## À lire aussi

- [DeepSeek-V4-Flash-0731 sur deux DGX Spark : la recette complète](/fr/blog/dsv4-2x-spark-recette/) : 96/100 de code juste sur 100 tirages jugés par exécution, 188,9 tok/s agrégés à seize flux, thermique et latence RDMA relevées sous charge.
- [Faire tourner vLLM proprement sur un DGX Spark](/fr/blog/dgx-spark-vllm-nvfp4/) : les trois bugs qui bloquent le build et le premier serveur sur l'architecture SM121, et comment les corriger.


## Sources

- Mesures sur notre DGX Spark (GB10), juillet 2026 : bench_laguna.py (débit, prefill, concurrence), compteurs vllm:spec_decode_* de /metrics (longueur d'acceptation), bench_guided.py (bench métier, 209 cas, 5 conditions)
- poolside/Laguna-S-2.1-NVFP4 : README du modèle, section DGX Spark (MAX_JOBS, CUTE_DSL_ARCH, pins FlashInfer, vLLM 0.25.1)

