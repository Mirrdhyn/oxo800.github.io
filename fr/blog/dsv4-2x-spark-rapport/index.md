# DeepSeek-V4-Flash 0731 sur 2× DGX Spark : montage, mesures, réglages

> Douze paramètres de service écartés sur seize, une falaise de latence de +31 s avec b12x et 188,9 tok/s à seize flux avec DeepSeek-V4-Flash sur deux DGX Spark.

- Date: 2026-08-04
- Language: fr
- URL: https://0x0800.com/fr/blog/dsv4-2x-spark-rapport/
- Tags: dgx-spark, deepseek-v4, vllm, tp2, inference-locale
- Translation (en): https://0x0800.com/blog/dsv4-2x-spark-rapport/


{{< keyfigures >}}
[
  {"label": "Qualité retenue", "value": "96/100", "detail": "jugée par exécution, avant tout débit"},
  {"label": "Mono-flux", "value": "78,55 tok/s", "detail": "code 512, au plafond imposé par k"},
  {"label": "Seize flux", "value": "188,9 tok/s", "detail": "le seul réglage qui ait produit un gain"},
  {"label": "Pool KV", "value": "1,79 M", "detail": "tokens, contexte servable 393 216"},
  {"label": "Paramètres écartés", "value": "12 sur 16", "detail": "la configuration la plus sobre est l’optimum"},
  {"label": "Falaise b12x", "value": "+31 s", "detail": "de coût fixe dès ~1850 tokens de contexte"}
]
{{< /keyfigures >}}

Rapport reproductible. Il décrit comment monter la stack de bout en bout, quels chiffres on obtient, ce que chaque paramètre change, et quelle combinaison sort le meilleur compromis performance/qualité.

Tous les chiffres viennent de mesures faites sur le matériel décrit ci-dessous. Ce qui est estimé ou non vérifié est signalé comme tel.

## Matériel et logiciel

Deux DGX Spark reliés en RoCE, un GPU par nœud, tensor-parallel 2.

| | spark-1 (tête) | spark-2 (worker) |
|---|---|---|
| rôle | `--node-rank 0`, sert l'API | `--node-rank 1 --headless` |
| noyau | 6.17.0-1026-nvidia | idem |
| CPU | Cortex-X925, 20 threads | idem |
| RAM | 121,6 GiB | idem |
| GPU | NVIDIA GB10, pilote 595.84, horloge plafond 3003 MHz | idem |
| CUDA | 13.2 | idem |
| RoCE | 2 HCA × 200 Gb/s | idem |

Stack logicielle mesurée dans ce rapport (voie A) :

| composant | version |
|---|---|
| Python | 3.12.13 |
| torch | 2.13.0+cu132 |
| NCCL | 2.29.7 |
| FlashInfer | 0.6.14 |
| vLLM | fork `jasl/vllm`, branche `b12x-nvfp4-mxfp4`, commit `48d274954` |
| checkpoint | `DeepSeek-V4-Flash-0731`, 155,43 GiB, experts MXFP4, dense/attn FP8 E4M3 |

Le GB10 est **sm_121**. Plusieurs chemins de code exigent du Blackwell datacenter (sm_10x) et sont donc indisponibles ici (voir §6).

## Voie A : monter la stack depuis le fork

C'est la voie utilisée pour toutes les mesures de ce rapport.

### 2.1 Pourquoi un fork

Le vLLM amont ne fait pas tourner ce modèle sur GB10 en l'état : `hyperconnection.hpp` rejette l'architecture. Le fork `jasl/vllm` porte le support sm_121 et suit l'amont : la branche `b12x-nvfp4-mxfp4` intègre `upstream/main` au 21/07/2026 et ses correctifs continus. C'est ce fork qui sert ici.

Le NCCL qui tourne est **2.29.7**, celui auquel torch 2.13.0+cu132 est lié (`torch.cuda.nccl.version()`). Le venv contient aussi un paquet `nvidia-nccl-cu13 2.30.4`, mais torch ne le charge pas : installer une version plus récente par pip ne change pas le NCCL effectif. Vérifier par l'appel torch, pas par `pip list`.

La pile mesurée dans ce rapport n'embarque **ni DeepGEMM ni correctif d'indexeur** : `import deep_gemm` échoue dans le venv de mesure, et aucun fichier d'indexeur n'est modifié. Ces deux éléments appartiennent à la variante requantifiée en NVFP4, qui est un autre chantier.

### 2.2 Étapes

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

L'étape 4 est celle sans laquelle rien ne démarre : le fork épingle `torch == 2.11.0` dans `pyproject.toml`, `requirements/build/cuda.txt` et `requirements/cuda.txt`, alors que le GB10 a besoin de 2.13.0+cu132. Le patch retire ces épinglages et les dépendances `torchaudio`/`torchcodec` qui entraînent la résolution avec elles. Il ne touche à aucune ligne de code du moteur.

**Contrainte TP=2 :** les deux nœuds doivent avoir des venvs **appariés** : même arbre source (même hash) et même version de torch. Un écart d'un commit suffit à faire échouer ou dériver le tensor-parallel. Vérifier avant chaque campagne :

```bash
for h in spark-1 spark-2; do
  ssh $h '~/.venv-dsv4/bin/python -c "import vllm,torch;print(vllm.__version__,torch.__version__)"'
done
```

### 2.3 Lancement

Le worker démarre en premier, la tête ensuite. **Toujours dans tmux** : un `ssh + nohup` sur un process long se fait couper, et une relance fantôme écrase le log.

```bash
# sur spark-2 (worker)
tmux new-session -d -s dsv4 -c ~ "ENVS ./dsv4_run.sh worker N0731 2>&1 | tee ~/worker.log"

# sur spark-1 (tête), 10 s plus tard
tmux new-session -d -s dsv4 -c ~ "APIHOST=127.0.0.1 APIPORT=8090 ENVS ./dsv4_run.sh head N0731 2>&1 | tee ~/head.log"
```

Arguments de service effectifs :

```args
--tensor-parallel-size 2 --distributed-executor-backend mp
--max-model-len 393216 --max-num-seqs 2 --max-num-batched-tokens 4096
--kv-cache-dtype fp8_ds_mla --block-size 256 --enable-prefix-caching
--gpu-memory-utilization 0.85
--speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"greedy"}'
--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'
```

Variables d'environnement :

```env
CUTE_DSL_ARCH=sm_121a
VLLM_USE_FLASHINFER_SAMPLER=1
VLLM_USE_BREAKABLE_CUDAGRAPH=0
```

### 2.4 Arrêt propre

Un `pkill -f dsv4_run.sh` ne suffit pas et se déclenche sur lui-même. La séquence correcte :

```bash
tmux kill-session -t dsv4                       # sur les deux nœuds
pkill -9 -f "[b]in/vllm"; pkill -9 -f "[V]LLM::" # le traînard Worker survit sinon
sync; sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
```

Le `drop_caches` avant chaque lancement évite une pression mémoire qui se traduit par un démarrage bloqué (voir §6.4).

## Voie B : la stack du dépôt amont (recette exacte)

Le dépôt d'un opérateur tiers publie ses chiffres sur un runtime différent : l'image `ghcr.io/anemll/dspark-vllm-gx10:0.1.1`, dont les sources sont dans `Anemll/dspark-vllm-gx10`. Son `upstream.lock` épingle tout :

```lock
VLLM_TAG=v0.25.1              VLLM_COMMIT=752a3a504485790a2e8491cacbb35c137339ad34
FLASHINFER_COMMIT=0472b9b3f2fba11b463f8526f390297d52a8aad7   FLASHINFER_VERSION=0.6.15
B12X_REPOSITORY=https://github.com/lukealonso/b12x.git
B12X_COMMIT=7dc6fb8fcc6446ea093537d1657df81985fa5f43         B12X_VERSION=0.15.3
CUTLASS_DSL_VERSION=4.5.2     CUDA_PYTHON_VERSION=13.3.1     TVM_FFI_VERSION=0.1.9
TORCH_VERSION=2.11.0+cu130    CUDA_VERSION=13.0.2            PYTHON_VERSION=3.12
```

**Point important : vLLM n'y est pas compilé.** Leur image part de la roue officielle `vllm/vllm-openai:v0.25.1` et applique un overlay de 12 fichiers Python. Transposé en venv natif, sans Docker :

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

Liste exhaustive de l'overlay (12 fichiers) :

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

Écart de code avec la voie A, mesuré par `diff` :

| fichier | Anemll | fork jasl | lignes divergentes |
|---|---:|---:|---:|
| `b12x_mxfp4_moe.py` | 877 | 785 | 83 |
| `flashmla_sparse.py` | 895 | 876 | 201 |
| `models/deepseek_v4/attention.py` | 825 | 921 | 141 |
| `models/deepseek_v4/sparse_mla.py` | 408 | 470 | 80 |
| `envs.py` | 2 179 | 2 328 | 210 |
| `config/cache.py` | 294 | 293 | 26 |

{{< hbars title="Lignes divergentes entre l'overlay Anemll et le fork jasl" caption="Les deux fichiers les plus divergents (b12x_mxfp4_moe.py, flashmla_sparse.py) sont ceux qui portent les comportements mesurés en §5.2 et §5.3." minwidth=600 barheight=16 marginleft=230 >}}
{"bars": [{"label": "envs.py", "value": 210, "text": "210 lignes", "color": "accent"}, {"label": "flashmla_sparse.py", "value": 201, "text": "201 lignes", "color": "accent"}, {"label": "models/deepseek_v4/attention.py", "value": 141, "text": "141 lignes", "color": "accent"}, {"label": "b12x_mxfp4_moe.py", "value": 83, "text": "83 lignes", "color": "accent"}, {"label": "models/deepseek_v4/sparse_mla.py", "value": 80, "text": "80 lignes", "color": "accent"}, {"label": "config/cache.py", "value": 26, "text": "26 lignes", "color": "accent"}]}
{{< /hbars >}}

Les deux fichiers les plus divergents (`b12x_mxfp4_moe.py`, `flashmla_sparse.py`) sont ceux qui portent les comportements mesurés en §5.2 et §5.3.

### 3.1 Points vérifiés au montage

**torch 2.11.0+cu130 fonctionne sur GB10.** La roue n'embarque pas de cubin sm_121 :

```python
>>> torch.cuda.get_arch_list()
['sm_80', 'sm_90', 'sm_100', 'sm_110', 'sm_120']
>>> torch.cuda.get_device_capability(0)
(12, 1)
```

Le GB10 est bien sm_121, et le support passe par la compatibilité binaire sm_120 : un `matmul` bfloat16 sur GPU donne le bon résultat. Aucun build torch maison n'est nécessaire pour la voie B, contrairement à la voie A.

Réserve restante : `b12x` et `flashinfer` s'installent en `--no-build-isolation` et compilent donc localement. Ils épinglent CUDA 13.0.2 ; sur une machine en 13.2, la compilation peut diverger.

### 3.2 Profil de service amont

Reproduit depuis leur `docker-compose.dspark.yml`, à poser tel quel sur un `vllm serve` natif :

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

Deux détails qui comptent :

- `MTP_NUM_TOKENS` vaut 3 par défaut dans leur compose, mais leur propre README précise que le `dspark_block_size` du checkpoint est 5 et que **k doit être ≥ 5**. Prendre 5.
- `VLLM_USE_BREAKABLE_CUDAGRAPH=0` est **obligatoire** ici, sinon leur code bascule tout seul sur le chemin lent (voir §5.5).

### 3.3 État

**Montage vérifié jusqu'à l'import** : `vllm 0.25.1`, `b12x 0.15.3`, `flashinfer-python 0.6.15`, overlay appliqué (11 fichiers écrasés, 1 nouveau, `b12x_mxfp4_moe.py`), imports de `b12x_mxfp4_moe` et `models.deepseek_v4.sparse_mla` concluants.

{{< callout label="État réel de la voie B" >}}
**Au 3 août 2026, la voie B n'a jamais été démarrée ni mesurée.** La réplication sur le second nœud, nécessaire au TP=2, n'a pas été faite. **Tous les chiffres de ce rapport sont ceux de la voie A**, sans exception. Cette section documente un montage vérifié jusqu'à l'import, rien de plus. Nous la publions parce que la recette exacte est difficile à reconstituer, pas parce qu'elle porte un résultat.
{{< /callout >}}

Conseil de reproductibilité : sauvegarder chaque fichier écrasé avant de poser l'overlay, ce qui permet de revenir au 0.25.1 stock sans réinstaller.

```bash
cd overlay/vllm && find . -name '*.py' | while read f; do
  cp -n "$SP/$f" "$SP/$f.orig" 2>/dev/null
done
cp -r overlay/vllm/. "$SP/"
```

## Protocole de mesure

Trois instruments, dans cet ordre. Le débit n'est jamais publié sans la qualité.

### 4.1 Barrière qualité : banc de code

20 tâches (10 Python, 10 Go) × 5 tirages, budget 16 000 tokens. Chaque sortie est compilée, vétée et testée. Le score est le nombre de tirages **justes** sur 100.

```bash
URL=http://127.0.0.1:8090/v1 MODEL=deepseek-v4-flash TAG=<bras> \
  python3 codebench/run.py --out resultats-<bras>.json
```

On relève aussi le nombre de **tâches instables** (ni 0/5 ni 5/5), qui détecte une dégradation que le score global masque.

### 4.2 Débit de décodage

Sonde en streaming, préfixe unique par requête pour défaire le cache de préfixe, 5 générations longues de préchauffage, puis niveaux de concurrence × répétitions.

Formule, qui exclut le prefill :

```formule
decode_tok_s = (tokens_out - 1) / (durée_totale - TTFT)
```

### 4.3 Latence au premier token

C'est l'instrument qui manque à la plupart des bancs, y compris celui du dépôt amont : les deux mesurent le décodage **à partir du** premier token. Une pénalité de prefill y est structurellement invisible.

La sonde envoie un prompt calé au token près via `/tokenize`, `max_tokens=16`, et chronomètre jusqu'au premier delta, en comptant `content` **et** `reasoning_content`, sans quoi le modèle raisonne sans jamais émettre de `content` et la mesure ne renvoie rien.

### 4.4 Télémétrie machine

Échantillonnage à 2 s : puissance GPU, horloge SM, température, occupation, RAM libre, débits RoCE (compteurs en mots de 4 octets).

Limites du GB10 à connaître : `power.draw` ne couvre que le domaine GPU ; `power.limit`, `memory.used` et `memory.total` renvoient `[N/A]` (mémoire unifiée) ; aucun capteur de puissance module dans `/sys/class/hwmon` ; pas de `dcgmi`.

## Résultats

### 5.1 Trois combinaisons, qualité et débit

Même serveur, même checkpoint, une variable à la fois.

| bras | MoE | KV cache | qualité /100 | Python | Go | instables | C1 tok/s | C2 tok/s | agrégé C2 | pool KV |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **A — base** | **MARLIN** | **`fp8_ds_mla`** | **98** | **50/50** | **48/50** | **1** | **47,86** | **35,80** | **68,71** | **1 859 337** |
| **B — b12x** | B12X_MXFP4 | `fp8_ds_mla` | **96** | 50/50 | 46/50 | 3 | **38,77** | 34,50 | 65,21 | 1 793 221 |
| **C — empilé** | B12X_MXFP4 | `nvfp4_ds_mla` | **92** | 48/50 | 44/50 | 6 | — | — | — | 1 811 142 |

Répétitions du bras B, pour montrer que l'écart n'est pas du bruit : C1 = 38,74 / 38,77 / 39,53 (dispersion 2 %) ; C2 = 34,39 / 34,57 / 34,50.

{{< hbars title="Qualité par bras, sur 100 tirages" caption="Qualité d'abord, mesurée par exécution du code sur 100 tirages. Empiler b12x et nvfp4 coûte six points au bras C, et fait passer les tâches instables de 1 à 6." minwidth=620 barheight=18 marginleft=160 max=100 marginright=170 >}}
{"bars": [{"label": "A — base", "value": 98, "text": "98/100 · 1 instable", "color": "signal"}, {"label": "B — b12x", "value": 96, "text": "96/100 · 3 instables", "color": "signal"}, {"label": "C — empilé", "value": 92, "text": "92/100 · 6 instables", "color": "alert"}]}
{{< /hbars >}}

{{< groupedbars title="Débit par bras, un et deux flux" caption="Débit des deux bras mesurés. b12x coûte 19 % à un flux ; l'écart se resserre à deux flux (−3,6 %), ce qui est cohérent avec un noyau taillé pour les gros lots." minwidth=620 marginleft=150 >}}
{"series": [{"label": "C1", "color": "accent"}, {"label": "C2", "color": "signal"}, {"label": "agrégé C2", "color": "accent-deep"}], "groups": [{"label": "A — base", "values": [47.86, 35.8, 68.71], "texts": ["47,86", "35,80", "68,71"]}, {"label": "B — b12x", "values": [38.77, 34.5, 65.21], "texts": ["38,77", "34,50", "65,21"]}]}
{{< /groupedbars >}}

Lectures :

- **b12x coûte 19 % de débit à C1** dans ce régime. L'écart se resserre avec la concurrence (−19 % à C1, −3,6 % à C2), ce qui est cohérent avec un noyau taillé pour les gros lots. Hypothèse non testée au-delà de C2.
- **`nvfp4_ds_mla` coûte de la qualité sans rendre de mémoire.** Le pool KV varie de 0,5 % entre B et C, dans la variabilité d'allocation entre deux démarrages. Le log de boot indique `padded nvfp4_ds_mla` : c'est la page de 584 octets, pas la disposition compacte de 416 octets. Le score passe de 96 à 92 et Python descend à 48/50 pour la première fois.

### 5.2 Falaise de latence au premier token

Même sonde, prompts calés au token près, 2 tirages par point.

| tokens réels | A — MARLIN + fp8 | B — b12x + fp8 | C — b12x + nvfp4 |
|---:|---:|---:|---:|
| 1 760 | 1,15 / 1,14 | 1,17 / 1,19 | 0,95 / 1,07 |
| 1 793 | 1,05 / 1,03 | 1,07 / 1,06 | 1,08 / 1,09 |
| 1 826 | 1,05 / 1,05 | 1,07 / 1,08 | 1,10 / 1,10 |
| **1 856** | **1,06 / 1,06** | **31,85 / 31,90** | **32,43 / 32,43** |
| 1 889 | 1,07 / 1,07 | 32,27 / 32,61 | 33,13 / 32,73 |
| 1 922 | 1,08 / 1,08 | 32,60 / 32,88 | 33,00 / 33,04 |

{{< linechart title="Latence au premier token selon la longueur du prompt" caption="Moyenne des deux tirages de chaque point. Une seule variable sépare A de B : le backend MoE. Activer b12x ajoute ~31 secondes de coût fixe par requête dès ~1850 tokens de contexte, de façon déterministe (dispersion sous 1 %)." minwidth=620 xlabel="tokens réels de prompt" ymin=0 ymax=36 ydec=0 yunit=" s" ylabel="TTFT" >}}
{"xs": [1760, 1793, 1826, 1856, 1889, 1922], "xticks": [1760, 1793, 1826, 1856, 1889, 1922], "yticks": [0, 9, 18, 27, 36], "series": [{"label": "A — MARLIN + fp8", "points": [1.145, 1.04, 1.05, 1.06, 1.07, 1.08], "color": "signal"}, {"label": "B — b12x + fp8", "points": [1.18, 1.065, 1.075, 31.875, 32.44, 32.74], "color": "alert"}, {"label": "C — b12x + nvfp4", "points": [1.01, 1.085, 1.1, 32.43, 32.93, 33.02], "color": "accent-deep"}]}
{{< /linechart >}}

**Une seule variable sépare A de B : le backend MoE.** Activer b12x ajoute **~31 secondes de coût fixe par requête** dès ~1850 tokens de contexte, de façon déterministe (dispersion sous 1 %). Le KV cache n'y est pour rien : `nvfp4_ds_mla` donne le même résultat que `fp8_ds_mla` une fois b12x actif.

Au-delà de la marche, la pente reprend : 1939 → 33,46 s, 2131 → 35,61 s, 2323 → 37,96 s, soit ~87 tok/s de prefill marginal.

Conséquence pratique : un dossier de quelques milliers de tokens dépasse trivialement ce seuil. Avec b12x, chaque requête paie une demi-minute avant le premier mot.

### 5.3 Effet du protocole de mesure

Sur le **même serveur, à la même minute, même stack** (bras B) :

| banc | tok/s |
|---|---:|
| banc de code (tâches exécutables) | 38,77 |
| **script du dépôt amont** | **55,35** |

{{< hbars title="Débit selon le banc, à stack identique" caption="+43 % pour un simple changement de charge, sur le même serveur à la même minute. Une énumération numérotée est le meilleur cas possible pour la spéculation ; du code exécutable est parmi les pires." minwidth=520 barheight=20 marginleft=250 >}}
{"bars":[
  {"label":"banc de code (exécutable)","value":38.77,"text":"38,77 tok/s","color":"accent"},
  {"label":"script du dépôt amont","value":55.35,"text":"55,35 tok/s","color":"signal"}]}
{{< /hbars >}}

+43 % pour un simple changement de charge. Le script amont demande *« exactly 128 numbered lowercase English words »* sur un contexte de remplissage répété : une énumération numérotée, que la spéculation MTP prédit très bien.

Deux caractéristiques de cette charge expliquent l'écart :

- Le prompt commence par `unique request {nonce}`, ce qui **défait volontairement le cache de préfixe**. Chaque requête paie son prefill.
- La métrique de sortie compte le `reasoning_content`. Les longueurs réelles observées vont de 592 à 7 871 tokens pour la même consigne : la variance inter-requête est forte.

**Une énumération numérotée est le meilleur cas possible pour la spéculation.** Du code exécutable est parmi les pires. Les deux chiffres sont honnêtes ; ils ne mesurent simplement pas la même chose. C'est la raison pour laquelle tout le reste de ce rapport publie ses charges avec ses débits.

### 5.4 Deux métriques d'acceptation à ne pas confondre

Le serveur expose deux grandeurs aux noms voisins, dans le même bloc de journal :

| métrique | plage observée | ce qu'elle compte |
|---|---:|---|
| `acceptance rate` | 0,718 – 0,771 | longueur d'acceptation rapportée à `k` |
| `Draft acceptance rate` | 40,9 – 42,3 % | tokens de brouillon acceptés sur émis |

Elles n'ont pas le même dénominateur et diffèrent d'un facteur proche de 1,8. Relever l'une en croyant lire l'autre suffit à fabriquer un écart qui n'existe pas. Les tableaux de ce rapport utilisent la seconde, et le précisent.

### 5.5 Graphes CUDA sécables

Le tableau du dépôt se lit **colonne « graphes classiques »**, colonne « sécables ». Les graphes classiques sont les plus rapides, et leur README le confirme : Anemll active automatiquement le chemin sécable, plus lent, quand la variable n'est pas posée.

{{< callout label="Note du 3 août 2026" >}}
Les valeurs que nous citions ici (95,9 et 74,55 tok/s) proviennent d'une publication amont désormais remplacée. La note de méthodologie de la publication du 3 août les désavoue explicitement : « *Historical C1≈95.9 on older images is not this freeze* », et donne 79,2 tok/s pour le gel courant. Nous retirons donc ces deux chiffres du corps du texte : le raisonnement de cette section porte sur le **mécanisme** de bascule vers le chemin sécable, qui lui est vérifié dans le code ci-dessous et reste valable.
{{< /callout >}}

La différence est dans le code, et elle est nette :

- **Voie B (overlay Anemll)** : `config/vllm.py:1113-1131` auto-active le chemin sécable quand l'architecture est `DeepseekV4ForCausalLM` ou `DeepSeekV4MTPModel`, *sauf* si la variable est explicitement posée. Sans `VLLM_USE_BREAKABLE_CUDAGRAPH=0`, on obtient la configuration lente par construction. D'où l'avertissement de leur README.
- **Voie A (fork jasl)** : même défaut `False` dans `envs.py`, mais la liste d'auto-activation (`config/vllm.py:79-86`) ne contient qu'Inkling et MiniMax. DeepSeek V4 n'y est pas : **la configuration rapide est déjà celle par défaut**, il n'y a rien à activer.

À noter si vous le testez quand même : l'activer **désactive le pipeline torch.compile** (`vllm.py:1218`, équivalent à `-cc.mode=none`), et réduit le pool KV de 1 816 971 à 1 718 754 tokens (−5 %), les graphes capturés passant de 0,12 à 0,34 GiB.

### 5.6 Télémétrie sous charge

Sonde de décodage du bras B, 88 échantillons :

| grandeur | médiane | min | max |
|---|---:|---:|---:|
| puissance GPU | 43,12 W | — | 45,70 W |
| horloge SM | 2522 MHz | 2411 | 2541 |
| température | 68 °C | — | — |
| occupation | 95 % | — | — |
| RAM libre | 9,54 GiB | — | — |
| RoCE | 0,70 Gb/s | — | — |

Aucun throttling (plafond 3003 MHz jamais approché mais horloge stable), aucune saturation d'interconnexion (0,70 Gb/s sur des liens à 200 Gb/s). Le déficit de b12x est dans le noyau, pas dans la plateforme.

Repère utile : le seuil SIGTERM d'earlyoom est à ~4,86 GiB de RAM libre, la marge observée est de 9,54 GiB.

### 5.7 Balayage à un bouton

Après avoir établi que `torch.compile` n'est pas le levier (voir 5.8), on reprend la question du débit par les réglages de service. Méthode : un bouton à la fois, même checkpoint, même serveur redémarré à chaque bras, porte de qualité avant tout chiffre.

Instrument de comparaison entre bras : `~/bench_mono.py`, mono-flux, streaming, trois charges fixes. Il applique la formule de 4.2. Les chiffres de cette section se comparent **entre eux** ; ils ne se comparent pas aux colonnes C1 de 5.1, qui portent sur la charge du banc de code.

Base de référence du balayage :

```bash
CGMODE=FULL_AND_PIECEWISE NOEP=1 SPECK=5 GPUUTIL=0.85 ./dsv4_run.sh <head|worker> N0731
```

| bras | qualité | code 512 | code 1024 | prose 512 | indice C2 | pool KV |
|---|---:|---:|---:|---:|---:|---:|
| **B0 — base** | **96/100 (k=5)** | **78,55** | **75,33** | **47,16** | **77,3** | **1 796 092** |
| **B1 — --async-scheduling** | 40/40 (k=2) | 78,30 | 76,64 | 48,19 | 76,9 | 1 830 803 |
| **B2 — SPECK=3** | refusé au démarrage | — | — | — | — | — |
| **B3 — CGCAPTURE=12** | 39/40 (k=2) | 77,52 | 75,96 | 49,46 | 74,6 | 1 803 747 |
| **B4 — BATCHTOK=8192** | 40/40 (k=2) | 77,18 | 76,92 | 48,05 | 76,7 | 1 029 078 |
| **B5 — REJSAMPLE=block** | 38/40 (k=2) | 76,40 | 73,97 | 48,87 | 77,4 | 1 790 611 |

{{< groupedbars title="Balayage à un bouton, trois charges et indice de concurrence" caption="Les cinq bras du balayage, tous mesurés de la même façon sur la même forme de rafale. Ils se comparent entre eux, pas aux colonnes C1 du §5.1. B2 (SPECK=3) est refusé au démarrage et n'a pas de chiffre." minwidth=660 marginleft=185 max=85 >}}
{"series": [{"label": "code 512", "color": "accent"}, {"label": "code 1024", "color": "accent-deep"}, {"label": "prose 512", "color": "signal"}, {"label": "indice C2", "color": "faint"}], "groups": [{"label": "B0 — base", "values": [78.55, 75.33, 47.16, 77.3], "texts": ["78,55", "75,33", "47,16", "77,3"]}, {"label": "B1 — --async-scheduling", "values": [78.3, 76.64, 48.19, 76.9], "texts": ["78,30", "76,64", "48,19", "76,9"]}, {"label": "B2 — SPECK=3", "values": [null, null, null, null], "texts": ["refusé au démarrage", " ", " ", " "]}, {"label": "B3 — CGCAPTURE=12", "values": [77.52, 75.96, 49.46, 74.6], "texts": ["77,52", "75,96", "49,46", "74,6"]}, {"label": "B4 — BATCHTOK=8192", "values": [77.18, 76.92, 48.05, 76.7], "texts": ["77,18", "76,92", "48,05", "76,7"]}, {"label": "B5 — REJSAMPLE=block", "values": [76.4, 73.97, 48.87, 77.4], "texts": ["76,40", "73,97", "48,87", "77,4"]}]}
{{< /groupedbars >}}

{{< callout label="Ce que l’« indice C2 » est, et n’est pas" >}}
**L'« indice C2 » n'est pas un débit à deux flux.** C'est la moyenne des lignes `Avg generation throughput` du journal serveur filtrées sur `Running: 2`, sur des rafales courtes. Or cette ligne du journal divise un nombre de tokens par une fenêtre de dix secondes, quoi qu'il se soit passé dedans : une fenêtre où les deux flux n'ont tourné que la moitié du temps compte comme les autres et tire la moyenne vers le bas. Sur une mesure soutenue de cinq minutes, le même serveur donne 100,4 tok/s et le client compte la même chose à 0,4 % près (§7.1).

La colonne reste utile pour ce à quoi elle sert ici, comparer les bras entre eux, tous mesurés de la même façon sur la même forme de rafale. Elle ne doit pas être lue comme la capacité de la machine. Le piège général : **une métrique moyennée sur une fenêtre fixe ne mesure le débit que si la charge remplit la fenêtre.**
{{< /callout >}}

Les bras du balayage passent la porte à k=2 : elle sert ici à détecter qu'un bouton casse le modèle, pas à mesurer une dispersion. Le bras retenu repasse à k=5 avant publication de son chiffre.

B0, détail : Python 50/50, Go 46/50, zéro troncature, écart-type inter-tâche 0,08. Les quatre tâches instables sont celles déjà identifiées en 5.1 (`ParseMontant`, `MergeIntervals`, `MapConcurrent`, `RetryContexte`) ; l'écart de deux tirages avec le 98 du bras A tient dans la non-reproductibilité documentée à température nulle.

Deux enseignements du seul bras de base :

**L'écart code/prose est de 1,67×** sur la même machine à la même minute (78,55 contre 47,16). C'est la confirmation la plus nette de 5.3 : la charge pèse davantage que n'importe quel réglage qu'on va tester ensuite. Toute annonce de débit sans description de la charge est ininterprétable.

**Le TTFT ne pèse presque rien à ces longueurs.** Écart entre la formule hors TTFT et une division par la durée totale : 5 % (78,55 contre 74,69). Les deux protocoles divergent, mais l'écart reste petit devant celui de la charge.

Sous charge, le journal du serveur donne aussi les métriques de spéculation, qui ne se lisent nulle part ailleurs :

| grandeur | plage observée |
|---|---|
| longueur d’acceptation moyenne (sur k=5) | 3,10 → 3,87 |
| taux d’acceptation moyen | 42,0 % → 57,4 % |
| débit brouillonné | 75 → 113 tok/s |
| acceptation par position | 0,82 / 0,63 / 0,47 / 0,33 / 0,21 |

{{< hbars title="Acceptation par position du brouillon" caption="Acceptation par position du brouillon, sur k=5. La cinquième position n'est acceptée qu'une fois sur cinq — mais la décroissance n'est pas exploitable vers le bas : num_speculative_tokens est borné par la taille de bloc DSpark." minwidth=520 barheight=18 marginleft=150 max=100 dec=2 >}}
{"bars": [{"label": "position 1", "value": 82, "text": "0,82", "color": "accent"}, {"label": "position 2", "value": 63, "text": "0,63", "color": "accent"}, {"label": "position 3", "value": 47, "text": "0,47", "color": "accent"}, {"label": "position 4", "value": 33, "text": "0,33", "color": "accent"}, {"label": "position 5", "value": 21, "text": "0,21", "color": "accent"}]}
{{< /hbars >}}

La décroissance par position est frappante : la cinquième position n'est acceptée qu'une fois sur cinq. Mais **elle n'est pas exploitable vers le bas** : `num_speculative_tokens` est borné par la taille de bloc DSpark. À `SPECK=3`, le serveur refuse de démarrer :

```log
DSpark requires num_speculative_tokens >= dspark_block_size (5); got 3.
Smaller values produce incorrect output. Use num_speculative_tokens=5 or larger.
```

Le garde-fou est dans le code et il échoue à l'ouverture plutôt que de produire du texte plausible et faux. `k=5` est donc le **plancher**. Qu'il soit aussi l'optimum n'est pas déductible de cette décroissance : il a fallu le mesurer vers le haut, ce que fait §7.4.

#### B1, `--async-scheduling` : le drapeau est déjà le défaut

Vérification faite sur le journal d'un bras lancé *sans* le drapeau, la ligne `Asynchronous scheduling is enabled.` est présente quand même. Idem pour `enable_chunked_prefill=True`. **Ces deux drapeaux ne sont pas des boutons sur cette version : les activer explicitement ne change rien parce qu'ils sont déjà actifs.** À vérifier dans le journal de démarrage avant de perdre un bras dessus.

B1 est donc un doublon de B0, et c'est ce qui le rend utile : deux démarrages successifs de la même configuration donnent la **reproductibilité du protocole**.

| charge | B0 | B1 | écart |
|---|---:|---:|---:|
| code 512 | 78,55 | 78,30 | 0,3 % |
| code 1024 | 75,33 | 76,64 | 1,7 % |
| prose 512 | 47,16 | 48,19 | 2,2 % |
| indice C2 | 77,3 | 76,9 | 0,5 % |
| acceptation moyenne | 50,3 % | 51,1 % | 0,8 pt |

**Barre d'erreur entre deux démarrages : ±2 %.** Aucun bras ne peut être déclaré gagnant en dessous de ce seuil. Les tailles de pool KV le confirment sur un autre axe : 1 796 092 / 1 830 803 / 1 803 747 pour des configurations dont aucune ne touche à la mémoire.

#### B3, `CGCAPTURE=12` : le mono-flux ne voit rien, la concurrence perd 3,5 %

Premier bras où la barre d'erreur sert vraiment à trancher, et dans les deux sens.

Le bouton `max_cudagraph_capture_size` **n'est pas honoré tel quel** : demandé à 12, le moteur le ramène à 8 (il n'accepte que des tailles de sa propre grille). La liste des tailles capturées passe de `[1, 2, 4, 8, 16, 24]` à `[1, 2, 4, 8]`. À vérifier dans le journal de démarrage : la valeur demandée n'est pas celle appliquée.

| charge | B0 | B3 | écart |
|---|---:|---:|---:|
| code 512 | 78,55 | 77,52 | −1,3 % |
| code 1024 | 75,33 | 75,96 | +0,8 % |
| prose 512 | 47,16 | 49,46 | +4,9 % |
| **indice C2** | **77,3** | **74,6** | **−3,5 %** |

Les trois charges mono-flux sont dans le bruit, y compris la prose à +4,9 %, qui n'est qu'à +2,6 % de B1 et dont l'intervalle recouvre celui de B0. **Sans le doublon B0/B1, ce +4,9 % aurait été annoncé comme un gain.** C'est l'usage exact d'une barre d'erreur.

L'agrégé à 2 flux, lui, sort du bruit vers le bas, et le mécanisme est lisible : un flux en décodage spéculatif consomme 1 + 5 = 6 créneaux, donc deux flux en demandent 12. Avec un plafond de capture à 8, aucun graphe capturé ne couvre plus ce lot ; avec la grille d'origine, la taille 16 l'absorbait. Le mono-flux (lot de 6) reste couvert par la taille 8 et ne perd rien.

**Ce que ça enseigne pour dimensionner le paramètre :** la plus grande taille capturée doit couvrir `max_num_seqs × (1 + num_speculative_tokens)`, soit 12 ici. Descendre en dessous économise de la mémoire de capture et se paie exactement là où la machine est censée servir, en concurrence. La grille par défaut est déjà correcte ; ce bouton n'est pas un levier de débit, c'est un levier de mémoire.

Qualité B3 : 39/40, une seule tâche en échec, `compile+vet` à 40/40, zéro troncature. Pas de régression.

#### B4, `BATCHTOK=8192` : rien en débit, 43 % du pool KV en moins

Ce bras a été lancé sur la foi d'un avertissement du moteur lui-même :

```log
max_num_scheduled_tokens is set to 4088 based on the speculative decoding
settings. This may lead to suboptimal performance. Consider increasing
max_num_batched_tokens to accommodate the additional draft token slots.
```

**L'avertissement ne se referme pas.** À 8192, il réapparaît mot pour mot avec `set to 8184`. Il ne signale pas un mauvais dimensionnement : il constate que les créneaux réservés au brouillon (`max_num_seqs × num_speculative_tokens`) rendent l'ordonnançable inférieur au demandé, ce qui est vrai à toute valeur. C'est un avertissement inextinguible, pas une recommandation. **Règle générale : vérifier qu'un avertissement se referme avant de le traiter comme un bouton.**

| charge | B0 | B4 | écart |
|---|---:|---:|---:|
| code 512 | 78,55 | 77,18 | −1,7 % |
| code 1024 | 75,33 | 76,92 | +2,1 % |
| prose 512 | 47,16 | 48,05 | +1,9 % |
| indice C2 | 77,3 | 76,7 | −0,8 % |
| acceptation moyenne | 50,3 % | 49,4 % | −0,9 pt |
| **pool KV** | 1 796 092 | **1 029 078** | **−43 %** |

Les trois charges mono-flux sont dans la barre, de signes opposés : le profil même d'un effet nul. Qualité 40/40, aucune régression.

Le coût, lui, est massif et lisible : le tampon de prefill est prélevé sur la même enveloppe mémoire que le cache KV, à `GPUUTIL` constant. Doubler le premier ampute le second de 43 %, soit environ 767 000 tokens de contexte servable échangés contre rien.

**Ce paramètre n'est pas un levier de débit sur ce profil.** Le raisonnement le dit d'ailleurs a posteriori : à `max_num_seqs=2`, le prefill n'atteint jamais 4096 tokens sur les charges testées, donc le plafond n'était pas contraignant. Il ne le deviendrait qu'avec beaucoup de flux ou des prompts très longs, et c'est alors le pool KV amputé qui devient le facteur limitant.

#### B5, `rejection_sample_method=block` : sans effet à température nulle

Dernier bouton du balayage, et le seul à viser la grandeur encore loin de son plafond : le taux d'acceptation, bloqué à 50 % pour une longueur d'acceptation de 3,1 sur 5. Le mode `block` vérifie les tokens de brouillon conjointement (Sun et al.) au lieu d'un par un. Il n'était pas exposé dans `dsv4_run.sh` ; il l'est désormais via `REJSAMPLE`, aux côtés de `SPECK` et `DRAFTMETH`.

| charge | B0 | B5 | écart |
|---|---:|---:|---:|
| code 512 | 78,55 | 76,40 | −2,7 % |
| code 1024 | 75,33 | 73,97 | −1,8 % |
| prose 512 | 47,16 | 48,87 | +3,6 % |
| indice C2 | 77,3 | 77,4 | +0,1 % |
| acceptation moyenne | 50,3 % | 51,1 % | +0,8 pt |

Écarts de signes opposés autour de la barre, agrégé identique au dixième, acceptation identique. **Le mode n'a aucun effet mesurable sur cette configuration**, et surtout il ne relève pas l'acceptation, ce qui était sa seule raison d'être ici.

Hypothèse mécanique, **non vérifiée** : le banc tourne à température nulle, où la vérification token-à-token et la vérification conjointe retiennent le même ensemble de tokens. `block` se distinguerait à température non nulle. Non testé.

Le drapeau a bien été appliqué : `rejection_sample_method': 'block'` apparaît dans la configuration effective au démarrage, et pas en B0. La vérification a été faite avant de conclure, après le précédent de `CGCAPTURE`.

Qualité 38/40, le plus bas relevé du balayage, sur deux échecs à la compilation en Go. **Ce n'est pas exploitable comme régression** : `ParseMontant` est l'une des quatre tâches déjà instables en 5.1, et deux tirages d'écart à k=2 tiennent dans la non-reproductibilité à température nulle. Départager aurait demandé un passage à k=5, non dépensé faute de gain de débit à défendre.

{{< callout label="Piège de lecture à connaître" >}}
Le même paramètre accepte la valeur `synthetic`, qui **accepte les tokens de brouillon selon un taux d'acceptation fixé par l'utilisateur, sans vérification réelle** :

`'synthetic' accepts draft tokens with a decaying probability calibrated to synthetic_acceptance_rate.`

C'est un outil de simulation d'ordonnancement. Il gonfle mécaniquement le débit affiché et la longueur d'acceptation journalisée, au prix de la sortie du modèle. Aucune mesure de ce rapport ne l'emploie ; toutes tournent en `standard`, sauf B5 en `block`. Un chiffre de débit en décodage spéculatif n'est interprétable qu'accompagné du mode de vérification, au même titre que de la charge.
{{< /callout >}}

{{< deviationbars title="Écarts à la configuration de base, tous bras et toutes charges" caption="Tous les écarts à B0 publiés par les quatre tableaux ci-dessus, sur une même échelle. La bande couvre la barre d'erreur de ±2 % établie par le doublon B0/B1. Le rapport ne retient comme significatifs que B3 en concurrence et B5 sur code 512 ; les dépassements positifs sont discutés bras par bras dans le texte." minwidth=640 bound=2 max=5.5 >}}
{"rows": [{"label": "B1 · code 512", "value": -0.3}, {"label": "B1 · code 1024", "value": 1.7}, {"label": "B1 · prose 512", "value": 2.2}, {"label": "B1 · indice C2", "value": -0.5}, {"label": "B3 · code 512", "value": -1.3}, {"label": "B3 · code 1024", "value": 0.8}, {"label": "B3 · prose 512", "value": 4.9}, {"label": "B3 · indice C2", "value": -3.5}, {"label": "B4 · code 512", "value": -1.7}, {"label": "B4 · code 1024", "value": 2.1}, {"label": "B4 · prose 512", "value": 1.9}, {"label": "B4 · indice C2", "value": -0.8}, {"label": "B5 · code 512", "value": -2.7}, {"label": "B5 · code 1024", "value": -1.8}, {"label": "B5 · prose 512", "value": 3.6}, {"label": "B5 · indice C2", "value": 0.1}]}
{{< /deviationbars >}}

**Bilan du balayage : aucun bouton n'a produit de gain.** Cinq bras, deux résultats hors barre d'erreur et tous deux négatifs (B3 en concurrence, B5 sur code 512), trois résultats nuls. Deux boutons sur cinq ne faisaient pas ce que leur nom annonçait : `--async-scheduling` déjà actif par défaut, `CGCAPTURE` silencieusement ramené à la grille du moteur. Et un troisième a été lancé sur la foi d'un avertissement du moteur qui ne se referme jamais.

**La configuration de base est l'optimum de service.** Ce résultat est cohérent avec la télémétrie : les deux GB10 tiennent 2463 MHz sur 3003, 95-96 % d'occupation, 36 et 46 W. Le décodage est limité par la bande passante mémoire, pas par l'ordonnancement. Aucun réglage de service ne déplace ce plafond ; seuls le quantifieur, le brouilleur et le matériel le peuvent.

### 5.8 torch.compile sur DeepSeek-V4 NVIDIA : impasse, et pourquoi

`torch.compile` est le levier de débit auquel on pense en premier sur une pile vLLM. Nous avons cherché à l'activer sur notre plateforme. Il ne s'active pas, et la raison tient au modèle : **DeepSeek-V4 n'est pas câblé pour la compilation sur NVIDIA.**

- La recette officielle vLLM lance DSv4 avec `--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'` et sans `--enforce-eager`. `cudagraph_mode` pilote la capture de graphes CUDA, pas la compilation Dynamo. Le modèle ne porte pas `@support_torch_compile`.
- Le décorateur existe dans l'arbre, mais sur la variante **XPU** du même modèle. Le portage NVIDIA ne l'a pas repris.

Sur la ligne 0.25, DSv4 embarque l'indexeur CuTe DSL (18 appels `.compile()` à l'exécution) et une orchestration multi-flux (8 sites dans l'attention). Les deux sont intraçables par Dynamo. Le modèle a troqué la compilation contre des noyaux fusionnés à la main : c'est un choix d'implémentation, pas un oubli.

**Le point dur, à connaître avant de tenter :** `fullgraph=True` dans `vllm/compilation/wrapper.py`. Sous ce réglage, une rupture de graphe Dynamo n'est pas un coût de performance, c'est une erreur fatale. Toute la famille « désactiver le traçage sur la fonction gênante » (`torch.compiler.disable`, `dynamo.disable`) est donc inopérante : Dynamo répond `Skip inlining torch.compiler.disable()d function` et s'arrête.

Cinq patches, chacun débloquant une couche et en révélant la suivante :

| # | patch | précédent dans l'arbre |
|---|---|---|
| 1 | `@support_torch_compile` sur `DeepseekV4Model` | oui (variante XPU) |
| 2 | six sites MHC passés en `torch.ops.vllm.*` | oui (`layers/mhc.py`) |
| 3 | enregistrement de `mhc_pre_broadcast_tilelang` | non |
| 4 | réservation du workspace à l'init de la couche | partiel |
| 5 | mémoïsation de `get_device_capability` par dict | non |

Blocages rencontrés dans l'ordre, tous du même genre, du code Python qui inspecte son propre source ou manipule un objet C au moment de l'appel : `torch.accelerator.empty_cache()` dans la croissance du workspace ; `functools._lru_cache_wrapper` sur `get_device_capability` ; le JIT CuTe DSL de l'indexeur (`IndexerQFp8Kernel.compile`), atteint via l'orchestration multi-flux ; puis `inspect.getsourcelines` appelé par TileLang pour décider de son mode JIT, qui retombe sur `_patched_getfile`, explicitement marquée non traçable par Dynamo.

La parade générique pour le troisième : rendre l'attention **opaque** en l'enregistrant comme custom op, sur le patron amont `unified_attention_with_output` : `@eager_break_during_capture`, implémentation factice vide, `mutates_args=["out"]`, couche retrouvée par son préfixe dans `static_forward_context` puisqu'un custom op ne peut pas recevoir de `nn.Module`. Ça fonctionne : le traçage traverse l'attention et va mourir plus loin, sur `hc_head_fused_kernel_tilelang` : un septième noyau TileLang à enregistrer.

Arrêté là. Le motif est sans fond : chaque noyau fusionné à la main réclame son propre op enregistré. Remonter la compilation revient à défaire une par une les fusions qui font la performance du fork.

**Conclusion utile :** `torch.compile` n'est pas la source du débit sur DSv4 moderne. L'amont s'en passe, et la recette qui le garde paie une version épinglée pour un résultat du même ordre. Le débit se cherche dans les réglages de service.

### 5.9 Longueur de contexte et débit : une courbe qu'on ne peut pas mesurer naïvement

Nous avions d'abord balayé la longueur de contexte de la façon évidente : faire résumer des textes de plus en plus longs, sortie bornée à 256 tokens, et lire le débit de décodage. La série obtenue descendait de 52 tok/s à 16 k jusqu'à 35 tok/s à 131 k. Trente pour cent de perte, et la cassure tombait juste au-dessus de 65 536, exactement le seuil YaRN du checkpoint. L'explication s'écrivait toute seule.

Elle était fausse. Deux vérifications l'ont démolie.

**Le code d'abord.** Le passage à l'échelle YaRN est un cache cos/sin construit une fois à l'initialisation (`yarn_scaling_rope.py::_compute_cos_sin_cache`), sur toute la plage de positions. À l'exécution, c'est une lecture de table, sans branche dépendant de la position. Le coût par token est identique à 1 k et à 131 k. Un seuil YaRN ne peut pas produire une chute de débit dépendante de la longueur, non pas parce que c'est improbable, mais parce qu'il n'y a pas de chemin de code par lequel cela arriverait.

**Les points intermédiaires ensuite.** En ajoutant 70 k, 80 k, 100 k et 115 k, la série cesse d'être monotone : 47,0 / 54,8 / 41,7 / 55,1 tok/s. Et le point à 131 k donne 35,1 puis 39,6 sur deux exécutions. L'écart entre tirages d'un même point atteint 27,8 %. Le bruit dépasse l'effet cherché : il n'y avait pas de courbe.

**La faute est dans le protocole.** Faire varier la longueur du contexte en changeant le texte à résumer fait varier autre chose en même temps : la prévisibilité de la sortie. Or la sortie est produite par spéculation, et le débit d'un décodage spéculatif est proportionnel au nombre de tokens acceptés par pas. Sur ce balayage, l'acceptation allait de 34,0 % à 63,6 %, un facteur 1,87, qui suffit largement à expliquer le facteur 1,32 observé sur le débit. Nous mesurions la prévisibilité du texte, en croyant mesurer la longueur du contexte.

**Le protocole correct tient la sortie constante.** Le prompt est un remplissage de longueur variable suivi de « ignore le texte ci-dessus, recopie mot pour mot le passage suivant », avec un passage fixe, à température 0. La sortie est alors strictement identique à toutes les longueurs, vérifié à chaque point, une seule chaîne distincte sur trois tirages. L'acceptation est relevée sur `/metrics` avant et après chaque point, donc mesurée exactement sur l'intervalle.

| entrée | TTFT | préremplissage | décodage | acceptation | long. d’acceptation | pas/s |
|---:|---:|---:|---:|---:|---:|---:|
| 4 240 | 2,30 s | 1 846 tok/s | 76,32 | 88,9 % | 4,44 | **14,02** |
| 16 520 | 8,67 | 1 905 | 71,05 | 82,1 % | 4,10 | **13,92** |
| 32 880 | 18,12 | 1 814 | 68,27 | 77,8 % | 3,89 | **13,97** |
| 65 680 | 40,16 | 1 636 | 61,23 | 69,5 % | 3,47 | **13,69** |
| 80 120 | 49,69 | 1 612 | 79,09 | 93,2 % | 4,66 | **13,97** |
| 100 120 | 63,60 | 1 574 | 76,26 | 91,1 % | 4,56 | **13,73** |
| 131 200 | 87,28 | 1 503 | 84,03 | 100,0 % | 5,00 | **14,00** |

{{< linechart title="Pas de décodage par seconde selon la longueur d'entrée" caption="La dernière colonne du tableau : le nombre de pas de décodage par seconde, soit le débit divisé par le multiplicateur de spéculation. Constante de 13,69 à 14,02, soit 2,4 % d'écart, sous la barre d'erreur de ±2 % entre deux démarrages." minwidth=620 xscale="log" xdiv="k" xdec=0 xlabel="tokens d'entrée" ymin=0 ymax=16 ylabel="pas/s" height=266 >}}
{"xs": [4240, 16520, 32880, 65680, 80120, 100120, 131200], "xticks": [4240, 16520, 32880, 65680, 131200], "yticks": [0, 4, 8, 12, 16], "band": {"from": 13.69, "to": 14.02, "label": "amplitude des sept points : 2,4 %"}, "series": [{"label": "pas par seconde", "points": [14.02, 13.92, 13.97, 13.69, 13.97, 13.73, 14.0], "color": "accent"}]}
{{< /linechart >}}

La dernière colonne est le nombre de pas de décodage par seconde, c'est-à-dire le débit divisé par le multiplicateur de spéculation `1 + longueur d'acceptation`.

**Elle est constante : 13,69 à 14,02, soit 2,4 % d'écart**, sous la barre d'erreur de ±2 % entre deux démarrages. Le résultat s'énonce simplement :

> **La longueur du contexte n'a aucun effet mesurable sur la vitesse d'un pas de décodage entre 4 k et 131 k tokens.** Toute la variation du débit en tokens par seconde vient de l'acceptation de la spéculation.

{{< linechart title="Débit de décodage et acceptation selon la longueur d'entrée" caption="Le débit suit l'acceptation, courbe pour courbe. Le point le plus long est d'ailleurs le plus rapide : 84,0 tok/s à 131 200 tokens d'entrée, avec 0,8 % d'écart entre les trois tirages, parce que la recopie y est acceptée à 100 %." minwidth=620 xscale="log" xdiv="k" xdec=0 xlabel="tokens d'entrée" ymin=50 ymax=95 ylabel="décodage, tok/s" rmin=60 rmax=105 rdec=0 runit=" %" rlabel="acceptation" >}}
{"xs": [4240, 16520, 32880, 65680, 80120, 100120, 131200], "xticks": [4240, 16520, 32880, 65680, 131200], "yticks": [50, 60, 70, 80, 90], "rticks": [60, 70, 80, 90, 100], "series": [{"label": "décodage, tok/s", "points": [76.32, 71.05, 68.27, 61.23, 79.09, 76.26, 84.03], "color": "accent"}, {"label": "acceptation, %", "points": [88.9, 82.1, 77.8, 69.5, 93.2, 91.1, 100.0], "color": "signal", "axis": "right"}]}
{{< /linechart >}}

Le point le plus long est d'ailleurs le plus rapide : 84,0 tok/s à 131 200 tokens d'entrée, avec 0,8 % d'écart entre les trois tirages, parce que la recopie y est acceptée à 100 %. Toute lecture qui ferait de 131 k un régime dégradé se trompe.

Ce qui dépend réellement de la longueur, c'est le préremplissage : 1 846 tok/s à 4 k contre 1 503 à 131 k, **−19 %**, et surtout 87 secondes d'attente avant le premier token. C'est là que se dimensionne un agent à contexte long, et c'est ce que le cache de préfixe évite de repayer (§5.6).

**Pourquoi c'est constant : l'attention lit un budget fixe.** Cette constance n'est pas une chance, c'est la signature attendue de l'attention sparse de DeepSeek-V4. Le checkpoint déclare `index_topk: 512`. Au décodage, le chemin `_fp8_decode` de `vllm/v1/attention/backends/mla/flashmla_sparse.py` passe au noyau un tenseur `topk_indices` de forme `(num_decodes, seq_len, 512)` : quelle que soit la longueur du contexte, **l'attention ne lit que 512 entrées de cache KV**, celles que l'indexeur a sélectionnées. Le journal de démarrage confirme le chemin actif :

```log
DeepSeek V4: using official FlashInfer SM120 packed sparse-MLA decode
DSA indexer decode path: use_flattening=True (next_n=6, ...)
```

Un pas de décodage coûte donc la lecture des poids plus un travail d'attention borné, indépendant de la longueur. Sur une attention dense, le même balayage donnerait une décroissance nette. C'est un point à retenir avant de transposer ces chiffres à un autre modèle : **le résultat vaut pour une attention sparse à budget fixe, pas pour n'importe quelle architecture.**

**Une loi à un seul paramètre.** Les sept points se résument à :

> `débit ≈ 13,90 × (1 + tokens acceptés par pas)`

{{< linechart title="Débit prédit par la loi contre débit mesuré" caption="Prédiction de la loi contre mesure, sur les sept longueurs. Écart maximal 1,5 %. Un pas dure 72 ms." minwidth=620 xscale="log" xdiv="k" xdec=0 xlabel="tokens d'entrée" ymin=55 ymax=90 ylabel="tok/s" height=286 >}}
{"xs": [4240, 16520, 32880, 65680, 80120, 100120, 131200], "xticks": [4240, 16520, 32880, 65680, 131200], "yticks": [55, 65, 75, 85], "series": [{"label": "mesuré", "points": [76.32, 71.05, 68.27, 61.23, 79.09, 76.26, 84.03], "color": "accent"}, {"label": "prédit par la loi", "points": [75.66, 70.93, 67.92, 62.17, 78.65, 77.2, 83.38], "color": "alert"}]}
{{< /linechart >}}

Prédiction contre mesure, sur les sept longueurs : 75,66 / 70,93 / 67,92 / 62,17 / 78,65 / 77,20 / 83,38 contre 76,32 / 71,05 / 68,27 / 61,23 / 79,09 / 76,26 / 84,03. **Écart maximal 1,5 %.** Un pas dure 72 ms.

Une seule chose déplace ce coefficient, et ce n'est aucun réglage d'ordonnancement : `k`. À `k=7` le pas passe à 82,5 ms et la loi devient `12,12 × (1 + acceptés)`, en restant tout aussi constante sur la plage de contexte (§7.4). Le terme mesure le coût de vérification d'un pas, rien d'autre.

**Conséquence directe : le plafond du mono-flux est fixé par `k`.** Avec `num_speculative_tokens = 5`, le débit ne peut pas dépasser `13,90 × 6 = 83,4` tok/s. Le point à 131 k, accepté à 100 %, mesure 84,03 : **nous sommes au plafond, pas près de lui.** Cela explique rétrospectivement le résultat du balayage (§5.7) : aucun réglage d'ordonnancement ne pouvait gagner, parce qu'aucun ne touche à ce mur. La seule dimension qui le déplace est `k` lui-même, et elle le déplace vers le bas (§7.4).

**Ce qu'il faut en retenir au-delà de cette machine.** Sur un moteur à décodage spéculatif, tout balayage qui laisse varier le contenu de la sortie mesure l'acceptation avant toute autre chose. Une courbe « débit en fonction de X » n'est interprétable que si X est la seule chose qui varie, et publier l'acceptation à côté du débit est le minimum pour qu'un lecteur puisse le vérifier. Nous avions écrit la mauvaise courbe et l'explication YaRN qui allait avec ; elles sont retirées de ce rapport, et cette section les remplace.

## Pièges et paramètres qui ne font rien

### 6.1 Variables d'environnement documentées mais inertes

Le profil publié en amont liste une quinzaine de variables. Vérification faite dans le runtime installé (`grep` sur l'arbre `site-packages` complet) :

- **`VLLM_USE_B12X_MOE`** est déclarée dans `envs.py` (lignes 50 et 1021) et lue par **aucun** fichier Python, ni dans le fork, ni dans l'overlay Anemll. Le backend MoE se choisit uniquement avec `--moe-backend flashinfer_b12x`.
- **`VLLM_USE_B12X_WO_PROJECTION`**, **`VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK`**, **`VLLM_DSPARK_LOCAL_ARGMAX`**, **`VLLM_DSPARK_REPLICATE_MARKOV_W1`**, **`VLLM_DSPARK_FUSED_MARKOV_ARGMAX`**, **`VLLM_DSPARK_CONFIDENCE_SCHEDULER`**, **`VLLM_DSV4_B12X_COMPRESSED_MLA`** ne sont **même pas déclarées** dans leur propre `envs.py`. Elles n'ont aucun effet sur le runtime 0.1.1.
- **`B12X_W4A16_TC_DECODE`** est la seule à avoir un vrai consommateur, et il est hors de vLLM : `b12x/integration/tp_moe.py` et `b12x/moe/fused/w4a16/kernel.py`.

Conclusion pratique : ne pas passer de temps à régler ces variables, et vérifier dans le log de boot que l'option a réellement pris.

### 6.2 b12x exige `NOEP=1`

`B12xExperts._supports_parallel_config` (`experts/b12x_mxfp4_moe.py:504`) exige `not use_ep and not use_all2all_kernels and not enable_eplb`. Avec l'expert parallelism actif, **le backend est silencieusement ignoré** et MARLIN est utilisé à sa place. Sans vérifier le log de boot, on croit mesurer b12x et on mesure MARLIN.

### 6.3 `max_cudagraph_capture_size` n'est pas un drapeau CLI

C'est un champ de `CompilationConfig`. Il doit aller **dans** le JSON de `--compilation-config`. Le profil amont le pose à `MAX_NUM_SEQS × (MTP+1)`, soit 36 à k=5 et 6 créneaux. À noter : vLLM dérive déjà cette valeur seul : un boot sans le paramètre affiche le même `FULL=6 (largest=36)`.

### 6.4 Démarrage bloqué après `Profiling CUDA graph memory`

Symptômes : `shm_broadcast.py:705 No available shared memory broadcast block found in 60 seconds` en boucle, GPU à 96 % d'occupation mais 12 W seulement (contre 40–45 W sous charge réelle), deux threads à ~100 % de CPU, et **aucune écriture dans le cache de compilation**, ce qui distingue une attente active d'une compilation légitime.

Correctif : arrêt complet des deux nœuds selon la séquence du §2.4, `drop_caches`, relance. Sur nos essais le blocage ne s'est pas reproduit après nettoyage ; la cause exacte n'est pas établie.

### 6.5 Chemins indisponibles sur GB10

- **`use_fp4_indexer_cache`** : une assertion (`v1/attention/backends/mla/indexer.py:307-314`) exige du Blackwell datacenter sm_10x. Le GB10 est sm_121 : le drapeau est inutilisable, indépendamment du KV cache choisi.
- **GPUDirect RDMA** : impossible sur GB10 (mémoire unifiée). Forcer `NCCL_NET_GDR_LEVEL=0`.
- **FlashInfer en attention** : non supporté, forcer `TRITON_ATTN`, aux deux endroits, y compris pour le drafter.

### 6.6 Coût des créneaux : ce n'est pas `max-num-seqs` qui coûte

Une version précédente de ce rapport écrivait que `--max-num-seqs 6` faisait tomber le pool KV de ~1,8 M à 1 016 517 tokens. **C'était une erreur d'attribution, et nous la corrigeons ici** : le journal de ce bras (`dsv4_seqs6b_head_20260802.log`) montre que `max_num_batched_tokens` y valait **8192** en même temps. Deux variables changées, une seule accusée.

Le bras B4 du balayage (§5.7) avait déjà isolé la vraie cause : `BATCHTOK=8192` seul, à `max-num-seqs 2`, donne 1 029 078 tokens, soit le même chiffre à 1,2 % près. Le tampon de préremplissage est prélevé sur la même enveloppe que le cache KV.

Mesure directe du paramètre isolé, à `BATCHTOK=4096` constant :

| `max-num-seqs` | pool KV | écart |
|---:|---:|---:|
| 2 | 1 793 656 | référence |
| 16 | 1 717 797 | **−4,2 %** |

{{< hbars title="Pool KV selon le paramètre isolé" caption="Le coût mémoire des créneaux est marginal (−4,2 %), là où doubler le lot de préremplissage ampute le pool de 43 %. C'est l'inverse de ce qu'on suppose spontanément." minwidth=520 barheight=18 marginleft=230 marginright=110 max=1900000 dec=0 >}}
{"bars":[
  {"label":"max-num-seqs 2","value":1793656,"text":"1 793 656","color":"signal"},
  {"label":"max-num-seqs 16","value":1717797,"text":"1 717 797","color":"signal"},
  {"label":"max-num-batched-tokens 8192","value":1029078,"text":"1 029 078","color":"alert"}]}
{{< /hbars >}}

**Augmenter le nombre de créneaux est presque gratuit en mémoire ; augmenter le lot de préremplissage coûte 43 %.** C'est l'inverse de ce qu'on suppose spontanément, et l'inverse de ce que ce rapport affirmait.

Le coût réel des créneaux est ailleurs : dans la capture de graphes CUDA, que le moteur étend seul de 6 à 27 tailles (jusqu'à 192) entre 2 et 16 créneaux, et dans le débit par flux (§7.1).

## Combinaison recommandée

Sur cette plateforme, avec la voie A, la meilleure performance à haute qualité est la configuration **la plus sobre** :

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

### 7.1 Ce que cette configuration donne

**Qualité**, mesurée avant tout chiffre de débit, sur 20 tâches exécutables × 5 tirages, code réellement compilé et testé :

| | juste | compile+vet | troncatures |
|---|---:|---:|---:|
| Python | 50/50 | 50/50 | 0 |
| Go | 46/50 | 47/50 | 0 |
| **Total** | **96/100** | **97/100** | **0** |

Quatre tâches Go instables, toujours les mêmes (`ParseMontant`, `MergeIntervals`, `MapConcurrent`, `RetryContexte`). Écart-type inter-tâche 0,08.

**Débit**, mono-flux, streaming, hors TTFT :

| charge | tok/s |
|---|---:|
| **code, 512 tokens** | **78,55** |
| code, 1024 tokens | 75,33 |
| prose, 512 tokens | 47,16 |

**Débit à deux flux concurrents : 100,4 tok/s agrégés**, soit ~50,2 par flux. Mesuré sur cinq minutes de charge soutenue, 108 requêtes, 30 558 tokens en 304,4 s de temps de mur. Le journal serveur donne 100,8 sur les mêmes fenêtres : les deux métriques concordent à 0,4 % près.

Deux flux est le plafond de cette configuration, qui sert `--max-num-seqs 2`. Au-delà, les requêtes attendent une place.

{{< callout label="Correction" >}}
**Ce plafond est un paramètre, pas une propriété de la machine, et nous nous sommes trompés dessus.** Une version précédente de ce rapport concluait, sur la foi d'une échelle mesurée à `max-num-seqs 2`, qu'« au-delà de deux flux on n'achète pas du débit, on achète de l'attente ». C'était vrai de la configuration, et faux de la plateforme. Deux défauts de protocole l'avaient rendue inattaquable en apparence : la seule variable disponible était bornée à 2, et les rafales utilisaient des complétions naturelles de longueurs très inégales (`[3342, 640, 5498, 7832]` sur un tirage), donc un créneau libéré tôt restait vide jusqu'à la fin du lot.
{{< /callout >}}

**Protocole corrigé.** Sortie fixée à 512 tokens exactement (`ignore_eos`), T=0, préfixe unique par requête, trois tirages par point : tous les créneaux d'un lot se libèrent ensemble, il n'y a plus de traînard. Une seule variable change entre les deux séries, `--max-num-seqs`, tout le reste étant la configuration retenue.

| flux | `seqs=2` | TTFT méd. | `seqs=16` | TTFT méd. |
|---:|---:|---:|---:|---:|
| 1 | 40,05 | 0,23 s | 43,15 | 0,25 s |
| 2 | 63,44 | 0,39 s | 62,80 | 0,35 s |
| 4 | 64,48 | **7,54 s** | **88,11** | 0,32 s |
| 6 | 62,63 | **15,58 s** | **116,51** | 0,51 s |
| 8 | 59,81 | **24,25 s** | **134,30** | 0,44 s |
| 16 | 60,34 | **56,64 s** | **188,88** | **0,75 s** |

{{< linechart title="Débit agrégé selon le nombre de flux" caption="Débit agrégé. Les deux séries sont superposables à C1 et C2, comme attendu : les deux configurations y font la même chose. Elles divergent exactement là où `max-num-seqs 2` sature." minwidth=620 xlabel="flux concurrents" ymax=205 ylabel="tok/s agrégés" >}}
{"xs": [1, 2, 4, 6, 8, 16], "xticks": [1, 2, 4, 6, 8, 16], "yticks": [0, 50, 100, 150, 200], "series": [{"label": "seqs=16", "points": [43.15, 62.8, 88.11, 116.51, 134.3, 188.88], "color": "accent"}, {"label": "seqs=2", "points": [40.05, 63.44, 64.48, 62.63, 59.81, 60.34], "color": "alert"}]}
{{< /linechart >}}

{{< linechart title="Latence au premier token selon le nombre de flux" caption="Latence médiane avant le premier token, échelle logarithmique. À seqs=2 elle passe de 0,39 s à 56,6 s, et jusqu'à 121,6 s au pire. À seqs=16 elle reste sous 0,8 s à tous les niveaux." minwidth=620 xlabel="flux concurrents" yscale="log" ymin=0.2 ymax=80 ydec="auto" ylabel="TTFT médian, s" >}}
{"xs": [1, 2, 4, 6, 8, 16], "xticks": [1, 2, 4, 6, 8, 16], "yticks": [0.2, 1, 5, 20, 80], "series": [{"label": "seqs=2", "points": [0.23, 0.39, 7.54, 15.58, 24.25, 56.64], "color": "alert"}, {"label": "seqs=16", "points": [0.25, 0.35, 0.32, 0.51, 0.44, 0.75], "color": "accent"}]}
{{< /linechart >}}

Les deux séries sont superposables à C1 et C2, comme attendu : les deux configurations y font la même chose. Elles divergent exactement là où `max-num-seqs 2` sature :

- À `seqs=2`, l'agrégat est **plat de C2 à C16** (63,4 → 60,3) pendant que l'attente avant le premier token passe de 0,39 s à 56,6 s, et jusqu'à 121,6 s au pire. On n'achète effectivement que de l'attente, mais parce que le paramètre l'impose.
- À `seqs=16`, l'agrégat **monte de façon monotone jusqu'à 188,88 tok/s**, soit 3,1× le plafond précédent, et le TTFT reste **sous 0,8 s à tous les niveaux**.

Le prix est le débit par flux : 44,1 tok/s seul, 13,1 à seize flux. On échange de la latence par utilisateur contre du débit total, ce qui est l'arbitrage normal d'un serveur, et il n'y avait aucune raison de croire qu'il n'existait pas ici. Le coût mémoire est négligeable (§6.6) : 1 717 797 tokens de pool KV contre 1 793 656, soit −4,2 %.

**Ce que ça change pour dimensionner.** Pour un agent seul, `--max-num-seqs 2` donne la meilleure latence et le meilleur débit mono-flux. Pour servir plusieurs utilisateurs, monter le paramètre est le levier le plus rentable de tout ce rapport : le seul, en réalité, qui ait produit un gain. Le reste de nos mesures mono-flux reste valable telle quelle, puisqu'elles ne dépendent pas de ce paramètre.

**Sur la charge soutenue à deux flux.** Les 100,4 tok/s cités plus haut viennent d'une charge de cinq minutes en régime continu, pas d'une rafale : les créneaux n'y sont jamais vides. C'est pour cette raison qu'ils dépassent les 63,4 tok/s de l'échelle ci-dessus, qui mesure des rafales courtes. Les deux ne se contredisent pas, elles mesurent deux régimes, et la charge du banc n'est pas la même (§5.3).

**Spéculation** : acceptation 50,3 %, longueur d'acceptation 3,10 à 3,87 sur k=5, brouillon stable autour de 111 tok/s.

**Capacité** : 1,8 M tokens de cache KV, contexte servable 393 216 tokens.

**Contexte long : le coût du premier token**, mono-flux, sortie fixée à 256 tokens, trois tirages par point, préfixe unique par tirage pour neutraliser le cache de préfixe :

| entrée | TTFT | préremplissage |
|---:|---:|---:|
| 308 | 0,38 s | 810 tok/s |
| 1 068 | 0,61 s | 1 751 |
| 4 148 | 2,25 s | 1 844 |
| 16 428 | 8,58 s | 1 915 |
| 65 588 | 40,13 s | 1 634 |
| 131 108 | 87,28 s | 1 502 |

{{< linechart title="Préremplissage et TTFT selon la longueur d'entrée" caption="Le préremplissage plafonne vers 1 900 tok/s autour de 16 k d'entrée puis redescend. Servir 131 k de contexte coûte 87 secondes avant le premier token : c'est le vrai coût du contexte long, et c'est ce que le cache de préfixe évite de repayer." minwidth=620 xscale="log" xdiv="k" xdec=0 xlabel="tokens d'entrée" ymin=600 ymax=2000 ydec=0 ylabel="préremplissage, tok/s" rmin=0 rmax=100 rdec=0 runit=" s" rlabel="TTFT" >}}
{"xs": [308, 1068, 4148, 16428, 65588, 131108], "xticks": [308, 1068, 4148, 16428, 65588, 131108], "yticks": [600, 950, 1300, 1650, 2000], "rticks": [0, 25, 50, 75, 100], "series": [{"label": "préremplissage, tok/s", "points": [810, 1751, 1844, 1915, 1634, 1502], "color": "accent"}, {"label": "TTFT, s", "points": [0.38, 0.61, 2.25, 8.58, 40.13, 87.28], "color": "alert", "axis": "right"}]}
{{< /linechart >}}

Le préremplissage plafonne vers 1 900 tok/s autour de 16 k d'entrée puis redescend. Servir 131 k de contexte coûte 87 secondes avant le premier token : c'est le vrai coût du contexte long, et c'est ce que le cache de préfixe évite de repayer (§5.6).

**Le débit de décodage en fonction de la longueur du contexte n'est pas mesurable avec ce protocole**, et nous ne publions donc pas de courbe. Le détail de cette impasse est en §5.9 : il vaut d'être lu avant de faire confiance à une courbe de ce type, publiée par nous ou par quiconque.

**Machine** : les deux GB10 tiennent 2463 MHz sur un plafond de 3003, 95-96 % d'occupation, 36 et 46 W, 66 et 73 °C.

### 7.2 Matrice complète : ce qui paie et ce qui ne paie pas

Chaque ligne est un bras mesuré sur cette plateforme, un paramètre à la fois, serveur redémarré, porte de qualité avant tout chiffre.

| paramètre testé | effet sur le débit | effet sur la qualité | retenu |
|---|---|---|---|
| `CGMODE=FULL_AND_PIECEWISE` | référence | 96/100 | ✅ |
| `NOEP=1` (pas d'expert parallelism) | positif en mono-flux | — | ✅ |
| `SPECK=5` | **optimum démontré (voir §7.4)** | 96/100 | ✅ |
| `GPUUTIL=0.85` | pool KV 1,8 M | — | ✅ |
| `--moe-backend flashinfer_b12x` | **−19 %** à C1, +31 s au-delà de ~1850 tokens | — | ❌ |
| `--kv-cache-dtype nvfp4_ds_mla` | nul | **−6 points** | ❌ |
| `--async-scheduling` | nul (déjà le défaut) | — | ➖ |
| `--enable-chunked-prefill` | nul (déjà le défaut) | — | ➖ |
| `SPECK=3` | refusé au démarrage | — | ❌ |
| `SPECK=7` | **−11,5 % à −17,1 %** | 98/100 | ❌ |
| `CGCAPTURE=12` | **−3,5 %** en concurrence | 39/40 | ❌ |
| `BATCHTOK=8192` | nul, **−43 % de pool KV** | 40/40 | ❌ |
| `REJSAMPLE=block` | nul, acceptation inchangée | 38/40 | ❌ |
| `torch.compile` | **non activable (voir 5.8)** | — | ❌ |

Douze paramètres écartés, quatre retenus, et les quatre retenus sont ceux du départ. **La configuration la plus sobre est l'optimum.**

Le mono-flux est plafonné par `k` (§5.9, §7.3), et c'est la seule dimension où un gain restait envisageable. Elle est mesurée : `SPECK=7` coûte 12,8 % de vitesse de pas sans rendre un seul token accepté de plus (§7.4). `SPECK=5` n'est donc pas seulement le plancher imposé par `dspark_block_size` : c'est l'optimum démontré.

Deux résultats méritent d'être isolés parce qu'ils sont contre-intuitifs :

- **`flashinfer_b12x` coûte 19 % de débit et une falaise de latence.** Le noyau MARLIN, qui n'est mis en avant nulle part, gagne sur les deux tableaux.
- **`nvfp4_ds_mla` coûte 6 points de qualité sans économiser de mémoire.** Le gain attendu d'un format plus compact ne se matérialise pas ici.

### 7.3 Pourquoi rien d'autre ne paie

La télémétrie donne la réponse et elle est cohérente avec toute la matrice : à 95-96 % d'occupation, 2463 MHz et 36-46 W, le décodage est limité par la **bande passante mémoire**, pas par le calcul ni par l'ordonnancement.

C'est structurel pour ce type de charge : un token de décodage relit tous les poids actifs pour produire un seul token. Aucun réglage de service ne déplace ce plafond. Les seuls leviers qui restent sont d'une autre nature : le format de quantification, la qualité du brouilleur spéculatif, et le matériel.

Ce qui explique aussi pourquoi la spéculation est, elle, rentable : elle est le seul mécanisme testé qui produise plusieurs tokens pour une seule relecture des poids. Elle rend 3,1 tokens par passe au lieu de 1.

**Et pourquoi le balayage cherchait dans la mauvaise dimension.** La loi établie en §5.9 (`débit ≈ 13,90 × (1 + tokens acceptés par pas)`, un pas à 72 ms) dit que le mono-flux est plafonné à `13,90 × (1 + k)`. À `k = 5`, cela fait 83,4 tok/s, et notre meilleur point mesure 84,03. Aucun des cinq boutons d'ordonnancement testés ne touche à ce terme : ils ne pouvaient pas gagner.

**La seule piste restante était `k > 5`. Elle est mesurée, et elle est négative** (voir §7.4).

### 7.4 SPECK=7 : le seul levier possible, mesuré, et il coûte

La loi de §5.9 dit que le mono-flux est plafonné par `k`. Le moteur exige `num_speculative_tokens >= dspark_block_size` (5 ici) et son propre message d'erreur suggère une valeur plus haute :

```log
DSpark requires num_speculative_tokens >= dspark_block_size (5); got 3.
Use num_speculative_tokens=5 or larger (e.g. 7).
```

**Ce qu'on savait déjà, et pourquoi ça ne suffisait pas.** Une campagne antérieure (1ᵉʳ août, base `PIECEWISE + EP`, publiée dans le tutoriel) avait mesuré `k=7` et `k=10` et les avait trouvés perdants : 38,92 et 23,90 tok/s en petit contexte contre 47,96 à `k=5`. Deux réserves empêchaient d'en conclure quoi que ce soit pour la configuration retenue ici. La base n'était pas la même, ni `FULL_AND_PIECEWISE` ni `NOEP=1`. Et surtout la seule explication avancée était « l'acceptation s'effondre », lue sur un taux, sans jamais séparer le coût du pas du nombre de tokens rendus. Le balayage à un bouton du §5.7, lui, n'a mesuré aucune valeur autre que 5 : son unique bras spéculatif visait `SPECK=3` et n'a jamais démarré.

Nous avons donc remonté un bras `SPECK=7` sur la base retenue, un seul paramètre changé, serveur redémarré, porte de qualité d'abord, cette fois pour obtenir le mécanisme, pas seulement le signe.

**Le moteur adapte seul la capture de graphes.** Inutile de toucher à `CGCAPTURE` : à `k=5` le défaut donne `cudagraph_capture_sizes [1,2,4,8,16,24]`, à `k=7` il passe à `[1,2,4,8,16,24,32]`. Il couvre `max_num_seqs × (1 + k)` dans les deux cas. Forcer `CGCAPTURE=12` faisait au contraire retomber sur 8, sous le défaut : c'est l'explication du bras B3.

**Qualité : intacte.** 98/100 (Python 50/50, Go 48/50), 99/100 compile+vet, zéro troncature, contre 96/100 pour la référence. Vérification nécessaire, car le moteur avertit qu'un `k` invalide produit une sortie corrompue plutôt qu'une acceptation dégradée. À `k=7` la sortie est saine. Pool KV 1 788 784 tokens contre 1 796 092, soit −0,4 %.

**Débit : nettement négatif.**

| charge | k=5 | k=7 | écart |
|---|---:|---:|---:|
| code, 512 tokens | 78,55 | 65,14 | **−17,1 %** |
| code, 1024 tokens | 75,33 | 65,95 | **−12,4 %** |
| prose, 512 tokens | 47,16 | 41,75 | **−11,5 %** |

{{< groupedbars title="Débit à k=5 et k=7, trois charges" caption="Trois charges, un seul paramètre changé sur la base retenue. k=7 perd de 11,5 % à 17,1 % selon la charge, à qualité intacte." minwidth=600 marginleft=170 >}}
{"series": [{"label": "k=5", "color": "signal"}, {"label": "k=7", "color": "alert"}], "groups": [{"label": "code, 512 tokens", "values": [78.55, 65.14], "texts": ["78,55", "65,14"]}, {"label": "code, 1024 tokens", "values": [75.33, 65.95], "texts": ["75,33", "65,95"]}, {"label": "prose, 512 tokens", "values": [47.16, 41.75], "texts": ["47,16", "41,75"]}]}
{{< /groupedbars >}}

**La cause, décomposée par la loi de §5.9.** En reprenant le protocole à sortie constante sur les sept longueurs :

| | k=5 | k=7 |
|---|---:|---:|
| pas par seconde | 13,90 | **12,12** |
| durée d'un pas | 72,0 ms | **82,5 ms** |
| **tokens acceptés par pas** | **4,30** | **4,30** |

{{< groupedbars title="Décomposition du débit à k=5 et k=7" caption="Le pas coûte 12,8 % de plus à k=7, et le nombre de tokens acceptés par pas ne bouge pas d'un centième. On paie la vérification plus cher sans rien gagner en sortie : le brouilleur DSpark sature autour de 4,3 tokens acceptés." minwidth=560 marginleft=210 max=15 >}}
{"series": [{"label": "k=5", "color": "signal"}, {"label": "k=7", "color": "alert"}], "groups": [{"label": "pas par seconde", "values": [13.9, 12.12], "texts": ["13,90", "12,12"]}, {"label": "tokens acceptés par pas", "values": [4.3, 4.3], "texts": ["4,30", "4,30"]}]}
{{< /groupedbars >}}

Deux faits, et le second est décisif :

- Le pas coûte 12,8 % de plus : vérifier huit positions au lieu de six ajoute 10,5 ms. Écart mesuré sur les sept longueurs : −10,9 % à −15,8 %.
- **Le nombre de tokens acceptés par pas ne bouge pas : 4,30 dans les deux cas.** Le brouilleur DSpark sature autour de 4,3 tokens acceptés. Lui en demander sept ne lui en fait pas rendre davantage.

On paie donc la vérification plus cher sans rien gagner en sortie. L'acceptation par position (0,82 / 0,63 / 0,47 / 0,33 / 0,21) laissait espérer une valeur résiduelle aux positions 6 et 7 ; en pratique elle est absorbée par le coût du pas.

**Et c'est ici que le piège du §5.4 se referme.** Dire « l'acceptation s'effondre à `k=7` » est exact si l'on parle du *taux* : 4,30 tokens sur 5 proposés font 86 %, les mêmes 4,30 sur 7 n'en font plus que 61 %. Mais ce taux ne dit rien de ce qui produit le débit, qui est le *nombre* de tokens rendus par pas. Et ce nombre, lui, ne bouge pas d'un centième. Le taux chute uniquement parce que son dénominateur augmente. Une conclusion tirée du taux seul aurait donné le bon signe pour la mauvaise raison, et n'aurait pas permis de prédire qu'aucun `k` supérieur ne peut gagner.

**Conclusion, et elle referme la question.** `SPECK=5` n'est pas seulement le plancher imposé par `dspark_block_size` : c'est **l'optimum démontré**. Puisque l'acceptation plafonne à 4,30 tokens par pas (en dessous même de `k=5`), aucune valeur supérieure ne peut gagner : elle ne fait que ralentir le pas. Le plancher du moteur se trouve être le meilleur réglage, et la seule dimension non explorée du balayage est désormais fermée.

La loi tient d'ailleurs à `k=7` aussi : le `pas/s` y reste constant sur toute la plage de contexte (11,80 à 12,38, soit 4,7 % d'écart). Seul le coefficient change, ce qui confirme que c'est bien le coût de vérification, et non la longueur du contexte, qui pilote ce terme.

## Reproduire les mesures

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

Deux règles de méthode qui changent les chiffres :

- **Préchauffer.** 5 générations longues minimum avant toute mesure. Le premier tirage à froid peut être 10× plus lent (10,66 s contre 0,83 s observés).
- **Isoler le serveur.** Le lier sur `127.0.0.1` pendant les campagnes. Un serveur de test joignable depuis le réseau se fait consommer des créneaux de file d'attente par du trafic applicatif, et les chiffres de concurrence deviennent faux.

---

Toutes les mesures de ce rapport sont celles de la voie A, sans exception, réalisées du 1er au 3 août 2026 sur deux DGX Spark (GB10, sm_121). Ce qui est estimé ou non vérifié est signalé comme tel dans le corps du texte. La version pas-à-pas, orientée mise en service, est dans la [recette complète](/fr/blog/dsv4-2x-spark-recette/).


## Sources

- Recette publique d'un opérateur tiers (DeepSeek-V4-Flash 0731 sur deux DGX Spark) : 74,55 et 95,9 tok/s (C1, décodage médian à chaud), cités en §5.5
- Mesures internes, deux NVIDIA DGX Spark (GB10, sm_121), TP=2, voie A, 1er au 3 août 2026 : banc de code codebench, decode_probe.py, seuil_probe.py, bench_mono.py, télémétrie machine
- jasl/vllm, branche b12x-nvfp4-mxfp4, commit 48d274954 (envs.py, config/vllm.py, experts/b12x_mxfp4_moe.py, flashmla_sparse.py, indexer.py, yarn_scaling_rope.py)
- Anemll/dspark-vllm-gx10 (upstream.lock, overlay de 12 fichiers, image ghcr.io/anemll/dspark-vllm-gx10:0.1.1) et la recette de l'opérateur tiers (profil de service, docker-compose.dspark.yml, README)
- vLLM v0.25.1 (752a3a504485790a2e8491cacbb35c137339ad34), FlashInfer 0472b9b3f2fba11b463f8526f390297d52a8aad7 (0.6.15), lukealonso/b12x 7dc6fb8fcc6446ea093537d1657df81985fa5f43 (0.15.3)
- Checkpoint DeepSeek-V4-Flash-0731 : configuration (index_topk 512, dspark_block_size 5)
- Journaux de démarrage et de service du banc, dont dsv4_seqs6b_head_20260802.log

