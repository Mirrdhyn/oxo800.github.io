# Faire tourner vLLM sur un DGX Spark (GB10) : du clone à l'inférence NVFP4

> Installation complète de vLLM sur un DGX Spark GB10 (SM121) : les trois bugs qui bloquent le build et le lancement d'un modèle NVFP4, et comment les corriger.

- Date: 2026-07-20
- Language: fr
- URL: https://0x0800.com/fr/blog/dgx-spark-vllm-nvfp4/
- Tags: dgx-spark, gb10, vllm, nvfp4, cuda, sm121
- Translation (en): https://0x0800.com/blog/dgx-spark-vllm-nvfp4/


Le DGX Spark embarque un GPU GB10 (Blackwell, architecture compute SM121) sur une carte aarch64. Une architecture assez récente pour que plusieurs bibliothèques de l'écosystème CUDA n'aient pas encore de kernel précompilé pour SM121. Trois correctifs qui ne sautent pas aux yeux dans la documentation officielle, expliqués ici du clone du dépôt au lancement d'un modèle quantifié NVFP4.

Cet article couvre l'installation complète, du clone du dépôt jusqu'au lancement réussi d'un modèle quantifié NVFP4 (Mistral-Small 119B), en expliquant pourquoi chaque étape est nécessaire. Toutes les commandes sont testées sur notre matériel ; adaptez les chemins (`~/models/...`, noms de modèles) au vôtre.

## Ce que le Spark fournit de base

À la sortie de la boîte, le DGX Spark tourne sous DGX OS (Ubuntu) avec le pilote NVIDIA et un toolkit CUDA déjà installés. Deux commandes suffisent pour connaître votre point de départ :

```bash
nvidia-smi -q | grep -i "CUDA Version"   # plafond de compilation JIT du pilote
nvcc --version                            # toolkit CUDA utilisé pour compiler
```

Sur notre matériel : pilote `595.71.05`, plafond JIT à **CUDA 13.2**, toolkit système installé en **13.3**. Cet écart d'un cran est la source du premier bug détaillé plus bas. Gardez ces deux nombres sous la main, ils comptent.

Autre particularité du GB10 à connaître avant d'aller plus loin : la mémoire GPU est la mémoire système (architecture unifiée). `nvidia-smi --query-gpu=memory.used,memory.free` renvoie `[N/A]` : pour surveiller la mémoire disponible, utilisez `free -h`. Et `--gpu-memory-utilization` dans vLLM se calcule sur la RAM totale de la machine, pas sur une VRAM dédiée. Sur une machine à 128 Gio, `0.85` demande donc environ 103 Gio, et l'OS ou l'environnement de bureau grignote une partie de la marge avant même de lancer quoi que ce soit.

## Environnement Python et premier build de vLLM

Nous utilisons `uv` pour la gestion de l'environnement virtuel, plus rapide que `venv`+`pip` classique pour un projet de cette taille :

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv ~/.venv --python 3.12
source ~/.venv/bin/activate
```

Clonez vLLM et installez-le en mode éditable :

```bash
git clone https://github.com/vllm-project/vllm.git ~/vllm-020
cd ~/vllm-020
export MAX_JOBS=18   # ajustez au nombre de cœurs disponibles moins une marge
~/.venv/bin/pip install --no-build-isolation -e .
```

**Point important** : `pip` doit être appelé via son chemin complet dans le venv (`~/.venv/bin/pip`), jamais un `pip` nu dans un script non interactif. Sous Debian/Ubuntu, un `pip` nu résout vers le pip système, protégé par PEP 668 (« externally-managed-environment »), et refuse l'installation.

Deux prérequis à vérifier avant de lancer ce build, sous peine d'échec en cours de compilation :

- **ninja installé au niveau système**, pas seulement dans le venv : `sudo apt install -y ninja-build`. Si `ninja` n'est visible que via le venv activé, un appel ultérieur au build (par exemple via un script qui n'active pas le venv) fera retomber CMake sur le générateur « Unix Makefiles », en conflit avec un cache CMake déjà généré en Ninja : erreur bloquante (« Does not match the generator used previously »).
- **$CUDA_HOME/bin sur le PATH**, pas seulement `$CUDA_HOME/lib64` sur le `LD_LIBRARY_PATH`. Sans ça, `nvcc` est invisible dans un shell de connexion normal alors même qu'un build précédent a pu fonctionner (via un PATH modifié à la main). Ajoutez dans `~/.bashrc`, juste après l'export de `CUDA_HOME` :

```bash
export CUDA_HOME="/usr/local/cuda"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
```

Le build complet prend un bon moment (comptez plusieurs dizaines de minutes selon le nombre de cœurs). Une fois terminé, vérifiez la version installée :

```bash
~/.venv/bin/python -c "import vllm; print(vllm.__version__)"
```

**Un dernier piège de premier build** : si votre installation monte `flashinfer-python` au passage (c'est le cas sur un `main` récent), vérifiez que `flashinfer-cubin` (le cache de kernels précompilés) suit la même version. Depuis la version 0.6.14, ce paquet n'est plus publié sur PyPI mais sur un index séparé :

```bash
~/.venv/bin/pip install flashinfer-cubin --extra-index-url https://flashinfer.ai/whl/
```

Sans cet index explicite, `pip` reste bloqué sur la dernière version PyPI (0.6.13) et vLLM démarre avec un cache de kernels qui ne correspond plus à `flashinfer-python` : erreur de version au lancement, pas au moment de l'installation.

## Premier lancement de contrôle

Un petit modèle suffit à valider que la chaîne complète fonctionne avant de passer aux choses sérieuses :

```bash
vllm serve <chemin_du_modele> --served-model-name mon-modele \
  --max-model-len 8192 --enforce-eager --gpu-memory-utilization 0.1 \
  --attention-backend FLASHINFER
```

Si ce lancement passe et répond correctement à une requête `/v1/chat/completions`, la base est saine. Si vous rencontrez un des trois bugs suivants, c'est le moment de les traiter.

## Bug n°1 — cudaErrorUnsupportedPtxVersion

**Symptôme** : le serveur démarre, charge les poids, puis plante au premier calcul matriciel réel (souvent visible sur un modèle avec bloc Mamba, mais le mécanisme touche potentiellement tout GEMM non pré-compilé pour SM121) :

```text
torch.AcceleratorError: CUDA error: the provided PTX was compiled with an unsupported toolchain.
```

**Cause** : sur une architecture aussi récente que SM121, plusieurs kernels n'ont pas de binaire natif et sont compilés en PTX au runtime (JIT), directement par le pilote. Si ce PTX a été généré par un toolkit CUDA plus récent que ce que le pilote sait interpréter en JIT (rappel : plafond 595.71.05 → CUDA 13.2), le pilote refuse de le charger. Le toolkit 13.3 utilisé pour compiler vLLM est un cran au-dessus de ce plafond, d'où le crash.

{{< callout label="Fix" >}}
Recompiler vLLM avec un toolkit CUDA aligné sur le plafond JIT du pilote (13.2 dans notre cas), pas avec la dernière version disponible.

```bash
# Installer le toolkit 13.2 à côté du 13.3 existant (même dépôt apt NVIDIA)
sudo apt install cuda-toolkit-13-2

# Basculer l'alternative active
sudo update-alternatives --config cuda
# → choisir /usr/local/cuda-13.2

# Vérifier
nvcc --version   # doit annoncer release 13.2

# Recompiler vLLM depuis ce nouvel environnement
cd ~/vllm-020
~/.venv/bin/pip install --no-build-isolation -e .
```

`update-alternatives` gère le symlink `/usr/local/cuda` : les deux toolkits coexistent sans conflit, et basculer l'un vers l'autre ne demande pas de retoucher `.bashrc` à chaque fois, tant que `CUDA_HOME` pointe sur ce symlink.

Relancez le repro de contrôle ci-dessus : l'erreur `cudaErrorUnsupportedPtxVersion` doit avoir disparu.
{{< /callout >}}

## Bug n°2 — FlashAttention-2 qui échoue en silence sur SM121

**Symptôme** : le serveur tourne, répond aux requêtes, mais le log affiche en boucle un message qui n'interrompt rien :

```text
FATAL: FlashAttention requires building with sm version sm80-sm90, but was built for < 8.0!
```

C'est plus grave qu'un simple message bruyant : les kernels FlashAttention-2 vendorisés dans vLLM sont gardés par une macro de compilation (`__CUDA_ARCH__ >= 800`). Pour toute variante de kernel qui ne matche pas nativement SM121, le code compilé se résume à afficher ce message et **retourner sans calculer l'attention**, sans lever d'exception. Le risque n'est pas un crash, c'est un résultat silencieusement faux si l'oracle de sélection automatique de vLLM route ne serait-ce qu'une partie du calcul vers FA2.

{{< callout label="Fix" >}}
Forcer explicitement un autre backend d'attention au lancement, pour exclure FA2 du chemin d'exécution :

```bash
vllm serve <chemin_du_modele> ... --attention-backend FLASHINFER
```

(ou `TRITON_MLA` selon l'architecture du modèle : les deux évitent FA2). Ce n'est pas un correctif du code vLLM, c'est un contournement au lancement, mais il est suffisant : plus aucune occurrence du message FATAL une fois le flag posé, réponses cohérentes vérifiées sur requête réelle.
{{< /callout >}}

## Bug n°3 — NVFP4/Marlin absents du build sur SM121 avec un toolkit CUDA ≥ 13.0

Celui-ci ne se voit qu'au lancement d'un modèle quantifié en NVFP4 (typiquement les checkpoints Mistral ou similaires livrés en `-NVFP4`).

**Symptôme** :

```text
WARNING: Your GPU does not have native support for FP4 computation...
NotImplementedError: Could not run '_C::gptq_marlin_repack' with arguments from the 'CUDA' backend
```

Ce message est trompeur : le GB10 supporte nativement NVFP4. Le problème est dans le build, pas dans le matériel.

**Cause** : dans `CMakeLists.txt` de vLLM, la liste d'architectures candidates testée pour activer les kernels NVFP4 (chemin CUTLASS SM120) et Marlin change selon la version du compilateur CUDA :

```cmake
# Avec toolkit CUDA >= 13.0, seul le candidat "family-compat" est testé :
cuda_archs_loose_intersection(MARLIN_ARCHS "8.0+PTX;12.0f" "${CUDA_ARCHS}")
# Avec toolkit CUDA < 13.0, l'arch explicite 12.1a est incluse :
cuda_archs_loose_intersection(MARLIN_ARCHS "8.0+PTX;12.0a;12.1a" "${CUDA_ARCHS}")
```

Avec un toolkit 13.2 ou plus (notre cas après le fix du bug n°1), seul `12.0f` est testé, pas `12.1a`. Si l'auto-détection des architectures cibles ne fait pas correspondre le GB10 à ce candidat, les deux kernels (CUTLASS NVFP4 et Marlin) sont silencieusement exclus du build.

{{< callout label="Fix" >}}
Forcer explicitement l'architecture cible au lieu de compter sur l'auto-détection, puis recompiler :

```bash
export TORCH_CUDA_ARCH_LIST="12.1a"
export MAX_JOBS=18
export CUTE_DSL_ARCH=sm_121a

cd ~/vllm-020
~/.venv/bin/pip install --no-build-isolation -e .
```

**Vérification** : plus fiable qu'une déduction à partir des symptômes, elle interroge directement le dispatcher PyTorch pour confirmer que le kernel CUDA est bien enregistré :

```python
import vllm
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import cutlass_fp4_supported
print(cutlass_fp4_supported())  # doit renvoyer True

import torch._C as C
print(C._dispatch_dump("_C::gptq_marlin_repack"))  # doit lister un enregistrement CUDA
```
{{< /callout >}}

## Lancer un modèle NVFP4 pour de bon

Avec les trois correctifs en place, un lancement réaliste ressemble à ceci (exemple avec un modèle Mistral 119B) :

```bash
vllm serve <chemin_du_modele> --served-model-name mon-modele-nvfp4 \
  --tokenizer-mode mistral --config-format mistral --load-format mistral \
  --host 0.0.0.0 --port 8086 \
  --max-model-len 65536 --gpu-memory-utilization 0.85 \
  --enforce-eager --attention-backend TRITON_MLA --moe_backend flashinfer_b12x \
  --enable-auto-tool-choice --tool-call-parser mistral --reasoning-parser mistral
```

Deux réglages à garder en tête pour un premier lancement sur un modèle de cette taille :

- Sur mémoire unifiée, restez prudents sur `--gpu-memory-utilization` (0.85 plutôt que 0.92) et sur `--max-model-len` : la marge réelle disponible est souvent inférieure à ce qu'un calcul naïf sur la RAM totale suggère.
- Si le modèle utilise un kernel FP4 sans cubin précompilé pour votre architecture, `flashinfer` le compile en JIT au premier lancement. Exportez `MAX_JOBS` (par exemple `4`) avant la commande `vllm serve` elle-même pour borner le nombre de compilations parallèles : une valeur trop haute peut saturer la mémoire pendant que le modèle charge en parallèle.

Une fois le serveur démarré (« Application startup complete » dans les logs), un test simple confirme que tout fonctionne :

```bash
curl -s http://localhost:8086/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "mon-modele-nvfp4",
  "messages": [{"role": "user", "content": "In one sentence, what is the capital of France?"}],
  "max_tokens": 50
}'
```

Une réponse cohérente, sans trace de `NotImplementedError`, de warning FP4, ni de message `FATAL` dans les logs, confirme que le chemin natif NVFP4 est utilisé de bout en bout.

## Les trois bugs, en un coup d'œil

| Bug | Symptôme | Fix |
|---|---|---|
| Toolkit CUDA trop récent | cudaErrorUnsupportedPtxVersion au premier GEMM | Aligner le toolkit sur le plafond JIT du pilote (nvidia-smi -q \| grep CUDA) via update-alternatives, puis rebuild |
| FlashAttention-2 non supporté sur SM121 | FATAL: FlashAttention requires... en boucle, sans crash | --attention-backend FLASHINFER (ou TRITON_MLA) au lancement |
| NVFP4/Marlin exclus du build (toolkit ≥ 13.0) | NotImplementedError sur gptq_marlin_repack, warning FP4 trompeur | Rebuild avec TORCH_CUDA_ARCH_LIST="12.1a", CUTE_DSL_ARCH=sm_121a |

Ces trois problèmes partagent une même racine : SM121 est une architecture trop récente pour que l'auto-détection et les chemins de compatibilité par défaut de l'écosystème CUDA/vLLM la couvrent correctement. Dans les trois cas, le contournement consiste à cesser de faire confiance à l'auto-détection (du toolkit, du backend d'attention, ou de la liste d'architectures) et à forcer explicitement la bonne valeur.

{{< callout label="Nous installons et réglons votre DGX Spark." >}}
Si ces correctifs vous font gagner du temps mais que vous préférez confier l'installation, le réglage des backends et la mise en production à quelqu'un qui l'a déjà fait, écrivez-nous.
{{< /callout >}}

## À lire aussi

- [DeepSeek-V4-Flash-0731 sur deux DGX Spark : la recette complète](/fr/blog/dsv4-2x-spark-recette/) : 96/100 de code juste sur 100 tirages jugés par exécution, 188,9 tok/s agrégés à seize flux, thermique et latence RDMA relevées sous charge.
- [Servir Laguna S 2.1 NVFP4 sur un seul DGX Spark](/fr/blog/laguna-s21-single-spark/) : un MoE de 117,6 Md de paramètres sur une seule box : hygiène mémoire, contexte natif 256K, décodage spéculatif DFlash.


## Sources

- Installation et vérifications sur notre DGX Spark (GB10), juillet 2026 : pilote 595.71.05, plafond JIT CUDA 13.2, toolkit système 13.3 (toutes les commandes de l'article y ont été testées)
- vllm-project/vllm : CMakeLists.txt, listes d'architectures NVFP4 (CUTLASS SM120) et Marlin selon la version du toolkit CUDA
- flashinfer-cubin : publié sur l'index flashinfer.ai/whl depuis la version 0.6.14 (PyPI s'arrête à 0.6.13)

