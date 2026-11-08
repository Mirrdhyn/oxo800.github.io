# TensorFold sur Intel Arc Pro B70 : du portage Python au moteur Zig

> Comment nous avons porté le moteur d'inférence TensorFold sur Intel Arc Pro B70 (Xe2), en Python puis en Zig sur Level Zero : Qwen3.8-27B à 122,9 tok/s, mesuré face à vLLM et llama.cpp.

- Date: 2026-10-23
- Language: fr
- URL: https://0x0800.com/fr/blog/tensorfold-xe2-b70/
- Tags: b70, tensorfold, xpu, zig, level-zero, vllm, llama-cpp
- Translation (en): https://0x0800.com/blog/tensorfold-xe2-b70/


TensorFold ne connaissait que les Mac et les GPU NVIDIA. Nous l'avons porté sur la carte Intel Arc Pro B70 (architecture Xe2, 32 Go), d'abord en Python par PyTorch, puis en Zig directement sur Level Zero, sans PyTorch ni Python à l'exécution. Le second portage est intégré au projet officiel depuis TensorFold 1.0.12. Ce texte en donne le chemin, les pièges et les mesures, y compris celles qui nous sont moins favorables.

{{< keyfigures >}}
[
  {"label": "Un flux, drafté", "value": "122,9 tok/s", "detail": "moteur Zig, code, glouton ; 99,3 tok/s en conversation"},
  {"label": "Concurrent == seul", "value": "64 / 64", "detail": "cellules égales au bit près, 1 à 8 flux, glouton et échantillonné"},
  {"label": "Statut", "value": "TensorFold 1.0.12", "detail": "backend intégré au projet officiel (PR #802), préversion à compiler"}
]
{{< /keyfigures >}}

## Une carte à 32 Go que les moteurs rapides ignoraient

[TensorFold](https://github.com/ashhart/TensorFold) est un moteur d'inférence open source (Apache-2.0) écrit par Ash Hart. Il sert des modèles de langage derrière une API compatible OpenAI, sur Apple Silicon et sur GPU NVIDIA. Sa particularité : chaque famille de modèles a ses propres noyaux et sa propre vérification des brouillons, et un brouillon n'est accepté que s'il égale le jeton que le moteur aurait produit seul. Sur Qwen3.8-27B, il s'appuie sur DFlash2, un petit modèle qui propose plusieurs jetons d'avance, organisés en arbre, que le grand modèle vérifie en une passe.

La carte visée est l'Intel Arc Pro B70 : puce Battlemage G31, architecture Xe2, 32 Go de mémoire, une bande passante mesurée de 577 Go/s en lecture et une limite de puissance fixée à 230 W par le micrologiciel. Le modèle est Qwen3.8-27B quantifié en 4 bits (GPTQ symétrique, groupes de 128), qui tient sur la carte avec son cache.

Le chemin CUDA de TensorFold pour ce modèle repose sur des noyaux écrits à la main et réglés pour NVIDIA. Rien de tout cela ne tourne tel quel sur une carte Intel. Au départ, la référence sur la B70 était vLLM-XPU : **33,8 tok/s** sans brouillon, **45 à 61 tok/s** avec son propre mécanisme de prédiction (MTP).

## Trois égalités à tenir, au bit près

Un backend n'entre dans TensorFold que s'il tient trois propriétés, vérifiées jeton par jeton, en glouton comme en échantillonné :

- **drafté == série** : une réponse produite avec brouillons est identique à la même réponse produite jeton par jeton ;
- **repris == frais** : un prompt repris depuis le cache donne la même suite qu'un prompt calculé de zéro ;
- **concurrent == seul** : une requête servie à côté d'autres donne la même réponse que la même requête servie seule.

La carte Intel résiste sur ce point. Le produit matriciel 4 bits de la bibliothèque vllm-xpu-kernels (`int4_gemm_w4a16`) atteint le plafond de bande passante, mais ses bits dépendent du nombre de lignes de l'appel, et parfois de la place d'une ligne dans son bloc. À 192 ou 256 lignes, il n'est même pas répétable d'un appel à l'autre. Sur la même pile, `F.linear` n'est pas déterministe non plus, et le compilateur fusionne des multiplications-additions différemment selon le noyau. Chacune de ces trois sources de divergence suffit à casser l'égalité drafté == série.

{{< callout label="Ce que vLLM fait de ce problème" >}}
vLLM ne promet pas cette propriété. Sur la même carte, nous avons comparé chaque réponse de vLLM v0.31 servie à quatre en même temps à la même requête servie seule : **27 réponses sur 48** identiques avec MTP4, **16 sur 48** sans brouillon. Deux exécutions seules successives de la même requête ne donnent la même réponse que dans **11 cas sur 16**. En code, elles concordent ; en conversation, elles divergent.
{{< /callout >}}

## Python et torch.xpu : garder le moteur, remplacer les noyaux

Le premier portage garde tout le moteur Python de TensorFold et lui ajoute un backend `--backend xpu`. Un module de compatibilité répond aux appels `torch.cuda` par `torch.xpu` et remplace les noyaux CUDA : la règle delta à portes (GDN), l'attention en arbre, l'attention du prompt, les projections denses, le brouillonneur. Bonne surprise : les noyaux Triton de TensorFold tournent presque sans modification sur la B70. Le premier jeton sort le 5 octobre, à 14 tok/s, exact dès le départ (12 réponses sur 12 en drafté == série, 12 sur 12 en repris == frais).

### Tenir l'exactitude avec un GEMM qui change ses bits

Le remède : ne jamais laisser le nombre de lignes varier là où il compte. Les appels de décodage et de vérification sont rembourrés à exactement 16 lignes (16 × N avec N flux, y compris pour un flux seul servi avec `--parallel`). Les appels du prompt utilisent des tailles de bloc dont les bits sont identiques pour toute la famille (512 et 1 024 lignes), et la projection `down` une grille ancrée sur les positions absolues, multiples de 1 024. Les noyaux Triton qui doivent s'accorder tournent sans fusion de multiplication-addition, et les projections denses passent par un noyau Triton à ordre de sommation fixe.

### Les pièges qui ont coûté le plus

- **Le premier `torch.topk` compile 38 secondes** sur cette pile, pour des tranches de 4 096 éléments et plus. La première requête prenait 50 s. Remplacé par un top-k en deux étages.
- **Débordements de registres** : les noyaux d'attention en arbre débordaient (256 registres par fil, environ 30 000 débordements). Répartir les colonnes de sortie sur quatre programmes et calculer les scores par tranches a rendu ces noyaux 5 à 14 fois plus rapides, aux mêmes bits.
- **Un masque sur un chargement dans la boucle** coûte un facteur 5 à 6 sur ce compilateur. Mieux vaut borner l'index et ne masquer que les poids.
- **Les conversions bf16 / fp16** autour de chaque appel 4 bits : environ 740 lancements par tour. Garder les activations en fp16 de bout en bout les a supprimés.

Résultat du portage Python, publié le 9 octobre : décodage série à 38 ms par jeton (26 tok/s), décodage drafté de **76 à 86 tok/s** en contexte court, préremplissage à **1 870 tok/s** à 2 000 jetons, et jusqu'à quatre flux exacts en parallèle.

## La comparaison, avec la ligne qui nous est moins favorable

Nos premières mesures opposaient TensorFold à vLLM v0.31 avec MTP4, son mécanisme de prédiction par défaut. Mais vLLM v0.31 sait aussi utiliser le brouillonneur DFlash2, à condition de tourner en bf16 (en fp16, son taux d'acceptation tombe à zéro sans erreur). Comparer sans cette ligne aurait gonflé l'écart. La voici, sur la même carte, le même checkpoint GPTQ et le même banc (192 jetons par flux, prompts de code et de conversation, température 0 et 1).

| Débit (tok/s) | TensorFold Python | vLLM + DFlash2 | vLLM + MTP4 | vLLM seul |
|---|---:|---:|---:|---:|
| **1 flux, code** | 116-119 | 87-99 (7 brouillons) | 62-63 | 33 |
| **1 flux, conversation** | 78-83 | 53-78 (7 brouillons) | 55-58 | 33 |
| **4 flux, agrégé** | 154-220 | 161-217 (3 brouillons) | 137-202 | 119-121 |

{{< hbars title="Débit d'un flux seul (tok/s)" caption="Un flux, prompt de code, glouton, même carte et même checkpoint GPTQ 4 bits. Les lignes Zig et llama.cpp viennent de la mesure de la PR #802 (13-14 octobre), les autres du banc du 6 octobre." barheight=16 marginleft=250 >}}
{"bars":[
  {"label":"TensorFold Zig, DFlash2","value":122.9,"text":"122,9","color":"accent"},
  {"label":"TensorFold Python, DFlash2","value":119.2,"text":"119,2","color":"accent"},
  {"label":"vLLM v0.31, DFlash2 (7 brouillons)","value":86.9,"text":"86,9","color":"signal"},
  {"label":"vLLM v0.31, MTP4","value":61.8,"text":"61,8","color":"signal"},
  {"label":"TensorFold Zig, sans brouillon","value":33.5,"text":"33,5","color":"faint"},
  {"label":"vLLM v0.31, sans brouillon","value":33.4,"text":"33,4","color":"faint"},
  {"label":"llama.cpp SYCL, Q4_0","value":27.7,"text":"27,7","color":"faint"}
]}
{{< /hbars >}}

Lus sans les arrondir dans notre sens, ces chiffres disent trois choses :

- **Pour une personne seule, TensorFold reste devant**, même face à vLLM avec le même brouillonneur : 116-119 contre 87-99 tok/s en code. L'écart vient des arbres de brouillons (4 à 5 jetons acceptés par tour), que vLLM ne fait pas.
- **À quatre flux, c'est l'égalité** : 154-220 contre 161-217 tok/s. Et vLLM doit choisir : avec 7 brouillons il est bon seul mais n'admet que deux requêtes à la fois (89-160 tok/s à quatre), avec 3 brouillons il tient la charge mais perd en flux seul (66-69 tok/s en code).
- **La différence qui ne se mesure pas en tok/s** : chaque cellule TensorFold vérifie aussi que chaque flux égale sa réponse seule et série. Aucun écart. vLLM ne tient pas cette propriété (voir plus haut).

Les fichiers bruts TensorFold et vLLM + MTP4 sont publiés dans `tools/xpu-bench/refs/` du dépôt. Les mesures vLLM + DFlash2 datent du même jour (6 octobre) et n'avaient pas encore été publiées.

## Zig sur Level Zero : plus de PyTorch, plus de Python

TensorFold a aussi un moteur natif écrit en Zig, d'abord pour Metal. Le second portage lui ajoute un backend `xpu` qui pilote la carte directement par **Level Zero**, l'interface bas niveau d'Intel : sans PyTorch, sans SYCL, sans Python à l'exécution. Le serveur `tensorfold-native` est un seul exécutable qui embarque tous ses noyaux.

- Les noyaux sont écrits en **OpenCL C** et compilés hors ligne par `ocloc` : GEMM 4 bits du décodage, GDN par morceaux et en série, attention, normes, brouillonneur.
- Les GEMM du prompt sont générés en ISA au moment du build par **nGEN** (le générateur de oneDNN), sur les unités matricielles DPAS de Xe2.
- L'attention du prompt est une FlashAttention-2 écrite pour DPAS, dans l'orientation de oneDNN (une requête par couloir, softmax dans le couloir). D'abord plus lente que le noyau Triton du portage Python (275 ms contre 185 à 6 700 jetons), elle est descendue à 155 ms, aux mêmes jetons.
- Le service à plusieurs flux partage les tours de vérification entre emplacements, chaque flux gardant son contexte DFlash2 ; un flux resté seul repasse en arbre.

La justesse est vérifiée contre une référence : les jetons gloutons égalent ceux d'une référence PyTorch fp32 du portage Python, en formats MLX et GPTQ. Pendant la revue, nous avons ajouté une référence FP64 des GEMM 4 bits.

### Les mesures du moteur Zig

Validation du 13 octobre, B70 à 230 W, serveur `--parallel 8 --context 16384`, format GPTQ 4 bits groupes de 128 :

| | GPTQ g128 |
|---|---:|
| Préremplissage 1 k / 4 k / 8 k jetons | 1 975 / 1 965 / 1 930 tok/s |
| Un flux, arbre DFlash2, code / conversation | 133 / 90 tok/s |
| 2 / 4 / 8 flux, code (agrégé) | 235 / 310 / 270 tok/s |
| 2 / 4 / 8 flux, conversation (agrégé) | 148 / 215 / 208 tok/s |
| Décodage bench_openai, 1 / 2 / 4 / 8 requêtes | 165 / 300 / 337 / 439 tok/s |

La validation couvre quatre configurations (MLX, GPTQ, GPTQ avec prompt en W4A8, poids tournés en W4A4), 1, 2, 4 et 8 flux, en glouton et en échantillonné : **64 cellules, toutes égales** à leur exécution seule et série, sans erreur dans les journaux. Le préremplissage est borné par la puissance : à 230 W, la carte tourne vers 2 180 MHz pendant un prompt.

### Face à llama.cpp, sur la même carte

Mesure demandée pendant la revue : llama.cpp (commit 8c2134a, `llama-bench`, Q4_0, attention flash) contre TensorFold à prompt froid, sans cache.

{{< groupedbars title="Préremplissage (tok/s)" caption="Préremplissage à froid, Qwen3.8-27B, Arc Pro B70 à 230 W ; TensorFold en GPTQ 4 bits groupes de 128, llama.cpp en Q4_0. llama.cpp Vulkan s'arrête à 32 000 jetons (attente de barrière)." >}}
{"series":[
  {"label":"TensorFold Zig","color":"accent"},
  {"label":"llama.cpp SYCL","color":"signal"},
  {"label":"llama.cpp Vulkan","color":"faint"}
],
"groups":[
  {"label":"2 000 jetons","values":[2051,788,645],"texts":["2 051","788","645"]},
  {"label":"8 000 jetons","values":[1985,787,484],"texts":["1 985","787","484"]},
  {"label":"32 000 jetons","values":[1719,696,null],"texts":["1 719","696",null]},
  {"label":"64 000 jetons","values":[1473,600,null],"texts":["1 473","600",null]}
]}
{{< /groupedbars >}}

En décodage sans brouillon, TensorFold fait **33,5 tok/s** contre 27,7 pour llama.cpp SYCL. Avec DFlash2 : **122,9 tok/s** en code et **99,3** en conversation.

## Du dépôt Hygur Labs au projet officiel

Le 13 octobre, nous avons proposé le backend à Ash Hart (issue [#801](https://github.com/ashhart/TensorFold/issues/801)) avec la pull request [#802](https://github.com/ashhart/TensorFold/pull/802) : uniquement des fichiers nouveaux, plus deux lignes de `build.zig`. La revue a demandé, entre autres, de retirer l'ancien noyau Triton capturé en SPIR-V, de dimensionner le nombre de flux sur la mémoire libre réelle de la carte (lue par Sysman), et de borner chaque paramètre d'entrée. Elle a aussi motivé la référence FP64 des GEMM.

Le backend est livré dans **TensorFold 1.0.12** (16 octobre) et la pull request a été fusionnée le 18 octobre. C'est une préversion à compiler soi-même : les archives de release ne la contiennent pas. Ash n'a pas de carte Intel ; l'intégration repose sur nos mesures sur la B70 et sur ses propres vérifications côté hôte (build, tests, analyse du code).

Le portage Python reste disponible dans [hygurlabs/tensorfold-xpu](https://github.com/hygurlabs/tensorfold-xpu) (branche `main`, Apache-2.0), avec ses scripts de mesure et ses résultats bruts. Il n'a pas été proposé au projet officiel : seul le moteur Zig l'a été.

## Ce que ce backend ne fait pas

- **Un seul modèle pour notre runner** : les dimensions de Qwen3.8-27B sont des constantes, et certaines formes sont écrites dans les noyaux. Depuis, un autre contributeur a ajouté Nemotron 3.5 Lightning au même serveur XPU, en version expérimentale, sur la même interface Level Zero.
- **Pas dans les archives** : le backend est dans le code de TensorFold depuis la 1.0.12, mais les archives publiées à chaque version (macOS et Linux) ne le contiennent pas. Il faut le compiler avec `zig build xpu-native`.
- **Formats 4 bits seulement** : MLX groupes de 64 et GPTQ symétrique groupes de 128. Le prompt en W4A8 demande le GPTQ.
- **Une seule carte** a servi aux mesures. Une B70 d'un autre fabricant, avec une autre limite de puissance, donnera d'autres chiffres de préremplissage.
- **Le débit drafté dépend du texte** : le code se prédit mieux que la conversation, d'où les fourchettes.

## Construire et lancer le moteur Zig

Prérequis : Zig 0.17, le pilote `xe` et le runtime de calcul Intel (Level Zero), `ocloc`, un compilateur C++17 et les en-têtes nGEN de oneDNN.

```bash
zig build xpu-native -Doptimize=ReleaseFast -Dngen-include=<chemin vers ngen>
zig build xpu-qwen27 -Doptimize=ReleaseFast -Dngen-include=<chemin vers ngen>

tensorfold-native serve <Qwen3.8-27B GPTQ> --drafter <DFlash2> \
  --parallel 8 --context 16384 --host 127.0.0.1 --port 8081
```

Les bancs (`bench_openai.py` pour le préremplissage et le décodage, `bench_concurrent.py --alone --serial` pour l'exactitude en concurrence) sont dans `tools/`. Si vous avez une Arc Pro B70, rejouez-les et envoyez-nous vos chiffres, y compris s'ils diffèrent des nôtres.

{{< callout label="Une B70 chez vous ?" >}}
Nous installons et réglons des serveurs d'inférence sur site, y compris sur carte Intel : dimensionnement, mise en service, connexion à vos sources, formation de vos équipes.
{{< /callout >}}

## À lire aussi

[Une IA privée sur une carte à 1 900 € : ce que la B70 change pour les professions réglementées](/fr/blog/ia-locale-b70-metiers/) : les mêmes mesures, lues pour un cabinet, une étude ou un service RH : coût, temps de réponse au bureau, et réponses qu'un auditeur peut rejouer.


## Sources

- hygurlabs/tensorfold-xpu, branche main : docs/recipes/xpu.md et tools/xpu-bench/refs/*.json (portage Python)
- hygurlabs/tensorfold-xpu, branche zig : docs/xpu/README.md (moteur Zig, validation du 2026-10-13)
- ashhart/TensorFold#802 : commentaire de mesure llama.cpp (13-14 octobre 2026), backend XPU fusionné le 18 octobre, TensorFold 1.0.12
- ashhart/TensorFold#801 : proposition du backend
- Mesure interne, vLLM v0.31 + DFlash2 et exactitude vLLM : fichiers bruts du banc (vllm_dflash_*.json, exact_vllm_*.json), 6 octobre 2026

