# Intel D270 contre RTX PRO 6000 : les 5,9x sont réels, sur un terrain étroit

> Le 5,9x d'Intel vient du FP64 appliqué à la tarification d'options. Sur l'inférence, l'écart tombe à 12 %, mais pour un peu plus de la moitié de l'énergie, et les 160 à 480 Go de LPDDR5X de la D270 comptent davantage pour le déploiement local.

- Date: 2026-11-21
- Language: fr
- URL: https://0x0800.com/fr/blog/intel-d270-5-9x/
- Tags: intel, d270, nvidia, fp64, inference-locale
- Translation (en): https://0x0800.com/blog/intel-d270-5-9x/


Intel annonce que sa D270 (350 W) offre jusqu'à 5,9 fois le débit de la RTX PRO 6000 Blackwell Server Edition de Nvidia (600 W). J'ai lu le rapport et ses notes : le chiffre est réel, mais il couvre un terrain très étroit.

Les 5,9x viennent du calcul en double précision (FP64) appliqué à la tarification d'options : Monte-Carlo européen (5,9x), Black-Scholes européen (5,6x), arbre binomial (4,6x). Sur l'option américaine, l'écart tombe à 1,92x, et la moyenne des quatre tests est de 4,5x. Nvidia limite le FP64 de ses cartes workstation à un ratio de 1:64, alors que la D270, basée sur l'architecture Xe3P, l'exécute à pleine vitesse, avec 64 FMA FP64 par cœur. Intel attaque là où Nvidia a laissé un vide.

Quelques réserves sur la méthode. Les mesures viennent d'Intel, sur du silicium de préproduction. Selon Intel, chaque test n'a mobilisé qu'une seule carte de chaque côté, mais les piles logicielles diffèrent : oneAPI côté Intel, un conteneur CUDA 12.8 avec DPC++ et oneMath côté Nvidia.

Sur l'inférence, l'écart se resserre. Sur Qwen3-30B-A3B en FP8, la D270 sort 12 % de débit en plus que la RTX PRO 6000, pour 350 W contre 600 W, et sans torch.compile, que la carte Nvidia utilisait. Sur un modèle de décision (Drex v2.0), Intel annonce une latence inférieure de 29 % à concurrence 1. Pour ces deux tests, la carte Nvidia tournait dans une VM AWS. Les 5,9x ont disparu, mais une carte Blackwell haut de gamme se fait battre sur un MoE de 30 milliards de paramètres avec un peu plus de la moitié de l'énergie.

Pour le déploiement local, la mémoire pèse le plus. La RTX PRO 6000 plafonne à 96 Go de GDDR7. Intel renonce aux HBM et empile de la LPDDR5X sur une carte PCIe refroidie par air : 160 Go en version de référence (1,5 To/s), jusqu'à 480 Go chez les partenaires. Des modèles dont les poids exigeaient un rack de cartes tiennent sur une seule carte de 350 W. À SC26, Intel et SambaNova présentent une architecture hybride : la D270 prend en charge le préremplissage des contextes massifs, la SN50 assure la génération des tokens.

Prix et date de disponibilité restent inconnus. Mais si Nvidia continue de limiter la mémoire de ses cartes, l'inférence à grande capacité pourrait glisser vers ces cartes LPDDR5X.


## Sources

- Intel Newsroom, « The case for Intel GPU D270 in high-performance computing », 16 novembre 2026, et ses notes de configuration
- Moyenne des quatre tests financiers : calcul à partir des chiffres d'Intel

