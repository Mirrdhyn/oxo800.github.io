# Une IA privée sur une carte à 1 900 € : ce que la B70 change pour les professions réglementées

> Un modèle de 27 milliards de paramètres sur une carte Intel à 1 900 € HT, sans qu'aucun document ne sorte : vitesse face à vLLM, cas de bureau et réponses rejouables pour l'audit.

- Date: 2026-10-23
- Language: fr
- URL: https://0x0800.com/fr/blog/ia-locale-b70-metiers/
- Tags: b70, tensorfold, ia-locale, professions-reglementees
- Translation (en): https://0x0800.com/blog/ia-locale-b70-metiers/


Sur une carte graphique à 1 900 € HT, un modèle de 27 milliards de paramètres répond à plus de 100 jetons par seconde, sans qu'aucun document ne quitte l'entreprise. Ce qui rend cela possible : notre portage du moteur TensorFold sur la carte Intel Arc Pro B70, aujourd'hui intégré au projet officiel.

{{< keyfigures >}}
[
  {"label": "Prix de la carte", "value": "1 900 € HT", "detail": "Intel Arc Pro B70, 32 Go, contre 5 700 € HT pour une RTX 5090 au même volume de mémoire"},
  {"label": "Une personne", "value": "90 à 133 tok/s", "detail": "Qwen3.8-27B, selon le texte produit (conversation ou code)"},
  {"label": "Réponses rejouables", "value": "100 %", "detail": "une réponse servie à côté d’autres égale la même réponse servie seule, à chaque essai"}
]
{{< /keyfigures >}}

## Le vrai frein, c'est l'hébergement

Dans un cabinet d'expertise comptable, une étude notariale, une mutuelle ou un service RH, l'IA bute d'abord sur une question simple : où partent les documents ? Les dossiers clients, les bulletins de paie ou les comptes rendus médicaux ne partent pas vers une API externe, parce que le secret professionnel et le RGPD l'interdisent.

On peut faire autrement : faire tourner le modèle sur place, dans les murs de l'entreprise. Les projets s'arrêtent alors souvent sur le prix du matériel.

## 32 Go de mémoire pour trois fois moins cher

Un modèle utile, comme Qwen3.8-27B en 4 bits, demande environ 32 Go de mémoire sur la carte graphique. La RTX 5090 de NVIDIA les a, pour 5 700 € HT au moins cher, en stock, au moment où nous écrivons. L'Intel Arc Pro B70 a les mêmes 32 Go pour **1 900 € HT**.

Jusqu'ici, l'écart de prix ne servait pas à grand-chose : la plupart des moteurs d'inférence rapides ne savaient travailler qu'avec CUDA, donc qu'avec NVIDIA. Nous avons porté TensorFold, un moteur open source écrit par Ash Hart, sur la B70. Le portage est intégré au projet officiel depuis la version 1.0.12, sous licence Apache-2.0. Le détail technique est dans [l'article qui raconte le portage](/fr/blog/tensorfold-xe2-b70/).

## Ce que vit l'utilisateur

Nous avons comparé TensorFold à vLLM v0.31, le moteur le plus répandu, sur la même carte et avec le même modèle. vLLM a été testé avec son propre mécanisme d'accélération (MTP4) et avec le même petit modèle d'avance que TensorFold (DFlash2), pour que la comparaison soit loyale.

| Débit (jetons/s) | TensorFold | vLLM, meilleur réglage |
|---|---:|---:|
| Une personne, rédaction de code | 116 à 133 | 87 à 99 |
| Une personne, conversation | 78 à 99 | 53 à 78 |
| Quatre personnes en même temps, total | 154 à 220 | 161 à 217 |

Pour une personne seule, TensorFold répond nettement plus vite. À quatre personnes en même temps, les deux moteurs font jeu égal.

{{< callout label="D'où viennent ces fourchettes" >}}
L'écart entre le bas et le haut de chaque fourchette tient au texte produit : le code se prédit mieux que la conversation. Les chiffres hauts de TensorFold pour une personne viennent de la version la plus récente du moteur ; ceux à quatre personnes, de la version mesurée face à vLLM le 6 octobre.
{{< /callout >}}

## Un dossier de vingt pages, quatre collègues

Nous avons simulé une journée de bureau : un dossier comptable fictif d'une vingtaine de pages (environ 18 700 jetons), interrogé en même temps par quatre collègues sur des points différents (échéances de TVA, montant d'une facture, etc.). Chaque réponse est vérifiée sur les faits du dossier.

| | TensorFold | vLLM + MTP4 |
|---|---:|---:|
| Le même dossier, déjà ouvert, quatre questions à la fois | 7,8 s | 11,7 s |
| Quatre dossiers différents ouverts au même moment | 54,9 s | 54,5 s |

Quand le dossier est déjà chargé, chacun a sa réponse en 7,8 secondes au lieu de 11,7. Quand quatre collègues ouvrent quatre dossiers différents au même moment, les deux moteurs sont à égalité : la lecture des documents domine, et aucun des deux ne l'accélère. Ce scénario n'a pas été mesuré avec vLLM + DFlash2.

## Une réponse qu'un auditeur peut rejouer

Dans un secteur réglementé, un autre résultat pèse autant que la vitesse. Avec TensorFold, une réponse calculée seule et la même réponse calculée à côté de trois autres sont **identiques, jeton pour jeton**. Nous l'avons vérifié dans chaque scénario, sans un seul écart. Un auditeur peut rejouer une réponse et retrouver le même texte, quelle que soit la charge du serveur au moment où elle a été produite.

vLLM ne tient pas cette propriété. Sur la même carte, il ne rend la même réponse en charge que seul dans 27 cas sur 48 avec son accélération, et dans 16 sur 48 sans. Même seul, deux essais successifs de la même question en conversation ne donnent pas toujours le même texte.

## Pour qui et jusqu'où

- **Une petite équipe** : une machine de bureau et une carte à 1 900 €, sans abonnement et sans données qui voyagent.
- **Des dizaines d'utilisateurs en continu** : il faudra mesurer la charge réelle avant d'acheter. Nos mesures vont jusqu'à huit flux simultanés sur une carte.
- **Deux modèles** pour l'instant sur cette carte : Qwen3.8-27B en 4 bits, celui de nos mesures, et Nemotron 3.5 Lightning, ajouté depuis par un autre contributeur en version expérimentale.
- **Une installation à compiler** : le backend Intel fait partie du code de TensorFold, mais pas encore des archives prêtes à l'emploi publiées à chaque version.

Le code, les scripts de mesure et les résultats bruts sont publics, pour que chacun puisse refaire les tests. Si vous avez une Arc Pro B70 sous la main, rejouez nos mesures et dites-nous ce que vous obtenez.

{{< callout label="Vos documents restent chez vous." >}}
Nous dimensionnons, installons et réglons le serveur sur site, le connectons à vos sources et formons vos équipes. Après une journée de cadrage sur site, nous vous disons si une B70 couvre votre usage.
{{< /callout >}}


## Sources

- hygurlabs/tensorfold-xpu : tools/xpu-bench/workflow_sim.py et résultats bruts wf_*.json (cas de bureau)
- ashhart/TensorFold#802 : backend Intel Arc Pro B70, intégré depuis la version 1.0.12 (Apache-2.0)
- Mesure interne, TensorFold contre vLLM v0.31 (MTP4, DFlash2) sur Intel Arc Pro B70, banc du 6 octobre 2026
- Prix de la RTX 5090 (5 700 €) : relevé chez LDLC à la rédaction, octobre 2026

