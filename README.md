# grn_balladeer

Pipeline GRN pour la classification TDAH par EEG (projet BALLADEER).

## Organisation

```
grn_balladeer/
├── data/          # Module 1 — labels, mapping demographics, split stratifie
├── connectivity/  # Module 3 — PLV, Laplacien magnetique (a venir)
├── model/         # Module 4-6 — CQT, GRNEncoder, tete de classification (a venir)
├── losses/        # Module 7, 7b — harmonique, symbolique, triplet (a venir)
├── training/       # Module 8, 9 — dual-branch, boucle CV (a venir)
├── eval/          # Module 10-13 — baselines, ablations, XAI
├── configs/       # 1 fichier YAML par dataset (BALLADEER, futur dataset X)
└── requirements.txt
```

## Reutiliser sur un autre dataset

Le code de `model/`, `losses/`, `connectivity/` est concu pour etre
agnostique au dataset. Pour adapter le pipeline a un nouveau dataset EEG :

1. Copier `configs/balladeer.yaml` vers `configs/mon_dataset.yaml`.
2. Adapter les champs (`labels.field`, `demographics_schema`, `eeg_devices`).
3. Ne PAS toucher au code de `model/`, `losses/`, `connectivity/`.

## Execution

**Developpement et entrainement : Google Colab Pro.**
Monter Drive avec `grn_balladeer.data.labels.mount_drive_colab()`, puis
utiliser les fonctions des sous-packages normalement.

**Reproductibilite de l'environnement : Docker.**
Le `Dockerfile` (a venir, une fois le pipeline complet valide sur Colab)
fixe les versions exactes de `requirements.txt`. Il a ete valide en CPU
sur un sous-echantillon ; l'entrainement complet se fait via le notebook
Colab Pro fourni, pas dans le conteneur.

## Etat d'avancement

- [x] Module 1 (labels) — verifie sur les 158 enregistrements reels
- [x] Module 2a (canaux CGX/Emotiv confirmes)
- [x] Module 10 (baselines SVM/RF/ratio theta-beta) — code pret, en attente
      des epochs pretraites (Module 2b) pour tourner reellement
- [ ] Module 2b, 3, 4-9, 11-13
