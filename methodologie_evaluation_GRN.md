# Méthodologie d'évaluation — GRN/BALLADEER (dataset complet, Colab)

Objectif de ce document : répondre proprement à "est-ce que le GRN donne
de bons résultats ou pas", avec un protocole qui tienne devant un
reviewer ESWA — pas juste un chiffre d'accuracy. Écrit pour être suivi
sur Colab avec les 138 sujets complets (Drive), en réutilisant les
briques déjà codées et testées ici sur les 4 sujets.

## 1. Protocole de validation croisée

**Niveau sujet, jamais niveau epoch.** Déjà en place :
`data.labels.stratified_subject_kfold(label_df, k=5, seed=42)` —
stratifie conjointement sur `label + sex + age_bin`, déjà re-vérifié sur
les 138 vrais sujets (répartition hommes/femmes cohérente ~63-70% dans
chaque fold train/val).

- **k=5** par défaut (ou k=10 si le temps de calcul le permet — plus de
  folds = estimation plus stable avec un dataset de cette taille).
- **CV imbriquée si des hyperparamètres sont réglés** (lambda1/2/3,
  learning rate, architecture) : boucle externe = évaluation finale,
  boucle interne = sélection d'hyperparamètres, jamais la même boucle
  pour les deux — sinon le score est optimiste.
- **Plusieurs seeds par fold** (au moins 3) pour séparer la variance due
  aux données (quel split) de la variance due à l'initialisation
  aléatoire du modèle — rapporter moyenne ± écart-type sur les
  seeds×folds, pas un seul chiffre.

## 2. Baselines et ablations — la structure qui prouve que le GRN sert à quelque chose

Sans ça, un bon score ne prouve rien : il faut montrer que chaque pièce
de l'architecture apporte quelque chose par rapport à plus simple.

**Baselines déjà codées** (`eval/baselines.py`) :
- `train_svm_baseline` / `train_rf_baseline` sur `extract_band_power_
  features` (theta/beta ratio inclus) — le baseline classique de la
  littérature EEG-TDAH, à battre.

**Ablations à faire tourner, dans cet ordre de priorité** :
1. **GRN EEG-only vs GRN dual-branch** (avec/sans branche EDA+comportementale)
2. **Avec/sans `L_harm`** (`lambda1=0`) — le harmonic loss apporte-t-il
   vraiment quelque chose ?
3. **Avec/sans `L_symb`** (`lambda2=0`)
4. **Option A (résonance apprise) vs Option B (`fixed_consonance_prior`,
   déjà codée dans `grn_encoder.py` comme baseline d'ablation prévue
   dès la conception)**
5. **Avec/sans `L_triplet`** (`lambda3=0`) — nécessite ≥2 sujets/classe
   par batch pour que le triplet mining ait quoi que ce soit à
   apprendre (voir `make_pk_batches`), donc seulement testable
   proprement sur le dataset complet, pas sur les 4 sujets actuels.
6. **Magnétique vs Laplacien classique** (si le temps le permet) — pour
   isoler la contribution spécifique de la préservation de phase.

Chaque ablation tourne sur le MÊME protocole de CV (section 1), pas sur
un split différent — sinon les comparaisons ne sont pas valides.

## 3. Métriques — jamais l'accuracy seule

Avec un déséquilibre de classes réel (88 TDAH / 50 Contrôle dans les 138
sujets, donc ~64%/36%), l'accuracy seule peut cacher un modèle qui
prédit presque toujours la classe majoritaire. `eval/baselines.py`
fournit déjà `evaluate()` → `EvalResult` — vérifier qu'il couvre :
- **Balanced accuracy** (moyenne du recall par classe — pas trompée par
  le déséquilibre)
- **F1 macro** ET F1 par classe
- **Sensibilité/spécificité séparément** (pertinent cliniquement : rater
  un vrai TDAH n'a pas le même coût qu'un faux positif)
- **AUC-ROC**
- **Matrice de confusion** (toujours la garder, même si on ne la cite
  pas dans le corps du papier — utile en annexe/discussion)

## 4. Rapport désagrégé — obligatoire, pas optionnel

Déjà codé (`training.evaluate::evaluate` avec `meta_df`, et
`eval.baselines::evaluate_disaggregated`) :
- **Par sexe** — vu le confound structurel de la cohorte (69.3% hommes
  TDAH vs 56.0% Contrôle) et la littérature sur la décodabilité du sexe
  depuis l'EEG (65-81% accuracy), ne jamais publier un score global sans
  vérifier qu'il ne cache pas un écart de performance homme/femme.
- **Par tranche d'âge** (`age_bin`, déjà dans `build_label_table`).
- **`training.leakage_probes::check_sex_leakage`** sur les embeddings
  appris (`z_eeg`/`omega`/`z_joint`) — première fois que ça aura un sens
  statistique réel avec 138 sujets (inutilisable sur les 4 actuels).
  Si le score dépasse nettement le hasard, creuser AVANT de conclure que
  c'est un biais à corriger — voir la nuance sur la littérature TDAH
  spécifique par sexe (v6/v7).

## 5. Significativité statistique — comparer GRN aux baselines proprement

Ne jamais conclure "le GRN est meilleur" sur un seul chiffre de
différence. Sur les scores par fold (accuracy ou F1, GRN vs SVM/RF) :
- **Test de Wilcoxon signé (non-paramétrique)** sur les paires de scores
  par fold — préférable à un t-test classique vu le petit nombre de
  folds (5-10).
- Si possible, **test de McNemar** sur les prédictions elles-mêmes
  (accord/désaccord par sujet entre GRN et baseline) — plus fin qu'une
  comparaison de moyennes.
- Reporter un intervalle de confiance (bootstrap sur les folds), pas
  juste un p-value isolé.

## 6. Diagnostics à logger à CHAQUE run, pas en debug ponctuel

Déjà codés, à intégrer systématiquement dans la boucle d'entraînement
complète (pas juste appelés à la main comme jusqu'ici) :
- `training.omega_diagnostics::check_omega_collapse` à chaque epoch —
  si `is_collapsed=True` réapparaît sur le vrai dataset, alerte immédiate.
- `training.leakage_probes::check_sex_leakage` en fin d'entraînement,
  sur le fold de validation.
- Courbe de perte complète (les 4 composantes séparément, comme dans les
  runs de cette session) — un `total_loss` qui baisse peut cacher un
  `L_symb` qui stagne pendant que `L_task` fait tout le travail.

## 7. Interprétabilité — le vrai argument différenciant du papier (BMI4DND)

Ne pas se limiter à l'accuracy — le positionnement du papier (échapper
au "paradoxe interprétabilité-précision") a besoin de preuves séparées :
- **`omega` appris est-il neuroscientifiquement plausible ?** Comparer
  aux bandes canoniques (theta~4-8Hz, alpha~8-13Hz, beta~13-30Hz) —
  est-ce que les `omega_i` convergent vers des valeurs qui correspondent
  à une activité EEG réelle, ou vers un optimum arbitraire dans [1,45]Hz ?
- **Le cluster frontal de `L_symb` reste-t-il pertinent après
  entraînement complet ?** (déjà trouvé empiriquement en Semaine 2 sur
  UB0136 seul — à re-vérifier sur les 138 sujets, pas juste supposer que
  ça tient à plus grande échelle).
- **`determine_rule_direction`** — enfin testable pour de vrai avec 138
  sujets (invalide statistiquement avec les 4 actuels, comme déjà noté).

## 8. Efficacité/déploiement — pertinent pour BMI4DND spécifiquement

Le papier positionne le GRN sur le compromis efficacité/précision/
interprétabilité (cadre Pareto). À mesurer et rapporter, même si ce
n'est pas le papier ESWA principal :
- Nombre de paramètres par composant (déjà su : `AuxBranchEncoder`
  ~9.4k, `CrossAttentionFusion` ~41.8k — à compléter avec `GRNEncoder`
  et les têtes).
- Latence d'inférence par epoch (CPU et si possible GPU Colab).
- Mémoire pic pendant l'entraînement à pleine échelle (138 sujets).

## 9. Ordre d'exécution recommandé sur Colab

1. Charger les 138 sujets, `stratified_subject_kfold(k=5)`.
2. Baselines SVM/RF (rapide, donne un plancher de référence immédiat).
3. GRN EEG-only, ablations 2-4 (section 2) — le cœur de la contribution.
4. GRN dual-branch (ablation 1) + triplet loss (ablation 5) — seulement
   une fois les EEG-only ablations propres, vu leur coût de calcul plus
   élevé et leur dépendance à `make_pk_batches`.
5. Diagnostics (section 6) + rapport désagrégé (section 4) sur le
   MEILLEUR modèle retenu, pas sur tous les runs d'ablation.
6. Tests de significativité (section 5) entre le meilleur GRN et les
   baselines.
7. Interprétabilité (section 7) en dernier — une fois qu'on sait que le
   modèle marche, pas avant.

## Ce qui manque encore côté code pour exécuter ça tel quel

- Une fonction d'orchestration qui enchaîne CV + ablations + logging
  systématique (rien de tel n'existe encore — chaque run a été fait
  manuellement jusqu'ici). À écrire une fois sur Colab, pas ici (pas de
  dataset complet disponible dans ce container).
- Confirmer que `eval/baselines.py::evaluate()` calcule bien TOUTES les
  métriques de la section 3 (balanced accuracy, F1 par classe,
  sensibilité/spécificité séparées) — à vérifier avant de s'appuyer
  dessus, pas supposé.
