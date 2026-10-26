# Statistical Analysis Plan (SAP) — v1.0, 2026-09-30

**Working title:** Beyond tumour size: how much does T2-weighted MRI AI add over lesion volume for muscle-invasion prediction in bladder cancer — and why published results on a public benchmark are inflated. A four-vendor leave-one-center-out study.

**Data:** FedBCa public dataset (Cao et al., Sci Data 2024, doi 10.1038/s41597-024-03971-0), 4 centers / 3 vendors, T2WI + expert lesion masks + pathology labels. No new patients, no reader input; fully automated pipeline.

**Disclosure of prior look:** Before this SAP was written, a volume-only logistic model was evaluated with LOCO (lesion-level AUC 0.75–0.84 per center; T1 vs T2 AUC 0.78). No image-based model was run before this SAP. This SAP is fixed before any image model is trained; any deviation will be reported as such.

## Part A — Incremental value over volume (primary question)

### Units and labels
- Lesion = one mask file (275). Patient = center + T2WI filename (221).
- Lesion label: MIBC (pT2–T4) = 1, NMIBC (pTa/pT1) = 0.
- Patient label: 1 if any lesion MIBC. Patient score = max lesion score.
- **Primary analysis: patient level.** Secondary: lesion level (patient-clustered inference).
- Predictors must be available before TURBT: image, mask-derived features, age, sex. **Grade and T stage are never used as predictors** (postoperative, outcome leakage).

### Validation
- 4-fold leave-one-center-out (LOCO). All preprocessing parameters, feature selection, hyperparameters and early stopping are learned **inside the training centers only** (inner patient-grouped 5-fold CV).
- Out-of-fold predictions from the 4 held-out centers are pooled for pooled metrics. Per-center metrics are also reported and combined with random-effects meta-analysis (DerSimonian–Laird on logit AUC).

### Arms (identical folds)
- A. Volume: logistic regression on log(mask volume, mL).
- A+. Clinical-volume: log-volume + age + sex + lesion count (patient-level only).
- B. Radiomics: IBSI-compliant PyRadiomics (original + LoG + wavelet, z-score normalised image, resampled 0.7×0.7×3.5 mm, fixed bin width), near-zero-variance and |ρ|>0.9 filtering, L1-penalised logistic regression (λ tuned inner CV).
- C. 3D CNN: ResNet-18 3D (MONAI), 2-channel input (image, mask), fixed physical-size crop centred on the lesion (preserves scale), 5 seeds ensembled.
- D. Foundation features: frozen DINOv2 ViT-B/14 slice embeddings of lesion-centred crops, mean+max pooled over lesion slices, PCA (fit on training centers), L2 logistic regression.
- For B, C, D: **"+volume" versions** (image score + log-volume in a logistic stacker fitted on inner OOF predictions) and **"residual"** versions (features orthogonalised to log-volume within training centers) to isolate volume-independent signal.

### Primary endpoint
ΔAUC (patient level, pooled LOCO) of the best pre-specified image arm+volume versus Arm A. The "best" arm is chosen on inner-CV performance within training centers, never on held-out centers. All other arms are reported as secondary.

Clinically meaningful increment: ΔAUC ≥ 0.05. Estimation framing: 95% CI from 2,000 patient-level cluster bootstrap resamples stratified by center. Conclusions:
- CI lower bound > 0.05 → clinically meaningful increment;
- CI upper bound < 0.05 → no clinically meaningful increment;
- otherwise inconclusive.

Precision: 221 patients / 83 events → approximate SE(AUC) ≈ 0.03; paired ΔAUC SE smaller given correlation. Minimum detectable ΔAUC ≈ 0.06–0.08.

### Secondary endpoints
- Calibration: per held-out center, calibration intercept (calibration-in-the-large) and slope; O/E ratio; pooled via random-effects; flexible calibration curves (loess). Brier score.
- Decision curve analysis: net benefit at threshold 0.10–0.50 (key thresholds 0.20, 0.30 for escalation to cystectomy/neoadjuvant discussion vs. re-TURBT).
- NRI (categorical, threshold 0.30) and IDI vs Arm A.
- Pre-specified subgroups (lesion level): lesions < 3 mL; T1 vs T2 boundary (pT1 vs pT2 only, 108 lesions) — AUC of each arm vs volume; exploratory.
- Sensitivity analyses: lesion-level analysis; exclusion of multifocal patients; center-recalibrated models (intercept update using held-out center prevalence — reported as "with local recalibration").

## Part B — Inflation audit (why published AUCs are high)

Single fixed backbone (2D ResNet-18, ImageNet-pretrained, lesion-containing slices, lesion score = mean slice probability) and the 3D CNN of Arm C, evaluated under evaluation designs found in the literature:
1. Random slice split (80/20, 5 repeats) — slice-level AUC.
2. Random slice split — lesion-aggregated AUC.
3. Random lesion split (lesions of one patient may cross train/test).
4. Patient-grouped 5-fold CV (pooled centers).
5. LOCO (reference).
6. LOCO + transductive test-time adaptation (AdaBN re-estimation of batch-norm statistics on the unlabelled held-out center).
7. LOCO with test-set model selection (best epoch chosen on held-out center — "optimistic selection").

Endpoint: AUC inflation = AUC(design) − AUC(LOCO), with bootstrap CIs (design-appropriate resampling unit). Also report volume-only performance under each design (to show that design inflation, not image learning, drives high numbers).

Literature table: all empirical studies using FedBCa (identified via Crossref/OpenAlex/Semantic Scholar citation search, 2026-09-30), extracting reported AUC, evaluation unit, split design, external/LOCO use, calibration reporting, volume baseline reporting.

## Reporting
TRIPOD+AI (primary), CLAIM 2024, CLEAR (radiomics arm). Code and fold assignments will be released.

## Software
Python 3.10; PyTorch 2.12, MONAI 1.6, PyRadiomics 3.0.1, scikit-learn 1.6, statsmodels 0.14, timm (DINOv2 lvd142m). Seeds fixed; GPU RTX 4080.
