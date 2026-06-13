# Motor-Imagery EEG Classification (EEGMMIDB)

Decoding **imagined movement** from scalp EEG, with an honest focus on the gap
between *within-subject* and *subject-independent* performance. Built on the
PhysioNet **EEG Motor Movement/Imagery Database** (EEGMMIDB; Schalk et al.,
2004 — 109 subjects, 64 channels, BCI2000), loaded programmatically via MNE.

> **Two readers in mind.** If you're a **neuroscientist**, the boxes labelled
> *Physiology* explain why each step exists. If you're an **ML engineer**, the
> boxes labelled *ML* explain the modelling and evaluation choices. You should be
> able to skip the other one.

---

## 1. The problem in one paragraph

When you move — or merely *imagine* moving — your hand, the sensorimotor rhythms
(**mu**, ~8–13 Hz, and **beta**, ~13–30 Hz) over the *opposite* side of your
brain briefly weaken. This is **event-related desynchronisation (ERD)**. Left-hand
imagery shows up near electrode **C4** (right hemisphere), right-hand imagery near
**C3** (left hemisphere). A brain–computer interface (BCI) tries to read that
spatial pattern from a few seconds of EEG and guess what the person imagined. We
benchmark several decoders on this task and — crucially — measure how well they
work on a **brand-new person the model has never seen**, which is the hard part.

**Task 1 (primary):** left fist vs. right fist, *imagined* (runs 4, 8, 12).
**Task 2 (extension):** 4-class — left fist / right fist / both fists / both feet,
*imagined* (left/right from runs 4/8/12 **plus** fists/feet from runs 6/10/14).

---

## 2. Results

> Numbers are produced by the pipeline, not hand-entered — they live in
> `artifacts/summary_*.csv`. The table below is the **classical-model pilot**:
> binary left/right-fist *imagery*, within-subject 5-fold CV, on a 20-subject
> cohort (18 usable — subjects 9 and 13 lost all epochs to amplitude rejection
> and were excluded, logged). Deep models (EEGNet, Denoising-EEGNet), the LOSO
> column, and the full 109-subject run are GPU-class jobs (see §6) and remain to
> be filled from their own `summary_*.csv`.

| Model | Within-subject acc (mean ± sd) | Kappa | LOSO acc | Notes |
|---|---|---|---|---|
| **Riemann + LR** | **0.620 ± 0.133** | 0.230 | _GPU run_ | covariance → tangent space; best & most consistent |
| CSP + LDA | 0.578 ± 0.132 | 0.131 | _GPU run_ | reference baseline |
| CSP + SVM (RBF) | 0.553 ± 0.121 | 0.090 | _GPU run_ | non-linear CSP |
| EEGNet | _GPU run_ | — | _GPU run_ | compact CNN |
| **Denoising-EEGNet** | _GPU run_ | — | _GPU run_ | custom; see §5 ablation |

_(n = 18 subjects; cohort = subjects 1–20. Reproduce with
`python -m mibci.run --config configs/binary_classical20.yaml --experiment binary`.)_

**Reading the pilot honestly.** Riemann leads, matching the literature, but no
pairwise difference survives Holm correction at n = 18 (CSP-SVM vs Riemann is the
closest, raw p = 0.023 → Holm p = 0.069). Accuracies are *modest* (~0.55–0.62):
these are imagined, not executed, movements (runs 4/8/12 — harder), the pipeline
is untuned, and it classifies the full 0–4 s window. The per-subject plot
(§6 / `artifacts/per_subject_classical20_binary_within.png`) is the real
takeaway — top subjects reach ~0.85–0.95 while several sit at chance, the classic
"BCI illiteracy" spread. Obvious levers to raise the mean: a tighter post-cue
window (e.g. 0.5–2.5 s), tuning `csp_components`, and the deep models.

Expectation to state up front: **within-subject ≫ LOSO**. Decoding someone you
calibrated on is easy; decoding a stranger with zero calibration is hard, because
brains differ in anatomy and rhythm topography (see §4.3). The smoke cohort
already shows this ordering (LOSO < within-subject).

---

## 3. How to run

```bash
make setup            # create .venv (Python 3.12) and install pinned deps
make test             # unit tests (offline, fast)
make smoke            # end-to-end on ~4 subjects — proves the pipeline runs

# Full benchmarks (heavy; intended for a GPU machine):
make binary           # within-subject, left vs right fist imagery, 109 subjects
make binary-loso      # subject-independent (LOSO) version
make fourclass        # 4-class experiment

# Or call the CLI directly:
python -m mibci.run --config configs/binary.yaml --experiment binary --cv loso
```

The `make` targets run the package straight from `src/` (`PYTHONPATH=src`), so
they work without an editable install. If you prefer the installed entry points
(`mibci ...` or `python -m mibci.run ...`), `make setup` also runs `pip install -e .`.

First run downloads EEGMMIDB into `~/mne_data` (cached thereafter). Preprocessed
epochs are cached under `artifacts/cache/<hash>/` keyed on the preprocessing
config, so re-running with a different *model* never re-pays preprocessing.

---

## 4. Method, decision by decision

Every parameter lives in one YAML config (`configs/`). There are **no magic
numbers in the code** — change the band, window, or thresholds in the YAML and the
run (and its cache key) update accordingly.

### 4.1 Data handling
- **All 109 subjects** fetched via `mne.datasets.eegbci`.
- **Known-bad subjects (88, 89, 92, 100) are excluded *and logged*** with the
  reason (timing/sampling inconsistent with the 160 Hz protocol, which
  desynchronises event annotations from the signal). We *also* re-verify every
  recording's sampling rate and annotations independently, so a *new* anomaly is
  caught rather than silently epoched into noise. Nothing is dropped silently.
- Channels standardised to the **10-10** montage so topomaps are spatially correct.

### 4.2 Preprocessing  (`src/mibci/preprocess/`)
Order is deliberate: **notch 60 Hz → band-pass 8–30 Hz (FIR) → bad-channel
interpolation → Common Average Reference**. Notch first so line noise doesn't
leak into the CAR; CAR last so the reference is computed from clean channels.

**Bad-channel interpolation** earns its place empirically. The peak-to-peak reject
operates across all 64 channels, so a *single* dead or noisy electrode (in this
dataset, peripheral eye/muscle sites like FT8, T10, AF7) vetoes the entire epoch —
and the same bad channel poisons the Common Average Reference. We flag channels
whose log-variance is a robust-z outlier (|z| > 4) and interpolate them from
their neighbours using the montage, *before* the CAR. Concretely, on the smoke
cohort this recovered a subject that had otherwise lost 100% of its epochs (one
399 µV temporal channel), and lifted overall epoch retention from ~45% to ~90%.
The reject threshold is then 200 µV p2p — still ~10× the physiological amplitude
in the 8–30 Hz band, so it only removes genuine transients, not real signal.

> **Physiology:** 8–30 Hz is exactly the mu+beta band where motor ERD lives. CAR
> approximates a reference-free view of the cortex, sharpening local activity.
>
> **ML:** band-limiting is the single biggest "feature engineering" lever here;
> it's a config parameter so you can ablate it.

**ICA artifact removal** is the subtle one: ICA is **fit on a 1 Hz high-passed
copy** (decomposition is unstable on the narrow 8–30 Hz band) and then **applied**
to the analysis data. Eye components are flagged via frontal channels Fp1/Fp2
(this dataset has no dedicated EOG electrode); muscle components via MNE's
`find_bads_muscle`. **The number of components removed is logged per subject** —
automatic cleaning you can't audit isn't trustworthy.

**Epoching** is cue-locked: a window of `[-0.5, 4.0] s` (config), baseline-corrected
on the pre-cue slice, with peak-to-peak amplitude rejection (default 150 µV) to
drop residual artifacts.

### 4.3 Models  (`src/mibci/models/`, `src/mibci/features/`)
All five share **identical CV splits**, so accuracy differences are about the
model, not the data partition (and the paired stats in §4.4 are valid).

1. **CSP + LDA** — the classical reference. CSP learns spatial filters maximising
   the between-class variance ratio; LDA draws a linear boundary on the
   log-variance features. Fast, interpretable, hard to beat on within-subject.
2. **CSP + SVM (RBF)** — same features, non-linear boundary.
3. **Riemannian + LR** — represent each epoch by its channel covariance matrix,
   project from the curved SPD manifold to the **tangent space**, classify with
   logistic regression. Strong and unusually robust across subjects.
4. **EEGNet** (Lawhern et al., 2018) — a compact CNN whose blocks mirror EEG
   intuition: temporal conv = learned band-pass, depthwise conv = learned spatial
   filter (CSP-like), separable conv = temporal dynamics.
5. **Denoising-EEGNet (custom)** — see §5.

### 4.4 Evaluation  (`src/mibci/evaluation/`)
Two regimes, both reported:
- **Within-subject** — per-subject *k*-fold. "If I calibrate on you, how well do I
  decode you?" The optimistic number.
- **Leave-One-Subject-Out (LOSO)** — train on everyone else, test on the held-out
  person. "Can I decode a stranger with no calibration?" The honest number, and
  usually **much lower** because of inter-subject variability.

Per model we report **mean accuracy ± std, Cohen's kappa, and confusion matrices**.
Models are compared with the **pairwise Wilcoxon signed-rank test** (paired,
non-parametric) across subjects, with **Holm-Bonferroni correction** for the
multiple comparisons. A **per-subject accuracy distribution** plot surfaces
*"BCI illiteracy"* — the well-documented fact that a minority of people decode
near chance no matter the model.

---

## 5. The custom model: Denoising-EEGNet

Three sentences, by design:

1. Single-trial EEG is dominated by non-task noise, and that noise differs
   between people — which is exactly what wrecks subject-independent accuracy.
2. So I prepend a small **convolutional denoising autoencoder**, pretrained to
   reconstruct each clean epoch from a noise-corrupted copy, learning to suppress
   noise before classification.
3. Its cleaned output feeds a standard EEGNet backbone and the whole thing is
   fine-tuned end-to-end — it's a *learned* denoiser, not a fixed filter.

**It has to earn its place.** The pipeline runs an **ablation** — `eegnet` vs
`denoising_eegnet` on the *same* splits — so the front-end is justified by
evidence (a significant Wilcoxon win), not decoration. If the ablation shows no
gain, that's a finding too, and the README should say so.

---

## 6. Interpretability & the neurophysiology check

- **CSP pattern topomaps** (`artifacts/csp_patterns_*.png`): the discriminative
  components should localise over **C3/C4**. If they do, the model is using real
  sensorimotor physiology; if they're frontal or occipital, suspect artifacts.
- **EEGNet filters**: temporal filters' frequency responses should peak in mu/beta;
  spatial filters should weight central electrodes.
- A short written interpretation accompanies the figures tying the patterns back
  to contralateral ERD.

---

## 7. Reproducibility & engineering

- **Pinned** `requirements.txt`; **Python 3.12** venv (PyTorch/MNE don't ship 3.14
  wheels yet).
- **Deterministic seeds** across NumPy / PyTorch / Python (CPU is bit-reproducible;
  GPU reproducibility additionally needs deterministic cuDNN kernels).
- **Cached** preprocessed epochs keyed on a hash of the preprocessing config.
- `src/` package, `configs/`, `notebooks/` (EDA only), `tests/` (real unit tests:
  filter shape, epoch count, CSP dims, config validation), `Makefile` + CLI.

---

## 8. Limitations (read this in the interview)

- **Offline only.** No real-time/online decoding; this is batch, epoched
  classification.
- **Automatic ICA is heuristic.** No human inspects components in the batch path;
  we log counts for auditability, but auto-detection can over- or under-clean.
- **Compute.** The full 109-subject LOSO × 5-model benchmark is GPU-class; the
  shipped `smoke` config proves correctness on a handful of subjects in minutes,
  and the heavy runs are meant for a GPU machine.
- **LOSO is genuinely hard.** Expect numbers far below the within-subject case;
  that's not a bug, it's the actual difficulty of zero-calibration BCI.

---

## 9. Layout

```
src/mibci/
  config.py          # typed, validated config (single source of truth)
  data/              # download, known-bad exclusions, loader + integrity checks
  preprocess/        # filters (notch/bandpass/CAR), ICA, epoching
  features/          # CSP, Riemannian covariance→tangent-space
  models/            # CSP+LDA/SVM, Riemann+LR, EEGNet, Denoising-EEGNet
  evaluation/        # within-subject + LOSO CV, metrics, Wilcoxon stats
  interpret/         # CSP topomaps, EEGNet filter visualisation
  viz/               # per-subject distribution, confusion matrices
  run.py             # CLI entry point
configs/             # binary.yaml, fourclass.yaml, smoke.yaml
tests/               # offline unit tests
```

## References
- Schalk et al. (2004), *BCI2000*. PhysioNet EEGMMIDB.
- Lawhern et al. (2018), *EEGNet*.
- Barachant et al., *Riemannian geometry for EEG* (pyRiemann).
