# Vocal Biomarkers for Parkinson's Detection

## Overview

A machine learning system for detecting indicators of Parkinson's disease from vocal biomarkers. It uses the UCI Parkinson's dataset (Little et al., 2007), comprising 195 recordings from 31 subjects, and extracts 22 acoustic measures from sustained phonations. It implements three classifiers (SVM, Random Forest, XGBoost) with a **group-aware split** to prevent data leakage across subjects.

**WARNING: This tool is intended solely for research and educational purposes. It is NOT a medical diagnostic instrument.**

## Scientific Background

Dysarthria and dysphonia are symptoms present in roughly 90% of patients with Parkinson's disease (Logemann et al., 1978). Vocal alterations include:

- **Jitter**: involuntary variations in the fundamental frequency (F0), indicating instability in vocal fold control
- **Shimmer**: variations in signal amplitude, reflecting irregularities in glottal vibration
- **HNR/NHR**: harmonics-to-noise ratio, measuring the presence of noise in the voice
- **RPDE, DFA, D2**: nonlinear measures that capture the dynamic complexity of the vocal signal
- **PPE, spread1, spread2**: pitch variation measures that detect vocal monotonicity (characteristic of PD)

These alterations can be detected before the classic motor symptoms (tremor, rigidity) appear, making vocal analysis a potential early biomarker.

## Dataset

**Source**: [UCI Machine Learning Repository - Parkinsons Dataset](https://archive.ics.uci.edu/ml/datasets/parkinsons)

| Property | Value |
|---|---|
| Samples | 195 sustained phonation recordings (/a/) |
| Subjects | 31 (23 with Parkinson's, 8 healthy) |
| Features | 22 vocal biomarkers |
| Distribution | 147 PD (75.4%) / 48 healthy (24.6%) |

### Extracted Features

| Group | Features | Description |
|---|---|---|
| Frequency | Fo, Fhi, Flo | Fundamental frequency (mean, maximum, minimum) |
| Jitter | Jitter(%), Jitter(Abs), RAP, PPQ, DDP | Cycle-to-cycle variations in frequency |
| Shimmer | Shimmer, Shimmer(dB), APQ3, APQ5, APQ, DDA | Cycle-to-cycle variations in amplitude |
| Noise | NHR, HNR | Noise-to-harmonics and harmonics-to-noise ratios |
| Nonlinear | RPDE, DFA, D2, spread1, spread2, PPE | Dynamic complexity and pitch variation |

## Methodology

### Group-Aware Split (CRITICAL)

The dataset contains multiple recordings per subject (~6 each). A conventional random split (`train_test_split`) would cause **data leakage**: recordings from the same subject would appear in both the training and test sets. The model would learn to recognize the individual's vocal identity rather than the Parkinson's biomarkers.

**Solution**: scikit-learn's `GroupShuffleSplit`, which ensures that **all** of a given subject's recordings stay exclusively in either the training OR the test set. The reported results represent the model's true ability to generalize to previously unseen subjects.

### Classifiers

1. **SVM (RBF)**: GridSearchCV over C=[0.1, 1, 10, 100], gamma=['scale', 'auto'], class_weight='balanced'
2. **Random Forest**: n_estimators=[100, 200, 500], max_depth=[5, 10, None], class_weight='balanced'
3. **XGBoost**: n_estimators=[100, 200], max_depth=[3, 5, 7], learning_rate=[0.01, 0.1], scale_pos_weight

All are tuned with StratifiedKFold(10) and scoring='roc_auc'. StandardScaler is fitted on the training set only.

## Results (Group-Aware Split)

| Model | Accuracy | Precision | Recall | F1-Score | AUC (test) | AUC (CV) |
|---|---|---|---|---|---|---|
| **SVM** | 0.9070 | 0.9444 | 0.9444 | 0.9444 | **0.9325** | 0.9955 |
| Random Forest | 0.9302 | 0.9459 | 0.9722 | 0.9589 | 0.9246 | 0.9705 |
| XGBoost | 0.9302 | 0.9459 | 0.9722 | 0.9589 | 0.9286 | 0.9636 |

**Note**: Results obtained with a group-aware split are typically lower than those reported with a random split in the literature (~95-97% accuracy), precisely because there is no information leakage between subjects. These results are more honest and realistic.

## Feature Analysis

The most discriminative features (by importance across the models):
- **spread1**: a measure of fundamental frequency variation — consistently the most important feature
- **PPE**: pitch period entropy — captures vocal monotonicity
- **MDVP:Fo(Hz)**: mean fundamental frequency
- **HNR**: harmonics-to-noise ratio
- **DFA**: detrended fluctuation analysis
- **RPDE**: recurrence entropy

## Installation and Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Train models (the dataset is downloaded automatically)
python train.py

# Start the web application
python app.py
# Open http://127.0.0.1:5000
```

### Audio Feature Extraction

The application accepts WAV files of a sustained vowel (/a/) and automatically extracts the 22 biomarkers using Parselmouth/Praat. For the nonlinear measures (RPDE, DFA, D2), it uses the nolds library when available.

## Project Structure

```
vocal-biomarkers/
├── app.py                  # Flask server
├── train.py                # Training pipeline
├── config.py               # Central configuration
├── ml/
│   ├── dataset.py          # Loading and group-aware split
│   ├── train_models.py     # SVM, RF, XGBoost training
│   ├── evaluate.py         # Evaluation and plot generation
│   └── predict.py          # Prediction with the saved model
├── audio/
│   └── extractor.py        # Feature extraction via Parselmouth
├── templates/              # HTML (Jinja2)
├── static/                 # CSS, JS, generated plots
├── data/raw/               # UCI dataset
└── models/                 # Trained models (.joblib)
```

## References

1. **Little, M.A., McSharry, P.E., Roberts, S.J., Costello, D.A.E., Moroz, I.M.** (2007). Exploiting Nonlinear Recurrence and Fractal Scaling Properties for Voice Disorder Detection. *BioMedical Engineering OnLine*, 6:23.

2. **Little, M.A., McSharry, P.E., Hunter, E.J., Spielman, J., Ramig, L.O.** (2009). Suitability of Dysphonia Measurements for Telemonitoring of Parkinson's Disease. *IEEE Transactions on Biomedical Engineering*, 56(4):1015-1022.

3. **Tsanas, A., Little, M.A., McSharry, P.E., Ramig, L.O.** (2010). Accurate Telemonitoring of Parkinson's Disease Progression by Noninvasive Speech Tests. *IEEE Transactions on Biomedical Engineering*, 57(4):884-893.

## License

MIT License. See the [LICENSE](LICENSE) file.