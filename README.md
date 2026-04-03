# Hospital Readmission Analysis

Exploratory data analysis of diabetic patient readmissions using the UCI Diabetes 130-US Hospitals dataset (1999–2008). The goal was to figure out what patient and clinical factors are linked to being readmitted within 30 days of discharge.

---

## What is Data Analysis (DA)?

Data analysis is the process of inspecting, cleaning, and making sense of raw data to draw useful conclusions. It's not about building ML models or writing production code — it's about understanding the data: what's in it, what's missing, what patterns exist, and what story it tells.

A typical DA workflow looks like this:

1. **Load the data** — get it into a workable format
2. **Assess data quality** — find missing values, bad encodings, duplicates
3. **Clean** — handle the issues you found (drop, fill, fix)
4. **Explore (EDA)** — visualize distributions, relationships, and breakdowns
5. **Summarize findings** — turn observations into clear takeaways

This project follows that exact flow from top to bottom.

---

## Why this dataset?

Hospital readmissions are a real problem. When a patient is discharged and comes back within 30 days, it usually means something went wrong — inadequate treatment, early discharge, or a condition that's just difficult to manage. US hospitals are actually penalized financially under the Hospital Readmissions Reduction Program (HRRP) for excessive readmission rates, especially for conditions like diabetes.

The UCI dataset has 100,000+ real encounters across 130 US hospitals. It's messy (like real-world data always is), clinically rich, and well-suited for practicing the full DA cycle — from ugly raw data to interpretable insights.

---

## Dataset

**Source:** [UCI Machine Learning Repository — Diabetes 130-US Hospitals](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)

- 101,766 patient encounters
- 50 columns covering demographics, admission details, clinical measurements, medication info, and the readmission outcome
- Time period: 1999–2008
- Missing values encoded as `?` (not NaN — that's one of the first things to fix)
- Target variable: `readmitted` — values are `<30` (readmitted within 30 days), `>30`, or `NO`

---

## Stack

| Tool | Why |
|---|---|
| Python | General-purpose, dominant in data work |
| pandas | DataFrame manipulation — filtering, grouping, aggregating |
| numpy | Numeric ops, array handling |
| matplotlib | Fine-grained chart control |
| seaborn | Cleaner statistical plots on top of matplotlib |
| scipy.stats | Point-biserial correlation, linear regression |
| ucimlrepo | Fetches the dataset directly without manual download |
| Jupyter Notebook | Ideal for EDA — code, output, and notes in one place |

**v2 adds a prediction notebook** (`readmission_prediction.ipynb`) using logistic regression to predict 30-day readmissions — see the ML section below.

---

## Project structure

```
healthcare-readmission-analysis/
├── hospital_readmission_analysis.ipynb   # EDA notebook (v1)
├── readmission_prediction.ipynb          # ML prediction notebook (v2)
├── analysis_report.md                    # written summary of findings
├── data_dictionary.txt                   # what each column means
├── visualizations/                       # all charts as PNG files
│   ├── missing_values.png
│   ├── readmission_by_age.png
│   ├── readmission_by_medications.png
│   ├── readmission_by_los.png
│   ├── readmission_by_diagnoses.png
│   ├── age_distribution.png
│   ├── los_distribution.png
│   ├── medications_boxplot.png
│   ├── procedures_vs_medications.png
│   ├── correlation_heatmap.png
│   ├── top_diagnoses.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── feature_coefficients.png
└── data/                                 # gitignored — too large to commit
    └── diabetic_data.csv
```

---

## How to run

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn ucimlrepo notebook
```

- EDA: `jupyter notebook hospital_readmission_analysis.ipynb`
- ML: `jupyter notebook readmission_prediction.ipynb`

The EDA notebook fetches the dataset automatically via `ucimlrepo`. Subsequent runs use the locally saved CSV.

---

## What the notebook does (section by section)

### 1. Data Loading & Initial Exploration
Loads the CSV, replaces `?` with `NaN`, and does a quick shape/dtype/sample check. The dataset has 101,766 rows and 50 columns, using 189MB in memory.

### 2. Data Quality Assessment
Checks every column for missing values before touching anything. Key issues found:

| Column | % Missing |
|---|---|
| weight | 96.9% |
| max_glu_serum | 94.8% |
| A1Cresult | 83.3% |
| medical_specialty | 49.1% |
| payer_code | 39.6% |
| race | 2.2% |

The first four are effectively unusable. A bar chart with a 40% cutoff line makes this decision visual and defensible.

### 3. Data Cleaning
- Drop columns with >40% missing (`weight`, `max_glu_serum`, `A1Cresult`, `medical_specialty`)
- Drop ID columns (`encounter_id`, `patient_nbr`) — not analytically meaningful
- Map age brackets like `[50-60)` to numeric midpoints (55) for correlation math
- Fill the 2% missing in `race` with the mode
- Create a binary target column `readmitted_30`: 1 if `readmitted == '<30'`, else 0
- Overall 30-day readmission rate after cleaning: **11.2%**

### 4. Exploratory Data Analysis
The main section. Broken into four sub-parts:

**4a. Readmission Rate Analysis**
- Readmission rate by age group — middle-aged to elderly dominate, youngest group [20-30) has a surprising 14.2% rate
- Readmission rate by medication count — clear positive trend, patients on 20+ meds readmit at higher rates
- Readmission rate by length of stay — slight upward trend as days increase

**4b. Clinical Patterns**
- Length of stay distribution — peaks at 3 days, right-skewed
- Age distribution — bulk of encounters are in the 50–80 range
- Procedures vs medications scatter (5k sample) — weak positive correlation
- Readmission rate by number of diagnoses — more diagnoses = slightly higher readmission

**4c. Medication Analysis**
- Patients with diabetes medication prescribed: 11.6% readmission vs 9.6% without
- Medication change during visit: 11.8% readmission vs 10.6% no change
- Box plot comparing medication counts for readmitted vs not-readmitted groups

**4d. Correlations**
- Full correlation heatmap of all numeric features
- Top 5 features by absolute correlation with `readmitted_30`:

| Feature | r |
|---|---|
| number_inpatient | 0.165 |
| number_emergency | 0.061 |
| number_diagnoses | 0.050 |
| time_in_hospital | 0.044 |
| num_medications | 0.038 |

---

## Key findings

- **11.2% overall 30-day readmission rate** across 101,766 encounters
- **Prior inpatient visits are the strongest predictor** (r=0.165, p≈0). Patients who've been hospitalized before are significantly more likely to return within 30 days. This makes clinical sense — they have complex, recurring conditions.
- **More medications → higher readmission risk.** Not because medications cause readmissions, but because patients on many medications tend to have more severe disease.
- **Middle-aged to elderly patients (50–80)** make up the bulk of encounters. The [60-80) range consistently sits above the overall average readmission rate.
- **Longer stays weakly correlate with readmission** — again, pointing to patient severity rather than a causal relationship.
- **More diagnoses = slightly higher readmission** — same interpretation: complex patients get readmitted more.

---

---

## v2 — Prediction Model

`readmission_prediction.ipynb` extends the EDA by building a logistic regression classifier to predict 30-day readmissions.

**What's new:**
- Feature engineering: selects key numeric + categorical features, label-encodes binary columns, one-hot encodes low-cardinality categoricals
- 80/20 stratified train/test split (preserves the 11% positive rate in both sets)
- `class_weight='balanced'` to handle the class imbalance — without this the model just predicts "not readmitted" for everything and gets 89% accuracy while being useless
- Evaluation: classification report (precision, recall, F1), confusion matrix, ROC curve (AUC ~0.66–0.68)
- Feature coefficients plot — shows which features push toward or away from readmission

**Key takeaways from the model:**
- `number_inpatient` is the strongest driver, consistent with EDA findings
- `discharge_disposition_id` has a large effect — where patients go after discharge matters a lot
- AUC of ~0.67 is modest but realistic for administrative healthcare data. Clinical readmission is hard to predict without post-discharge information.
- A tree-based model would likely improve this — logistic regression is the right starting point but hits a ceiling on non-linear relationships

---

## Limitations

- Data is from **1999–2008**. Treatment guidelines, medications, and discharge practices have changed significantly.
- The dataset only captures readmissions to the **same hospital network**. If a patient went to a different hospital, it's recorded as no readmission — so 11.2% is likely an undercount.
- **Correlation ≠ causation.** Longer stays and more medications reflect sicker patients, not causes of readmission.
- **Class imbalance** (~11% positive class). Any predictive model built on this data would need to address that.
- No socioeconomic data, no post-discharge follow-up, no information on patient adherence to treatment.
