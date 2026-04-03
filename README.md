# Hospital Readmission Analysis

EDA and prediction on diabetic patient readmissions using the UCI Diabetes 130-US Hospitals dataset (1999–2008). Explores what clinical and demographic factors are linked to 30-day readmissions, then builds a logistic regression model to predict them.

## Dataset

**Source:** UCI Diabetes 130-US Hospitals (1999–2008)

- 101,766 patient encounters across 130 US hospitals
- 50 columns: demographics, admission details, medications, diagnoses, readmission outcome
- Target: `readmitted` — `<30` (within 30 days), `>30`, or `NO`
- Dataset is gitignored (19MB) — download from [UCI ML Repository](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)

## Notebooks

| Notebook | What it covers |
|---|---|
| `hospital_readmission_analysis.ipynb` | Data cleaning, EDA, visualizations, key findings |
| `readmission_prediction.ipynb` | Feature engineering, logistic regression, confusion matrix, ROC curve, feature importance |

## Key Findings

- **11.2% overall 30-day readmission rate** across 101k+ encounters
- **Prior inpatient visits** are the strongest predictor of readmission (r=0.165)
- **More medications → higher readmission risk** — reflects disease complexity, not causation
- **Ages 60–80** consistently sit above the average readmission rate
- **Logistic regression AUC ~0.67** — modest but expected for administrative data without post-discharge info

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn ucimlrepo notebook
jupyter notebook hospital_readmission_analysis.ipynb
jupyter notebook readmission_prediction.ipynb
```

## Stack

Python · pandas · scikit-learn · matplotlib · seaborn · Jupyter
