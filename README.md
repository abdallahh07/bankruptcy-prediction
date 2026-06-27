# Company Bankruptcy Prediction

Binary classification model predicting corporate bankruptcy using 94 financial ratios from 6,819 Taiwan Stock Exchange companies (1999–2009).

## Results

| Model               | CV AUC | Test AUC |
| ------------------- | ------ | -------- |
| LightGBM (Tuned)    | 0.9355 | 0.9512   |
| XGBoost             | 0.9238 | 0.9452   |
| Logistic Regression | 0.8739 | 0.9171   |

**LightGBM selected** as final model. Hyperparameter tuning reduced overfitting (train score 1.0 → 0.98) while improving test AUC.

## Dataset

- Source: [Company Bankruptcy Prediction](https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction)
- 6,819 companies, 96 financial ratios, Taiwan Economic Journal 1999–2009
- 3.2% positive class (bankrupt) — severe class imbalance handled with `class_weight='balanced'`

## Financial Analysis

Unlike typical ML notebooks, this project includes CFA-level financial analysis comparing bankrupt vs non-bankrupt companies across:

- Income statement ratios (ROA, Operating Margin, EPS)
- Balance sheet ratios (Current Ratio, Debt Ratio, Net Worth/Assets)
- Cash flow ratios (Cash/Current Liability, CFO to Assets)

## Key Findings

- Balance sheet ratios show clearest separation between bankrupt and healthy companies
- Top predictors: Borrowing Dependency, Interest Expense Ratio, Net Income to Total Assets
- Income statement ratios alone are insufficient — consistent with Altman Z-Score theory

## Project Structure

bankruptcy_model/

├── config/config.yml # model parameters and paths

├── processing/

│ ├── data_manager.py # load data, save/load model

│ └── features.py # feature selection and target extraction

├── pipeline.py # sklearn pipeline definition

train_pipeline.py # run training end-to-end

predict.py # load model and predict

research.ipynb # EDA, financial analysis, modelling

charts/ # saved visualizations

## Usage

```bash
python train_pipeline.py
```

## Key Decisions

- **AUC over accuracy** — 96.8% class imbalance makes accuracy misleading
- **No scaling** — LightGBM handles raw features natively
- **Net Income Flag dropped** — constant value across all rows, zero variance
- **Stratified split** — preserves 3.2% bankruptcy rate in train and test sets
