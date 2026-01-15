# Telco Churn Model Benchmark

Benchmark multiple model families on tabular customer churn data and document practical tradeoffs for deployment.

## What this shows
- **Model comparison on tabular data** (linear models vs tree ensembles vs neural net)
- **Metric-driven selection** using AUC
- **Training diagnostics** (gradient statistics for the Keras MLP)
- **Deployment thinking** via a serverless inference architecture (AWS S3 + Lambda + API Gateway)

## Results (validation AUC)
| Model | AUC |
|---|---:|
| GradientBoostingClassifier | 0.8434 |
| LogisticRegression | 0.8419 |
| SGDClassifier | 0.8403 |
| keras_mlp | 0.8341 |
| RandomForestClassifier | 0.8162 |

Source: `docs/artifacts/model_leaderboard_telco.json`

## Repo structure
- `src/` — code
- `data/` — small sample data (no large raw datasets)
- `docs/` — reports (Markdown + PDF) and figures

## Reports
- Main writeup: `docs/REPORT.md` (print: `docs/REPORT.pdf`)
- AWS serverless writeup: `docs/REPORT_AWS.md` (print: `docs/REPORT_AWS.pdf`)

## Artifacts
- `docs/artifacts/model_leaderboard_telco.json`
- `docs/artifacts/keras_gradients.json`

