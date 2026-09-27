# AI Job Market & Salary Prediction

A supervised machine learning project that predicts salaries (`salary_usd`) for AI and Data Science job postings based on experience, location, company attributes, and job characteristics.

## Business Problem

Companies hiring for AI/Data Science roles, and job seekers evaluating offers, often lack a clear reference for what a fair, competitive salary looks like given a role's experience level, location, and company size. This project builds a data-driven salary estimator to close that gap.

## Dataset

The dataset contains 15,000 AI/Data Science job postings with 19 features, including:

- `job_title`, `experience_level`, `employment_type`
- `company_location`, `company_size`, `employee_residence`, `remote_ratio`
- `required_skills`, `education_required`, `years_experience`
- `industry`, `posting_date`, `application_deadline`
- `job_description_length`, `benefits_score`
- `salary_usd` (target), `salary_currency`

## Project Structure

```
├── AI_Job_Salary_Prediction.ipynb   # Full notebook (EDA + modeling), pre-executed
├── ai_job_dataset.csv               # Dataset used
├── requirements.txt                 # Python dependencies
└── README.md
```

## Approach

**Phase 1 — Exploratory Data Analysis**
- Distribution analysis of salaries (right-skewed, most postings $50K–$150K)
- Correlation analysis: `years_experience` is the strongest numeric driver (r = 0.74)
- Salary breakdowns by experience level, education, company size, industry, and remote work ratio

**Phase 2 — Feature Engineering & Modeling**
- Engineered features: `days_to_deadline` (posting-to-deadline window), `skills_count` (from parsed skill list), `same_location` (employee vs. company country match)
- Preprocessing pipeline: `StandardScaler` for numeric features, `OneHotEncoder` for categorical features
- Trained and compared 5 regression models:
  - Linear Regression
  - Ridge Regression
  - Random Forest
  - Gradient Boosting
  - XGBoost
- Evaluated with RMSE, MAE, R², and 5-fold cross-validation
- Interpreted the best model using feature importance

## Results

| Model | RMSE | MAE | R² |
|---|---|---|---|
| **Gradient Boosting** | **20,851** | **14,973** | **0.881** |
| Random Forest | 21,152 | 15,166 | 0.877 |
| XGBoost | 22,468 | 15,873 | 0.862 |
| Linear Regression | 23,197 | 17,054 | 0.852 |
| Ridge Regression | 23,198 | 17,046 | 0.852 |

**Best model: Gradient Boosting Regressor**, explaining ~88% of salary variance (5-fold CV R² = 0.878 ± 0.003).

**Top salary drivers:** `years_experience` and `experience_level` (especially Executive level) dominate, consistent with the Phase 1 EDA. `company_location` is a secondary factor reflecting regional pay differences.

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook AI_Job_Salary_Prediction.ipynb
```

## Future Work

- Hyperparameter tuning (GridSearchCV / Optuna)
- Log-transforming `salary_usd` to further reduce right-skew
- Deploying the trained pipeline behind a simple API or Streamlit app

## Author

Saja Akool
