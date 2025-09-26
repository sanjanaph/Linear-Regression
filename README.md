## Multiple Linear Regression — Housing Case Study

### Overview
This project builds and explains a multiple linear regression (MLR) model to predict housing prices using a structured end‑to‑end workflow: data understanding, exploratory data analysis (EDA), feature engineering, model training, diagnostics, interpretation, and validation. The work is implemented in the notebook `Multiple Linear Regression - Housing Case Study.ipynb`.

### Problem Statement
- **Goal**: Predict house prices using multiple explanatory variables (e.g., area, bedrooms, bathrooms, location indicators, quality scores).
- **Type**: Supervised regression.
- **Outputs**: A parsimonious, interpretable linear model with validated performance metrics and diagnostic checks.

### Data
- **Dataset**: Housing dataset containing numeric and categorical predictors describing properties and their sale price.
- **Target variable**: `price` (continuous).
- **Typical features** (examples):
  - Numeric: `area_sqft`, `lot_size`, `bedrooms`, `bathrooms`, `year_built`, `age_years`, `garage_cars`, quality/condition indices
  - Categorical: neighborhoods/locations, building types, exterior materials (encoded via one-hot/dummies)
- **Expected issues**: Missing values, outliers, skewed distributions, multicollinearity among size/quality variables.

### Modeling Approach
1. Data audit and cleaning
   - Identify missingness and impute or drop (documented per feature).
   - Coerce data types; standardize categorical levels.
2. Exploratory Data Analysis (EDA)
   - Univariate distributions; log transforms where skewed.
   - Bivariate relationships with `price` (scatter plots, box plots, correlations).
   - Correlation heatmap to surface multicollinearity.
3. Feature engineering
   - One‑hot encoding for categoricals (drop one level to avoid the dummy variable trap).
   - Optional transformations: log(price), log(area), polynomial terms only if justified by diagnostics.
   - Interaction terms if domain suggests combined effects (kept only if significant and stable).
4. Train/validation split
   - Stratified or random split (e.g., 70/30 or 80/20) with fixed random seed.
   - Optionally K‑fold cross‑validation for robust generalization estimates.
5. Model training
   - Baseline OLS (ordinary least squares) multiple linear regression.
   - Compare with regularized variants (Ridge/Lasso/ElasticNet) if needed to stabilize coefficients under multicollinearity.
6. Model selection
   - Iterative feature selection guided by p‑values, adjusted R², AIC/BIC, and domain plausibility.
   - Remove redundant or unstable predictors (high VIF, low incremental value, or non‑robust significance).
7. Evaluation
   - Metrics: R², Adjusted R², RMSE, MAE on train and validation/test.
   - Residual diagnostics and assumption checks (see below).

### Multiple Linear Regression Assumptions and Checks
1. **Linearity**: Relationship between predictors and target is linear in parameters.
   - Check: Residuals vs fitted should be patternless; partial residual plots.
2. **Independence**: Observations and residuals are independent.
   - Check: Study design/time; Durbin–Watson for autocorrelation (if temporal order exists).
3. **Homoscedasticity**: Constant variance of residuals across fitted values.
   - Check: Residuals vs fitted scatter; Breusch‑Pagan/White tests.
4. **Normality of residuals**: Residuals are approximately normal (for inference).
   - Check: Q–Q plot; Shapiro–Wilk on large out‑of‑sample residuals interpreted cautiously.
5. **No multicollinearity**: Predictors are not excessively correlated.
   - Check: VIF (Variance Inflation Factor); correlation matrix; condition number.

If violations occur:
- Apply transformations (e.g., log of `price` and/or skewed predictors).
- Remove or combine collinear variables; prefer more interpretable/broader constructs.
- Consider regularization (Ridge/Lasso) to stabilize coefficients.
- Re‑specify model to better capture nonlinearity (interactions, splines) while preserving interpretability.

### Diagnostics Performed
- **Residual analysis**: Residuals vs fitted, scale‑location, leverage plots.
- **Influential points**: Cook’s distance, leverage (hat values), DFFITS/DFBETAS; remove only if data errors or non‑representative anomalies.
- **Multicollinearity**: VIF thresholding (commonly VIF > 5 or 10 prompts review), drop/merge features.
- **Cross‑validation**: K‑fold CV to validate stability of RMSE/MAE and detect overfitting.

### Metrics and Interpretation
- **R²**: Proportion of variance explained by the model.
- **Adjusted R²**: Penalizes for number of predictors; prefer over R² for model comparison.
- **RMSE**: Typical magnitude of prediction error in price units.
- **MAE**: Average absolute error; robust to outliers compared to RMSE.

When using a log‑price model, back‑transform predictions and interpret coefficients multiplicatively. For a predictor \(x_j\) with coefficient \(\beta_j\):
- If target is log(price), a 1‑unit increase in \(x_j\) changes price by approximately \((e^{\beta_j} - 1) \times 100\%\), holding others constant.

### Final Model (Illustrative Structure)
- Response: `price` or `log_price`.
- Predictors: a curated subset of size, quality, age, and key location dummies, chosen for significance, stability, low VIF, and interpretability.
- Coefficients are interpreted as ceteris paribus effects; categorical coefficients are relative to the dropped baseline level.

### Reproducibility
- Fixed random seed for splits.
- All data prep and modeling steps contained in the notebook with ordered cells.
- Figures generated programmatically to ensure repeatability.

### Repository Structure
- `Multiple Linear Regression - Housing Case Study.ipynb` — complete analysis and modeling workflow
- `README.md` — this document
- Optional (if present): `data/` for raw/processed datasets, `figures/` for exported plots

### How to Run
1. Open the notebook in Jupyter or VS Code.
2. Run cells from top to bottom. Ensure the dataset path(s) inside the notebook point to your local `data/` location.
3. Review the diagnostics and metrics outputs at the end to validate assumptions and performance.

### Requirements
Install the following Python packages (versions indicative):

```bash
pip install numpy pandas matplotlib seaborn scikit-learn statsmodels scipy
```

If you plan to export figures:

```bash
pip install plotly
```

### Key Takeaways
- Start broad with EDA, then narrow to a stable, interpretable subset of predictors.
- Validate all linear model assumptions; do not rely solely on R².
- Use VIF and domain knowledge to manage multicollinearity.
- Prefer simpler models with comparable adjusted R² and lower generalization error.

### References
- Freedman, D. (2009). Statistical Models: Theory and Practice.
- Kutner, Nachtsheim, Neter, Li (2004). Applied Linear Statistical Models.
- James, Witten, Hastie, Tibshirani (2013). An Introduction to Statistical Learning.
- Statsmodels and scikit‑learn documentation.

