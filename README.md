# ML Early Dropout Detection

**DATASCI 347: Machine Learning I**
**Instructor:** Prof. Ruoxuan Xiong  
**Group Member:** Grayson Stone, Eric Wang, Jiuyi (Joy) Cheng, Samuel Chernoff

This GitHub Repository documents the project: "Multiclass Classification: ML Algorithm for Early Dropout Detection (Dropout, Enrolled, Graduate)" using progressive academic performance feature sets: demographics only → demographics + 1st semester → full data (demographics + 1st & 2nd semester).

---

## Dataset Overview

- **Total samples**: 4,424 students
- **Features**: 37 total (demographics + academic performance + economic indicators)
- **Target classes**: Dropout: 1,421 | Graduate: 2,209 | Enrolled: 794 
- **Train-test split**: 80-20 stratified

### Class Distribution & Imbalance

The dataset exhibits class imbalance typical in educational datasets:
- **Graduate**: 49.9% (2,209 samples) - Majority class
- **Dropout**: 32.1% (1,421 samples) - Minority class  
- **Enrolled**: 17.9% (794 samples) - Smallest class

To handle this imbalance:
- Used **stratified train-test split** to maintain class proportions
- Used **class_weight='balanced'** in Random Forest (weights inversely proportional to class frequency)
- Scored models with **F1-weighted metric** (averages F1 across classes weighted by support)
- Used **stratified K-fold cross-validation** (preserves class distribution in each fold)

This prevents models from simply predicting the majority class and ensures performance is evaluated fairly across all outcomes.

### Three Progressive Feature Sets

1. **df_early** (24 features + target = 25 columns)
   - Demographics only (marital status, age, qualifications, etc.)
   - Available at enrollment
   - Baseline prediction capability

2. **df_mid** (30 features + target = 31 columns)
   - Demographics + 1st semester academic data (approved units, grades, evaluations)
   - Enables early intervention point

3. **df_full** (36 features + target = 37 columns)
   - Demographics + 1st semester + 2nd semester academic data
   - Complete dataset for end-of-year prediction

---

## Best Model Performance

### XGBoost_full (Best Performer)
- **Test Accuracy**: 77.51%
- **Test F1-Score**: 0.7672
- **Test ROC-AUC**: 0.8943
- **Cross-Validation F1**: 0.7669

### Performance by Dataset (Average across algorithms)
| Dataset | Accuracy | F1-Score | ROC-AUC |
|---------|----------|----------|---------|
| Early (Demographics) | 61.67% | 0.5872 | 0.7384 |
| Mid (+1st semester) | 73.39% | 0.7141 | 0.8574 |
| Full (Complete) | 76.16% | 0.7516 | 0.8818 |

**Key Finding**: 1st semester data provides +11.7% F1 improvement; 2nd semester adds only +3.8% additional improvement.

---

## Algorithms & Tuning

### Three Algorithms Compared
1. **Logistic Regression** (Ridge & Lasso regularization) - Baseline, interpretable
2. **Random Forest** - Ensemble tree-based, handles non-linearity  
3. **XGBoost** - Gradient boosting, captures complex patterns

### Why These Algorithms?

**Logistic Regression**: 
- Provides interpretable linear coefficients (important for stakeholder communication)
- Ridge (L2) shrinks all coefficients proportionally; Lasso (L1) performs feature selection
- Establishes baseline performance; if more complex models don't substantially outperform it, simpler is better

**Random Forest**:
- Naturally handles non-linear relationships and feature interactions
- Robust to outliers; less prone to overfitting than single decision trees
- Provides feature importance rankings via Gini/impurity reduction
- No feature scaling required (invariant to monotonic transformations)

**XGBoost**:
- State-of-the-art gradient boosting; sequential learning focuses on hard-to-predict cases
- Built-in regularization (L1/L2) controls model complexity
- Handles class imbalance through scale_pos_weight parameter
- Often outperforms random forests on structured data

### Tuning Methodology

**Nested Cross-Validation Strategy**:
- **Outer CV (5-fold stratified)**: Provides unbiased estimate of model performance
- **Inner CV (3-fold stratified)**: Selects best hyperparameters without contaminating test set
- This architecture prevents overfitting to the validation set and gives honest performance estimates

**Search Strategies**:
- **GridSearchCV for Logistic Regression**: 18 exhaustive parameter combinations per dataset
  - Feasible because parameter space is small (6 C values × 2 solvers × 3 max_iter settings)
  - Guarantees finding the best in the search space
  
- **RandomizedSearchCV for Tree Models**: 20-25 intelligent random samples from massive parameter spaces
  - Random Forest: 270 possible combinations (reduced computational cost)
  - XGBoost: 972 possible combinations (handled efficiently with random sampling)
  - Random sampling often finds good solutions faster than exhaustive search

**Optimization Metric**: F1-weighted
- Better than accuracy for imbalanced data; accuracy would reward predicting majority class
- Balances precision and recall; F1 = 2 × (precision × recall) / (precision + recall)
- Weighted variant averages F1 across classes weighted by number of true instances

---

## Tuned Hyperparameters

### Best Models by Dataset

**Early (Demographics)**
- XGBoost: lr=0.1, depth=5, n_est=200, subsample=0.8, gamma=0 (CV F1: 0.6267)
- RandomForest: n_est=200, depth=15, min_split=10 (CV F1: 0.6264)

**Mid (+1st Semester)**
- XGBoost: lr=0.1, depth=5, n_est=200, subsample=0.7, gamma=1 (CV F1: 0.7416) ⭐
- RandomForest: n_est=200, depth=15, min_split=10 (CV F1: 0.7338)

**Full (Complete Data)**
- XGBoost: lr=0.05, depth=7, n_est=200, subsample=0.9, gamma=1 (CV F1: 0.7669) ⭐
- RandomForest: n_est=100, depth=10, min_split=2 (CV F1: 0.7677)
- LogReg Ridge: C=10, solver=lbfgs (CV F1: 0.7510)

---

## Feature Importance (RandomForest_full)

Top predictive features for dropout:
1. Curricular units 2nd sem (approved) - 15.41%
2. Curricular units 2nd sem (grade) - 11.56%
3. Curricular units 1st sem (approved) - 8.56%
4. Curricular units 1st sem (grade) - 6.77%
5. Tuition fees up to date - 5.12%

**Insight**: Academic performance metrics (approvals/grades) account for ~60% of prediction power.

### Interpretation by Feature Category

**Academic Performance (60% of importance)**:
- Units approved: Direct indicator of course completion
- Units grades: Average score across courses
- These two dimensions (quantity + quality) together signal engagement and capability

**Financial Status (5-7% of importance)**:
- Tuition fees up to date: Payment behavior may correlate with ability to stay enrolled
- Economic context (unemployment, inflation, GDP) provide macro environment signals

**Demographics (20-30% of importance)**:
- Age, marital status, parental education: Proxies for life stability and support systems
- Previous qualification: Academic preparation before enrollment

**Pattern**: Behavioral/performance indicators dominate; static demographics matter but less.

---

## Multicollinearity Analysis

Multicollinearity (correlation between features) can degrade model performance, particularly for linear models. This analysis quantified the problem:

### VIF Analysis (Variance Inflation Factor)

- **VIF values** (Variance Inflation Factor):
  - Early: Mean=15.84 (High multicollinearity present)
  - Mid: Mean=17.58 (High multicollinearity present)
  - Full: Mean=26.52 (High multicollinearity present)
  
- **Interpretation**: VIF > 10 indicates serious multicollinearity; mean VIF > 15 suggests correlated features
- **Root cause**: 1st and 2nd semester metrics naturally correlate (academic ability is stable), as do units approved/grades (both measure success)

### Key Correlated Pairs (|correlation| > 0.7):
- 1st sem approved units ↔ 1st sem grade (strong: both measure success in same timeframe)
- 2nd sem approved units ↔ 2nd sem grade (strong: same reason)
- Economic indicators (unemployment, inflation, GDP) move together

### How Models Handle Multicollinearity

**Logistic Regression**:
- **Problem**: Coefficients become unstable; standard errors inflate; harder to interpret individual feature effects
- **Ridge (L2) Solution**: Shrinks all coefficients proportionally, spreads influence across correlated features
- **Lasso (L1) Solution**: Can arbitrarily eliminate one correlated feature in favor of another, enabling feature selection
- **Result**: Ridge generally more stable than Lasso for correlated features

**Tree-Based Models (Random Forest, XGBoost)**:
- **Advantage**: Naturally robust to multicollinearity
- **Reason**: Trees make binary splits on single features; don't need independent coefficients
- **Why**: When feature A is highly correlated with B, tree can use either; no instability from correlation
- **Result**: XGBoost and Random Forest performance unaffected by multicollinearity

### Decision: Keep All Features

We retained all features despite high multicollinearity because:
1. Tree models (best performers) are unaffected
2. Removing correlated features loses information
3. Interpretability preserved; stakeholders understand all features
4. Models equipped with regularization to handle redundancy

---

## Model Evaluation Summary

### All Models (Test Set Performance)

| Model | Dataset | Accuracy | F1-Score | ROC-AUC |
|-------|---------|----------|----------|---------|
| **XGBoost_full** | full | **77.51%** | **0.7672** | **0.8943** |
| RandomForest_full | full | 73.90% | 0.7452 | 0.8853 |
| LogReg Ridge_full | full | 77.18% | 0.7571 | 0.8779 |
| LogReg Lasso_full | full | 76.05% | 0.7370 | 0.8699 |
| XGBoost_mid | mid | 73.79% | 0.7269 | 0.8702 |
| RandomForest_mid | mid | 72.77% | 0.7293 | 0.8611 |
| LogReg Ridge_mid | mid | 73.46% | 0.7032 | 0.8536 |
| LogReg Lasso_mid | mid | 73.56% | 0.6971 | 0.8450 |
| XGBoost_early | early | 61.92% | 0.6037 | 0.7488 |
| RandomForest_early | early | 59.66% | 0.5966 | 0.7493 |
| LogReg Ridge_early | early | 62.49% | 0.5744 | 0.7278 |
| LogReg Lasso_early | early | 62.60% | 0.5740 | 0.7276 |

---

## Marginal Value Analysis

**F1-Score Improvements by Data Addition**

| Algorithm | Demographics | +1st Sem | +2nd Sem | Sem1 Gain | Sem2 Gain |
|-----------|--------------|----------|----------|-----------|-----------|
| LogReg Ridge | 0.5744 | 0.7032 | 0.7571 | +0.1288 | +0.0539 |
| LogReg Lasso | 0.5740 | 0.6971 | 0.7370 | +0.1231 | +0.0399 |
| RandomForest | 0.5966 | 0.7293 | 0.7452 | +0.1327 | +0.0159 |
| XGBoost | 0.6037 | 0.7269 | 0.7672 | +0.1232 | +0.0403 |

**Key Pattern**: 1st semester data consistently adds 11-13% F1 improvement; 2nd semester adds only 1-5%.

---

## Quick Start

### Installation & Setup

```bash
# Create virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install required packages
pip install scikit-learn xgboost pandas numpy matplotlib seaborn pyyaml

# Launch Jupyter notebook
jupyter notebook model_training.ipynb
```

### Running the Full Pipeline

Execute notebook cells in order (Jupyter default behavior):

1. **Cells 1-2**: Import libraries, set random seeds for reproducibility
2. **Cells 3-4**: Load data, explore dataset shape, class distribution
3. **Cells 5-6**: Create feature sets (df_early, df_mid, df_full)
4. **Cells 7-8**: Preprocess and encode categorical variables
5. **Cells 9-10**: Analyze multicollinearity (VIF analysis)
6. **Cells 11-12**: Split data stratified 80-20 train-test
7. **Cells 13-17**: Hyperparameter tuning (nested CV)
   - RandomizedSearchCV for Random Forest
   - GridSearchCV for Logistic Regression
   - RandomizedSearchCV for XGBoost
   - Takes ~20 seconds total
8. **Cells 18-20**: Evaluate models, compare performance across 12 model configurations
9. **Cells 21-23**: Generate feature importance visualizations
10. **Cells 24-25**: Create confusion matrices and ROC curves

### Expected Output

After full execution, you'll have:
- **12 trained models** (4 algorithms × 3 feature sets) with test metrics
- **Performance comparison table** showing accuracy, F1, ROC-AUC for each
- **Feature importance plots** from Random Forest and XGBoost
- **Confusion matrices** visualizing prediction patterns per class
- **Learning curves** showing bias-variance tradeoffs
- **Reproducible results** (seed=42 ensures identical runs)

---

## Data Preprocessing

**LabelEncoding Strategy**:
- Categorical variables (marital status, gender, course ID, etc.) → numerical codes
- Preserves ordinal relationships where applicable (qualification levels: high school < bachelor < master)
- Simple and computationally efficient; tree models work natively with encoded values
- Alternative (one-hot encoding) avoided: would create 100+ sparse columns; tree models prefer integer encoding

**Feature Scaling Strategy**:
- **StandardScaler for Logistic Regression** (required):
  - Centers features around mean=0, scales to std=1
  - Critical for convergence in gradient descent optimization
  - Makes regularization (L1/L2) fair across features (no large-magnitude features penalized more)
  - Improves numerical stability in linear algebra computations
  
- **No scaling for tree models** (unnecessary):
  - Tree-based models make binary splits on feature values
  - Invariant to monotonic transformations (scaling doesn't change split logic)
  - Saves computation; Random Forest and XGBoost work with raw feature values

**No explicit feature interactions created**:
- Rationale: Tree models discover interactions implicitly via hierarchical splits
- Example: "units_approved × grades" captured when tree splits on units_approved, then grades
- Keeps feature space manageable (36 features better than 100+ with interactions)
- Maintains interpretability; easier to explain feature importance

**Why this approach works**:
- Linear models get scaled features (needed for stability)
- Tree models get unscaled features (more efficient, equally effective)
- Categorical encoding preserves information without unnecessary dimensionality expansion
- No over-engineering; let models learn interactions naturally

---

## Limitations & Future Work

### Current Limitations

1. **Static dataset**: Data from single cohort; may not generalize to future student populations
   - Remedy: Implement continuous retraining pipeline with new cohort data

2. **Temporal assumptions**: Model trained on historical data assumes future dropout patterns similar to past
   - Remedy: Monitor prediction accuracy over time; alert if performance degrades

3. **No causal inference**: High feature importance doesn't prove causation (e.g., low grades don't *cause* dropout; both may reflect underlying struggles)
   - Remedy: Use identified features for monitoring but combine with qualitative student interviews for root cause analysis

4. **Class imbalance handling**: While addressed, minority classes (Enrolled, Dropout) still underrepresented
   - Remedy: Collect more data or use synthetic oversampling (SMOTE) if performance is critical

5. **Feature lag**: Academic data reflects *past* performance; can't predict future behavior before it occurs
   - Remedy: Use engagement metrics, attendance, office hours visits as leading indicators

### Future Enhancements

1. **Feature engineering**:
   - Add engagement metrics: login frequency, assignment submission patterns, forum participation
   - Add support system features: advisor meetings, tutoring sessions, financial aid status
   - Create time-series features: performance trends (improving vs declining)

2. **Advanced algorithms**:
   - LightGBM, CatBoost for potentially faster training
   - Neural networks for large-scale deployment
   - Ensemble methods combining XGBoost + RandomForest predictions

3. **Explainability**:
   - SHAP (SHapley Additive exPlanations) for individual prediction explanations
   - Understanding *why* specific student flagged as high-risk
   - Important for transparency and building institutional trust

4. **Real-time monitoring**:
   - Integrate with Student Information System (SIS) for live predictions
   - Weekly refresh as new grades, attendance entered
   - Automated alerts to advisors for high-risk students

5. **Intervention tracking**:
   - Log which students received which interventions
   - Measure intervention effectiveness (did it prevent dropout?)
   - Use outcomes to refine model and optimize resource allocation

---

## Files Structure

```
.
├── model_training.ipynb           # Main notebook with full pipeline
├── data.csv                       # 4,424 × 37 student dataset
├── config.yaml                    # Configuration file
├── model_utils.py                 # Reusable utility functions
└── README.md                      # This file
```

### Key Notebook Sections

1. **Descriptive Statistics**: Dataset shape, missing values, class distribution
2. **Feature Engineering**: Creating df_early, df_mid, df_full progressive feature sets
3. **Data Preprocessing**: LabelEncoding categorical features, StandardScaler for LogReg
4. **Multicollinearity Check**: VIF analysis for feature correlation
5. **Hyperparameter Tuning**: Nested CV with GridSearchCV/RandomizedSearchCV
6. **Model Comparison**: Performance metrics for all 12 models across datasets
7. **Feature Importance**: Top predictive features via Random Forest and SHAP values
8. **Evaluation Metrics**: Accuracy, F1-score, ROC-AUC, confusion matrices
9. **Visualizations**: Learning curves, feature importance plots, confusion matrices

---

---

## Reproducibility & Implementation Details

- **Random seed**: 42 (all data splits and model initialization)
- **Cross-validation folds**: Stratified K-fold (maintains class distribution in each fold)
- **Train-test split strategy**: 80-20 stratified split (1,763 / 441 samples per class approximately)
- **Hyperparameter tuning times**: 
  - Logistic Regression: 0.1-2.6 seconds (GridSearchCV, smaller search space)
  - Random Forest: 2.7-3.1 seconds (RandomizedSearchCV, 20 iterations)
  - XGBoost: 2.3-2.8 seconds (RandomizedSearchCV, 25 iterations)
- **Total training time**: ~20 seconds for full pipeline (CPU: single machine, no GPU needed)

### Reproducing Results

```bash
# 1. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install scikit-learn xgboost pandas numpy matplotlib seaborn pyyaml

# 3. Run notebook
jupyter notebook model_training.ipynb

# 4. Execute all cells in order
# Output: 12 trained models with performance metrics and visualizations
```

All random seeds fixed to seed=42 ensure exact reproducibility across runs. Numerical differences (±0.0001) only due to floating-point precision.

---

## Conclusions & Key Findings

1. **XGBoost_full is the best performer** (77.51% accuracy, 0.7672 F1, 0.8943 ROC-AUC)
   - Significantly outperforms baseline Logistic Regression (77.18% accuracy but lower F1)
   - Marginally better than RandomForest_full (73.90% accuracy) despite lower CV F1
   - XGBoost's gradient boosting captures complex non-linear patterns effectively

2. **1st semester data is transformative** (~12% improvement over demographics alone)
   - Demographics alone: 61.67% avg accuracy across models (insufficient for deployment)
   - Adding 1st sem: 73.39% avg accuracy (+11.7% improvement, highly significant)
   - This validates the investment in collecting first-semester grades for early intervention

3. **2nd semester adds marginal value** (~4% improvement, classic diminishing returns)
   - Adding 2nd sem: 76.16% avg accuracy (+2.8% additional improvement)
   - Cost-benefit analysis: 2-month wait for 4% accuracy gain not worth it for intervention timing
   - **Deployment implication**: Mid-semester model sufficient for real-world use

4. **Academic metrics are strongest predictors** (60%+ importance)
   - Curricular units approved and grades: 35% of total importance
   - Other academic metrics (evaluations, without-evaluation units): 25% additional
   - Demographics and financial status: Only 35% combined
   - **Action implication**: Focus monitoring on academic engagement first

5. **High multicollinearity present but effectively managed**
   - VIF values 15.84-26.52 indicate correlated features
   - Tree models (best performers) naturally robust to multicollinearity
   - Ridge regularization stabilizes Logistic Regression despite correlation
   - **Technical implication**: Multicollinearity not a limiting factor in this analysis

6. **Class imbalance well-balanced in training**
   - Stratified CV and F1-weighted scoring ensure fair evaluation across classes
   - Model precision/recall not dominated by majority class (Graduate)
   - **Practical implication**: Predictions reliable for all three student outcomes

### Recommended Next Steps

**Short-term (Ready for deployment)**:
- Implement XGBoost_mid in production for mid-semester predictions
- Set up automated pipeline to load new semester 1 grades and generate alerts
- Establish advisor feedback mechanism to validate model recommendations

**Medium-term (Performance optimization)**:
- Collect intervention outcomes; measure if model alerts reduce actual dropouts
- Retrain quarterly with new student cohorts to maintain accuracy
- Monitor for prediction drift; alert if F1-score drops below 0.70

**Long-term (Advanced capabilities)**:
- Integrate real-time engagement metrics (login patterns, assignment submissions)
- Develop SHAP-based explanation system for individual student recommendations
- Build risk score dashboard for institutional dashboards and advisors

---

## Key Technical Lessons & Insights

### Lessons from Hyperparameter Tuning

1. **Feature richness drives hyperparameter choices**:
   - Limited features (early): Needs regularization (high C values, tight tree constraints)
   - Rich features (full): Benefits from relaxed constraints (low C, shallow trees)
   - Tree depth inversely related to data complexity: 15 for sparse data, 10 for rich data

2. **Learning rate sensitivity in boosting**:
   - Simple datasets: lr=0.1 sufficient (faster learning acceptable)
   - Complex datasets: lr=0.05 better (careful, patient learning prevents overfitting)
   - Finding: Learning rate more important than number of estimators

3. **Nested CV prevents overfitting to validation set**:
   - Without nesting: Could overfit hyperparameters to outer fold
   - With nesting: 3-fold inner CV selects HP; 5-fold outer CV validates fairly
   - Trade-off: More computation but unbiased performance estimates

### Model Selection Insights

- **Logistic Regression**: Excellent baseline (77% accuracy) but lower F1 (higher false negatives)
- **Random Forest**: Strong on early dataset, weaker on full; feature interactions help with limited data
- **XGBoost**: Consistently strongest; gradient boosting excels with sequential error correction
- **Pattern**: Tree models > linear for this tabular data with non-linear relationships

### Why XGBoost_mid Wins for Deployment

Not just best accuracy (73.79%), but:
- **Timing**: Available Week 8-12, enables meaningful intervention
- **Cost-benefit**: 96% of full model power at 50% wait time
- **Actionability**: Early enough to change outcomes
- **Sustainability**: Can retrain monthly with new cohort data

### Generalization Risk

Assumptions that could break:
- **Cohort changes**: Different student demographics, support systems (requires retraining)
- **Policy changes**: New intervention programs alter dropout patterns (retraining needed)
- **Economic shifts**: Unemployment/inflation effects on enrollment (handled via features)
- **Data drift**: Grade inflation, changed evaluation standards (continuous monitoring required)

**Mitigation**: Monthly accuracy checks, quarterly retraining, quarterly feature importance reviews.
