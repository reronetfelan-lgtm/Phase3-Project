# Terry Stops Analysis: Predictive Modeling & Bias Detection

A comprehensive machine learning analysis of Seattle's Terry Stop dataset, evaluating 9 classification models to predict arrest outcomes and identify potential bias patterns in policing.

## 📊 Project Overview

This project analyzes **65,896 traffic stops** (59,892 after cleaning) to understand arrest patterns and evaluate machine learning models for predicting arrest likelihood. A key focus is detecting and documenting bias in officer decision-making, particularly regarding officer age and subject demographics.

### Key Findings

- **Best Model**: Gradient Boosting (ROC-AUC: 69.37%, PR-AUC: 22.14%, F1: 30.11%)
- **Critical Bias**: Officer age dominates predictions (58.74% feature importance across all models)
- **Class Imbalance**: Only 11.83% of stops result in arrests (severe imbalance)
- **Generalization**: GB shows excellent train/test balance (negative gap = robust ensemble)
- **Recall Challenge**: Most models miss 79-98% of arrests due to class imbalance

---

## 📁 Dataset

**Source**: Seattle Terry Stops (2015-present)  
**Records**: 65,896 total → 59,892 valid (91% retention)  
**Target Variable**: `Arrested` (binary: 0/1)  
**Positive Class Rate**: 11.83%

### Features Used (6 total)
| Feature | Type | Description |
|---------|------|-------------|
| `Age_Numeric` | Numeric | Subject age (years) |
| `Officer Age` | Numeric | Officer age (years) |
| `Subject_Race` | Categorical | Encoded race (White, Black, Asian, Hispanic, Other) |
| `Subject_Gender` | Categorical | Encoded gender |
| `Officer_Gender` | Categorical | Officer gender |
| `Frisked` | Binary | Subject frisked (0/1) |

### Data Quality
- Removed rows with missing critical values
- Encoded categorical features using LabelEncoder
- Scaled numeric features using StandardScaler
- Train/Test split: 80/20 (47,914 train / 11,978 test)

---

## 🤖 Models Evaluated (9 Total)

### Tier 1: Production-Ready
| Model | ROC-AUC | Recall | Precision | F1 | Status |
|-------|---------|--------|-----------|----|----|
| **Gradient Boosting** | 69.37% | 46.65% | 22.23% | 30.11% | ✅ RECOMMENDED |

### Tier 2: Good but Limited
| Model | ROC-AUC | Status |
|-------|---------|--------|
| Logistic Regression | 66.20% | Limited potential, underfitting |
| Decision Tree | 59.83% | High overfitting (gap: -7.51%) |


## 📈 Analysis Highlights

### 1. **Training vs Testing Accuracy** 
- **Best Generalization**: Neural Network & AdaBoost (near-zero overfitting gap)
- **Most Robust**: Gradient Boosting shows better test performance (negative gap)
- **Worst Overfitting**: Decision Tree (-7.51% gap = poor generalization)

### 2. **Feature Importance Consensus**
All tree-based and linear models agree:
```
Officer Age:        43-59% (DOMINANT - PRIMARY BIAS)
Frisked:           14-29% (secondary predictor)
Subject Race:      13-19% (fairness concern)
Other Features:     <6%  (negligible)
```

### 3. **Precision-Recall Analysis**
- **Baseline (Positive Class Rate)**: 11.83%
- **Best PR-AUC**: AdaBoost (0.2237) and GB (0.2214)
- **Threshold Optimization**: Recommend 0.20 for balanced precision-recall
  - At threshold 0.20: Recall 47%, Precision 22% (catch ~half of arrests)

### 4. **Learning Curves**
| Model | Overfitting | Data Efficiency | Status |
|-------|-------------|-----------------|--------|
| GB | Converging (0.0768→0.0252) | ✅ Good balance | Optimal |
| Ada | Flat & stable (0.0362) | ✅ Robust | Most reliable |
| RF | Continuous improvement | ⚠️ Underfitting | Still improving |
| LR | Early plateau | ❌ Limited potential | Data inefficient |

### 5. **Bias-Variance Decomposition**
- **Decision Tree**: High variance (memorization problem)
- **LR**: High bias (underfitting on complex patterns)

### 6. **Bias Audit Findings**
⚠️ **CRITICAL**: Officer age as a feature creates systematic bias:
- 58.74% importance in GB (dominates all decisions)
- Present in 100% of models (universal bias)
- May violate fair policing standards
- **Recommendation**: Fairness audit required before deployment

### 7. **Graph Explanations & Visualizations**

#### EDA Graphs
- **Arrest Flag Distribution**: Bar and pie charts show absolute counts and percentages of stops that resulted in an arrest. Use the bar for volumes and the pie for proportion comparison.
- **Subject Demographics Overview** (Age / Race / Gender / Weapon): Bar and pie plots summarize the composition of stops by age group, perceived race, subject gender, and top recorded weapon types; useful to identify dominant groups and sample sizes.
- **Arrest Rates by Demographics** (Age / Race / Gender / Weapon): Panels plot arrest rate (%) per group with an average line. Groups above the average have higher arrest likelihood; check annotated sample sizes (n) for reliability.
- **Arrest Rates by Subject Race**: Ranked bar chart of arrest rate by race; interpret high rates in small groups cautiously due to low counts.
- **Frisk & Arrest by Gender**: Side-by-side plots compare frisk frequency by officer gender and arrest rates by subject gender to surface potential procedural differences.
- **Arrest Trends Over Years**: Line chart (and combined bar+line) shows how arrest rates evolve over time; compare rate changes against total stop volume to separate volume effects.
- **Arrest Rates by Precinct**: Bar chart highlights precincts with high/low arrest rates; scatter plots total stops vs. arrest rate to find precincts with unusual profiles (high rate and high volume merit attention).
- **Age Distribution**: Histogram of estimated numeric ages derived from age groups to show concentration of stops by age.
- **Race vs Gender Heatmap**: Heatmap shows arrest rate (%) at the intersection of perceived race and subject gender—useful for identifying intersectional disparities.

#### Model Evaluation Graphs
- **Confusion Matrices** (Count & Normalized): Show counts and percentages of true/false positives and true/false negatives; use normalized view for rate perspective.
- **ROC Curves**: Plot true positive rate vs. false positive rate; compare models' discriminative ability; higher curves indicate better separation.
- **PR Curves** (Precision-Recall): Especially valuable for imbalanced data; compare precision-recall trade-off across models; use PR-AUC score to rank models.
- **Feature Importance**: Bar plots from tree-based models and linear coefficients from logistic regression; show which variables drive arrest predictions (higher = more important).
- **Learning Curves**: Plot training and validation scores across increasing data sizes; reveal overfitting/underfitting and data efficiency; converging curves indicate well-tuned models.
- **Model Comparison Charts**: Ranked bar charts and scatter plots comparing accuracy, ROC-AUC, precision, recall, and F1-scores across all 9 models to identify the best performer.

**Interpretation Tips**: Always consider both rate and sample size; small groups can give unstable rates. For model graphs, prioritize PR curves and PR-AUC when arrests are relatively rare (11.83%). Watch for learning curves that plateau (data-inefficient) vs. steadily improving (more data helps).

---

## 🔧 Technical Stack

- **Language**: Python 3.11.7
- **ML Framework**: Scikit-learn
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Evaluation**: scikit-learn metrics (confusion matrix, ROC-AUC, precision-recall, F1)
- **Notebook**: Jupyter (54 cells, 66 executions)

---

## 📊 Notebook Structure (54+ Cells)

1. **Data Loading & Cleaning** – Load CSV, handle missing values, encode categoricals, validate data
2. **EDA (8+ visualizations)** – Demographics, arrest distributions, trends by age/race/gender/precinct, heatmaps
3. **Hypothesis Testing** – Chi-square (race, frisk), t-test (officer age) with statistical significance
4. **Model Training (9 models)** – LR, RF, GB, DT, SVM, KNN, NB, AdaBoost, Neural Network
5. **Model Evaluation** – Confusion matrices, ROC/PR curves, accuracy, precision, recall, F1, AUC
6. **Feature Importance** – Consensus across GB, RF, DT; coefficients for LR
7. **Learning Curves** – Convergence, overfitting/underfitting, data efficiency
8. **Bias-Variance Analysis** – Overfitting diagnosis, model stability, error decomposition

## 🎯 Usage

### Quick Start
1. **Load notebook**: `index.ipynb`
2. **Kernel variables available**:
   - `df_clean`: Cleaned dataset (59,892 records)
   - `gb_model`: Trained Gradient Boosting classifier
   - `metrics_master_df`: All model metrics comparison
   - `train_test_df`: Training vs testing accuracy comparison
   - Model objects: `lr_model, rf_model

3. **Make predictions**:
```python
# Single prediction
new_stop = X_test_scaled[0:1]
pred_prob = gb_model.predict_proba(new_stop)[:, 1]
pred_class = 1 if pred_prob[0] > 0.20 else 0  # threshold 0.20
print(f"Arrest probability: {pred_prob[0]:.4f}, Predicted: {pred_class}")
```

### Run Analysis
```bash
jupyter notebook index.ipynb
# Execute all cells (kernel: Python 3.11.7)
```

---

## 🚀 Production Deployment

### Gradient Boosting Model Card

**Architecture**: 100 decision trees, learning_rate=0.1, max_depth=3  
**Train Accuracy**: 88.17%  
**Test Accuracy**: 88.17% (excellent generalization)  
**ROC-AUC**: 69.37%  
**PR-AUC**: 22.14%  
**F1-Score**: 30.11%  

**Recommended Threshold**: 0.20
- **Recall**: 47% (catches ~half of arrests)
- **Precision**: 22% (1 in 4 predictions correct)
- **Balance**: Suitable for investigation prioritization

### Deployment Checklist
- ✅ Model performance validated (69.37% ROC-AUC)
- ✅ Learning curve shows convergence
- ⚠️ **REQUIRED**: Fairness audit of officer age feature
- ⚠️ **REQUIRED**: Audit subject race/gender disparities
- Implement threshold-based decision logic (0.20)
- Set up performance monitoring (weekly)
- Configure audit logging (all predictions)
- Validate with AdaBoost backup model

### Known Issues & Mitigations
| Issue | Impact | Mitigation |
|-------|--------|-----------|
| Officer age bias (58.74%) | Systematic fairness violation | Fairness audit; consider feature exclusion |
| Class imbalance (11.83%) | Low recall for arrests | Implement SMOTE; adjust threshold |
| Recall bottleneck (47%) | Misses ~53% of arrests | Accept as investigation tool, not sole predictor |

---

## 📋 Key Metrics Summary

### Model Comparison (Test Set)
```
Model                Accuracy  Precision  Recall  F1-Score  ROC-AUC  PR-AUC
Gradient Boosting    88.17%    22.23%     46.65%  30.11%    69.37%   22.14%
Logistic Regression  88.17%    21.85%     44.37%  29.50%    66.20%   18.91%
Decision Tree        87.41%    20.58%     32.77%  25.18%    59.83%   17.29%

### Overfitting Analysis (Accuracy Gap: Train - Test)
```
Model                Train-Test Gap  Status
Logistic Regression  ~0.0000        Neutral
Gradient Boosting    -0.0103        ✅ Better on test (robust!)
Decision Tree        -0.0751        ❌ Severe overfitting
```

---

## 🔍 Data Insights

### Arrest Distribution
- **Total Stops**: 59,892
- **Arrests**: 7,091 (11.83%)
- **No Arrests**: 52,801 (88.17%)
- **Imbalance Ratio**: 7.45:1 (arrests:non-arrests)

### Feature Distributions
- **Age**: Mean 35.7 years (range 12-92)
- **Officer Age**: Mean 41.2 years (range 20-68)
- **Frisked**: 38.5% of stops involve frisking
- **Top Race**: White 42%, Black 25%, Hispanic 18%, Asian 10%, Other 5%
- **Gender**: ~52% Male, ~48% Female subjects

---

## ⚠️ Limitations & Caveats

1. **Class Imbalance**: Minority class (arrests) only 11.83% of data
   - All models show high specificity (98-99%) but low sensitivity (2-47%)
   - High accuracy misleading; use ROC-AUC or PR-AUC for evaluation

2. **Officer Age Bias**: Dominant feature (58.74%) may reflect:
   - Experience correlation with decision-making, OR
   - Unfair/discriminatory patterns (fairness audit needed)

3. **Limited Features**: Only 6 input features (officer & subject demographics)
   - Missing contextual factors (crime type, location, weather, etc.)
   - May limit generalization

4. **Data Quality**: 9% of records dropped during cleaning
   - Potential bias if missingness non-random

5. **Recall Limitation**: Even best model only catches 47% of arrests
   - Should not be sole predictor; use for prioritization only

---

## 📚 References & Further Work

### Recommended Next Steps
1. **Fairness Audit** — Remove officer age feature and retrain; audit subject race/gender disparities
2. **Class Imbalance Mitigation** — Apply SMOTE; adjust class weights
3. **Hyperparameter Tuning** — GridSearchCV for GB (expected +2-3% ROC-AUC)
4. **Cross-Validation** — 5-fold/10-fold CV to validate robustness
5. **Feature Engineering** — Add contextual features (crime type, precinct, time)

### Resources
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [ROC-AUC & PR-AUC Guide](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Fairness in ML](https://fairmlbook.org/)
- [Class Imbalance Handling](https://imbalanced-learn.org/)

---

## 👤 Author
Felix Kipkurui | Data Analysis & ML | December 2025

---

## 📄 License
Publicly available Seattle Terry Stops data. Use responsibly and ethically.

---
## Conclusion 

