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
| **AdaBoost** | 68.71% | 46.88% | 22.05% | 29.79% | ✅ BACKUP |

### Tier 2: Good but Limited
| Model | ROC-AUC | Status |
|-------|---------|--------|
| Neural Network (MLP) | 68.50% | Consider for ensembles |
| Random Forest | 64.18% | Data-hungry, still improving |
| Logistic Regression | 66.20% | Limited potential, underfitting |
| Decision Tree | 59.83% | High overfitting (gap: -7.51%) |

### Tier 3: Not Recommended
| Model | ROC-AUC | Issue |
|-------|---------|-------|
| Naive Bayes | 54.51% | Naive assumption fails on imbalanced data |
| K-Nearest Neighbors | 53.96% | Performs poorly on imbalanced data |
| SVM | 50.19% | Complete failure (worse than random) |

---

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
- **GB**: Balanced bias-variance tradeoff (moderate overfitting converges)
- **Ada**: Lowest variance across all models (most stable)
- **Decision Tree**: High variance (memorization problem)
- **LR**: High bias (underfitting on complex patterns)

### 6. **Bias Audit Findings**
⚠️ **CRITICAL**: Officer age as a feature creates systematic bias:
- 58.74% importance in GB (dominates all decisions)
- Present in 100% of models (universal bias)
- May violate fair policing standards
- **Recommendation**: Fairness audit required before deployment

---

## 🔧 Technical Stack

- **Language**: Python 3.11.7
- **ML Framework**: Scikit-learn
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Evaluation**: scikit-learn metrics (confusion matrix, ROC-AUC, precision-recall, F1)
- **Notebook**: Jupyter (54 cells, 66 executions)

---

## 📊 Notebook Structure

| Cell Range | Analysis | Status |
|------|----------|--------|
| 1-10 | Data Loading & Cleaning | ✅ Complete |
| 11-30 | Exploratory Data Analysis (EDA) | ✅ Complete |
| 31-45 | Hypothesis Testing & Visualizations | ✅ Complete |
| 46-56 | Model Training (9 models) | ✅ Complete |
| 47-49 | Precision-Recall & Feature Importance | ✅ Complete |
| 50-51 | Learning Curves & Model Profiles | ✅ Complete |
| 52 | Gradient Boosting Deployment Guide | ✅ Complete |
| 53 | Bias-Variance Analysis | ✅ Complete |
| 54 | Training vs Testing Accuracy | ✅ Complete |

---

## 🎯 Usage

### Quick Start
1. **Load notebook**: `index.ipynb`
2. **Kernel variables available**:
   - `df_clean`: Cleaned dataset (59,892 records)
   - `gb_model`: Trained Gradient Boosting classifier
   - `metrics_master_df`: All model metrics comparison
   - `train_test_df`: Training vs testing accuracy comparison
   - Model objects: `lr_model, rf_model, dt_model, svm_model, knn_model, nb_model, ada_model, nn_model`

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
AdaBoost             88.17%    22.05%     46.88%  29.79%    68.71%   22.37%
Neural Network       88.17%    21.94%     47.17%  30.14%    68.50%   20.56%
Random Forest        88.12%    21.87%     44.59%  29.60%    64.18%   19.82%
Logistic Regression  88.17%    21.85%     44.37%  29.50%    66.20%   18.91%
Decision Tree        87.41%    20.58%     32.77%  25.18%    59.83%   17.29%
Naive Bayes          88.05%    20.69%     28.14%  23.81%    54.51%   16.56%
K-Nearest Neighbors  86.99%    18.66%     24.08%  21.03%    53.96%   14.90%
SVM                  88.17%    16.71%     5.73%   8.52%     50.19%   11.91%
```

### Overfitting Analysis (Accuracy Gap: Train - Test)
```
Model                Train-Test Gap  Status
Neural Network       +0.0006        ✅ Excellent generalization
AdaBoost             0.0000         ✅ Perfect balance
Naive Bayes          0.0000         ✅ Simple model
K-Nearest Neighbors  +0.0055        ⚠️ Minor overfitting
Random Forest        +0.0055        ⚠️ Minor overfitting
Logistic Regression  ~0.0000        Neutral
Gradient Boosting    -0.0103        ✅ Better on test (robust!)
SVM                  ~0.0000        Neutral
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

## ✅ Project Status

**Analysis**: ✅ Complete (54 cells, 66 executions)  
**Models**: ✅ 9 models trained & evaluated  
**README**: ✅ Comprehensive documentation  
**GitHub**: ✅ Ready to push  

