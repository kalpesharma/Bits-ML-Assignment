# ML Classification Models - Assignment 2

## Problem Statement

This project implements and evaluates **5 different machine learning classification models** on a real-world classification dataset. The goal is to compare the performance of different algorithms using standardized evaluation metrics and identify the best-performing model for the given dataset.

The project demonstrates end-to-end machine learning workflow including:
- Data loading and preprocessing
- Model training and evaluation
- Metrics calculation (6 different metrics per model)
- Interactive web deployment using Streamlit
- Model comparison and analysis

## Dataset Description

**Dataset:** Wine Quality (Red Wine)

**Source:** UCI Machine Learning Repository  
**Dataset Link:** [Wine Quality Dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv)

**Features:** 12 physicochemical properties  
**Observations:** 1599 samples  
**Target Variable:** Binary classification (Quality ≥ 6 = Good/1, Quality < 6 = Bad/0)

### Dataset Features:
1. Fixed Acidity
2. Volatile Acidity
3. Citric Acid
4. Residual Sugar
5. Chlorides
6. Free Sulfur Dioxide
7. Total Sulfur Dioxide
8. Density
9. pH
10. Sulphates
11. Alcohol
12. Quality (Target)

**Data Split:**
- Training Set: 80% (1279 samples)
- Test Set: 20% (320 samples)

## GitHub Repository Link

[GitHub Repository](https://github.com/kalpesharma/Bits-ML-Assignment)

Access the complete source code, documentation, and trained models at the repository above.

## Models Used

### Classification Models Implemented:

1. **Logistic Regression**
   - Linear classification model using logistic function
   - Efficient for binary classification problems
   - Provides probability estimates

2. **Decision Tree Classifier**
   - Non-parametric, tree-based classifier
   - Interpretable decision boundaries
   - Parameters: max_depth=10

3. **K-Nearest Neighbors (KNN)**
   - Instance-based learning algorithm
   - Classifies based on majority vote of k nearest neighbors
   - Parameters: n_neighbors=5

4. **Naive Bayes Classifier**
   - Probabilistic classifier based on Bayes' theorem
   - Assumes feature independence
   - Type: Gaussian Naive Bayes (for continuous features)

5. **Random Forest (Ensemble)**
   - Ensemble method combining multiple decision trees
   - Reduces overfitting through bagging
   - Parameters: n_estimators=100, max_depth=10

### Evaluation Metrics (6 Parameters):

1. **Accuracy** - Overall correctness of predictions
   - Formula: (TP + TN) / (TP + TN + FP + FN)

2. **AUC Score** - Area Under the Receiver Operating Characteristic Curve
   - Measures classification performance across all thresholds
   - Range: 0 to 1 (higher is better)

3. **Precision** - Proportion of positive predictions that are correct
   - Formula: TP / (TP + FP)
   - Important when false positives are costly

4. **Recall** - Proportion of actual positives correctly identified
   - Formula: TP / (TP + FN)
   - Important when false negatives are costly

5. **F1 Score** - Harmonic mean of precision and recall
   - Formula: 2 × (Precision × Recall) / (Precision + Recall)
   - Balanced metric for imbalanced datasets

6. **Matthews Correlation Coefficient (MCC)** - Correlation between predicted and actual
   - Range: -1 to 1 (1 is perfect prediction)
   - Considers all four confusion matrix elements

## Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7423 | 0.8219 | 0.7634 | 0.7509 | 0.7571 | 0.4829 |
| Decision Tree | 0.9068 | 0.9440 | 0.9243 | 0.8994 | 0.9117 | 0.8134 |
| KNN | 0.7999 | 0.8812 | 0.7943 | 0.8444 | 0.8186 | 0.5973 |
| Naive Bayes | 0.7342 | 0.7971 | 0.7688 | 0.7193 | 0.7432 | 0.4695 |
| Random Forest (Ensemble) | 0.9400 | 0.9841 | 0.9470 | 0.9404 | 0.9437 | 0.8794 |

### Model Performance Observations

#### Logistic Regression
- **Strengths:** Good AUC score (0.8219), solid baseline model, provides probability estimates
- **Weaknesses:** Lower accuracy (74.23%), assumes linear relationships, doesn't capture complex patterns
- **Metrics:** Accuracy: 74.23%, F1: 75.71%
- **Use Case:** Good baseline model, interpretable, fast inference
- **Performance:** Moderate performer, competitive with simpler models

#### Decision Tree
- **Strengths:** Good interpretability, captures non-linear patterns, high precision (92.43%)
- **Weaknesses:** Moderate accuracy (90.68%), prone to overfitting, decent AUC (0.9440)
- **Metrics:** Accuracy: 90.68%, F1: 91.17%
- **Use Case:** When interpretability is critical, understanding feature importance
- **Performance:** Mid-range performer, good F1 score

#### K-Nearest Neighbors (KNN)
- **Strengths:** Good AUC score (0.8812), simple to implement
- **Weaknesses:** Higher accuracy than Logistic Regression, Navie Bayes, slow predictions, sensitive to scaling
- **Metrics:** Accuracy: 79.99%, F1: 81.86%
- **Use Case:** Non-parametric classification, baseline comparison
- **Performance:** Competitive but slower than linear models

#### Naive Bayes
- **Strengths:** Fastest training, good precision (76.88%), works with small datasets
- **Weaknesses:** Lowest accuracy (73.42%), assumes feature independence (violated here), lowest score across except Precision
- **Metrics:** Accuracy: 73.42%, F1: 74.32%
- **Use Case:** Text classification, quick prototyping
- **Performance:** Weakest performer on this dataset, feature independence assumption problematic

#### Random Forest (Ensemble) - **WINNER** 🏆
- **Strengths:** 
  - **Highest Accuracy (94.00%)** - Best overall correctness
  - **Highest AUC (0.9841)** - Excellent discrimination ability
  - **Highest Precision (94.70%)** - Fewer false positives
  - **Highest Recall (94.04%)** - Better detection rate
  - **Highest F1 Score (94.37%)** - Best balanced performance
  - **Highest MCC (0.87940)** - Strongest correlation
  - Robust to outliers and feature scaling
  - Handles non-linear relationships well
- **Weaknesses:** More complex model, longer training time, requires tuning
- **Metrics:** Accuracy: 94.00%, F1: 74.32%
- **Use Case:** Production systems where accuracy is critical
- **Performance:** **BEST OVERALL PERFORMER** - Outperforms all models across ALL 6 metrics

### Overall Winner for your Dataset?

**🏆 Random Forest Classifier (Ensemble Method)**

**Justification:**
The Random Forest model definitively outperforms all other models across all 6 evaluation metrics:
- Achieved the **highest accuracy (94.00%)** - 3.66% better than Decision Tree
- **Strongest AUC score (0.9841)** - indicating excellent discrimination ability (0.0401 better than Decision tree)
- **Best precision-recall tradeoff** - highest F1 score (94.37%)
- **Most robust** across different evaluation criteria through ensemble averaging
- **Reduced overfitting** through bagging and boosting principles
- Handles the complex relationships in wine quality data effectively

**Key Performance Advantages:**
- Significant improvement in accuracy over baseline
- Significant improvement in AUC over single models
- Consistent performance across all metrics (no weak areas)
- Best suited for real-world deployment

**Recommendation:** Use Random Forest for production deployment on wine quality classification tasks.

## Project Structure

```
ml_assignment/
├── app.py                      # Streamlit web application
├── train_models.py            # Model training and evaluation script
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── test_data.csv              # Test dataset (12 features, 500+ observations)
└── models/                    # Saved model files
    ├── Logistic_Regression.pkl
    ├── Decision_Tree.pkl
    ├── KNN.pkl
    ├── Naive_Bayes.pkl
    ├── Random_Forest.pkl
    └── scaler.pkl
```

### Key Findings
1. Ensemble methods (Random Forest) significantly outperform single models
2. Logistic Regression provides good baseline performance
3. Decision Tree shows competitive AUC score 
4. Decision Trees are interpretable but prone to overfitting
5. Naive Bayes assumes feature independence which may not be always


## Student
Your Kalpesh Sharma / 2025da04264  
Date: August 18, 2026
