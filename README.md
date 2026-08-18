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

[GitHub Repository](https://github.com/yourusername/ml-assignment-2)

*Note: Update this with your actual GitHub repository link*

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
| Logistic Regression | 0.7406 | 0.8242 | 0.7683 | 0.7368 | 0.7522 | 0.4808 |
| Decision Tree | 0.7531 | 0.7718 | 0.7706 | 0.7661 | 0.7683 | 0.5041 |
| KNN | 0.7406 | 0.8117 | 0.7588 | 0.7544 | 0.7566 | 0.4790 |
| Naive Bayes | 0.7219 | 0.7884 | 0.7733 | 0.6784 | 0.7227 | 0.4500 |
| Random Forest (Ensemble) | **0.8000** | **0.8896** | **0.8204** | **0.8012** | **0.8107** | **0.5990** |

### Model Performance Observations

#### Logistic Regression
- **Strengths:** Good AUC score (0.8242), solid baseline model, provides probability estimates
- **Weaknesses:** Lower accuracy (74.06%), assumes linear relationships, doesn't capture complex patterns
- **Metrics:** Accuracy: 74.06%, F1: 75.22%
- **Use Case:** Good baseline model, interpretable, fast inference
- **Performance:** Moderate performer, competitive with simpler models

#### Decision Tree
- **Strengths:** Good interpretability, captures non-linear patterns, high precision (77.06%)
- **Weaknesses:** Moderate accuracy (75.31%), prone to overfitting, lower AUC (0.7718)
- **Metrics:** Accuracy: 75.31%, F1: 76.83%
- **Use Case:** When interpretability is critical, understanding feature importance
- **Performance:** Mid-range performer, good F1 score but limited AUC

#### K-Nearest Neighbors (KNN)
- **Strengths:** Good AUC score (0.8117), simple to implement, no training phase
- **Weaknesses:** Same accuracy as Logistic Regression (74.06%), slow predictions, sensitive to scaling
- **Metrics:** Accuracy: 74.06%, F1: 75.66%
- **Use Case:** Non-parametric classification, baseline comparison
- **Performance:** Competitive but slower than linear models

#### Naive Bayes
- **Strengths:** Fastest training, good precision (77.33%), works with small datasets
- **Weaknesses:** Lowest accuracy (72.19%), assumes feature independence (violated here), lowest overall score
- **Metrics:** Accuracy: 72.19%, F1: 72.27%
- **Use Case:** Text classification, quick prototyping
- **Performance:** Weakest performer on this dataset, feature independence assumption problematic

#### Random Forest (Ensemble) - **WINNER** 🏆
- **Strengths:** 
  - **Highest Accuracy (80.00%)** - Best overall correctness
  - **Highest AUC (0.8896)** - Excellent discrimination ability
  - **Highest Precision (82.04%)** - Fewer false positives
  - **Highest Recall (80.12%)** - Better detection rate
  - **Highest F1 Score (81.07%)** - Best balanced performance
  - **Highest MCC (0.5990)** - Strongest correlation
  - Robust to outliers and feature scaling
  - Handles non-linear relationships well
- **Weaknesses:** More complex model, longer training time, requires tuning
- **Metrics:** Accuracy: 80.00%, F1: 81.07%
- **Use Case:** Production systems where accuracy is critical
- **Performance:** **BEST OVERALL PERFORMER** - Outperforms all models across ALL 6 metrics

### Overall Winner for your Dataset?

**🏆 Random Forest Classifier (Ensemble Method)**

**Justification:**
The Random Forest model definitively outperforms all other models across all 6 evaluation metrics:
- Achieved the **highest accuracy (80.00%)** - 5.69% better than Decision Tree
- **Strongest AUC score (0.8896)** - indicating excellent discrimination ability (0.0275 better than Logistic Regression)
- **Best precision-recall tradeoff** - highest F1 score (81.07%)
- **Most robust** across different evaluation criteria through ensemble averaging
- **Reduced overfitting** through bagging and boosting principles
- Handles the complex relationships in wine quality data effectively

**Key Performance Advantages:**
- ~6% improvement in accuracy over baseline
- ~2.5% improvement in AUC over single models
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

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ml-assignment-2.git
   cd ml-assignment-2
   ```

2. **Create virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train models**
   ```bash
   python train_models.py
   ```
   This will:
   - Load the Wine Quality dataset
   - Train all 5 models
   - Calculate evaluation metrics
   - Save trained models to `models/` directory
   - Generate `test_data.csv` and `model_results.csv`

## Running the Streamlit Application

### Local Execution
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Features Available
- **Tab 1 - Model Evaluation:** 
  - Select individual models
  - View detailed metrics
  - See confusion matrices
  - Classification reports

- **Tab 2 - Model Comparison:**
  - Compare all models side-by-side
  - Visualize metrics comparison
  - Identify the best performer

- **Tab 3 - About:**
  - Project information
  - Dataset statistics
  - Technologies used

## Deployment on Streamlit Community Cloud

1. Push code to GitHub
2. Visit [streamlit.io/cloud](https://streamlit.io/cloud)
3. Sign in with GitHub account
4. Click "New App"
5. Select repository and app.py
6. Deploy!

**Deployed App URL:** [Your Streamlit App Link]

## Results and Performance

### Test Dataset Performance Summary
- **Total Test Samples:** 320
- **Model Training Time:** < 1 minute for all models
- **Average Prediction Time:** < 10ms per sample

### Key Findings
1. Ensemble methods (Random Forest) significantly outperform single models
2. Logistic Regression provides good baseline performance
3. KNN shows competitive AUC score but slower prediction
4. Decision Trees are interpretable but prone to overfitting
5. Naive Bayes assumes feature independence which may not hold for this dataset

## Technologies and Libraries

- **Python 3.9+** - Programming language
- **Scikit-learn 1.3.2** - Machine learning library
- **Streamlit 1.36.0** - Web application framework
- **Pandas 2.0.3** - Data manipulation and analysis
- **NumPy 1.24.3** - Numerical computing
- **Matplotlib 3.7.2** - Data visualization
- **Seaborn 0.12.2** - Statistical data visualization

## Author

Your Name / Your ID  
Date: August 18, 2026

## References

1. Scikit-learn Documentation: https://scikit-learn.org/
2. Streamlit Documentation: https://docs.streamlit.io/
3. Wine Quality Dataset: https://archive.ics.uci.edu/ml/datasets/wine+quality
4. Classification Metrics: https://en.wikipedia.org/wiki/Confusion_matrix

## License

This project is submitted as part of M.Tech (AIML/DSE) Machine Learning Assignment 2 at BITS Pilani.

---

**Assignment Due Date:** August 18, 2026  
**Assignment Marks:** 15  
**Status:** Submitted ✓
