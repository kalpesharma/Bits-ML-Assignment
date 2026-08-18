"""
ML Classification Models Training Script
Trains 5 different classification models and evaluates them using 6 metrics
Dataset: Wine Quality (12+ features, 500+ observations)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and prepare the Wine Quality dataset"""
    # Download the wine quality dataset from UCI
    try:
        # Using red wine quality dataset
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
        df = pd.read_csv(url, sep=';')
        
        # Convert regression problem to classification (quality >= 6 = Good, < 6 = Bad)
        df['quality'] = (df['quality'] >= 6).astype(int)
        
        print(f"Dataset shape: {df.shape}")
        print(f"Number of features: {df.shape[1] - 1}")
        print(f"Number of observations: {df.shape[0]}")
        print(f"\nClass distribution:\n{df['quality'].value_counts()}")
        
        return df
    except Exception as e:
        print(f"Error loading from URL: {e}")
        print("Creating synthetic dataset with required specifications...")
        return create_synthetic_dataset()

def create_synthetic_dataset():
    """Create a synthetic classification dataset if download fails"""
    np.random.seed(42)
    
    n_samples = 1000  # 2x the minimum requirement
    n_features = 12
    
    # Create features
    X = np.random.randn(n_samples, n_features)
    # Create target with some correlation to features
    y = (X[:, 0] + X[:, 1] - X[:, 2] + np.random.randn(n_samples) * 0.5 > 0).astype(int)
    
    feature_names = [f'feature_{i+1}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    print(f"Synthetic Dataset shape: {df.shape}")
    print(f"Number of features: {df.shape[1] - 1}")
    print(f"Number of observations: {df.shape[0]}")
    print(f"\nClass distribution:\n{df['target'].value_counts()}")
    
    return df

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train all 5 models and evaluate with 6 metrics"""
    
    results = {}
    models = {}
    
    # 1. Logistic Regression
    print("\n" + "="*60)
    print("Training Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    y_pred_proba_lr = lr_model.predict_proba(X_test)[:, 1]
    
    results['Logistic Regression'] = evaluate_model(y_test, y_pred_lr, y_pred_proba_lr)
    models['Logistic Regression'] = lr_model
    print(f"✓ Logistic Regression trained - Accuracy: {results['Logistic Regression']['Accuracy']:.4f}")
    
    # 2. Decision Tree Classifier
    print("\nTraining Decision Tree Classifier...")
    dt_model = DecisionTreeClassifier(max_depth=10, random_state=42)
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)
    y_pred_proba_dt = dt_model.predict_proba(X_test)[:, 1]
    
    results['Decision Tree'] = evaluate_model(y_test, y_pred_dt, y_pred_proba_dt)
    models['Decision Tree'] = dt_model
    print(f"✓ Decision Tree trained - Accuracy: {results['Decision Tree']['Accuracy']:.4f}")
    
    # 3. K-Nearest Neighbor Classifier
    print("\nTraining K-Nearest Neighbor Classifier...")
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    y_pred_knn = knn_model.predict(X_test)
    y_pred_proba_knn = knn_model.predict_proba(X_test)[:, 1]
    
    results['KNN'] = evaluate_model(y_test, y_pred_knn, y_pred_proba_knn)
    models['KNN'] = knn_model
    print(f"✓ KNN trained - Accuracy: {results['KNN']['Accuracy']:.4f}")
    
    # 4. Naive Bayes Classifier
    print("\nTraining Naive Bayes Classifier...")
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    y_pred_nb = nb_model.predict(X_test)
    y_pred_proba_nb = nb_model.predict_proba(X_test)[:, 1]
    
    results['Naive Bayes'] = evaluate_model(y_test, y_pred_nb, y_pred_proba_nb)
    models['Naive Bayes'] = nb_model
    print(f"✓ Naive Bayes trained - Accuracy: {results['Naive Bayes']['Accuracy']:.4f}")
    
    # 5. Random Forest Classifier (Ensemble)
    print("\nTraining Random Forest Classifier...")
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]
    
    results['Random Forest'] = evaluate_model(y_test, y_pred_rf, y_pred_proba_rf)
    models['Random Forest'] = rf_model
    print(f"✓ Random Forest trained - Accuracy: {results['Random Forest']['Accuracy']:.4f}")
    
    return results, models

def evaluate_model(y_true, y_pred, y_pred_proba):
    """Evaluate model with 6 metrics"""
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'AUC': roc_auc_score(y_true, y_pred_proba),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0),
        'MCC': matthews_corrcoef(y_true, y_pred)
    }

def save_models(models, directory='./models'):
    """Save trained models to disk"""
    os.makedirs(directory, exist_ok=True)
    for model_name, model in models.items():
        filepath = os.path.join(directory, f'{model_name.replace(" ", "_")}.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        print(f"✓ Saved {model_name} to {filepath}")

def print_results(results):
    """Print results in table format"""
    print("\n" + "="*100)
    print("MODEL COMPARISON TABLE - EVALUATION METRICS")
    print("="*100)
    
    # Create DataFrame for better visualization
    results_df = pd.DataFrame(results).T
    results_df = results_df.round(4)
    print(results_df.to_string())
    
    # Save to CSV
    results_df.to_csv('model_results.csv')
    print("\n✓ Results saved to model_results.csv")
    
    # Find winner
    print("\n" + "="*100)
    print("MODEL WINNER (Based on Average Score)")
    print("="*100)
    
    # Calculate average score (exclude MCC which can be negative)
    avg_scores = results_df[['Accuracy', 'AUC', 'Precision', 'Recall', 'F1']].mean(axis=1)
    winner = avg_scores.idxmax()
    print(f"\n🏆 WINNER: {winner} with average score of {avg_scores[winner]:.4f}")
    print(f"\nWinner Metrics:")
    for metric, value in results[winner].items():
        print(f"  {metric}: {value:.4f}")
    
    return winner

def main():
    print("="*60)
    print("ML CLASSIFICATION MODELS TRAINING")
    print("="*60)
    
    # Load and prepare data
    print("\nLoading dataset...")
    df = load_and_prepare_data()
    
    # Save test data as CSV
    df.to_csv('test_data.csv', index=False)
    print(f"✓ Test data saved to test_data.csv")
    
    # Prepare features and target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    with open('models/scaler.pkl', 'wb') as f:
        os.makedirs('models', exist_ok=True)
        pickle.dump(scaler, f)
    
    print(f"\nTraining set size: {X_train_scaled.shape[0]}")
    print(f"Test set size: {X_test_scaled.shape[0]}")
    
    # Train and evaluate models
    results, models = train_and_evaluate_models(
        X_train_scaled, X_test_scaled, y_train, y_test
    )
    
    # Save models
    print("\n" + "="*60)
    print("Saving trained models...")
    print("="*60)
    save_models(models)
    
    # Print and save results
    winner = print_results(results)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)

if __name__ == "__main__":
    main()
