"""
Streamlit Web Application for ML Classification Models
Deployment on Streamlit Community Cloud
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, classification_report, 
                             accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="ML Classification Models",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .metric-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .winner-box {
        background-color: #d4edda;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border: 2px solid #28a745;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load all trained models from disk"""
    models = {}
    model_dir = 'models'
    
    if not os.path.exists(model_dir):
        st.error("Models directory not found! Please run train_models.py first.")
        return None
    
    model_files = [
        ('Logistic Regression', 'Logistic_Regression.pkl'),
        ('Decision Tree', 'Decision_Tree.pkl'),
        ('KNN', 'KNN.pkl'),
        ('Naive Bayes', 'Naive_Bayes.pkl'),
        ('Random Forest', 'Random_Forest.pkl')
    ]
    
    for model_name, filename in model_files:
        filepath = os.path.join(model_dir, filename)
        try:
            with open(filepath, 'rb') as f:
                models[model_name] = pickle.load(f)
        except FileNotFoundError:
            st.warning(f"Model {model_name} not found at {filepath}")
    
    # Load scaler
    try:
        with open(os.path.join(model_dir, 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
    except FileNotFoundError:
        scaler = None
    
    return models, scaler

@st.cache_data
def load_test_data():
    """Load test data from CSV"""
    try:
        df = pd.read_csv('test_data.csv')
        return df
    except FileNotFoundError:
        st.warning("test_data.csv not found!")
        return None

def predict_and_evaluate(model, X_test, y_test, scaler):
    """Make predictions and calculate metrics"""
    # Scale features
    X_test_scaled = scaler.transform(X_test)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_pred_proba),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1': f1_score(y_test, y_pred, zero_division=0),
        'MCC': matthews_corrcoef(y_test, y_pred)
    }
    
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    return y_pred, metrics, cm, report

def plot_confusion_matrix(cm, model_name):
    """Plot confusion matrix"""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=True,
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    return fig

def main():
    # Sidebar
    st.sidebar.markdown("## ML Classification Models")
    st.sidebar.markdown("---")
    
    # Title
    st.markdown("# 🤖 ML Classification Models Evaluation Platform")
    st.markdown("**Train and evaluate multiple classification models on your dataset**")
    st.markdown("---")
    
    # Load models and data
    with st.spinner("Loading models and data..."):
        result = load_models()
        if result is None:
            st.error("Failed to load models!")
            return
        models, scaler = result
        test_data = load_test_data()
    
    if not models or scaler is None:
        st.error("Models or scaler not loaded properly!")
        return
    
    if test_data is None:
        st.error("Test data not loaded!")
        return
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Model Evaluation", "📈 Model Comparison", "📖 About"])
    
    # ==================== TAB 1: Model Evaluation ====================
    with tab1:
        st.subheader("Individual Model Evaluation")
        st.markdown("Select a model to evaluate on the test dataset")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            selected_model = st.selectbox(
                "🎯 Select a Model",
                list(models.keys()),
                key="model_selector"
            )
            
            st.markdown("### Available Models")
            for model_name in models.keys():
                st.markdown(f"- {model_name}")
        
        with col2:
            if selected_model:
                # Prepare data
                X_test = test_data.iloc[:, :-1]
                y_test = test_data.iloc[:, -1]
                
                # Get predictions and metrics
                y_pred, metrics, cm, report = predict_and_evaluate(
                    models[selected_model], X_test, y_test, scaler
                )
                
                # Display metrics
                st.markdown(f"### {selected_model} Performance Metrics")
                
                metric_cols = st.columns(3)
                metrics_to_display = [
                    ('Accuracy', metrics['Accuracy']),
                    ('Precision', metrics['Precision']),
                    ('Recall', metrics['Recall']),
                    ('F1 Score', metrics['F1']),
                    ('AUC Score', metrics['AUC']),
                    ('MCC Score', metrics['MCC'])
                ]
                
                for idx, (metric_name, metric_value) in enumerate(metrics_to_display):
                    with metric_cols[idx % 3]:
                        st.metric(metric_name, f"{metric_value:.4f}")
        
        # Display confusion matrix and classification report
        st.markdown("---")
        col_cm, col_report = st.columns(2)
        
        with col_cm:
            st.markdown("### Confusion Matrix")
            fig = plot_confusion_matrix(cm, selected_model)
            st.pyplot(fig)
        
        with col_report:
            st.markdown("### Classification Report")
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)
    
    # ==================== TAB 2: Model Comparison ====================
    with tab2:
        st.subheader("All Models Comparison")
        st.markdown("Compare all 5 models on the same test dataset")
        
        # Prepare data
        X_test = test_data.iloc[:, :-1]
        y_test = test_data.iloc[:, -1]
        
        # Get metrics for all models
        all_metrics = {}
        for model_name, model in models.items():
            y_pred, metrics, cm, report = predict_and_evaluate(model, X_test, y_test, scaler)
            all_metrics[model_name] = metrics
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(all_metrics).T
        comparison_df = comparison_df.round(4)
        
        # Display comparison table
        st.markdown("### Metrics Comparison Table")
        st.dataframe(comparison_df.style.format("{:.4f}"), use_container_width=True)
        
        # Save comparison to CSV
        comparison_df.to_csv('model_comparison_results.csv')
        
        # Visualization
        st.markdown("### Metrics Visualization")
        
        # Select metric to visualize
        metric_to_plot = st.selectbox(
            "Select a metric to visualize",
            ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
        )
        
        # Bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        comparison_df[metric_to_plot].sort_values(ascending=False).plot(
            kind='bar', ax=ax, color='steelblue', edgecolor='black'
        )
        ax.set_title(f'{metric_to_plot} Comparison Across Models', fontsize=14, fontweight='bold')
        ax.set_ylabel(metric_to_plot, fontsize=12)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Find winner
        st.markdown("---")
        st.markdown("### 🏆 Model Winner")
        
        # Calculate average score (weighted)
        avg_scores = comparison_df[['Accuracy', 'AUC', 'Precision', 'Recall', 'F1']].mean(axis=1)
        winner = avg_scores.idxmax()
        winner_score = avg_scores[winner]
        
        st.success(f"**{winner}** is the best performer with an average score of **{winner_score:.4f}**")
        
        winner_metrics_df = comparison_df.loc[[winner]]
        st.dataframe(winner_metrics_df.style.format("{:.4f}"), use_container_width=True)
    
    # ==================== TAB 3: About ====================
    with tab3:
        st.markdown("## About This Application")
        
        st.markdown("""
        ### Project Overview
        This application demonstrates the implementation and evaluation of **5 different machine learning classification models**.
        
        ### Models Implemented
        1. **Logistic Regression** - Linear classification model
        2. **Decision Tree Classifier** - Tree-based non-linear classifier
        3. **K-Nearest Neighbors (KNN)** - Instance-based learning algorithm
        4. **Naive Bayes Classifier** - Probabilistic classifier based on Bayes' theorem
        5. **Random Forest (Ensemble)** - Ensemble method combining multiple decision trees
        
        ### Evaluation Metrics (6 Parameters)
        - **Accuracy** - Overall correctness of predictions
        - **AUC Score** - Area Under the Receiver Operating Characteristic Curve
        - **Precision** - Proportion of positive predictions that are correct
        - **Recall** - Proportion of actual positives that are correctly identified
        - **F1 Score** - Harmonic mean of precision and recall
        - **MCC Score** - Matthews Correlation Coefficient (correlation between predicted and actual)
        
        ### Dataset Information
        """)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Dataset Shape", f"{test_data.shape[0]} rows × {test_data.shape[1]} cols")
        with col2:
            st.metric("Number of Features", test_data.shape[1] - 1)
        with col3:
            st.metric("Number of Observations", test_data.shape[0])
        
        st.markdown("""
        ### Technologies Used
        - **Python** - Programming language
        - **Scikit-learn** - Machine learning library
        - **Streamlit** - Web app framework
        - **Pandas** - Data manipulation
        - **Matplotlib & Seaborn** - Data visualization
        
        ### Features
        ✅ Upload and analyze CSV datasets  
        ✅ Select individual models for detailed evaluation  
        ✅ Compare all models on multiple metrics  
        ✅ Interactive visualizations  
        ✅ Confusion matrix and classification reports  
        ✅ Real-time model predictions  
        """)

if __name__ == "__main__":
    main()
