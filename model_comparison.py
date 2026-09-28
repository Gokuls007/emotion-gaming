"""
Fast Emotion Recognition Model Comparison
Optimized for quick completion (15-20 minutes)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, roc_curve, auc, 
                             classification_report)
from sklearn.preprocessing import label_binarize
import xgboost as xgb
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# LOAD DATA (FASTER - WITH SAMPLING)
# ============================================================================

def load_fer2013_data(sample_per_class=1000):
    """Load FER-2013 with sampling for faster training"""
    print("Loading FER-2013 dataset (sampled for speed)...")
    
    emotions = []
    images = []
    
    train_dir = 'train'
    emotion_folders = sorted(os.listdir(train_dir))
    
    emotion_map = {}
    
    for emotion_label, emotion_name in enumerate(emotion_folders):
        emotion_path = os.path.join(train_dir, emotion_name)
        if not os.path.isdir(emotion_path):
            continue
        
        emotion_map[emotion_label] = emotion_name
        print(f"Loading {emotion_name}...", end=' ')
        
        img_files = os.listdir(emotion_path)[:sample_per_class]  # Sample
        
        for img_file in img_files:
            try:
                img_path = os.path.join(emotion_path, img_file)
                img = load_img(img_path, color_mode='grayscale', target_size=(48, 48))
                img_array = img_to_array(img).flatten() / 255.0
                images.append(img_array)
                emotions.append(emotion_label)
            except:
                continue
        
        print(f"{len([e for e in emotions if e == emotion_label])} images")
    
    X = np.array(images)
    y = np.array(emotions)
    
    print(f"\nTotal samples: {len(X)}")
    return X, y, emotion_map

# ============================================================================
# MODEL 1: LOGISTIC REGRESSION
# ============================================================================

def train_logistic_regression(X_train, y_train, X_test, y_test):
    print("\n" + "="*70)
    print("MODEL 1: LOGISTIC REGRESSION (L2 Regularization)")
    print("="*70)
    
    # Use best practices parameters (no extensive grid search)
    lr = LogisticRegression(
        C=0.1,  # L2 regularization strength
        penalty='l2',
        solver='lbfgs',
        max_iter=1000,
        random_state=42
    )
    
    print("Training Logistic Regression...")
    lr.fit(X_train, y_train)
    
    y_pred = lr.predict(X_test)
    y_pred_proba = lr.predict_proba(X_test)
    
    results = evaluate_model("Logistic Regression", y_test, y_pred, y_pred_proba)
    return lr, results

# ============================================================================
# MODEL 2: RANDOM FOREST
# ============================================================================

def train_random_forest(X_train, y_train, X_test, y_test):
    print("\n" + "="*70)
    print("MODEL 2: RANDOM FOREST")
    print("="*70)
    
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    print("Training Random Forest...")
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    y_pred_proba = rf.predict_proba(X_test)
    
    # Feature importance
    print("\nTop 20 most important pixel positions:")
    top_indices = np.argsort(rf.feature_importances_)[-20:]
    for idx in reversed(top_indices):
        row = idx // 48
        col = idx % 48
        print(f"  Pixel ({row}, {col}): importance = {rf.feature_importances_[idx]:.4f}")
    
    results = evaluate_model("Random Forest", y_test, y_pred, y_pred_proba)
    return rf, results

# ============================================================================
# MODEL 3: XGBOOST
# ============================================================================

def train_xgboost(X_train, y_train, X_test, y_test):
    print("\n" + "="*70)
    print("MODEL 3: XGBOOST")
    print("="*70)
    
    xgb_model = xgb.XGBClassifier(
        max_depth=7,
        learning_rate=0.1,
        n_estimators=200,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss',
        n_jobs=-1
    )
    
    print("Training XGBoost...")
    xgb_model.fit(X_train, y_train)
    
    y_pred = xgb_model.predict(X_test)
    y_pred_proba = xgb_model.predict_proba(X_test)
    
    results = evaluate_model("XGBoost", y_test, y_pred, y_pred_proba)
    return xgb_model, results

# ============================================================================
# MODEL 4: DECISION TREE (SIMPLE BASELINE)
# ============================================================================

def train_decision_tree(X_train, y_train, X_test, y_test):
    print("\n" + "="*70)
    print("MODEL 4: DECISION TREE (Baseline)")
    print("="*70)
    
    dt = DecisionTreeClassifier(
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    
    print("Training Decision Tree...")
    dt.fit(X_train, y_train)
    
    y_pred = dt.predict(X_test)
    y_pred_proba = dt.predict_proba(X_test)
    
    results = evaluate_model("Decision Tree", y_test, y_pred, y_pred_proba)
    return dt, results

# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model_name, y_true, y_pred, y_pred_proba):
    """Comprehensive evaluation"""
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    print(f"\n{model_name} Performance:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    
    emotion_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=emotion_names, zero_division=0))
    
    return {
        'model': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

# ============================================================================
# VISUALIZATIONS
# ============================================================================

def plot_confusion_matrices(results_list, emotion_names):
    """Confusion matrices for all models"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.ravel()
    
    for idx, results in enumerate(results_list):
        cm = confusion_matrix(results['y_true'], results['y_pred'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=emotion_names, yticklabels=emotion_names,
                   ax=axes[idx], cbar_kws={'label': 'Count'})
        axes[idx].set_title(f'{results["model"]}', fontsize=14, weight='bold')
        axes[idx].set_ylabel('True Emotion', fontsize=11)
        axes[idx].set_xlabel('Predicted Emotion', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('confusion_matrices_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: confusion_matrices_comparison.png")
    plt.close()

def plot_roc_curves(results_list, n_classes=7):
    """ROC curves comparison (micro-average)"""
    
    plt.figure(figsize=(10, 8))
    
    for results in results_list:
        y_true_bin = label_binarize(results['y_true'], classes=range(n_classes))
        y_pred_proba = results['y_pred_proba']
        
        # Micro-average ROC
        fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_pred_proba.ravel())
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, linewidth=2.5, 
                label=f'{results["model"]} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - All Models (Micro-Average)', fontsize=14, weight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(alpha=0.3)
    plt.savefig('roc_curves_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: roc_curves_comparison.png")
    plt.close()

def plot_performance_bars(results_list):
    """Performance metrics bar chart"""
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    model_names = [r['model'] for r in results_list]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(model_names))
    width = 0.2
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    
    for idx, metric in enumerate(metrics):
        values = [r[metric] for r in results_list]
        ax.bar(x + idx*width, values, width, label=metric.capitalize(), 
               color=colors[idx], alpha=0.8)
        
        # Add value labels on bars
        for i, v in enumerate(values):
            ax.text(i + idx*width, v + 0.01, f'{v:.3f}', 
                   ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, weight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: performance_comparison.png")
    plt.close()

def create_performance_table(results_list):
    """Performance table"""
    
    df = pd.DataFrame([
        {
            'Model': r['model'],
            'Accuracy': f"{r['accuracy']:.4f}",
            'Precision': f"{r['precision']:.4f}",
            'Recall': f"{r['recall']:.4f}",
            'F1-Score': f"{r['f1']:.4f}"
        }
        for r in results_list
    ])
    
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON TABLE")
    print("="*70)
    print(df.to_string(index=False))
    
    df.to_csv('model_performance_comparison.csv', index=False)
    print("\n✓ Saved: model_performance_comparison.csv")
    
    return df

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*70)
    print("   EMOTION RECOGNITION - FAST MODEL COMPARISON")
    print("   4 Models | Complete in ~15-20 minutes")
    print("="*70)
    
    # Load sampled data (1000 per class = 7000 total)
    X, y, emotion_map = load_fer2013_data(sample_per_class=1000)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set:  {X_test.shape[0]} samples")
    
    # Train all models
    all_results = []
    
    # Model 1: Logistic Regression (~3 min)
    lr_model, lr_results = train_logistic_regression(X_train, y_train, X_test, y_test)
    all_results.append(lr_results)
    
    # Model 2: Decision Tree (~2 min)
    dt_model, dt_results = train_decision_tree(X_train, y_train, X_test, y_test)
    all_results.append(dt_results)
    
    # Model 3: Random Forest (~5 min)
    rf_model, rf_results = train_random_forest(X_train, y_train, X_test, y_test)
    all_results.append(rf_results)
    
    # Model 4: XGBoost (~8 min)
    xgb_model, xgb_results = train_xgboost(X_train, y_train, X_test, y_test)
    all_results.append(xgb_results)
    
    # Generate visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    emotion_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    
    plot_confusion_matrices(all_results, emotion_names)
    plot_roc_curves(all_results)
    plot_performance_bars(all_results)
    performance_df = create_performance_table(all_results)
    
    # Final recommendation
    print("\n" + "="*70)
    print("FINAL RECOMMENDATION")
    print("="*70)
    
    best_model = max(all_results, key=lambda x: x['f1'])
    print(f"\nBest Model: {best_model['model']}")
    print(f"F1-Score: {best_model['f1']:.4f}")
    print(f"Accuracy: {best_model['accuracy']:.4f}")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print("\nGenerated files for your report:")
    print("  1. confusion_matrices_comparison.png - Visual comparison of predictions")
    print("  2. roc_curves_comparison.png - ROC curves showing model discrimination")
    print("  3. performance_comparison.png - Bar chart of all metrics")
    print("  4. model_performance_comparison.csv - Table for your report")
    print("\nUse these visualizations in your data mining assignment report!")

if __name__ == "__main__":
    main()