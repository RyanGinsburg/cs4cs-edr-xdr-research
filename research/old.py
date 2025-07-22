import pandas as pd #type: ignore
import numpy as np #type: ignore
from sklearn.model_selection import train_test_split, cross_val_score #type: ignore
from sklearn.linear_model import LogisticRegression #type: ignore
from sklearn.preprocessing import StandardScaler, LabelEncoder #type: ignore
from sklearn.metrics import (accuracy_score, precision_score, recall_score, #type: ignore
                           f1_score, confusion_matrix, classification_report)
import matplotlib.pyplot as plt #type: ignore
import seaborn as sns #type: ignore
from sklearn.feature_selection import SelectKBest, f_classif #type: ignore
from sklearn.ensemble import RandomForestClassifier #type: ignore
import warnings
import base64
import io
import datetime
warnings.filterwarnings('ignore')

class HTMLReportGenerator:
    def __init__(self):
        self.html_content = []
        self.plots = []
        
    def add_title(self, title, level=1):
        self.html_content.append(f'<h{level} class="section-title">{title}</h{level}>')
    
    def add_text(self, text, style_class="content"):
        self.html_content.append(f'<div class="{style_class}">{text}</div>')
    
    def add_metrics_table(self, metrics_data, title="Model Metrics"):
        html = f'<div class="metrics-container"><h3>{title}</h3>'
        html += '<table class="metrics-table">'
        html += '<thead><tr>'
        
        # Add Model header first
        html += '<th>Model</th>'
        
        # Header row for all columns
        for col in metrics_data.columns:
            html += f'<th>{col.title()}</th>'
        html += '</tr></thead><tbody>'
        
        # Data rows
        for idx, row in metrics_data.iterrows():
            html += '<tr>'
            html += f'<td class="model-name">{idx}</td>'
            for col in metrics_data.columns:
                html += f'<td>{row[col]:.4f}</td>'
            html += '</tr>'
        html += '</tbody></table></div>'
        self.html_content.append(html)
    
    def add_confusion_matrix_table(self, results):
        html = '<div class="confusion-container"><h3>Confusion Matrix Analysis</h3>'
        html += '<table class="confusion-table">'
        html += '<thead><tr><th>Model</th><th>TN</th><th>FP</th><th>FN</th><th>TP</th><th>FP Rate</th><th>Miss Rate</th></tr></thead>'
        html += '<tbody>'
        
        for model_name in results.keys():
            tn = results[model_name]['true_negatives']
            fp = results[model_name]['false_positives']
            fn = results[model_name]['false_negatives']
            tp = results[model_name]['true_positives']
            fp_rate = results[model_name]['false_positive_rate']
            fn_rate = results[model_name]['false_negative_rate']
            
            html += f'<tr><td class="model-name">{model_name}</td>'
            html += f'<td>{tn}</td><td>{fp}</td><td>{fn}</td><td>{tp}</td>'
            html += f'<td>{fp_rate:.2f}%</td><td>{fn_rate:.2f}%</td></tr>'
        
        html += '</tbody></table></div>'
        self.html_content.append(html)
    
    def add_plot(self, fig, title=""):
        img = io.BytesIO()
        fig.savefig(img, format='png', dpi=150, bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        
        html = f'<div class="plot-container">'
        if title:
            html += f'<h3>{title}</h3>'
        html += f'<img src="data:image/png;base64,{plot_url}" alt="{title}" class="plot-image"/>'
        html += '</div>'
        self.html_content.append(html)
        plt.close(fig)
    
    def add_best_models_summary(self, best_models, comparison_df, results):
        html = '<div class="best-models-container"><h3>Best Models by Metric</h3>'
        html += '<div class="best-models-grid">'
        
        metric_columns = {
            'Accuracy': 'accuracy',
            'Precision': 'precision',
            'Recall': 'recall',
            'F1 Score': 'f1'
        }
        
        for metric, column in metric_columns.items():
            model = best_models.get(metric, '')
            if model and column in comparison_df.columns:
                score = comparison_df.loc[model, column]
                html += f'<div class="metric-card">'
                html += f'<div class="metric-name">{metric}</div>'
                html += f'<div class="metric-model">{model}</div>'
                html += f'<div class="metric-score">{score:.4f}</div>'
                html += f'</div>'
        
        # Add error rates
        if 'Lowest False Positive Rate' in best_models:
            model = best_models['Lowest False Positive Rate']
            score = results[model]['false_positive_rate']
            html += f'<div class="metric-card error-rate">'
            html += f'<div class="metric-name">Lowest FP Rate</div>'
            html += f'<div class="metric-model">{model}</div>'
            html += f'<div class="metric-score">{score:.2f}%</div>'
            html += f'</div>'
        
        if 'Lowest Miss Rate' in best_models:
            model = best_models['Lowest Miss Rate']
            score = results[model]['false_negative_rate']
            html += f'<div class="metric-card error-rate">'
            html += f'<div class="metric-name">Lowest Miss Rate</div>'
            html += f'<div class="metric-model">{model}</div>'
            html += f'<div class="metric-score">{score:.2f}%</div>'
            html += f'</div>'
        
        html += '</div></div>'
        self.html_content.append(html)
    
    def generate_html(self, output_file="results.html"):
        css = """
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 15px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                overflow: hidden;
            }
            
            .header {
                background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }
            
            .header h1 {
                font-size: 2.5em;
                margin-bottom: 10px;
                font-weight: 300;
            }
            
            .header .subtitle {
                font-size: 1.1em;
                opacity: 0.9;
            }
            
            .content {
                padding: 30px;
            }
            
            .section-title {
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
                margin: 30px 0 20px 0;
                font-weight: 500;
            }
            
            .metrics-container, .confusion-container {
                margin: 20px 0;
                background: #f8f9fa;
                border-radius: 10px;
                padding: 20px;
                border-left: 4px solid #3498db;
            }
            
            .metrics-table, .confusion-table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 15px;
                background: white;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }
            
            .metrics-table th, .confusion-table th {
                background: #34495e;
                color: white;
                padding: 15px;
                text-align: left;
                font-weight: 600;
            }
            
            .metrics-table td, .confusion-table td {
                padding: 12px 15px;
                border-bottom: 1px solid #eee;
            }
            
            .model-name {
                font-weight: 600;
                color: #2c3e50;
            }
            
            .metrics-table tr:hover, .confusion-table tr:hover {
                background: #f1f2f6;
            }
            
            .plot-container {
                margin: 30px 0;
                text-align: center;
                background: white;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            }
            
            .plot-image {
                max-width: 100%;
                height: auto;
                border-radius: 8px;
            }
            
            .best-models-container {
                margin: 20px 0;
                background: #f8f9fa;
                border-radius: 10px;
                padding: 20px;
                border-left: 4px solid #e74c3c;
            }
            
            .best-models-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-top: 15px;
            }
            
            .metric-card {
                background: white;
                border-radius: 8px;
                padding: 20px;
                text-align: center;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                border-top: 4px solid #3498db;
            }
            
            .metric-card.error-rate {
                border-top-color: #e74c3c;
            }
            
            .metric-name {
                font-weight: 600;
                color: #2c3e50;
                margin-bottom: 8px;
            }
            
            .metric-model {
                color: #7f8c8d;
                margin-bottom: 8px;
                font-size: 0.9em;
            }
            
            .metric-score {
                font-size: 1.5em;
                font-weight: 700;
                color: #3498db;
            }
            
            .error-rate .metric-score {
                color: #e74c3c;
            }
            
            .footer {
                background: #ecf0f1;
                padding: 20px;
                text-align: center;
                color: #7f8c8d;
                border-top: 1px solid #ddd;
            }
            
            .info-box {
                background: #e8f6f3;
                border-left: 4px solid #1abc9c;
                padding: 15px;
                margin: 15px 0;
                border-radius: 4px;
            }

            .warning-box {
                background: #fdf2e9;
                border-left: 4px solid #f39c12;
                padding: 15px;
                margin: 15px 0;
                border-radius: 4px;
            }
        </style>
        """
        
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Machine Learning Analysis Results</title>
            {css}
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Machine Learning Analysis Report</h1>
                    <div class="subtitle">Generated on {datetime.datetime.now().strftime('%B %d, %Y at %I:%M %p')}</div>
                </div>
                <div class="content">
                    {''.join(self.html_content)}
                </div>
                <div class="footer">
                    <p>Generated by ML Analysis Pipeline | CS4CS Research</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)
        
        return output_file

# Global HTML generator instance
html_generator = HTMLReportGenerator()

def load_and_preprocess_data(file_path):
    """Load the CSV file and preprocess the data"""
    html_generator.add_title("Dataset Loading and Preprocessing", 2)
    
    # Load the CSV file
    df = pd.read_csv(file_path)
    
    # Check class distribution
    class_counts = df['Class'].value_counts()
    balance = class_counts.min() / class_counts.max() * 100
    
    # Store data for later use - don't add to HTML yet
    return df, class_counts, balance

def prepare_features_and_target(df):
    """Prepare features and target variable"""
    
    # Separate features and target
    feature_columns = [col for col in df.columns if col not in ['Category', 'Class']]
    X = df[feature_columns].copy()
    y = df['Class'].copy()
    
    # Encode target variable
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Handle any infinite or very large values
    X = X.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN values with median for numeric columns
    numeric_columns = X.select_dtypes(include=[np.number]).columns
    X[numeric_columns] = X[numeric_columns].fillna(X[numeric_columns].median())
    
    # Return data without adding to HTML yet
    return X, y_encoded, le, feature_columns

def feature_selection(X, y, k=30):
    """Select top k features using statistical tests"""
    selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
    X_selected = selector.fit_transform(X, y)
    
    # Get selected feature names
    selected_features = X.columns[selector.get_support()].tolist()
    
    return X_selected, selected_features

def train_multiple_models(X_train, X_test, y_train, y_test):
    """Train multiple ML models and compare performance"""
    
    # STANDARDIZED: Use same scaler instance for all models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # STANDARDIZED: All models use same random_state for reproducibility
    RANDOM_STATE = 42
    
    # Define models to compare with standardized parameters
    models = {
        'Logistic Regression': LogisticRegression(
            random_state=RANDOM_STATE, 
            max_iter=1000,
            solver='lbfgs'  # Explicitly set solver for consistency
        ),
        'Random Forest': RandomForestClassifier(
            random_state=RANDOM_STATE, 
            n_estimators=100,
            max_depth=None,  # Explicitly set for consistency
            min_samples_split=2,  # Default but explicit
            min_samples_leaf=1,   # Default but explicit
            n_jobs=-1  # Use all cores for consistent timing
        )
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        # STANDARDIZED: Same training process for all models
        model.fit(X_train_scaled, y_train)
        trained_models[name] = model
        
        # STANDARDIZED: Same prediction process for all models
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # STANDARDIZED: Same evaluation metrics for all models
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate metrics
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),  # Handle edge cases consistently
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'confusion_matrix': cm,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'true_positives': tp,
            'false_positive_rate': fp / (fp + tn) * 100 if (fp + tn) > 0 else 0,
            'false_negative_rate': fn / (fn + tp) * 100 if (fn + tp) > 0 else 0
        }
    
    return trained_models, scaler, results

def compare_models(results, y_test):
    """Compare performance across all models"""
    html_generator.add_title("Model Performance Comparison", 2)
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results).T
    comparison_df = comparison_df[['accuracy', 'precision', 'recall', 'f1']].round(4)
    
    # Add to HTML
    html_generator.add_metrics_table(comparison_df, "Model Performance Metrics")
    html_generator.add_confusion_matrix_table(results)
    
    # Find best models
    metric_columns = {
        'Accuracy': 'accuracy',
        'Precision': 'precision',
        'Recall': 'recall',
        'F1 Score': 'f1'
    }
    
    best_models = {}
    for metric, column in metric_columns.items():
        best_models[metric] = comparison_df[column].idxmax()
    
    # Add best models for error rates
    error_comparison_df = pd.DataFrame({
        model: {
            'fp_rate': results[model]['false_positive_rate'],
            'fn_rate': results[model]['false_negative_rate']
        } for model in results.keys()
    }).T
    
    best_models['Lowest False Positive Rate'] = error_comparison_df['fp_rate'].idxmin()
    best_models['Lowest Miss Rate'] = error_comparison_df['fn_rate'].idxmin()
    
    html_generator.add_best_models_summary(best_models, comparison_df, results)
    
    # Create comparison plots
    fig = plt.figure(figsize=(18, 12))
    
    # Performance comparison
    plt.subplot(2, 3, 1)
    comparison_df.plot(kind='bar', ax=plt.gca())
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    
    # Best model highlight
    plt.subplot(2, 3, 2)
    best_f1_model = comparison_df['f1'].idxmax()
    plt.bar(range(len(comparison_df)), comparison_df['f1'], 
            color=['red' if model == best_f1_model else 'lightblue' for model in comparison_df.index])
    plt.xticks(range(len(comparison_df)), comparison_df.index, rotation=45)
    plt.title(f'F1 Scores (Best: {best_f1_model})')
    plt.ylabel('F1 Score')
    
    # Confusion Matrix
    plt.subplot(2, 3, 3)
    model_names = list(results.keys())
    if len(model_names) == 1:
        cm = results[model_names[0]]['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Benign', 'Malware'], 
                   yticklabels=['Benign', 'Malware'])
        plt.title(f'{model_names[0]}\nConfusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
    
    # Error rates comparison
    plt.subplot(2, 3, 4)
    error_rates = pd.DataFrame({
        model: [results[model]['false_positive_rate'], results[model]['false_negative_rate']]
        for model in results.keys()
    }, index=['False Positive Rate', 'False Negative Rate']).T
    
    error_rates.plot(kind='bar', ax=plt.gca())
    plt.title('Error Rates Comparison')
    plt.ylabel('Error Rate (%)')
    plt.legend()
    plt.xticks(rotation=45)
    
    # Prediction breakdown
    plt.subplot(2, 3, 5)
    if len(model_names) == 1:
        model = model_names[0]
        categories = ['TN\n(Correct\nBenign)', 'FP\n(False\nAlarms)', 
                     'FN\n(Missed\nMalware)', 'TP\n(Caught\nMalware)']
        values = [results[model]['true_negatives'], results[model]['false_positives'],
                 results[model]['false_negatives'], results[model]['true_positives']]
        colors = ['lightgreen', 'orange', 'red', 'darkgreen']
        
        bars = plt.bar(categories, values, color=colors, alpha=0.7)
        plt.title(f'{model}\nPrediction Breakdown')
        plt.ylabel('Count')
        
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(values),
                    str(value), ha='center', va='bottom', fontweight='bold')
    
    # Legend
    plt.subplot(2, 3, 6)
    plt.text(0.1, 0.9, "Confusion Matrix Legend:", fontsize=12, fontweight='bold')
    plt.text(0.1, 0.8, "• TN: True Negatives (Correct benign)", fontsize=10)
    plt.text(0.1, 0.7, "• FP: False Positives (False alarms)", fontsize=10, color='orange')
    plt.text(0.1, 0.6, "• FN: False Negatives (Missed threats)", fontsize=10, color='red')
    plt.text(0.1, 0.5, "• TP: True Positives (Caught threats)", fontsize=10, color='green')
    plt.text(0.1, 0.3, "Key Metrics:", fontsize=12, fontweight='bold')
    plt.text(0.1, 0.2, "• Precision: TP/(TP+FP)", fontsize=10)
    plt.text(0.1, 0.1, "• Recall: TP/(TP+FN)", fontsize=10)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    
    plt.tight_layout()
    html_generator.add_plot(fig, "Model Performance Analysis")
    
    return best_f1_model

def plot_confusion_matrix(y_test, y_pred, le, model_name="Model"):
    """Create and display confusion matrix"""
    html_generator.add_title(f"{model_name} - Confusion Matrix Analysis", 3)
    
    cm = confusion_matrix(y_test, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    fig = plt.figure(figsize=(15, 10))
    
    # Plot confusion matrix counts
    plt.subplot(2, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Confusion Matrix (Counts)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    # Plot confusion matrix percentages
    plt.subplot(2, 2, 2)
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Confusion Matrix (Percentages)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    # Detailed breakdown
    tn, fp, fn, tp = cm.ravel()
    
    plt.subplot(2, 2, 3)
    categories = ['True\nNegatives\n(Correct\nBenign)', 'False\nPositives\n(False\nAlarms)', 
                  'False\nNegatives\n(Missed\nMalware)', 'True\nPositives\n(Caught\nMalware)']
    values = [tn, fp, fn, tp]
    colors = ['lightgreen', 'orange', 'red', 'darkgreen']
    
    bars = plt.bar(categories, values, color=colors, alpha=0.7)
    plt.title('Confusion Matrix Breakdown')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(values),
                str(value), ha='center', va='bottom', fontweight='bold')
    
    # Performance summary
    plt.subplot(2, 2, 4)
    plt.text(0.1, 0.8, "Confusion Matrix Analysis:", fontsize=14, fontweight='bold')
    plt.text(0.1, 0.7, f"• True Negatives (TN):  {tn:,}", fontsize=11)
    plt.text(0.1, 0.6, f"• False Positives (FP): {fp:,}", fontsize=11)
    plt.text(0.1, 0.5, f"• False Negatives (FN): {fn:,}", fontsize=11)
    plt.text(0.1, 0.4, f"• True Positives (TP):  {tp:,}", fontsize=11)
    plt.text(0.1, 0.2, f"False Alarm Rate: {fp/(fp+tn)*100:.2f}%", fontsize=11, color='orange')
    plt.text(0.1, 0.1, f"Miss Rate: {fn/(fn+tp)*100:.2f}%", fontsize=11, color='red')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    
    plt.tight_layout()
    html_generator.add_plot(fig, f"{model_name} Detailed Confusion Matrix")

def cross_validation_analysis(X, y, model, scaler, model_name="Model"):
    """Perform cross-validation analysis"""
    html_generator.add_title(f"{model_name} - Cross-Validation Analysis", 3)
    
    # STANDARDIZED: Use same scaler for CV (create new instance to avoid data leakage)
    cv_scaler = StandardScaler()
    X_scaled = cv_scaler.fit_transform(X)
    
    # STANDARDIZED: Same CV parameters for all models
    CV_FOLDS = 5
    
    # Perform cross-validation with same parameters (removed random_state)
    cv_scores = cross_val_score(
        model, X_scaled, y, 
        cv=CV_FOLDS, 
        scoring='accuracy'
    )
    
    cv_info = f"""
    <div class="info-box">
        <strong>{model_name} - {CV_FOLDS}-Fold Cross-Validation Results:</strong><br>
        • Accuracy scores: {[f'{score:.4f}' for score in cv_scores]}<br>
        • Mean accuracy: {cv_scores.mean():.4f} ± {cv_scores.std()*2:.4f}<br>
        • Performance is {'consistent' if cv_scores.std() < 0.05 else 'variable'} across data splits
    </div>
    """
    html_generator.add_text(cv_info)

def feature_importance_analysis(model, feature_names, model_name="Model"):
    """Analyze feature importance"""
    html_generator.add_title(f"{model_name} - Feature Importance Analysis", 3)
    
    # Check if model has coefficients (like Logistic Regression) or feature_importances (like Random Forest)
    if hasattr(model, 'coef_'):
        # For linear models like Logistic Regression
        coefficients = model.coef_[0]
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': np.abs(coefficients),
            'coefficient': coefficients
        }).sort_values('importance', ascending=False)
        
        # Create feature importance table for HTML
        top_features = feature_importance.head(10)
        html_table = f'<div class="info-box"><strong>{model_name} - Top 10 Most Important Features:</strong><br>'
        for i, (_, row) in enumerate(top_features.iterrows()):
            html_table += f"{i+1}. {row['feature']} (coef: {row['coefficient']:.4f})<br>"
        html_table += '</div>'
        html_generator.add_text(html_table)
        
        # Plot top features
        fig = plt.figure(figsize=(12, 10))
        
        plt.subplot(2, 1, 1)
        top_features_plot = feature_importance.head(15)
        plt.barh(range(len(top_features_plot)), top_features_plot['importance'])
        plt.yticks(range(len(top_features_plot)), top_features_plot['feature'])
        plt.xlabel('Absolute Coefficient Value')
        plt.title(f'{model_name} - Top 15 Most Important Features')
        plt.gca().invert_yaxis()
        
        plt.subplot(2, 1, 2)
        plt.barh(range(len(top_features_plot)), top_features_plot['coefficient'], 
                 color=['red' if x < 0 else 'blue' for x in top_features_plot['coefficient']])
        plt.yticks(range(len(top_features_plot)), top_features_plot['feature'])
        plt.xlabel('Coefficient Value (Red=Benign indicator, Blue=Malware indicator)')
        plt.title(f'{model_name} - Feature Coefficients (Direction of Influence)')
        plt.gca().invert_yaxis()
        
    elif hasattr(model, 'feature_importances_'):
        # For tree-based models like Random Forest
        importances = model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Create feature importance table for HTML
        top_features = feature_importance.head(10)
        html_table = f'<div class="info-box"><strong>{model_name} - Top 10 Most Important Features:</strong><br>'
        for i, (_, row) in enumerate(top_features.iterrows()):
            html_table += f"{i+1}. {row['feature']} (importance: {row['importance']:.4f})<br>"
        html_table += '</div>'
        html_generator.add_text(html_table)
        
        # Plot top features
        fig = plt.figure(figsize=(12, 8))
        
        top_features_plot = feature_importance.head(15)
        plt.barh(range(len(top_features_plot)), top_features_plot['importance'])
        plt.yticks(range(len(top_features_plot)), top_features_plot['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'{model_name} - Top 15 Most Important Features')
        plt.gca().invert_yaxis()
        
    else:
        # For models without feature importance
        html_table = f'<div class="info-box"><strong>{model_name} - Feature importance analysis not available for this model type.</strong></div>'
        html_generator.add_text(html_table)
        return
    
    plt.tight_layout()
    html_generator.add_plot(fig, f"{model_name} - Feature Importance Analysis")

def validate_results(X_train, X_test, y_train, y_test, results):
    """Validate if results are realistic"""
    
    # Check 1: Cross-validation on training data
    from sklearn.model_selection import cross_val_score
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    lr = LogisticRegression(random_state=42, max_iter=1000)
    cv_scores = cross_val_score(lr, X_train_scaled, y_train, cv=5)
    
    # Check 2: Simple baseline
    from sklearn.dummy import DummyClassifier
    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(X_train_scaled, y_train)
    dummy_score = dummy.score(scaler.transform(X_test), y_test)
    
    # Check 3: Feature importance extremes
    rf = RandomForestClassifier(random_state=42, n_estimators=100)
    rf.fit(X_train_scaled, y_train)
    max_importance = rf.feature_importances_.max()
    
    # Add validation info to HTML report only
    validation_info = f"""
    <div class="warning-box">
        <strong>Validation Checks:</strong><br>
        • CV mean accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}<br>
        • Baseline accuracy: {dummy_score:.4f}<br>
        • Max feature importance: {max_importance:.4f}<br>
        {'• WARNING: One feature dominates - possible data leakage!' if max_importance > 0.5 else '• No single feature dominates'}
    </div>
    """
    html_generator.add_text(validation_info)

def main():
    """Main function to run the complete ML pipeline"""
    
    file_path = "edr.csv"
    
    # STANDARDIZED: Set global random seed for reproducibility
    GLOBAL_RANDOM_STATE = 42
    np.random.seed(GLOBAL_RANDOM_STATE)
    
    try:
        html_generator.add_title("Machine Learning Analysis Pipeline")
        
        # Step 1: Load and preprocess data
        df, class_counts, balance = load_and_preprocess_data(file_path)
        
        # Step 2: Prepare features and target
        X, y, label_encoder, feature_columns = prepare_features_and_target(df)
        
        # Step 3: Use ALL features (standardized feature set)
        X_selected = X.copy()
        selected_features = feature_columns
        
        # STANDARDIZED: Same train/test split parameters for all runs
        TEST_SIZE = 0.2  # Changed back to 20% for better training data
        
        # Step 4: Split the data with consistent parameters
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, 
            test_size=TEST_SIZE, 
            random_state=GLOBAL_RANDOM_STATE, 
            stratify=y,  # Maintains class distribution
            shuffle=True  # Explicitly set for consistency
        )
        
        # Add preprocessing info (updated with standardized split)
        preprocessing_boxes = f"""
        <div class="info-box">
            <strong>Dataset loaded successfully!</strong><br>
            • Shape: {df.shape}<br>
            • Total samples: {len(df):,}<br>
            • Number of features: {df.shape[1] - 1}<br>
            • Missing values: {df.isnull().sum().sum()}
        </div>
        <div class="info-box">
            <strong>Class Distribution:</strong><br>
            • {class_counts.index[0]}: {class_counts.iloc[0]:,}<br>
            • {class_counts.index[1]}: {class_counts.iloc[1]:,}<br>
            • Class balance: {balance:.1f}%
        </div>
        <div class="info-box">
            <strong>Feature Preparation:</strong><br>
            • Total features: {len(feature_columns)}<br>
            • Target encoding: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}<br>
            • Data preprocessing: Infinite values handled, missing values filled
        </div>
        <div class="info-box">
            <strong>Standardized Experimental Setup:</strong><br>
            • Training set: {X_train.shape[0]:,} samples ({(1-TEST_SIZE)*100:.0f}%)<br>
            • Test set: {X_test.shape[0]:,} samples ({TEST_SIZE*100:.0f}%)<br>
            • Random state: {GLOBAL_RANDOM_STATE} (for reproducibility)<br>
            • Stratified split: Yes (maintains class balance)<br>
            • Feature scaling: StandardScaler (applied to all models)
        </div>
        """
        
        # Add all boxes at once
        html_generator.html_content.append(preprocessing_boxes)
        
        # Step 5: Train models
        trained_models, scaler, results = train_multiple_models(
            X_train, X_test, y_train, y_test
        )
        
        # ADD THIS VALIDATION
        validate_results(X_train, X_test, y_train, y_test, results)
        
        # Step 6: Compare models
        best_model_name = compare_models(results, y_test)
        
        # Step 7-10: Analyze ALL models instead of just the best
        for model_name, model in trained_models.items():
            y_pred = results[model_name]['predictions']
            
            # Add model-specific section title
            html_generator.add_title(f"{model_name} - Detailed Analysis", 2)
            
            # Plot confusion matrix for this model
            plot_confusion_matrix(y_test, y_pred, label_encoder, model_name)
            
            # Cross-validation analysis for this model
            cross_validation_analysis(X_selected, y, model, scaler, model_name)
            
            # Feature importance analysis for this model
            feature_importance_analysis(model, selected_features, model_name)
            
            # Classification report for this model
            html_generator.add_title(f"{model_name} - Classification Report", 3)
            report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
            
            # Convert classification report to HTML table
            report_df = pd.DataFrame(report).transpose()
            report_df = report_df.round(4)
            html_generator.add_metrics_table(report_df, f"{model_name} Classification Report")
        
        # Step 11: Generate HTML report
        output_file = html_generator.generate_html("results.html")
        print(f"HTML report generated: {output_file}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()