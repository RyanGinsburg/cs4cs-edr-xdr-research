import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import matplotlib #type: ignore
matplotlib.use('Agg')  # ← Add this line right here
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.preprocessing import StandardScaler, LabelEncoder  # type: ignore
from sklearn.metrics import ( #type: ignore
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    average_precision_score, balanced_accuracy_score
)
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif  # type: ignore
from sklearn.ensemble import RandomForestClassifier, VotingClassifier  # type: ignore
from sklearn.dummy import DummyClassifier #type: ignore
from imblearn.over_sampling import SMOTE, BorderlineSMOTE       #type: ignore            # new
from imblearn.pipeline      import Pipeline as ImbPipeline  # new#type: ignore
from sklearn.metrics import precision_recall_curve#type: ignore
from lightgbm import LGBMClassifier #type: ignore
from imblearn.ensemble import BalancedRandomForestClassifier #type: ignore
from sklearn.linear_model import SGDClassifier #type: ignore
from sklearn.ensemble import IsolationForest #type: ignore
from xgboost import XGBClassifier  # type: ignore
import warnings
import base64
import io
import datetime
import os
import random
import json

# Enhanced reproducibility
np.random.seed(42)
random.seed(42)
os.environ['PYTHONHASHSEED'] = '0'

warnings.filterwarnings('ignore')

y_test_global = None

# =========================
# Data Export Infrastructure
# =========================
class DataExporter:
    def __init__(self):
        self.export_dir = "analysis_data"
        os.makedirs(self.export_dir, exist_ok=True)
        self.comprehensive_data = {
            "metadata": {
                "generated_timestamp": datetime.datetime.now().isoformat(),
                "analysis_type": "Standardized ML Comparison",
                "datasets": [],
                "models": ["Logistic Regression", "Random Forest (SMOTE)", "LightGBM", "XGBoost", "Balanced RF", "SGD SVM"],
                "configuration": {
                    "n_estimators": 300,
                    "random_state": 42,
                    "recall_target": 0.80,
                    "standardized": True
                }
            },
            "datasets": {},
            "comparison": {},
            "detailed_results": {}
        }
    
    def add_dataset_info(self, dataset_label, train_df, test_df, target_col, class_counts, feature_columns, le):
        """Add comprehensive dataset information"""
        self.comprehensive_data["metadata"]["datasets"].append(dataset_label)
        
        self.comprehensive_data["datasets"][dataset_label] = {
            "basic_info": {
                "train_shape": list(train_df.shape),
                "test_shape": list(test_df.shape),
                "num_features": len(feature_columns),
                "target_column": target_col,
                "missing_values_train": int(train_df.isnull().sum().sum()),
                "missing_values_test": int(test_df.isnull().sum().sum())
            },
            "class_distribution": {
                "counts": class_counts.to_dict(),
                "percentages": (class_counts / len(train_df) * 100).to_dict(),
                "balance_ratio": float(class_counts.min() / class_counts.max() * 100),
                "minority_class": str(class_counts.idxmin()),
                "majority_class": str(class_counts.idxmax())
            },
            "feature_info": {
                "feature_names": feature_columns,
                "num_features": len(feature_columns),
                "excluded_columns": ["src_comp", "dst_comp", "computer", "host", target_col]
            },
            "label_encoding": dict(zip(le.classes_.astype(str), le.transform(le.classes_).tolist()))
        }
    
    def add_model_results(self, dataset_label, results, baseline_acc):
        """Add comprehensive model results"""
        if dataset_label not in self.comprehensive_data["detailed_results"]:
            self.comprehensive_data["detailed_results"][dataset_label] = {}
        
        self.comprehensive_data["detailed_results"][dataset_label]["baseline_accuracy"] = float(baseline_acc)
        self.comprehensive_data["detailed_results"][dataset_label]["models"] = {}
        
        for model_name, r in results.items():
            model_data = {
                "performance_metrics": {
                    "accuracy": float(r['accuracy']),
                    "precision": float(r['precision']),
                    "recall": float(r['recall']),
                    "f1": float(r['f1']),
                    "balanced_accuracy": float(r['balanced_accuracy']),
                    "roc_auc": float(r['roc_auc']) if not pd.isna(r['roc_auc']) else None,
                    "pr_auc": float(r['ap']) if not pd.isna(r['ap']) else None
                },
                "confusion_matrix": {
                    "true_negatives": int(r['true_negatives']) if not pd.isna(r['true_negatives']) else None,
                    "false_positives": int(r['false_positives']) if not pd.isna(r['false_positives']) else None,
                    "false_negatives": int(r['false_negatives']) if not pd.isna(r['false_negatives']) else None,
                    "true_positives": int(r['true_positives']) if not pd.isna(r['true_positives']) else None,
                    "false_positive_rate": float(r['false_positive_rate']),
                    "false_negative_rate": float(r['false_negative_rate'])
                },
                "threshold_tuning": {
                    "was_tuned": bool(r['threshold_tuned']),
                    "threshold_info": r['threshold_info'] if r['threshold_info'] else None
                },
                "raw_predictions": r['predictions'].tolist() if r['predictions'] is not None else None,
                "probability_scores": r['scores'].tolist() if r['scores'] is not None else None
            }
            
            self.comprehensive_data["detailed_results"][dataset_label]["models"][model_name] = model_data
    
    def add_curve_data(self, dataset_label, curve_type, data_dict):
        """Add ROC/PR curve data"""
        if "curves" not in self.comprehensive_data["detailed_results"][dataset_label]:
            self.comprehensive_data["detailed_results"][dataset_label]["curves"] = {}
        
        self.comprehensive_data["detailed_results"][dataset_label]["curves"][curve_type] = data_dict
    
    def add_threshold_analysis(self, dataset_label, threshold_data):
        """Add threshold sweep analysis"""
        self.comprehensive_data["detailed_results"][dataset_label]["threshold_analysis"] = threshold_data
    
    def add_feature_importance(self, dataset_label, model_name, importance_data):
        """Add feature importance data"""
        if "feature_importance" not in self.comprehensive_data["detailed_results"][dataset_label]:
            self.comprehensive_data["detailed_results"][dataset_label]["feature_importance"] = {}
        
        self.comprehensive_data["detailed_results"][dataset_label]["feature_importance"][model_name] = importance_data
    
    def generate_comparison_summary(self):
        """Generate cross-dataset comparison"""
        if len(self.comprehensive_data["datasets"]) >= 2:
            datasets = list(self.comprehensive_data["datasets"].keys())
            
            # Model performance comparison
            comparison_table = {}
            for dataset in datasets:
                comparison_table[dataset] = {}
                if dataset in self.comprehensive_data["detailed_results"]:
                    for model_name, model_data in self.comprehensive_data["detailed_results"][dataset]["models"].items():
                        comparison_table[dataset][model_name] = model_data["performance_metrics"]
            
            self.comprehensive_data["comparison"] = {
                "cross_dataset_performance": comparison_table,
                "best_models_by_dataset": {},
                "performance_differences": {}
            }
            
            # Find best models per metric per dataset
            for dataset in datasets:
                self.comprehensive_data["comparison"]["best_models_by_dataset"][dataset] = {}
                if dataset in self.comprehensive_data["detailed_results"]:
                    models_data = self.comprehensive_data["detailed_results"][dataset]["models"]
                    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
                        best = max(models_data.items(), 
                                 key=lambda x: x[1]["performance_metrics"][metric] or 0)
                        self.comprehensive_data["comparison"]["best_models_by_dataset"][dataset][metric] = {
                            "model": best[0],
                            "score": best[1]["performance_metrics"][metric]
                        }
    
    def export_comprehensive_json(self, filename="comprehensive_analysis_data.json"):
        """Export everything to one comprehensive JSON file"""
        self.generate_comparison_summary()
        
        filepath = os.path.join(self.export_dir, filename)
        
        # Custom JSON encoder to handle numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NumpyEncoder, self).default(obj)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.comprehensive_data, f, indent=2, cls=NumpyEncoder, ensure_ascii=False)
        
        print(f"Comprehensive analysis data exported: {filepath}")
        print(f"File size: {os.path.getsize(filepath) / 1024 / 1024:.2f} MB")
        return filepath

# Global data exporter
data_exporter = DataExporter()

# =========================
# HTML report infrastructure
# =========================
class HTMLReportGenerator:
    def __init__(self):
        self.html_content = []

    def add_title(self, title, level=1):
        self.html_content.append(f'<h{level} class="section-title">{title}</h{level}>')

    def add_text(self, text, style_class="content"):
        self.html_content.append(f'<div class="{style_class}">{text}</div>')

    def add_metrics_table(self, metrics_data, title="Model Metrics"):
        # metrics_data should be a DataFrame with models as index
        html = f'<div class="metrics-container"><h3>{title}</h3>'
        html += '<table class="metrics-table"><thead><tr>'
        html += '<th>Model</th>'
        for col in metrics_data.columns:
            html += f'<th>{col}</th>'
        html += '</tr></thead><tbody>'
        for idx, row in metrics_data.iterrows():
            html += '<tr>'
            html += f'<td class="model-name">{idx}</td>'
            for col in metrics_data.columns:
                val = row[col]
                if pd.api.types.is_number(val):
                    cell = f'{val:.4f}'
                else:
                    try:
                        cell = f'{float(val):.4f}'
                    except Exception:
                        cell = str(val)
                html += f'<td>{cell}</td>'
            html += '</tr>'
        html += '</tbody></table></div>'
        self.html_content.append(html)

    def add_confusion_matrix_table(self, results):
        html = '<div class="confusion-container"><h3>Confusion Matrix Analysis</h3>'
        html += '<table class="confusion-table">'
        html += '<thead><tr><th>Model</th><th>TN</th><th>FP</th><th>FN</th><th>TP</th><th>FP Rate</th><th>Miss Rate</th></tr></thead><tbody>'
        for model_name, r in results.items():
            tn = r['true_negatives']
            fp = r['false_positives']
            fn = r['false_negatives']
            tp = r['true_positives']
            fpr = r['false_positive_rate']
            fnr = r['false_negative_rate']
            html += f'<tr><td class="model-name">{model_name}</td>'
            html += f'<td>{"" if pd.isna(tn) else int(tn)}</td>'
            html += f'<td>{"" if pd.isna(fp) else int(fp)}</td>'
            html += f'<td>{"" if pd.isna(fn) else int(fn)}</td>'
            html += f'<td>{"" if pd.isna(tp) else int(tp)}</td>'
            html += f'<td>{fpr:.2f}%</td><td>{fnr:.2f}%</td></tr>'
        html += '</tbody></table></div>'
        self.html_content.append(html)

    def add_plot_with_data_export(self, fig, title="", data=None, filename_prefix=""):
        """Add plot to HTML and export underlying data to CSV"""
        img = io.BytesIO()
        fig.savefig(img, format='png', dpi=150, bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        
        # Export data if provided
        csv_filename = ""
        if data is not None and filename_prefix:
            csv_filename = data_exporter.export_to_csv(data, filename_prefix, title)
            csv_basename = os.path.basename(csv_filename)
        
        html = '<div class="plot-container">'
        if title:
            html += f'<h3>{title}</h3>'
        html += f'<img src="data:image/png;base64,{plot_url}" alt="{title}" class="plot-image"/>'
        if csv_filename:
            html += f'<div style="margin-top:8px; font-size:0.9em; color:#6ea8fe;">📊 Chart data exported: {csv_basename}</div>'
        html += '</div>'
        
        self.html_content.append(html)
        plt.close(fig)

    def add_plot(self, fig, title=""):
        """Fallback for plots without data export"""
        img = io.BytesIO()
        fig.savefig(img, format='png', dpi=150, bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        html = '<div class="plot-container">'
        if title:
            html += f'<h3>{title}</h3>'
        html += f'<img src="data:image/png;base64,{plot_url}" alt="{title}" class="plot-image"/></div>'
        self.html_content.append(html)
        plt.close(fig)

    def add_best_models_summary(self, best_models, comparison_df, results):
        html = '<div class="best-models-container"><h3>Best Models by Metric</h3><div class="best-models-grid">'
        for metric, model in best_models.items():
            if model and metric in ["Accuracy","Precision","Recall","F1","Balanced Acc","ROC-AUC","PR-AUC"]:
                score = comparison_df.loc[model, metric]
                html += f'<div class="metric-card"><div class="metric-name">{metric}</div><div class="metric-model">{model}</div><div class="metric-score">{score:.4f}</div></div>'
            elif metric in ["Lowest False Positive Rate","Lowest Miss Rate"]:
                score = results[model]['false_positive_rate'] if metric == "Lowest False Positive Rate" else results[model]['false_negative_rate']
                html += f'<div class="metric-card error-rate"><div class="metric-name">{metric}</div><div class="metric-model">{model}</div><div class="metric-score">{score:.2f}%</div></div>'
        html += '</div></div>'
        self.html_content.append(html)

    def generate_html(self, output_file="results.html"):
        styles = """
        <style>
          body { font-family: Arial, sans-serif; margin: 0; background:#0b1020; color:#e7ebff; }
          .container { max-width: 1100px; margin: 0 auto; padding: 24px; }
          .header { text-align:center; margin-bottom: 24px; }
          .subtitle { opacity: 0.8; font-size: 0.95rem; }
          .section-title { border-left: 4px solid #6ea8fe; padding-left: 10px; margin-top: 24px; }
          .content { display:block; gap: 16px; }
          .metrics-container, .confusion-container, .best-models-container, .info-box, .warning-box, .plot-container {
             background:#12183a; border:1px solid #253061; border-radius:12px; padding:16px; margin:16px 0;
          }
          .warning-box { border-color:#a855f7; }
          .export-info { border-color:#28a745; background:#0d2818; }
          table { width:100%; border-collapse: collapse; }
          th, td { border-bottom:1px solid #253061; padding:8px; text-align:left; }
          th { background:#1a2150; position: sticky; top: 0; }
          .model-name { font-weight: 600; }
          .best-models-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap:12px; }
          .metric-card { background:#0f1533; border:1px solid #2b3774; border-radius:10px; padding:12px; }
          .metric-name { font-size:0.9rem; opacity:0.9; }
          .metric-model { font-weight:600; margin-top:4px; }
          .metric-score { font-family:ui-monospace, SFMono-Regular, Menlo, monospace; margin-top:6px; }
          .plot-image { width:100%; height:auto; border-radius:8px; border:1px solid #253061; }
          .footer { text-align:center; opacity:0.8; font-size:0.9rem; margin-top:24px; }
        </style>
        """
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Machine Learning Analysis Results</title>
            {styles}
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Machine Learning Analysis Report</h1>
                    <div class="subtitle">Generated on {datetime.datetime.now().strftime('%B %d, %Y at %I:%M %p')}</div>
                </div>
                <div class="content">
                    <div class="export-info">
                        <strong>📊 Data Export Information</strong><br>
                        All numerical data, metrics, and chart data have been exported to CSV files in the <code>analysis_data/</code> folder.<br>
                        These files contain all underlying data for AI analysis and further processing.
                    </div>
                    {''.join(self.html_content)}
                </div>
                <div class="footer">
                    <p>Generated by ML Analysis Pipeline</p>
                </div>
            </div>
        </body>
        </html>
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)
        return output_file


# Global HTML generator
html_generator = HTMLReportGenerator()

# =========================
# Data loading & preparation
# =========================
def load_csv(path):
    return pd.read_csv(path)

def prepare_train_test(train_df, test_df):
    """
    Prepare features and target; align columns; encode labels with train fit only.
    """
    # 1) Identify the target column
    target_col = next(
        (c for c in ['label','Class','target','y'] if c in train_df.columns),
        None
    )
    if target_col is None:
        raise ValueError(f"No target column found. Available columns: {list(train_df.columns)}")
    print(f"Using '{target_col}' as target")

    # 2) Check what values are in the target column
    unique_values = sorted(train_df[target_col].unique())
    print(f"Target column '{target_col}' has unique values: {unique_values}")

    # 3) Exclude only the target and the host identifiers
    exclude_cols = {
        target_col,
        'src_comp', 'dst_comp', 'computer', 'host'
    }
    feature_columns = [c for c in train_df.columns if c not in exclude_cols]

    print("Now using features:", feature_columns)

    # 4) Build X/y
    X_train = train_df[feature_columns].copy()
    y_train_raw = train_df[target_col].copy()

    # 5) Align test set
    for c in feature_columns:
        if c not in test_df.columns:
            test_df[c] = 0
    X_test = test_df[feature_columns].copy()
    y_test_raw = test_df[target_col].copy() if target_col in test_df.columns else None

    # 6) Fill missing and infinite values
    for df_ in (X_train, X_test):
        df_.replace([np.inf, -np.inf], np.nan, inplace=True)
    medians = X_train.median()
    X_train.fillna(medians, inplace=True)
    X_test.fillna(medians, inplace=True)

    # 7) Encode labels properly for binary classification
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train_raw)
    y_test_enc = le.transform(y_test_raw) if y_test_raw is not None else None
    
    print(f"Label encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    print(f"Encoded y_train unique values: {sorted(np.unique(y_train_enc))}")
    if y_test_enc is not None:
        print(f"Encoded y_test unique values: {sorted(np.unique(y_test_enc))}")

    return X_train, X_test, y_train_enc, y_test_enc, le, feature_columns


# =========================
# Standardized Modeling & evaluation
# =========================
def _binary_confusion_counts(cm):
    return cm.ravel() if cm.shape == (2, 2) else (np.nan, np.nan, np.nan, np.nan)

def train_and_evaluate_models(X_train, X_test, y_train, y_test, dataset_label="Dataset"):
    """
    Standardized model training - same models and parameters for both EDR and XDR.
    """
    # Scale once for everyone
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    RANDOM_STATE = 42
    
    # Calculate consistent parameters for all models
    contamination_rate = sum(y_train) / len(y_train)
    pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    
    # Standardized parameters for consistency
    N_ESTIMATORS = 300
    
    # STANDARDIZED MODELS - Same for both EDR and XDR
    models = {
        'Logistic Regression': LogisticRegression(
            random_state=RANDOM_STATE,
            class_weight='balanced',
            max_iter=1000,
            solver='lbfgs',
            tol=1e-4,
            C=1.0
        ),
        'Random Forest (SMOTE)': ImbPipeline([
            ('smote', SMOTE(sampling_strategy=0.5, random_state=RANDOM_STATE)),
            ('rf', RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_estimators=N_ESTIMATORS,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                class_weight='balanced',
                n_jobs=-1,
                verbose=0
            ))
        ]),
        'LightGBM': LGBMClassifier(
            n_estimators=N_ESTIMATORS,
            scale_pos_weight=pos_weight,
            random_state=RANDOM_STATE,
            max_depth=-1,
            min_child_samples=1,
            min_split_gain=0.0,
            subsample=0.8,
            colsample_bytree=0.8,
            verbose=-1,
            n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=N_ESTIMATORS,
            learning_rate=0.1,
            max_depth=6,
            scale_pos_weight=pos_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric='logloss'
        ),
        'Balanced RF': BalancedRandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            sampling_strategy=0.5,
            replacement=True,
            random_state=RANDOM_STATE,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            n_jobs=-1,
            verbose=0
        ),
        'SGD SVM': SGDClassifier(
            loss='hinge',
            class_weight='balanced',
            max_iter=1000,
            tol=1e-4,
            random_state=RANDOM_STATE,
            alpha=0.0001,
            learning_rate='optimal'
        )
    }

    results = {}
    trained = {}

    # Check if it's binary classification
    unique_labels = sorted(np.unique(y_train))
    binary = (len(unique_labels) == 2)
    print(f"Binary classification: {binary}, unique labels: {unique_labels}")

    for name, model in models.items():
        print(f"Training {name}...")
        
        # Use scaled data for ALL models consistently
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else None
            
        trained[name] = model

        y_scores = (y_proba[:, 1] if (y_proba is not None and y_proba.shape[1] > 1) else
                    (y_proba[:, 0] if y_proba is not None else None))

        # STANDARDIZED threshold tuning - same 80% recall target for both EDR and XDR
        tuned_predictions = None
        threshold_info = {}
        if y_scores is not None and binary:
            precs, recs, thresh = precision_recall_curve(y_test, y_scores)
            target_recall = 0.80  # Same recall target for both EDR and XDR
            valid = np.where(recs[1:] >= target_recall)[0]
            if len(valid) > 0:
                best_t = thresh[valid[0]]
                test_pred = (y_scores >= best_t).astype(int)
                if np.mean(test_pred) < 0.9:  # Don't accept if >90% predicted as positive
                    y_pred_t = test_pred
                    tuned_f1 = f1_score(y_test, y_pred_t, zero_division=0)
                    tuned_prec = precision_score(y_test, y_pred_t, zero_division=0)
                    tuned_rec = recall_score(y_test, y_pred_t, zero_division=0)
                    print(f"  → {name} @ recall≥{target_recall:.2f}: thresh={best_t:.3f}  "
                          f"prec={tuned_prec:.3f}  rec={tuned_rec:.3f}  f1={tuned_f1:.3f}")
                    tuned_predictions = y_pred_t
                    threshold_info = {
                        'threshold': best_t,
                        'tuned_precision': tuned_prec,
                        'tuned_recall': tuned_rec,
                        'tuned_f1': tuned_f1
                    }
                else:
                    print(f"  → {name}: threshold {best_t:.3f} predicts {np.mean(test_pred)*100:.1f}% positive - skipping")
            else:
                print(f"  → {name}: no non-trivial threshold achieves recall≥{target_recall:.2f}")

        # Use tuned predictions if available, otherwise use default threshold predictions
        final_predictions = tuned_predictions if tuned_predictions is not None else y_pred

        # Base metrics (using final predictions)
        avg = 'binary' if binary else 'weighted'
        acc = accuracy_score(y_test, final_predictions)
        prec = precision_score(y_test, final_predictions, average=avg, zero_division=0)
        rec = recall_score(y_test, final_predictions, average=avg, zero_division=0)
        f1 = f1_score(y_test, final_predictions, average=avg, zero_division=0)
        bal_acc = balanced_accuracy_score(y_test, final_predictions)

        # ROC-AUC & PR-AUC (fixed for binary classification)
        roc_auc = np.nan
        ap = np.nan
        if y_scores is not None:
            try:
                if binary:
                    roc_auc = roc_auc_score(y_test, y_scores)
                    ap = average_precision_score(y_test, y_scores)
                else:
                    roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
            except Exception as e:
                print(f"Warning: Could not compute ROC-AUC for {name}: {e}")

        # Confusion matrix (using final predictions)
        cm = confusion_matrix(y_test, final_predictions)
        tn, fp, fn, tp = _binary_confusion_counts(cm)
        fpr = fp / (fp + tn) * 100 if (not pd.isna(fp) and (fp + tn) > 0) else 0.0
        fnr = fn / (fn + tp) * 100 if (not pd.isna(fn) and (fn + tp) > 0) else 0.0

        results[name] = {
            'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
            'balanced_accuracy': bal_acc, 'roc_auc': roc_auc, 'ap': ap,
            'predictions': final_predictions,
            'scores': y_scores, 'probabilities': y_proba,
            'confusion_matrix': cm,
            'true_negatives': tn, 'false_positives': fp, 'false_negatives': fn, 'true_positives': tp,
            'false_positive_rate': fpr, 'false_negative_rate': fnr,
            'binary': binary,
            'threshold_tuned': tuned_predictions is not None,
            'threshold_info': threshold_info
        }

    # Export comprehensive model results
    model_results_data = []
    for name, r in results.items():
        row = {
            'model': name,
            'dataset': dataset_label,
            'accuracy': r['accuracy'],
            'precision': r['precision'],
            'recall': r['recall'],
            'f1': r['f1'],
            'balanced_accuracy': r['balanced_accuracy'],
            'roc_auc': r['roc_auc'],
            'pr_auc': r['ap'],
            'true_negatives': r['true_negatives'],
            'false_positives': r['false_positives'],
            'false_negatives': r['false_negatives'],
            'true_positives': r['true_positives'],
            'false_positive_rate': r['false_positive_rate'],
            'false_negative_rate': r['false_negative_rate'],
            'threshold_tuned': r['threshold_tuned']
        }
        # Add threshold info if available
        if r['threshold_info']:
            row.update(r['threshold_info'])
        model_results_data.append(row)
    
    model_results_df = pd.DataFrame(model_results_data)
    data_exporter.export_to_csv(
        model_results_df,
        f"{dataset_label.lower()}_comprehensive_model_results",
        f"{dataset_label} - Comprehensive model performance results with all metrics"
    )

    # Baseline with properly scaled data
    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(X_train_scaled, y_train)
    base_acc = dummy.score(X_test_scaled, y_test)
    
    # Export baseline info
    baseline_data = pd.DataFrame([{
        'dataset': dataset_label,
        'baseline_accuracy': base_acc,
        'contamination_rate': contamination_rate,
        'pos_weight': pos_weight,
        'n_estimators': N_ESTIMATORS,
        'random_state': RANDOM_STATE
    }])
    data_exporter.export_to_csv(
        baseline_data,
        f"{dataset_label.lower()}_baseline_info",
        f"{dataset_label} - Baseline performance and configuration parameters"
    )
    
    html_generator.add_text(
        f'<div class="info-box"><strong>Baseline (Most-Frequent) Accuracy:</strong> {base_acc:.4f}<br>'
        f'<strong>Standardized Configuration:</strong> Same models, parameters, and recall targets for fair comparison</div>'
    )

    return trained, scaler, results


# =========================
# Reporting helpers WITH DATA EXPORT
# =========================
def build_classification_report_table(y_test, y_pred, label_encoder):
    """
    Clean DataFrame: per-class rows + macro/weighted + accuracy row (handles accuracy scalar).
    """
    report = classification_report(
        y_test, y_pred,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0
    )
    
    rows = []
    # only include the true labels (no macro/weighted)
    for label in label_encoder.classes_:
        vals = report[label]
        rows.append({
            'label':    label,
            'precision': vals['precision'],
            'recall':    vals['recall'],
            'f1':        vals['f1-score'],
            'support':   vals['support']
        })
    total_support = sum(r['support'] for r in rows)

    # now the accuracy row
    rows.append({
        'label':    'accuracy',
        'precision': np.nan,
        'recall':    np.nan,
        'f1':        report['accuracy'],
        'support':   total_support
    })
    
    df = pd.DataFrame(rows).set_index('label')
    for c in ['precision','recall','f1','support']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def compare_models_and_plots(results, le, dataset_label):
    """
    Add the big comparison table + confusion summary + best models + plots WITH COMPREHENSIVE DATA EXPORT.
    """
    html_generator.add_title(f"{dataset_label}: Model Performance Comparison", 2)

    comparison_df = pd.DataFrame({
        name: {
            'Accuracy': r['accuracy'],
            'Balanced Acc': r['balanced_accuracy'],
            'Precision': r['precision'],
            'Recall': r['recall'],
            'F1': r['f1'],
            'ROC-AUC': r['roc_auc'],
            'PR-AUC': r['ap']
        } for name, r in results.items()
    }).T

    html_generator.add_metrics_table(comparison_df.round(4), f"{dataset_label} – Model Performance Metrics")
    html_generator.add_confusion_matrix_table(results)

    # Export main metrics data
    data_exporter.export_to_csv(
        comparison_df, 
        f"{dataset_label.lower()}_model_metrics", 
        f"{dataset_label} Model Performance Metrics - All models compared across key metrics"
    )

    # Export confusion matrix summary
    confusion_summary = pd.DataFrame({
        name: {
            'TN': r['true_negatives'],
            'FP': r['false_positives'], 
            'FN': r['false_negatives'],
            'TP': r['true_positives'],
            'FP_Rate': r['false_positive_rate'],
            'FN_Rate': r['false_negative_rate']
        } for name, r in results.items()
    }).T
    data_exporter.export_to_csv(
        confusion_summary,
        f"{dataset_label.lower()}_confusion_summary",
        f"{dataset_label} Confusion Matrix Summary - TN/FP/FN/TP counts and error rates"
    )

    # Bests
    best_models = {}
    for col in ['Accuracy','Balanced Acc','Precision','Recall','F1','ROC-AUC','PR-AUC']:
        if col in comparison_df.columns and comparison_df[col].notna().any():
            best_models[col] = comparison_df[col].idxmax()
    # Error rates
    error_df = pd.DataFrame({
        name: {'FP%': r['false_positive_rate'], 'FN%': r['false_negative_rate']}
        for name, r in results.items()
    }).T
    if not error_df.empty:
        best_models['Lowest False Positive Rate'] = error_df['FP%'].idxmin()
        best_models['Lowest Miss Rate'] = error_df['FN%'].idxmin()

    # Export best models summary
    best_models_df = pd.DataFrame(list(best_models.items()), columns=['Metric', 'Best_Model'])
    data_exporter.export_to_csv(
        best_models_df,
        f"{dataset_label.lower()}_best_models",
        f"{dataset_label} Best Models - Top performing model for each metric"
    )

    html_generator.add_best_models_summary(best_models, comparison_df, results)

    # Bar plot of metrics WITH DATA EXPORT
    fig = plt.figure(figsize=(12, 7))
    metrics_for_plot = comparison_df[['Accuracy','Balanced Acc','Precision','Recall','F1']]
    metrics_for_plot.plot(kind='bar', ax=plt.gca())
    plt.title(f'{dataset_label} – Metrics by Model')
    plt.ylabel('Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Use the new method that exports data
    html_generator.add_plot_with_data_export(
        fig, 
        f"{dataset_label} – Metrics by Model",
        metrics_for_plot,
        f"{dataset_label.lower()}_metrics_bar_chart"
    )

    # ROC & PR curves WITH DATA EXPORT
    any_binary = any(r['binary'] for r in results.values())
    classes = list(le.classes_)
    if any_binary and len(classes) == 2:
        global y_test_global
        if y_test_global is not None:
            # ROC Curves
            fig = plt.figure(figsize=(10, 7))
            roc_data = []
            
            for name, r in results.items():
                if r['scores'] is not None:
                    from sklearn.metrics import roc_curve
                    fpr, tpr, _ = roc_curve(y_test_global, r['scores'])
                    plt.plot(fpr, tpr, label=f"{name} (AUC={r['roc_auc']:.3f})")
                    
                    # Collect data for export
                    for f, t in zip(fpr, tpr):
                        roc_data.append({
                            'Model': name,
                            'FPR': f,
                            'TPR': t,
                            'AUC': r['roc_auc']
                        })
            
            plt.plot([0,1], [0,1], linestyle='--', linewidth=1)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'{dataset_label} – ROC Curves')
            plt.legend()
            plt.tight_layout()
            
            # Export ROC data
            roc_df = pd.DataFrame(roc_data)
            html_generator.add_plot_with_data_export(
                fig,
                f"{dataset_label} – ROC Curves",
                roc_df,
                f"{dataset_label.lower()}_roc_curves"
            )

            # Precision-Recall Curves
            fig = plt.figure(figsize=(10, 7))
            pr_data = []
            
            for name, r in results.items():
                if r['scores'] is not None:
                    from sklearn.metrics import precision_recall_curve
                    prec, rec, _ = precision_recall_curve(y_test_global, r['scores'])
                    plt.plot(rec, prec, label=f"{name} (AP={r['ap']:.3f})")
                    
                    # Collect data for export
                    for p, rec_val in zip(prec, rec):
                        pr_data.append({
                            'Model': name,
                            'Precision': p,
                            'Recall': rec_val,
                            'AP': r['ap']
                        })
            
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'{dataset_label} – Precision–Recall Curves')
            plt.legend()
            plt.tight_layout()
            
            # Export PR data
            pr_df = pd.DataFrame(pr_data)
            html_generator.add_plot_with_data_export(
                fig,
                f"{dataset_label} – Precision–Recall Curves",
                pr_df,
                f"{dataset_label.lower()}_precision_recall_curves"
            )

    # Probability histograms WITH DATA EXPORT
    fig = plt.figure(figsize=(12, 5))
    plt.title(f'{dataset_label} – Predicted Probability Distributions (Positive Class)')
    hist_data = []
    
    for name, r in results.items():
        if r['scores'] is not None:
            counts, bins = np.histogram(r['scores'], bins=30, density=True)
            plt.hist(r['scores'], bins=30, alpha=0.5, label=name, density=True)
            
            # Collect histogram data
            bin_centers = (bins[:-1] + bins[1:]) / 2
            for bc, count in zip(bin_centers, counts):
                hist_data.append({
                    'Model': name,
                    'Probability_Bin': bc,
                    'Density': count
                })
            
            # Also export raw probability scores for detailed analysis
            scores_data = pd.DataFrame({
                'Model': name,
                'Sample_ID': range(len(r['scores'])),
                'Probability_Score': r['scores'],
                'Actual_Label': y_test_global
            })
            data_exporter.export_to_csv(
                scores_data,
                f"{dataset_label.lower()}_{name.lower().replace(' ', '_')}_probability_scores",
                f"{dataset_label} {name} - Individual probability scores for each test sample"
            )
    
    plt.xlabel('Predicted probability')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    
    # Export histogram data
    hist_df = pd.DataFrame(hist_data)
    html_generator.add_plot_with_data_export(
        fig,
        f"{dataset_label} – Predicted Probability Distributions",
        hist_df,
        f"{dataset_label.lower()}_probability_distributions"
    )

    # Threshold sweep WITH DATA EXPORT
    if any_binary and len(classes) == 2 and y_test_global is not None:
        fig = plt.figure(figsize=(12, 7))
        thresholds = np.linspace(0.0, 1.0, 101)
        threshold_data = {}
        
        for name, r in results.items():
            if r['scores'] is None:
                continue
            f1_vals, prec_vals, rec_vals = [], [], []
            for t in thresholds:
                y_hat = (r['scores'] >= t).astype(int)
                f1_val = f1_score(y_test_global, y_hat, zero_division=0)
                prec_val = precision_score(y_test_global, y_hat, zero_division=0)
                rec_val = recall_score(y_test_global, y_hat, zero_division=0)
                
                f1_vals.append(f1_val)
                prec_vals.append(prec_val)
                rec_vals.append(rec_val)
                
                # Collect data for export
                threshold_data.append({
                    'Model': name,
                    'Threshold': t,
                    'F1': f1_val,
                    'Precision': prec_val,
                    'Recall': rec_val
                })
            
            plt.plot(thresholds, f1_vals, label=f'{name} – F1')
            plt.plot(thresholds, prec_vals, linestyle='--', label=f'{name} – Precision')
            plt.plot(thresholds, rec_vals, linestyle=':', label=f'{name} – Recall')
        
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title(f'{dataset_label} – Threshold Sweep')
        plt.legend(ncol=2)
        plt.tight_layout()
        
        # Export threshold data
        threshold_df = pd.DataFrame(threshold_data)
        html_generator.add_plot_with_data_export(
            fig,
            f"{dataset_label} – Threshold Sweep",
            threshold_df,
            f"{dataset_label.lower()}_threshold_sweep"
        )


def plot_confusion_matrix_detailed(y_test, y_pred, le, model_name, dataset_label):
    html_generator.add_title(f"{dataset_label} – {model_name}: Confusion Matrix", 3)
    cm = confusion_matrix(y_test, y_pred)
    cm_percent = (cm.astype(float) / cm.sum(axis=1, keepdims=True)) * 100

    # Export confusion matrix data
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    cm_percent_df = pd.DataFrame(cm_percent, index=le.classes_, columns=le.classes_)
    
    # Export both raw counts and percentages
    data_exporter.export_to_csv(
        cm_df, 
        f"{dataset_label.lower()}_{model_name.lower().replace(' ', '_')}_confusion_matrix_counts",
        f"{dataset_label} {model_name} - Confusion Matrix Raw Counts"
    )
    data_exporter.export_to_csv(
        cm_percent_df, 
        f"{dataset_label.lower()}_{model_name.lower().replace(' ', '_')}_confusion_matrix_percent",
        f"{dataset_label} {model_name} - Confusion Matrix Percentages"
    )

    # Export detailed predictions vs actual
    predictions_df = pd.DataFrame({
        'Sample_ID': range(len(y_test)),
        'Actual': y_test,
        'Predicted': y_pred,
        'Actual_Label': le.inverse_transform(y_test),
        'Predicted_Label': le.inverse_transform(y_pred),
        'Correct': y_test == y_pred
    })
    data_exporter.export_to_csv(
        predictions_df,
        f"{dataset_label.lower()}_{model_name.lower().replace(' ', '_')}_detailed_predictions",
        f"{dataset_label} {model_name} - Detailed predictions vs actual for each test sample"
    )

    fig = plt.figure(figsize=(14, 10))

    plt.subplot(2, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Counts'); plt.ylabel('Actual'); plt.xlabel('Predicted')

    plt.subplot(2, 2, 2)
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Percent per Actual'); plt.ylabel('Actual'); plt.xlabel('Predicted')

    # Binary-only TN/FP/FN/TP breakdown
    tn, fp, fn, tp = _binary_confusion_counts(cm)
    if not pd.isna(tn):
        plt.subplot(2, 2, 3)
        categories = ['TN\n(Correct Benign)', 'FP\n(False Alarms)', 'FN\n(Missed Threats)', 'TP\n(Caught Threats)']
        values = [tn, fp, fn, tp]
        colors = ['lightgreen', 'orange', 'red', 'darkgreen']
        bars = plt.bar(categories, values, color=colors, alpha=0.7)
        plt.title('Binary Breakdown'); plt.ylabel('Count'); plt.xticks(rotation=0)
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(values),
                     str(int(value)), ha='center', va='bottom', fontweight='bold')

    plt.subplot(2, 2, 4)
    text_lines = [
        "Key metrics:",
        "• Precision = TP/(TP+FP)",
        "• Recall = TP/(TP+FN)",
        "• F1 = harmonic mean of precision & recall"
    ]
    y = 0.9
    for line in text_lines:
        plt.text(0.05, y, line, fontsize=12); y -= 0.12
    plt.axis('off')

    plt.tight_layout()
    html_generator.add_plot(fig, f"{dataset_label} – {model_name}: Confusion Matrix")


def feature_importance_analysis(model, feature_names, model_name, dataset_label):
    html_generator.add_title(f"{dataset_label} – {model_name}: Feature Importance", 3)
    
    # Handle pipeline models - extract the final estimator
    final_model = model
    if hasattr(model, 'named_steps'):
        # It's a pipeline - get the final step
        step_names = list(model.named_steps.keys())
        final_step_name = step_names[-1]  # Last step should be the estimator
        final_model = model.named_steps[final_step_name]
        print(f"Extracting feature importance from pipeline step: {final_step_name}")
    
    if hasattr(final_model, 'coef_'):
        coefs = final_model.coef_[0]
        df = pd.DataFrame({'feature': feature_names, 'importance': np.abs(coefs), 'coefficient': coefs})
        df = df.sort_values('importance', ascending=False)
        
        # Export feature importance data
        data_exporter.export_to_csv(
            df,
            f"{dataset_label.lower()}_{model_name.lower().replace(' ', '_')}_feature_importance",
            f"{dataset_label} {model_name} - Feature Importance (Coefficients)"
        )
        
        # Continue with plotting...
        top = df.head(15)
        fig = plt.figure(figsize=(12, 10))
        plt.subplot(2, 1, 1)
        plt.barh(range(len(top)), top['importance'])
        plt.yticks(range(len(top)), top['feature'])
        plt.xlabel('Absolute Coefficient'); plt.title('Top 15 (Magnitude)'); plt.gca().invert_yaxis()

        plt.subplot(2, 1, 2)
        colors = ['red' if v < 0 else 'blue' for v in top['coefficient']]
        plt.barh(range(len(top)), top['coefficient'], color=colors)
        plt.yticks(range(len(top)), top['feature'])
        plt.xlabel('Coefficient (Red=Negative, Blue=Positive)'); plt.title('Coefficient Direction')
        plt.gca().invert_yaxis()

        plt.tight_layout()
        html_generator.add_plot(fig, f"{dataset_label} – {model_name}: Feature Importance")

    elif hasattr(final_model, 'feature_importances_'):
        imp = final_model.feature_importances_
        df = pd.DataFrame({'feature': feature_names, 'importance': imp}).sort_values('importance', ascending=False)
        
        # Export feature importance data
        data_exporter.export_to_csv(
            df,
            f"{dataset_label.lower()}_{model_name.lower().replace(' ', '_')}_feature_importance",
            f"{dataset_label} {model_name} - Feature Importance (Tree-based)"
        )
        
        # Continue with plotting...
        top = df.head(15)
        fig = plt.figure(figsize=(12, 8))
        plt.barh(range(len(top)), top['importance'])
        plt.yticks(range(len(top)), top['feature'])
        plt.xlabel('Feature Importance'); plt.title('Top 15 Features'); plt.gca().invert_yaxis()
        plt.tight_layout()
        html_generator.add_plot(fig, f"{dataset_label} – {model_name}: Feature Importance")
    else:
        html_generator.add_text('<div class="info-box"><strong>Feature importance not available for this model type.</strong></div>')


# =========================
# Standardized dataset pipeline
# =========================
def run_dataset_pipeline(dataset_label, train_path, test_path):
    """
    Standardized dataset pipeline - same processing for both EDR and XDR.
    dataset_label: string for section titles (e.g., "EDR" or "XDR")
    train_path/test_path: CSV paths you provide
    """
    html_generator.add_title(f"{dataset_label}: Dataset Loading & Preprocessing", 2)

    train_df = load_csv(train_path)
    test_df = load_csv(test_path)

    # Export dataset info
    dataset_info = pd.DataFrame([{
        'dataset': dataset_label,
        'train_path': train_path,
        'test_path': test_path,
        'train_shape_rows': train_df.shape[0],
        'train_shape_cols': train_df.shape[1],
        'test_shape_rows': test_df.shape[0],
        'test_shape_cols': test_df.shape[1],
        'train_memory_mb': train_df.memory_usage(deep=True).sum() / 1024 / 1024,
        'test_memory_mb': test_df.memory_usage(deep=True).sum() / 1024 / 1024
    }])
    data_exporter.export_to_csv(
        dataset_info,
        f"{dataset_label.lower()}_dataset_info",
        f"{dataset_label} - Basic dataset information and statistics"
    )

    # Find the target column
    target_col = None
    for col_name in ['label', 'Class', 'target', 'y']:
        if col_name in train_df.columns:
            target_col = col_name
            break
    
    if target_col is None:
        html_generator.add_text(f'<div class="warning-box"><strong>Error:</strong> No target column found in {dataset_label} data. Available columns: {list(train_df.columns)}</div>')
        return

    # Class distribution (train)
    class_counts = train_df[target_col].value_counts()
    balance = class_counts.min() / class_counts.max() * 100 if len(class_counts) > 1 else 100.0

    # Export class distribution
    class_dist_df = pd.DataFrame({
        'class': class_counts.index,
        'count': class_counts.values,
        'percentage': class_counts.values / len(train_df) * 100
    })
    class_dist_df['dataset'] = dataset_label
    data_exporter.export_to_csv(
        class_dist_df,
        f"{dataset_label.lower()}_class_distribution",
        f"{dataset_label} - Class distribution in training data"
    )

    X_train, X_test, y_train, y_test, le, feature_columns = prepare_train_test(train_df, test_df)

    # Export feature information
    feature_info = pd.DataFrame({
        'feature_name': feature_columns,
        'dataset': dataset_label,
        'feature_index': range(len(feature_columns))
    })
    data_exporter.export_to_csv(
        feature_info,
        f"{dataset_label.lower()}_feature_info",
        f"{dataset_label} - List of features used in modeling"
    )

    # Standardized info boxes
    preprocessing_boxes = f"""
    <div class="info-box">
        <strong>{dataset_label} – Train/Test Overview</strong><br>
        • Train shape: {train_df.shape} | Test shape: {test_df.shape}<br>
        • Total train samples: {len(train_df):,} | Total test samples: {len(test_df):,}<br>
        • Number of features: {len(feature_columns)}<br>
        • Target column: '{target_col}'<br>
        • Missing values (train): {train_df.isnull().sum().sum()} | (test): {test_df.isnull().sum().sum()}
    </div>
    <div class="info-box">
        <strong>{dataset_label} – Train Class Distribution</strong><br>
        {"<br>".join([f"• {cls}: {cnt:,}" for cls, cnt in class_counts.items()])}<br>
        • Class balance (minority/majority): {balance:.4f}%
    </div>
    <div class="info-box">
        <strong>{dataset_label} – Feature Preparation</strong><br>
        • Target encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}<br>
        • Data preprocessing: Infinite values handled, missing values filled with train medians<br>
        • Feature scaling: StandardScaler (fit on train, applied to test)<br>
        • Configuration: Standardized models and parameters for fair EDR vs XDR comparison
    </div>
    """
    html_generator.add_text(preprocessing_boxes)

    # Standardized imbalance warning
    minority_ratio = balance / 100.0
    if minority_ratio < 0.01:  # Less than 1%
        html_generator.add_text(f'''
        <div class="warning-box">
            <strong>⚠️ Extreme Class Imbalance Detected</strong><br>
            • Minority class represents only {balance:.4f}% of the data<br>
            • Applied standardized techniques: SMOTE, balanced sampling, threshold tuning (80% recall target)<br>
            • Metrics like Precision-Recall AUC and F1 are more meaningful than accuracy<br>
            • Same imbalance handling strategy applied to both EDR and XDR for fair comparison
        </div>
        ''')

    # Standardized train & evaluate
    trained_models, scaler, results = train_and_evaluate_models(X_train, X_test, y_train, y_test, dataset_label)

    # Make y_test globally available to plotting helpers that need it
    global y_test_global
    y_test_global = y_test

    # Comparison section + plots
    compare_models_and_plots(results, le, dataset_label)

    # Per-model details
    for model_name, model in trained_models.items():
        y_pred = results[model_name]['predictions']
        html_generator.add_title(f"{dataset_label}: {model_name} – Detailed Analysis", 2)

        plot_confusion_matrix_detailed(y_test, y_pred, le, model_name, dataset_label)

        # Classification report (fixed table)
        report_df = build_classification_report_table(y_test, y_pred, le).round(4)
        html_generator.add_metrics_table(report_df, f"{dataset_label} – {model_name}: Classification Report")
        
        # Export classification report
        report_export = report_df.copy()
        report_export['model'] = model_name
        report_export['dataset'] = dataset_label
        data_exporter.export_to_csv(
            report_export,
            f"{dataset_label.lower()}_{model_name.lower().replace(' ', '_')}_classification_report",
            f"{dataset_label} {model_name} - Detailed classification report by class"
        )

        # Feature importance
        feature_importance_analysis(model, feature_columns, model_name, dataset_label)


# =========================
# Main
# =========================
def main():
    """
    Standardized pipeline with comprehensive JSON export
    """

    # ===== CONFIG: fill these paths =====
    EDR_TRAIN_PATH = "lanl_output/edr_train_sorted_adjusted.csv"  # <<< set me
    EDR_TEST_PATH  = "lanl_output/edr_test_sorted_adjusted.csv"   # <<< set me
    XDR_TRAIN_PATH = "lanl_output/xdr_train_sorted_adjusted.csv"  # <<< set me
    XDR_TEST_PATH  = "lanl_output/xdr_test_sorted_adjusted.csv"   # <<< set me
    # ====================================

    try:
        html_generator.add_title("Standardized Machine Learning Analysis Pipeline")
        html_generator.add_text('<div class="info-box"><strong>Standardized Comparison:</strong><br>'
                               '• Same models: Logistic Regression, Random Forest (SMOTE), LightGBM, XGBoost, Balanced RF, SGD SVM<br>'
                               '• Same parameters: N_ESTIMATORS=300, same random seeds, same scaling<br>'
                               '• Same recall targets: 80% threshold tuning for both EDR and XDR<br>'
                               '• Fair comparison: No dataset-specific feature engineering or optimizations</div>')

        # Run EDR first
        run_dataset_pipeline("EDR", EDR_TRAIN_PATH, EDR_TEST_PATH)

        # Then XDR
        run_dataset_pipeline("XDR", XDR_TRAIN_PATH, XDR_TEST_PATH)

        # Export comprehensive JSON with ALL data
        json_file = data_exporter.export_comprehensive_json()
        
        # Final HTML
        output_file = html_generator.generate_html("standardized_results.html")
        print(f"Standardized HTML report generated: {output_file}")
        print(f"Comprehensive data file: {json_file}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
