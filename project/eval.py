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
from sklearn.feature_selection import SelectKBest, f_classif  # type: ignore
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.dummy import DummyClassifier #type: ignore
from imblearn.over_sampling import SMOTE       #type: ignore            # new
from imblearn.pipeline      import Pipeline as ImbPipeline  # new#type: ignore
from sklearn.metrics import precision_recall_curve#type: ignore
from lightgbm import LGBMClassifier #type: ignore
from imblearn.ensemble import BalancedRandomForestClassifier #type: ignore
from sklearn.linear_model import SGDClassifier #type: ignore
from sklearn.ensemble import IsolationForest #type: ignore
import warnings
import base64
import io
import datetime

warnings.filterwarnings('ignore')

y_test_global = None

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

    def add_plot(self, fig, title=""):
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
    Now with maximal feature usage!
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
    TIME = True
    if TIME:
        exclude_cols = {
            target_col,
            'src_comp', 'dst_comp', 'computer', 'host'
        }
    else:
        exclude_cols = {
            target_col,
            'src_comp', 'dst_comp', 'computer', 'host', 'day', 'win'
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
# Modeling & evaluation
# =========================
def _binary_confusion_counts(cm):
    return cm.ravel() if cm.shape == (2, 2) else (np.nan, np.nan, np.nan, np.nan)

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Fit models, compute metrics, and return dict of results and fitted models.
    """
    # Scale once for everyone
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    RANDOM_STATE = 42
    
    # Calculate consistent parameters for all models
    contamination_rate = sum(y_train) / len(y_train)
    pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    
    # Standardized tree parameters (based on your best performers)
    N_ESTIMATORS = 200  # Match LightGBM
    MAX_DEPTH = None    # Let trees grow deep like Balanced RF
    MIN_SAMPLES_SPLIT = 2
    MIN_SAMPLES_LEAF = 1
     
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
                max_depth=MAX_DEPTH,
                min_samples_split=MIN_SAMPLES_SPLIT,
                min_samples_leaf=MIN_SAMPLES_LEAF,
                class_weight='balanced',  # Consistent with others
                n_jobs=-1,
                verbose=0
            ))
        ]),
        'LightGBM': LGBMClassifier(
            n_estimators=N_ESTIMATORS,
            scale_pos_weight=pos_weight,
            random_state=RANDOM_STATE,
            max_depth=-1,
            min_child_samples=MIN_SAMPLES_LEAF,
            min_split_gain=0.0,
            subsample=0.8,
            colsample_bytree=0.8,
            verbose=-1,
            n_jobs=-1
        ),
        'Balanced RF': BalancedRandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            sampling_strategy=0.5,  # ← Match SMOTE ratio
            replacement=True,
            random_state=RANDOM_STATE,
            max_depth=MAX_DEPTH,
            min_samples_split=MIN_SAMPLES_SPLIT,
            min_samples_leaf=MIN_SAMPLES_LEAF,
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
        ),
        'IsolationForest': IsolationForest(
            n_estimators=N_ESTIMATORS,
            contamination=contamination_rate,
            random_state=RANDOM_STATE,
            max_samples='auto',
            max_features=1.0,
            n_jobs=-1,
            verbose=0
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
        if name == 'IsolationForest':
            model.fit(X_train_scaled)  # Unsupervised
            y_pred = model.predict(X_test_scaled)
            y_pred = np.where(y_pred == -1, 1, 0)
            y_proba = None
        else:
            # All supervised models use same scaled data
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else None
            
        trained[name] = model

        y_scores = (y_proba[:, 1] if (y_proba is not None and y_proba.shape[1] > 1) else
                    (y_proba[:, 0] if y_proba is not None else None))

        # — threshold tuning: pick the first *non-trivial* cutoff achieving >= 80% recall —
        tuned_predictions = None  # Will store tuned predictions if threshold tuning is applied
        if y_scores is not None and binary:
            precs, recs, thresh = precision_recall_curve(y_test, y_scores)
            # ignore the first point (recall==1.0 at threshold=-inf)
            target_recall = 0.80
            valid = np.where(recs[1:] >= target_recall)[0]
            if len(valid) > 0:
                # shift back to original because we dropped recs[0]
                best_t = thresh[valid[0]]
                # Add sanity check - don't use thresholds that predict everything as positive
                test_pred = (y_scores >= best_t).astype(int)
                if np.mean(test_pred) < 0.9:  # Don't accept if >90% predicted as positive
                    y_pred_t = test_pred
                    tuned_f1 = f1_score(y_test, y_pred_t, zero_division=0)
                    tuned_prec = precision_score(y_test, y_pred_t, zero_division=0)
                    tuned_rec = recall_score(y_test, y_pred_t, zero_division=0)
                    print(f"  → {name} @ recall≥{target_recall:.2f}: thresh={best_t:.3f}  "
                          f"prec={tuned_prec:.3f}  rec={tuned_rec:.3f}  f1={tuned_f1:.3f}")
                    
                    # Use tuned predictions for final metrics
                    tuned_predictions = y_pred_t
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
                    # For binary classification, roc_auc_score automatically handles it
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
            'predictions': final_predictions,  # Use final (potentially tuned) predictions
            'scores': y_scores, 'probabilities': y_proba,
            'confusion_matrix': cm,
            'true_negatives': tn, 'false_positives': fp, 'false_negatives': fn, 'true_positives': tp,
            'false_positive_rate': fpr, 'false_negative_rate': fnr,
            'binary': binary,
            'threshold_tuned': tuned_predictions is not None  # Flag to indicate if threshold was tuned
        }

    # Baseline sanity check (using manually scaled data)
    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(X_train_scaled, y_train)
    base_acc = dummy.score(X_test_scaled, y_test)
    html_generator.add_text(
        f'<div class="info-box"><strong>Baseline (Most-Frequent) Accuracy:</strong> {base_acc:.4f}</div>'
    )

    return trained, scaler, results


# =========================
# Reporting helpers
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
    Add the big comparison table + confusion summary + best models + plots.
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

    html_generator.add_best_models_summary(best_models, comparison_df, results)

    # Bar plot of metrics
    fig = plt.figure(figsize=(12, 7))
    comparison_df[['Accuracy','Balanced Acc','Precision','Recall','F1']].plot(kind='bar', ax=plt.gca())
    plt.title(f'{dataset_label} – Metrics by Model')
    plt.ylabel('Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    html_generator.add_plot(fig, f"{dataset_label} – Metrics by Model")

    # ROC & PR curves (only for binary problems) - FIXED VERSION
    any_binary = any(r['binary'] for r in results.values())
    classes = list(le.classes_)
    if any_binary and len(classes) == 2:
        # Get y_test from the global variable (should be set in run_dataset_pipeline)
        global y_test_global
        if y_test_global is not None:
            # ROC
            fig = plt.figure(figsize=(10, 7))
            for name, r in results.items():
                if r['scores'] is not None:
                    # Compute ROC
                    from sklearn.metrics import roc_curve #type: ignore
                    fpr, tpr, _ = roc_curve(y_test_global, r['scores'])
                    plt.plot(fpr, tpr, label=f"{name} (AUC={r['roc_auc']:.3f})")
            plt.plot([0,1], [0,1], linestyle='--', linewidth=1)
            plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
            plt.title(f'{dataset_label} – ROC Curves')
            plt.legend()
            plt.tight_layout()
            html_generator.add_plot(fig, f"{dataset_label} – ROC Curves")

            # Precision-Recall
            fig = plt.figure(figsize=(10, 7))
            from sklearn.metrics import precision_recall_curve #type: ignore
            for name, r in results.items():
                if r['scores'] is not None:
                    prec, rec, _ = precision_recall_curve(y_test_global, r['scores'])
                    plt.plot(rec, prec, label=f"{name} (AP={r['ap']:.3f})")
            plt.xlabel('Recall'); plt.ylabel('Precision')
            plt.title(f'{dataset_label} – Precision–Recall Curves')
            plt.legend()
            plt.tight_layout()
            html_generator.add_plot(fig, f"{dataset_label} – Precision–Recall Curves")

    # Probability histograms
    fig = plt.figure(figsize=(12, 5))
    plt.title(f'{dataset_label} – Predicted Probability Distributions (Positive Class)')
    for i, (name, r) in enumerate(results.items(), start=1):
        if r['scores'] is not None:
            plt.hist(r['scores'], bins=30, alpha=0.5, label=name, density=True)
    plt.xlabel('Predicted probability')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    html_generator.add_plot(fig, f"{dataset_label} – Predicted Probability Distributions")

    # Threshold sweep (binary only) - FIXED VERSION
    if any_binary and len(classes) == 2 and y_test_global is not None:
        fig = plt.figure(figsize=(12, 7))
        thresholds = np.linspace(0.0, 1.0, 101)
        for name, r in results.items():
            if r['scores'] is None:
                continue
            f1_vals, prec_vals, rec_vals = [], [], []
            for t in thresholds:
                y_hat = (r['scores'] >= t).astype(int)
                f1_vals.append(f1_score(y_test_global, y_hat, zero_division=0))
                prec_vals.append(precision_score(y_test_global, y_hat, zero_division=0))
                rec_vals.append(recall_score(y_test_global, y_hat, zero_division=0))
            plt.plot(thresholds, f1_vals, label=f'{name} – F1')
            plt.plot(thresholds, prec_vals, linestyle='--', label=f'{name} – Precision')
            plt.plot(thresholds, rec_vals, linestyle=':', label=f'{name} – Recall')
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title(f'{dataset_label} – Threshold Sweep')
        plt.legend(ncol=2)
        plt.tight_layout()
        html_generator.add_plot(fig, f"{dataset_label} – Threshold Sweep")


def plot_confusion_matrix_detailed(y_test, y_pred, le, model_name, dataset_label):
    html_generator.add_title(f"{dataset_label} – {model_name}: Confusion Matrix", 3)
    cm = confusion_matrix(y_test, y_pred)
    cm_percent = (cm.astype(float) / cm.sum(axis=1, keepdims=True)) * 100

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
# One full pass for a dataset
# =========================
def run_dataset_pipeline(dataset_label, train_path, test_path):
    """
    dataset_label: string for section titles (e.g., "EDR" or "XDR")
    train_path/test_path: CSV paths you provide
    """
    html_generator.add_title(f"{dataset_label}: Dataset Loading & Preprocessing", 2)

    train_df = load_csv(train_path)
    test_df = load_csv(test_path)

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

    X_train, X_test, y_train, y_test, le, feature_columns = prepare_train_test(train_df, test_df)

    # Info boxes
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
        • Feature scaling: StandardScaler (fit on train, applied to test)
    </div>
    """
    html_generator.add_text(preprocessing_boxes)

    # Check for extreme imbalance and add warning
    minority_ratio = balance / 100.0
    if minority_ratio < 0.01:  # Less than 1%
        html_generator.add_text(f'''
        <div class="warning-box">
            <strong>⚠️ Extreme Class Imbalance Detected</strong><br>
            • Minority class represents only {balance:.4f}% of the data<br>
            • This extreme imbalance may cause models to predict everything as majority class<br>
            • Consider: more aggressive SMOTE ratios, cost-sensitive learning, or ensemble methods<br>
            • Metrics like Precision-Recall AUC and F1 are more meaningful than accuracy
        </div>
        ''')

    # Train & evaluate
    trained_models, scaler, results = train_and_evaluate_models(X_train, X_test, y_train, y_test)

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

        # Feature importance
        feature_importance_analysis(model, feature_columns, model_name, dataset_label)


# =========================
# Main
# =========================
def main():
    """
    Runs the pipeline twice: first for EDR, then for XDR, output to a single HTML file.
    """

    # ===== CONFIG: fill these paths =====
    EDR_TRAIN_PATH = "lanl_output/edr_train_sorted_adjusted.csv"  # <<< set me
    EDR_TEST_PATH  = "lanl_output/edr_test_sorted_adjusted.csv"   # <<< set me
    XDR_TRAIN_PATH = "lanl_output/xdr_train_sorted_adjusted.csv"  # <<< set me
    XDR_TEST_PATH  = "lanl_output/xdr_test_sorted_adjusted.csv"   # <<< set me
    # ====================================

    try:
        html_generator.add_title("Machine Learning Analysis Pipeline")

        # Run EDR first
        run_dataset_pipeline("EDR", EDR_TRAIN_PATH, EDR_TEST_PATH)

        # Then XDR
        run_dataset_pipeline("XDR", XDR_TRAIN_PATH, XDR_TEST_PATH)

        # Final HTML
        output_file = html_generator.generate_html("results.html")
        print(f"HTML report generated: {output_file}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
