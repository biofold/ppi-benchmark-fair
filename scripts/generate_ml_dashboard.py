#!/usr/bin/env python3
"""
Script to generate an interactive HTML dashboard for displaying results from
ppi_ml_croissant.py with the same style as index.html.
"""

import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import base64
from io import BytesIO
import seaborn as sns
import sys
from datetime import datetime

def load_results_json(results_path):
    """Load results JSON file."""
    with open(results_path, 'r') as f:
        results = json.load(f)
    return results

def create_figure_image(fig):
    """Convert matplotlib figure to base64 encoded image."""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return img_str

def generate_performance_plots(results, output_dir=None):
    """Generate performance plots from results."""
    plots = {}
    
    # Get model names
    model_names = [name for name in results.keys() if name not in ['cross_validation_settings']]
    
    if not model_names:
        return plots
    
    # 1. Metrics comparison bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    x = np.arange(len(model_names))
    width = 0.2
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = []
        for name in model_names:
            if name in results:
                values.append(results[name]['cv_metrics'].get(metric, 0))
        
        if values:
            ax.bar(x + i*width - width*1.5, values, width, label=label)
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Score')
    ax.set_title('Average Cross-Validation Metrics by Model')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    plots['metrics_comparison'] = create_figure_image(fig)
    
    # 2. F1-Score per fold (line plot)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for name in model_names:
        if name in results and 'fold_metrics' in results[name]:
            fold_f1s = results[name]['fold_metrics'].get('f1_scores', [])
            if fold_f1s:
                ax.plot(range(1, len(fold_f1s) + 1), fold_f1s, 
                       marker='o', linewidth=2, markersize=8, label=name)
    
    ax.set_xlabel('Fold')
    ax.set_ylabel('F1-Score')
    ax.set_title('F1-Score per Fold by Model')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, max([len(results[name]['fold_metrics'].get('f1_scores', [])) 
                                for name in model_names if name in results], default=5) + 1))
    
    plots['f1_per_fold'] = create_figure_image(fig)
    
    # 3. Metric distribution across folds (box plot)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    data = []
    for name in model_names:
        if name in results and 'fold_metrics' in results[name]:
            metrics_data = results[name]['fold_metrics']
            for metric_name, values in metrics_data.items():
                if isinstance(values, list) and values:
                    for fold_idx, value in enumerate(values, 1):
                        metric_label = metric_name.replace('_', ' ').title().replace('F1 Scores', 'F1-Score')
                        data.append({
                            'Model': name,
                            'Metric': metric_label,
                            'Value': value,
                            'Fold': fold_idx
                        })
    
    if data:
        df_plot = pd.DataFrame(data)
        
        # Pivot for boxplot
        pivot_data = []
        for name in model_names:
            for metric in ['Accuracies', 'Precisions', 'Recalls', 'F1 Scores']:
                if metric in df_plot[df_plot['Model'] == name]['Metric'].unique():
                    values = df_plot[(df_plot['Model'] == name) & (df_plot['Metric'] == metric)]['Value'].tolist()
                    pivot_data.extend([{'Model': name, 'Metric': metric, 'Value': v} for v in values])
        
        if pivot_data:
            pivot_df = pd.DataFrame(pivot_data)
            pivot_df['Metric'] = pivot_df['Metric'].map({
                'Accuracies': 'Accuracy',
                'Precisions': 'Precision', 
                'Recalls': 'Recall',
                'F1 Scores': 'F1-Score'
            })
            
            # Filter out any NaN values
            pivot_df = pivot_df.dropna(subset=['Value'])
            
            if not pivot_df.empty:
                sns.boxplot(x='Model', y='Value', hue='Metric', data=pivot_df, ax=ax)
                ax.set_title('Metric Distributions Across Folds')
                ax.set_ylabel('Score')
                ax.set_xlabel('Model')
                ax.set_ylim(0, 1.1)
                ax.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.grid(True, alpha=0.3, axis='y')
    
    plots['metric_distribution'] = create_figure_image(fig)
    
    # 4. Overall metrics radar chart
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Prepare data for radar chart
    categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    
    for name in model_names:
        if name in results and 'overall_metrics' in results[name]:
            metrics = results[name]['overall_metrics']
            values = [
                metrics.get('accuracy', 0),
                metrics.get('precision', 0),
                metrics.get('recall', 0),
                metrics.get('f1', 0),
                metrics.get('roc_auc', 0) if metrics.get('roc_auc') is not None else 0
            ]
            
            # Complete the loop
            values = values + values[:1]
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            angles = angles + angles[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=name)
            ax.fill(angles, values, alpha=0.25)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title('Overall Model Performance (Radar Chart)')
    ax.legend(bbox_to_anchor=(1.3, 1), loc='upper right')
    ax.grid(True)
    
    plots['radar_chart'] = create_figure_image(fig)
    
    # Save plots to files if output_dir provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, (name, img_data) in enumerate(plots.items()):
            # Convert base64 back to image and save
            img_bytes = base64.b64decode(img_data)
            img_path = output_dir / f"{name}.png"
            with open(img_path, 'wb') as f:
                f.write(img_bytes)
    
    return plots

def generate_model_details_table(results):
    """Generate HTML table with detailed model results."""
    model_names = [name for name in results.keys() if name not in ['cross_validation_settings']]
    
    if not model_names:
        return "<p>No model results found.</p>"
    
    html = """
    <div class="table-container">
        <table>
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Avg Accuracy</th>
                    <th>Avg Precision</th>
                    <th>Avg Recall</th>
                    <th>Avg F1-Score</th>
                    <th>Avg ROC-AUC</th>
                    <th>Overall Accuracy</th>
                    <th>Overall F1-Score</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for name in model_names:
        if name in results:
            cv_metrics = results[name].get('cv_metrics', {})
            overall_metrics = results[name].get('overall_metrics', {})
            
            # Format ROC-AUC
            roc_auc = cv_metrics.get('roc_auc')
            roc_auc_display = f"{roc_auc:.4f}" if roc_auc is not None else 'N/A'
            
            html += f"""
                <tr>
                    <td><strong>{name}</strong></td>
                    <td>{cv_metrics.get('accuracy', 0):.4f}</td>
                    <td>{cv_metrics.get('precision', 0):.4f}</td>
                    <td>{cv_metrics.get('recall', 0):.4f}</td>
                    <td><strong>{cv_metrics.get('f1', 0):.4f}</strong></td>
                    <td>{roc_auc_display}</td>
                    <td>{overall_metrics.get('accuracy', 0):.4f}</td>
                    <td><strong>{overall_metrics.get('f1', 0):.4f}</strong></td>
                </tr>
            """
    
    html += """
            </tbody>
        </table>
    </div>
    """
    
    return html

def generate_fold_details_table(results):
    """Generate HTML table with fold-by-fold details."""
    model_names = [name for name in results.keys() if name not in ['cross_validation_settings']]
    
    if not model_names:
        return "<p>No fold details available.</p>"
    
    # Check if we have fold metrics
    has_fold_metrics = any('fold_metrics' in results[name] for name in model_names if name in results)
    
    if not has_fold_metrics:
        return "<p>No fold-level metrics available.</p>"
    
    html = """
    <div class="table-container">
        <table>
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Fold</th>
                    <th>Accuracy</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1-Score</th>
                    <th>ROC-AUC</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for name in model_names:
        if name in results and 'fold_metrics' in results[name]:
            fold_metrics = results[name]['fold_metrics']
            
            # Get number of folds
            n_folds = len(fold_metrics.get('accuracies', []))
            
            for fold_idx in range(n_folds):
                # Get values for this fold with safe indexing
                accuracies = fold_metrics.get('accuracies', [])
                precisions = fold_metrics.get('precisions', [])
                recalls = fold_metrics.get('recalls', [])
                f1_scores = fold_metrics.get('f1_scores', [])
                roc_aucs = fold_metrics.get('roc_aucs', [])
                
                # Format each metric
                accuracy_val = f"{accuracies[fold_idx]:.4f}" if fold_idx < len(accuracies) else 'N/A'
                precision_val = f"{precisions[fold_idx]:.4f}" if fold_idx < len(precisions) else 'N/A'
                recall_val = f"{recalls[fold_idx]:.4f}" if fold_idx < len(recalls) else 'N/A'
                f1_val = f"{f1_scores[fold_idx]:.4f}" if fold_idx < len(f1_scores) else 'N/A'
                
                # Format ROC-AUC
                if fold_idx < len(roc_aucs) and roc_aucs[fold_idx] is not None:
                    roc_auc_val = f"{roc_aucs[fold_idx]:.4f}"
                else:
                    roc_auc_val = 'N/A'
                
                html += f"""
                    <tr>
                        <td>{'→' if fold_idx > 0 else name}</td>
                        <td>{fold_idx + 1}</td>
                        <td>{accuracy_val}</td>
                        <td>{precision_val}</td>
                        <td>{recall_val}</td>
                        <td><strong>{f1_val}</strong></td>
                        <td>{roc_auc_val}</td>
                    </tr>
                """
    
    html += """
            </tbody>
        </table>
    </div>
    """
    
    return html

def generate_cv_settings_section(results):
    """Generate HTML section for CV settings."""
    if 'cross_validation_settings' not in results:
        return ""
    
    settings = results['cross_validation_settings']
    
    html = f"""
    <div class="fair-section">
        <h3>Cross-Validation Settings</h3>
        <ul class="fair-checklist">
            <li><strong>Method:</strong> {settings.get('method', 'ClusterID-aware stratified cross-validation')}</li>
            <li><strong>Number of Folds:</strong> {settings.get('n_splits', 5)}</li>
            <li><strong>Random State:</strong> {settings.get('random_state', 42)}</li>
            <li><strong>Best Model:</strong> <span class="badge fair">{settings.get('best_model', 'Not determined')}</span></li>
        </ul>
    </div>
    """
    
    return html

def generate_feature_analysis_section(feature_eval_path=None):
    """Generate HTML section for feature analysis."""
    html = ""
    
    if feature_eval_path and Path(feature_eval_path).exists():
        try:
            with open(feature_eval_path, 'r') as f:
                feature_data = json.load(f)
            
            html += """
            <div class="fair-section">
                <h3>Feature Analysis Summary</h3>
            """
            
            if 'dataset_info' in feature_data:
                ds_info = feature_data['dataset_info']
                html += f"""
                <ul class="fair-checklist">
                    <li><strong>Samples:</strong> {ds_info.get('n_samples', 'N/A')}</li>
                    <li><strong>Features:</strong> {ds_info.get('n_features', 'N/A')}</li>
                </ul>
                """
            
            if 'correlation_analysis' in feature_data:
                corr_info = feature_data['correlation_analysis']
                html += f"""
                <h4>Correlation Analysis</h4>
                <ul class="fair-checklist">
                    <li><strong>Highly correlated feature pairs (&gt;0.8):</strong> {corr_info.get('n_highly_correlated', 0)}</li>
                </ul>
                """
            
            if 'feature_target_analysis' in feature_data:
                ft_info = feature_data['feature_target_analysis']
                html += f"""
                <h4>Feature-Target Relationship</h4>
                <ul class="fair-checklist">
                    <li><strong>Significant features (ANOVA p&lt;0.05):</strong> {len(ft_info.get('anova_significant_features', []))}</li>
                </ul>
                """
                
                if ft_info.get('mutual_info_top_features'):
                    html += """
                    <h4>Top Features by Mutual Information</h4>
                    <ol>
                    """
                    for i, feature in enumerate(ft_info['mutual_info_top_features'][:5], 1):
                        html += f"<li>{feature}</li>"
                    html += "</ol>"
            
            if 'pca_analysis' in feature_data:
                pca_info = feature_data['pca_analysis']
                html += f"""
                <h4>PCA Analysis</h4>
                <ul class="fair-checklist">
                    <li><strong>Components for 95% variance:</strong> {pca_info.get('components_needed_95', 'N/A')}</li>
                </ul>
                """
            
            html += "</div>"
            
        except Exception as e:
            html += f"""
            <div class="fair-section">
                <h3>Feature Analysis</h3>
                <p>Error loading feature analysis: {e}</p>
            </div>
            """
    
    return html

def generate_html_dashboard(results, plots, feature_eval_path=None, output_path="ml_results_dashboard.html"):
    """Generate the HTML dashboard."""
    
    # Get current date
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    # Get best model
    best_model = results.get('cross_validation_settings', {}).get('best_model', 'Not determined')
    
    # Calculate average F1 score across all models
    model_names = [name for name in results.keys() if name not in ['cross_validation_settings']]
    avg_f1_scores = []
    for name in model_names:
        if name in results and 'cv_metrics' in results[name]:
            avg_f1_scores.append(results[name]['cv_metrics'].get('f1', 0))
    
    avg_f1 = np.mean(avg_f1_scores) if avg_f1_scores else 0
    
    # Get CV settings
    cv_settings = results.get('cross_validation_settings', {})
    n_splits = cv_settings.get('n_splits', 5)
    
    # Generate table content
    model_details_table = generate_model_details_table(results)
    fold_details_table = generate_fold_details_table(results)
    cv_settings_section = generate_cv_settings_section(results)
    feature_analysis_section = generate_feature_analysis_section(feature_eval_path)
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ELIXIR - PPI Benchmark ML Results Dashboard</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        :root {{
            --primary-color: #2c3e50;
            --secondary-color: #4689a3;
            --accent-color: #e74c3c;
            --light-bg: #f8f9fa;
            --success-color: #27ae60;
            --warning-color: #f39c12;
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: var(--light-bg);
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        /* Header - Matching index.html */
        header {{
            background: linear-gradient(135deg, var(--primary-color),#4689a3);
            color: white;
            padding: 50px 0;
            text-align: center;
            position: relative;
            overflow: hidden;
            margin-bottom: 40px;
            border-radius: 12px;
        }}
        
        header::before {{
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-image: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" preserveAspectRatio="none"><path d="M0,0 L100,0 L100,100 Z" fill="rgba(255,255,255,0.05)"/></svg>');
            background-size: cover;
        }}
        
        .header-content {{
            position: relative;
            z-index: 1;
            padding: 0 20px;
        }}
        
        h1 {{
            font-size: 2.5rem;
            margin-bottom: 20px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        
        .badges {{
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
            gap: 10px;
            margin: 20px 0;
        }}
        
        .badge {{
            display: inline-block;
            padding: 8px 16px;
            border-radius: 50px;
            font-size: 0.9rem;
            font-weight: 600;
            text-decoration: none;
            transition: transform 0.3s ease;
        }}
        
        .badge:hover {{
            transform: translateY(-2px);
        }}
        
        .badge.fair {{
            background: linear-gradient(135deg, var(--success-color), #219653);
            color: white;
            box-shadow: 0 4px 10px rgba(39, 174, 96, 0.3);
        }}
        
        .badge.ml {{
            background: linear-gradient(135deg, #9b59b6, #8e44ad);
            color: white;
            box-shadow: 0 4px 10px rgba(155, 89, 182, 0.3);
        }}
        
        .badge.cv {{
            background: linear-gradient(135deg, var(--warning-color), #e67e22);
            color: white;
            box-shadow: 0 4px 10px rgba(243, 156, 18, 0.3);
        }}
        
        .badge.best {{
            background: linear-gradient(135deg, var(--accent-color), #c0392b);
            color: white;
            box-shadow: 0 4px 10px rgba(231, 76, 60, 0.3);
        }}
        
        .dashboard-link {{
            display: inline-flex;
            align-items: center;
            gap: 10px;
            background-color: rgba(255,255,255,0.15);
            color: white;
            padding: 12px 20px;
            border-radius: 10px;
            text-decoration: none;
            font-weight: 400;
            transition: all 0.3s ease;
            border: 2px solid rgba(255,255,255,0.3);
            margin: 10px;
        }}
        
        .dashboard-link:hover {{
            background-color: rgba(255,255,255,0.25);
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }}
        
        /* Section Cards - Matching index.html */
        .section {{
            background-color: white;
            margin: 30px 0;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.08);
            border-left: 5px solid var(--secondary-color);
        }}
        
        h2 {{
            color: var(--primary-color);
            margin-bottom: 20px;
            font-size: 1.8rem;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        h2::before {{
            content: "📊";
            font-size: 1.5rem;
        }}
        
        h3 {{
            color: var(--primary-color);
            margin: 20px 0 10px;
            font-size: 1.3rem;
        }}
        
        h4 {{
            color: #555;
            margin: 15px 0 10px;
            font-size: 1.1rem;
        }}
        
        /* Tables */
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 5px 15px rgba(0,0,0,0.05);
        }}
        
        th, td {{
            padding: 15px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }}
        
        th {{
            background-color: #f8f9fa;
            color: var(--primary-color);
            font-weight: 600;
        }}
        
        tr:hover {{
            background-color: #f8f9fa;
        }}
        
        /* Lists */
        ul, ol {{
            margin: 15px 0;
            padding-left: 20px;
        }}
        
        li {{
            margin-bottom: 8px;
            line-height: 1.6;
        }}
        
        .fair-section {{
            margin: 25px 0;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid;
        }}
        
        .fair-section.findable {{ background-color: #e8e8e8; border-left-color: #e74c3c; }}
        .fair-section.accessible {{ background-color: #e8e8e8; border-left-color: #3498db; }}
        .fair-section.interoperable {{ background-color: #e8e8e8; border-left-color: #9b59b6; }}
        .fair-section.reusable {{ background-color: #e8e8e8; border-left-color: #27ae60; }}
        
        .fair-score {{
            font-size: 1.5rem;
            font-weight: bold;
            margin-bottom: 10px;
        }}
        
        .fair-checklist li {{
            list-style-type: none;
            padding-left: 25px;
            position: relative;
        }}
        
        .fair-checklist li:before {{
            content: "✅";
            position: absolute;
            left: 0;
        }}
        
        .fair-checklist li.needs-improvement:before {{
            content: "⚠️";
        }}
        
        /* Figure containers */
        .figure-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
            gap: 30px;
            margin: 30px 0;
        }}
        
        .figure-container {{
            background-color: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.05);
            border: 1px solid #e0e0e0;
        }}
        
        .figure-container img {{
            width: 100%;
            height: auto;
            border-radius: 6px;
        }}
        
        .figure-title {{
            text-align: center;
            margin-bottom: 15px;
            color: var(--primary-color);
            font-weight: 600;
        }}
        
        /* Footer */
        footer {{
            background-color: #386277;
            color: white;
            padding: 10px 0;
            text-align: center;
            margin-top: 60px;
            border-radius: 12px;
        }}
        
        .footer-content {{
            margin-top: 20px;
            padding: 0 10px;
        }}
        
        .footer-links {{
            display: flex;
            justify-content: center;
            gap: 30px;
            margin: 30px 0;
            flex-wrap: wrap;
        }}
        
        .footer-link {{
            color: rgba(255,255,255,0.8);
            text-decoration: none;
            transition: color 0.3s ease;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .footer-link:hover {{
            color: white;
        }}
        
        .copyright {{
            margin-top: 20px;
            color: rgba(255,255,255,0.6);
            font-size: 0.9rem;
            line-height: 1.6;
        }}
        
        /* Responsive */
        @media (max-width: 768px) {{
            h1 {{
                font-size: 2rem;
            }}
            
            .section {{
                padding: 20px;
            }}
            
            table {{
                display: block;
                overflow-x: auto;
            }}
            
            .badges {{
                flex-direction: column;
                align-items: center;
            }}
            
            .badge {{
                width: 90%;
                text-align: center;
            }}
            
            .figure-grid {{
                grid-template-columns: 1fr;
            }}
            
            .footer-links {{
                flex-direction: column;
                gap: 15px;
            }}
        }}
        
        /* Model performance highlights */
        .performance-highlights {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        
        .highlight-card {{
            background: linear-gradient(135deg, #f8f9fa, #e9ecef);
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            border-left: 5px solid var(--secondary-color);
        }}
        
        .highlight-value {{
            font-size: 2.5rem;
            font-weight: bold;
            color: var(--primary-color);
            margin-bottom: 10px;
        }}
        
        .highlight-label {{
            font-size: 1rem;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        /* Tabs for different views */
        .tab-container {{
            margin: 30px 0;
        }}
        
        .tab-buttons {{
            display: flex;
            border-bottom: 2px solid #e0e0e0;
            margin-bottom: 20px;
        }}
        
        .tab-button {{
            padding: 12px 24px;
            background: none;
            border: none;
            cursor: pointer;
            font-size: 1rem;
            color: #666;
            border-bottom: 3px solid transparent;
            transition: all 0.3s ease;
        }}
        
        .tab-button.active {{
            color: var(--secondary-color);
            border-bottom: 3px solid var(--secondary-color);
            font-weight: 600;
        }}
        
        .tab-button:hover:not(.active) {{
            color: var(--primary-color);
            background-color: #f8f9fa;
        }}
        
        .tab-content {{
            display: none;
        }}
        
        .tab-content.active {{
            display: block;
        }}
    </style>
    
    <script>
        function showTab(tabName) {{
            // Hide all tab contents
            const tabContents = document.querySelectorAll('.tab-content');
            tabContents.forEach(content => {{
                content.classList.remove('active');
            }});
            
            // Remove active class from all tab buttons
            const tabButtons = document.querySelectorAll('.tab-button');
            tabButtons.forEach(button => {{
                button.classList.remove('active');
            }});
            
            // Show the selected tab content
            document.getElementById(tabName).classList.add('active');
            
            // Mark the clicked button as active
            event.currentTarget.classList.add('active');
        }}
        
        // Initialize first tab as active on page load
        document.addEventListener('DOMContentLoaded', function() {{
            showTab('summary');
        }});
    </script>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <header>
            <div class="header-content">
                <h3 style="color:white;"><strong>ELIXIR Protein-Protein Interaction Benchmark</strong></h3>
                <h1><strong>Machine Learning Results Dashboard</strong></h1>
                
                <div class="badges">
                    <span class="badge ml"><i class="fas fa-brain"></i> ML Analysis</span>
                    <span class="badge cv"><i class="fas fa-crosshairs"></i> ClusterID-Aware CV</span>
                    <span class="badge best"><i class="fas fa-trophy"></i> Best Model: {best_model}</span>
                    <span class="badge fair"><i class="fas fa-chart-line"></i> Avg F1: {avg_f1:.3f}</span>
                </div>
                
                <div>
                    <a href="#summary" class="dashboard-link" onclick="showTab('summary'); return false;">
                        <i class="fas fa-home"></i> Summary
                    </a>
                    <a href="#performance" class="dashboard-link" onclick="showTab('performance'); return false;">
                        <i class="fas fa-chart-bar"></i> Performance
                    </a>
                    <a href="#models" class="dashboard-link" onclick="showTab('models'); return false;">
                        <i class="fas fa-cogs"></i> Model Details
                    </a>
                    <a href="#features" class="dashboard-link" onclick="showTab('features'); return false;">
                        <i class="fas fa-chart-line"></i> Feature Analysis
                    </a>
                </div>
            </div>
        </header>

        <main>
            <!-- Tab Container -->
            <div class="tab-container">
                <div class="tab-buttons">
                    <button class="tab-button active" onclick="showTab('summary')">Summary</button>
                    <button class="tab-button" onclick="showTab('performance')">Performance Plots</button>
                    <button class="tab-button" onclick="showTab('models')">Model Details</button>
                    <button class="tab-button" onclick="showTab('features')">Feature Analysis</button>
                    <button class="tab-button" onclick="showTab('data')">Dataset Info</button>
                </div>
                
                <!-- Summary Tab -->
                <div id="summary" class="tab-content active">
                    <section class="section">
                        <h2><i class="fas fa-home"></i> Analysis Summary</h2>
                        
                        <div class="performance-highlights">
                            <div class="highlight-card">
                                <div class="highlight-value">{len(model_names)}</div>
                                <div class="highlight-label">Models Trained</div>
                            </div>
                            <div class="highlight-card">
                                <div class="highlight-value">{n_splits}</div>
                                <div class="highlight-label">CV Folds</div>
                            </div>
                            <div class="highlight-card">
                                <div class="highlight-value">{avg_f1:.3f}</div>
                                <div class="highlight-label">Average F1-Score</div>
                            </div>
                            <div class="highlight-card">
                                <div class="highlight-value">{best_model}</div>
                                <div class="highlight-label">Best Model</div>
                            </div>
                        </div>
                        
                        <h3>Key Findings</h3>
                        <ul class="fair-checklist">
                            <li>All models trained using <strong>ClusterID-aware cross-validation</strong> to prevent data leakage</li>
                            <li>Performance evaluated across {n_splits} folds</li>
                            <li><strong>{best_model}</strong> achieved the highest average F1-score</li>
                            <li>Models show consistent performance across different folds</li>
                        </ul>
                        
                        {cv_settings_section}
                    </section>
                </div>
                
                <!-- Performance Plots Tab -->
                <div id="performance" class="tab-content">
                    <section class="section">
                        <h2><i class="fas fa-chart-bar"></i> Performance Visualizations</h2>
                        
                        <div class="figure-grid">
    """
    
    # Add plots to HTML
    if 'metrics_comparison' in plots:
        html_content += f"""
                            <div class="figure-container">
                                <div class="figure-title">Metrics Comparison Across Models</div>
                                <img src="data:image/png;base64,{plots['metrics_comparison']}" alt="Metrics Comparison">
                                <p>Comparison of average cross-validation metrics across all trained models. Higher values indicate better performance.</p>
                            </div>
        """
    
    if 'f1_per_fold' in plots:
        html_content += f"""
                            <div class="figure-container">
                                <div class="figure-title">F1-Score Per Fold</div>
                                <img src="data:image/png;base64,{plots['f1_per_fold']}" alt="F1-Score Per Fold">
                                <p>F1-Score for each fold across different models. Consistency across folds indicates robust model performance.</p>
                            </div>
        """
    
    if 'metric_distribution' in plots:
        html_content += f"""
                            <div class="figure-container">
                                <div class="figure-title">Metric Distribution Across Folds</div>
                                <img src="data:image/png;base64,{plots['metric_distribution']}" alt="Metric Distribution">
                                <p>Box plots showing distribution of metrics across folds for each model. Tighter distributions indicate more consistent performance.</p>
                            </div>
        """
    
    if 'radar_chart' in plots:
        html_content += f"""
                            <div class="figure-container">
                                <div class="figure-title">Overall Model Performance (Radar Chart)</div>
                                <img src="data:image/png;base64,{plots['radar_chart']}" alt="Radar Chart">
                                <p>Radar chart comparing overall performance across multiple metrics. Larger areas indicate better overall performance.</p>
                            </div>
        """
    
    html_content += f"""
                        </div>
                    </section>
                </div>
                
                <!-- Model Details Tab -->
                <div id="models" class="tab-content">
                    <section class="section">
                        <h2><i class="fas fa-cogs"></i> Model Performance Details</h2>
                        
                        <h3>Average Cross-Validation Metrics</h3>
                        <p>The following table shows average performance metrics across all cross-validation folds:</p>
                        {model_details_table}
                        
                        <h3>Fold-by-Fold Performance</h3>
                        <p>Detailed performance metrics for each fold of cross-validation:</p>
                        {fold_details_table}
                        
                        <h3>Performance Insights</h3>
                        <ul class="fair-checklist">
                            <li>F1-Score is used as the primary metric for model comparison</li>
                            <li>Precision and Recall provide insight into error types</li>
                            <li>ROC-AUC indicates overall ranking performance (when available)</li>
                            <li>Consistency across folds indicates model robustness</li>
                        </ul>
                    </section>
                </div>
                
                <!-- Feature Analysis Tab -->
                <div id="features" class="tab-content">
                    <section class="section">
                        <h2><i class="fas fa-chart-line"></i> Feature Analysis</h2>
                        {feature_analysis_section}
                        
                        <h3>Feature Importance</h3>
                        <p>Understanding which features contribute most to model predictions:</p>
                        <ul class="fair-checklist">
                            <li>Feature importance varies by model type</li>
                            <li>Random Forest provides intrinsic feature importance scores</li>
                            <li>Mutual information identifies features with strong target relationships</li>
                            <li>PCA analysis reveals feature redundancy and dimensionality</li>
                        </ul>
                        
                        <div class="fair-section">
                            <h3>Next Steps for Feature Engineering</h3>
                            <ul class="fair-checklist">
                                <li>Consider removing highly correlated features</li>
                                <li>Focus on top features identified by importance analysis</li>
                                <li>Explore feature interactions and polynomial features</li>
                                <li>Test dimensionality reduction techniques</li>
                            </ul>
                        </div>
                    </section>
                </div>
                
                <!-- Dataset Info Tab -->
                <div id="data" class="tab-content">
                    <section class="section">
                        <h2><i class="fas fa-database"></i> Dataset Information</h2>
                        
                        <h3>ML Pipeline Overview</h3>
                        <pre><code>Croissant Dataset → Feature Extraction → Preprocessing → 
ClusterID-aware CV → Model Training → Evaluation → Visualization</code></pre>
                        
                        <h3>Key Pipeline Components</h3>
                        <table>
                            <thead>
                                <tr>
                                    <th>Component</th>
                                    <th>Purpose</th>
                                    <th>Implementation</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td>CroissantLoader</td>
                                    <td>Load and parse Croissant-formatted dataset</td>
                                    <td>Extracts features, labels, ClusterIDs</td>
                                </tr>
                                <tr>
                                    <td>FeatureProcessor</td>
                                    <td>Clean and preprocess features</td>
                                    <td>Handles missing values, categorical encoding</td>
                                </tr>
                                <tr>
                                    <td>ClusterAwareCV</td>
                                    <td>Prevent data leakage</td>
                                    <td>Ensures same-cluster samples stay together</td>
                                </tr>
                                <tr>
                                    <td>ModelTrainer</td>
                                    <td>Train multiple ML models</td>
                                    <td>Random Forest, SVM, Logistic Regression</td>
                                </tr>
                                <tr>
                                    <td>Evaluator</td>
                                    <td>Comprehensive model evaluation</td>
                                    <td>Multiple metrics, visualizations, statistical tests</td>
                                </tr>
                            </tbody>
                        </table>
                        
                        <h3>Methodology Notes</h3>
                        <ul class="fair-checklist">
                            <li><strong>ClusterID-aware cross-validation:</strong> Prevents data leakage by ensuring all interfaces from the same sequence cluster stay together in training or testing sets</li>
                            <li><strong>Stratified sampling:</strong> Maintains class balance across folds</li>
                            <li><strong>Multiple metrics:</strong> Evaluates models using Accuracy, Precision, Recall, F1-Score, and ROC-AUC</li>
                            <li><strong>Feature scaling:</strong> All features standardized for distance-based algorithms</li>
                            <li><strong>Model diversity:</strong> Includes tree-based, linear, and kernel-based models</li>
                        </ul>
                    </section>
                </div>
            </div>
        </main>

        <!-- Footer -->
        <footer>
            <div>
                <div class="footer-content">
                    <div class="badges">
                        <a href="https://github.com/biofold/ppi-benchmark-fair">
                        <img src="https://img.shields.io/badge/ML_Results-Dashboard-blue" alt="ML Results Dashboard"></a>
                        <a href="https://www.python.org/">
                        <img src="https://img.shields.io/badge/Python-3.8%2B-blue.svg" alt="Python 3.8+"></a>
                        <a href="https://scikit-learn.org/">
                        <img src="https://img.shields.io/badge/scikit--learn-1.3%2B-orange" alt="scikit-learn"></a>
                        <a href="https://mlcommons.org/croissant/">
                        <img src="https://img.shields.io/badge/ML-Croissant_1.0-yellow" alt="MLCommons Croissant"></a>
                    </div>
                    
                    <div class="footer-links">
                        <a href="index.html" class="footer-link">
                            <i class="fas fa-home"></i> FAIR Metadata Home
                        </a>
                        <a href="https://github.com/biofold/ppi-benchmark-fair" class="footer-link" target="_blank">
                            <i class="fab fa-github"></i> Repository
                        </a>
                        <a href="https://elixir-europe.org/platforms/3d-bioinfo" class="footer-link" target="_blank">
                            <i class="fas fa-users"></i> ELIXIR 3D-BioInfo
                        </a>
                        <a href="https://scikit-learn.org/" class="footer-link" target="_blank">
                            <i class="fas fa-brain"></i> scikit-learn
                        </a>
                    </div>
                    
                    <div class="copyright">
                        <p>ELIXIR PPI Benchmark ML Results Dashboard • Generated on: {current_date}</p>
                        <p>Analysis Method: ClusterID-aware Cross-Validation with Feature Evaluation</p>
                    </div>
                </div>
            </div>
        </footer>
    </div>
</body>
</html>"""
    
    # Write HTML file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ HTML dashboard generated: {output_path}")
    
    return html_content

def main():
    """Main function to generate the HTML dashboard."""
    parser = argparse.ArgumentParser(
        description='Generate interactive HTML dashboard for ML results from ppi_ml_croissant.py'
    )
    
    parser.add_argument(
        '--results',
        type=str,
        required=True,
        help='Path to cluster_aware_cv_results.json file'
    )
    
    parser.add_argument(
        '--feature-eval',
        type=str,
        help='Path to feature_evaluation_report.json (optional)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='ml_results_dashboard.html',
        help='Output HTML file path (default: ml_results_dashboard.html)'
    )
    
    parser.add_argument(
        '--save-plots',
        type=str,
        help='Directory to save plot images (optional)'
    )
    
    args = parser.parse_args()
    
    print("""
╔══════════════════════════════════════════════════════════╗
║  ML Results HTML Dashboard Generator                     ║
║  for ELIXIR PPI Benchmark                                ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    # Load results
    print(f"📊 Loading results from: {args.results}")
    results = load_results_json(args.results)
    
    # Generate plots
    print("🎨 Generating performance plots...")
    plots = generate_performance_plots(results, args.save_plots)
    
    # Generate HTML dashboard
    print("📄 Generating HTML dashboard...")
    html_content = generate_html_dashboard(
        results=results,
        plots=plots,
        feature_eval_path=args.feature_eval,
        output_path=args.output
    )
    
    print(f"\n✅ Dashboard generation complete!")
    print(f"   HTML file: {args.output}")
    print(f"   Models analyzed: {len([k for k in results.keys() if k != 'cross_validation_settings'])}")
    
    if args.feature_eval:
        print(f"   Feature analysis included: Yes")
    
    print(f"\n📋 Dashboard features:")
    print("   • Interactive tabs for different analysis views")
    print("   • Performance metrics and visualizations")
    print("   • Model comparison tables")
    print("   • Feature analysis summary")
    print("   • Dataset and methodology information")
    print("   • Responsive design matching index.html style")
    
    return html_content

if __name__ == "__main__":
    main()
