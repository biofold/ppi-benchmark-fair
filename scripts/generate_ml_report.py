#!/usr/bin/env python3
"""
Dual-mode script to generate interactive HTML dashboards for:
1. ML prediction results (original functionality)
2. Feature evaluation reports (new functionality)

Updated styling to match FAIR Interactive Dashboard HTML template.
"""

import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
import os
import math
from typing import Dict, List, Any, Optional

# Set Plotly template
#pio.templates.default = "plotly_white"
#pio.templates.default = "plotly" #"seaborn"

# Add at the top of the script after imports
STANDARD_MARGINS = {
    'feature_plots': dict(t=60, b=140, l=80, r=40),
    'prediction_plots': dict(t=60, b=140, l=80, r=40),
    'tall_plots': dict(t=60, b=60, l=40, r=40),  # For plots with many items
}


# CSS / page style matching FAIR Interactive Dashboard
PAGE_STYLE = """
:root {
    --primary-color: #2c3e50;
    --secondary-color: #4689a3;
    --accent-color: #e74c3c;
    --light-bg: #f8f9fa;
    --success-color: #27ae60;
    --warning-color: #f39c12;
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    line-height: 1.6;
    color: #333;
    background-color: var(--light-bg);
    padding: 20px;
}

.container {
    max-width: 1400px;
    margin: 0 auto;
}

/* Header - Matching FAIR dashboard */
header {
    background: linear-gradient(135deg, var(--primary-color), #4689a3);
    color: white;
    padding: 30px 0;
    text-align: center;
    position: relative;
    overflow: hidden;
    margin-bottom: 40px;
    border-radius: 12px;
}

header::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-image: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" preserveAspectRatio="none"><path d="M0,0 L100,0 L100,100 Z" fill="rgba(255,255,255,0.05)"/></svg>');
    background-size: cover;
}

.header-content {
    position: relative;
    z-index: 1;
    padding: 0 20px;
}

h1 {
    font-size: 2.5rem;
    margin-bottom: 5px;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}

h2 {
    color: var(--primary-color);
    margin-bottom: 20px;
    font-size: 1.8rem;
    display: flex;
    align-items: center;
    gap: 10px;
}

h4 {
    color: white;
    margin: 5px 0 5px;
    font-size: 1.1rem;
}

.tagline {
    font-size: 1.3rem;
    opacity: 0.9;
    max-width: 800px;
    margin: 0 auto 30px;
}

.dashboard-link {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    background-color: rgba(255,255,255,0.15);
    color: white;
    padding: 6px 6px;
    border-radius: 10px;
    text-decoration: none;
    font-weight: 500;
    transition: all 0.3s ease;
    border: 2px solid rgba(255,255,255,0.3);
    margin: 10px 5px;
    margin-top: 25px;
}

.dashboard-link:hover {
    background-color: rgba(255,255,255,0.25);
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
}

/* Section styling */
.section {
    background-color: white;
    margin: 30px 0;
    padding: 30px;
    border-radius: 12px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
    border-left: 5px solid var(--secondary-color);
}

/* Key Metrics - Matching FAIR dashboard style */
.key-metrics {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 15px;
    margin: 40px 0;
}

.metric-card {
    background: linear-gradient(135deg, var(--primary-color), #4689a3);
    color: white;
    padding: 25px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
    transition: transform 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-5px);
}

.metric-value {
    font-size: 2.5rem;
    font-weight: bold;
    margin-bottom: 10px;
}

.metric-label {
    font-size: 1rem;
    opacity: 0.9;
}


/* Performance Table Styling */
.performance-table {
    width: 100%;
    text-align: right;
    border-collapse: collapse;
    margin: 30px 0;
    box-shadow: 0 5px 15px rgba(0,0,0,0.05);
    border-radius: 8px;
    overflow: hidden;
}

.performance-table th {
    background: linear-gradient(135deg, var(--primary-color), #4689a3);
    color: white;
    padding: 15px;
    text-align: right;
    font-weight: 600;
    border-bottom: 2px solid var(--secondary-color);
}

.performance-table td {
    padding: 15px;
    border-bottom: 1px solid #eee;
}

.performance-table tr:last-child td {
    border-bottom: none;
}

.performance-table tr:hover {
    background-color: #f8f9fa;
}

.performance-table .best-metric {
    background-color: rgba(39, 174, 96, 0.1);
    font-weight: 600;
    color: var(--success-color);
}

/* Figure Containers - Matching FAIR dashboard cards */
.figure-container {
    background-color: white;
    margin: 40px 0;
    padding: 30px;
    border-radius: 12px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
    border-left: 5px solid var(--secondary-color);
    transition: transform 0.3s ease;
}

.figure-container:hover {
    transform: translateY(-3px);
}

.figure-title {
    color: var(--primary-color);
    margin-bottom: 15px;
    font-size: 1.5rem;
    display: flex;
    align-items: center;
    gap: 10px;
}

.figure-title::before {
    content: "\\F080";
    font-family: "Font Awesome 6 Free";
}

.figure-description {
    color: #7f8c8d;
    margin-bottom: 25px;
    font-size: 1rem;
    line-height: 1.7;
}

.figure-embed {
    width: 100%;
    min-height: 400px;
    border: none;
    border-radius: 8px;
    box-shadow: 0 5px 15px rgba(0,0,0,0.05);
    overflow: visible;
}

/* Navigation Bar */
.nav-bar {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
    margin-bottom: 40px;
    position: sticky;
    top: 20px;
    z-index: 100;
}

.nav-bar ul {
    list-style-type: none;
    padding: 0;
    margin: 0;
    display: flex;
    flex-wrap: wrap;
    gap: 15px;
    justify-content: center;
}

.nav-bar li {
    display: inline;
}

.nav-bar a {
    text-decoration: none;
    color: var(--secondary-color);
    padding: 8px 16px;
    border-radius: 10px;
    border: 2px solid var(--secondary-color);
    transition: all 0.3s ease;
    font-weight: 600;
    font-size: 0.90rem;
}

.nav-bar a:hover {
    background-color: var(--secondary-color);
    color: white;
}

/* Control Buttons */
.controls {
    text-align: center;
    margin: 30px 0;
    padding: 25px;
    background-color: white;
    border-radius: 12px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
}

.toggle-button {
    background-color: var(--secondary-color);
    color: white;
    border: none;
    padding: 12px 25px;
    border-radius: 50px;
    cursor: pointer;
    font-size: 1rem;
    margin: 5px 10px;
    transition: all 0.3s ease;
    font-weight: 600;
}

.toggle-button:hover {
    background-color: #2980b9;
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(52, 152, 219, 0.3);
}

.toggle-button.hidden {
    background-color: var(--accent-color);
}

/* Badge for top performer */
.top-badge {
    display: inline-block;
    background: linear-gradient(135deg, var(--success-color), #219653);
    color: white;
    padding: 8px 20px;
    border-radius: 50px;
    font-weight: 700;
    font-size: 0.9rem;
    letter-spacing: 1px;
    margin: 15px 0;
    box-shadow: 0 4px 10px rgba(39, 174, 96, 0.3);
}

/* Footer */
footer {
    background-color: #386277;
    color: white;
    padding: 10px 0;
    text-align: center;
    margin-top: 60px;
    border-radius: 12px;
}

.footer-content {
    margin: 15px;
    padding: 0 10px;
}

.footer-links {
    display: flex;
    justify-content: center;
    gap: 30px;
    margin: 30px 0;
    flex-wrap: wrap;
}

.footer-link {
    color: rgba(255,255,255,0.8);
    text-decoration: none;
    transition: color 0.3s ease;
    display: flex;
    align-items: center;
    gap: 8px;
}

.footer-link:hover {
    color: white;
}

.copyright {
    margin: 10px;
    color: rgba(255,255,255,0.6);
    font-size: 0.9rem;
    line-height: 1.6;
}

/* Responsive Design */
@media (max-width: 768px) {
    h1 {
        font-size: 2.2rem;
    }

    .tagline {
        font-size: 1.1rem;
    }

    .nav-bar ul {
        flex-direction: column;
        align-items: center;
    }

    .nav-bar li {
        width: 100%;
        text-align: center;
    }

    .nav-bar a {
        display: block;
        width: 90%;
        margin: 5px auto;
    }

    .figure-embed {
        height: 500px;
    }

    .key-metrics {
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 15px;
    }

    .metric-value {
        font-size: 2rem;
    }

    .performance-table {
        font-size: 0.9rem;
    }

    .performance-table th,
    .performance-table td {
        padding: 10px;
    }
}

@media (max-width: 480px) {
    .figure-embed {
        height: 400px;
    }

    .toggle-button {
        display: block;
        width: 90%;
        margin: 10px auto;
    }

    .performance-table {
        display: block;
        overflow-x: auto;
    }
}
"""

JS_SCRIPT = """
<script>
    // JavaScript for interactive controls
    function toggleAllFigures(action) {
        const figures = document.querySelectorAll('.figure-container iframe');
        const buttons = document.querySelectorAll('.toggle-button');

        if (action === 'show') {
            figures.forEach(fig => {
                fig.style.display = 'block';
                fig.parentElement.style.display = 'block';
            });
            buttons[0].classList.add('hidden');
            buttons[1].classList.remove('hidden');
        } else {
            figures.forEach(fig => {
                fig.style.display = 'none';
            });
            buttons[0].classList.remove('hidden');
            buttons[1].classList.add('hidden');
        }
    }

    function expandAllFigures() {
        const figures = document.querySelectorAll('.figure-embed');
        figures.forEach(fig => {
            fig.style.height = '600px';
        });
        // Visual feedback
        showNotification('All figures expanded to full view');
    }

    function collapseAllFigures() {
        const figures = document.querySelectorAll('.figure-embed');
        figures.forEach(fig => {
            fig.style.height = '400px';
        });
        // Visual feedback
        showNotification('All figures collapsed to compact view');
    }

    // Show a temporary notification
    function showNotification(message) {
        // Remove existing notification if any
        const existingNotification = document.querySelector('.notification');
        if (existingNotification) {
            existingNotification.remove();
        }

        // Create new notification
        const notification = document.createElement('div');
        notification.className = 'notification';
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: var(--secondary-color);
            color: white;
            padding: 15px 25px;
            border-radius: 50px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            z-index: 1000;
            font-weight: 600;
            animation: slideIn 0.3s ease;
        `;

        // Add to body
        document.body.appendChild(notification);

        // Remove after 3 seconds
        setTimeout(() => {
            notification.style.animation = 'slideOut 0.3s ease';
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }

    // Add CSS for animations
    const style = document.createElement('style');
    style.textContent = `
        @keyframes slideIn {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        @keyframes slideOut {
            from { transform: translateX(0); opacity: 1; }
            to { transform: translateX(100%); opacity: 0; }
        }
    `;
    document.head.appendChild(style);

    // Smooth scrolling for navigation
    document.querySelectorAll('.nav-bar a').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            if (targetId !== '#') {
                const targetElement = document.querySelector(targetId);
                if (targetElement) {
                    window.scrollTo({
                        top: targetElement.offsetTop - 120,
                        behavior: 'smooth'
                    });
                }
            }
        });
    });

    // Lazy loading for iframes
    document.addEventListener("DOMContentLoaded", function() {
        const iframes = document.querySelectorAll('.figure-embed');

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const iframe = entry.target;
                    if (!iframe.dataset.loaded) {
                        iframe.dataset.loaded = true;
                        // Iframes load on src attribute, so no additional action needed
                    }
                }
            });
        }, { rootMargin: '100px' });

        iframes.forEach(iframe => observer.observe(iframe));

        // Set initial state
        document.querySelectorAll('.toggle-button')[1].classList.add('hidden');
    });
</script>
"""

def safe_log10_p(p_value: float) -> float:
    """Return -log10(p_value) safely. Allows p_value == 0 by capping to 1e-300."""
    if p_value is None or (isinstance(p_value, float) and math.isnan(p_value)):
        return 0.0
    try:
        capped = max(float(p_value), 1e-300)
        return -math.log10(capped)
    except Exception:
        return 0.0

def create_feature_plots(report: dict, output_dir: str, verbose: bool = False) -> Dict[str, str]:
    """
    Create feature analysis plots and save each plot as an independent HTML file.

    Args:
        report: Feature report dictionary loaded from JSON.
        output_dir: Directory path where figure HTML files will be written.
        verbose: If True, print progress messages.

    Returns:
        A mapping of plot keys to relative HTML file paths (relative to output_dir).
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    figs_dir = Path(output_dir) / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    plots_map: Dict[str, str] = {}

    if verbose:
        print("📊 Creating feature analysis visualizations...")

    # 1. Feature Importance Comparison (grouped bars)
    try:
        importance_data = report.get('feature_importance', {})
        methods = [('random_forest', 'Random Forest', '#4689a3'),
                   ('logistic_regression', 'Logistic Regression', '#e74c3c'),
                   ('mutual_information', 'Mutual Information', '#27ae60')]

        all_features_data = []
        for method_key, method_name, color in methods:
            method_entry = importance_data.get(method_key, {})
            top_features = method_entry.get('top_features', [])
            for feature, importance in top_features[:20]:
                all_features_data.append({'Feature': feature, 'Importance': importance, 'Method': method_name, 'Color': color})

        if all_features_data:
            df_imp = pd.DataFrame(all_features_data)
            unique_features = df_imp['Feature'].unique().tolist()

            fig = go.Figure()
            for method_name, group in df_imp.groupby('Method'):
                series = group.set_index('Feature')['Importance'].reindex(unique_features).fillna(0)
                color = group['Color'].iloc[0]
                fig.add_trace(go.Bar(
                    name=method_name,
                    x=unique_features,
                    y=series.values,
                    marker_color=color,
                    hovertemplate='<b>%{x}</b><br>Method: ' + method_name + '<br>Importance: %{y:.4f}<extra></extra>'
                ))

            # Add axis labels
            fig.update_layout(
                barmode='group',
                xaxis_tickangle=-45,
                height=520,
                margin=dict(t=20, b=180, l=60, r=20),
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family="Segoe UI, Tahoma, Geneva, Verdana, sans-serif"),
                xaxis_title="Features",
                yaxis_title="Importance Score"
            )

            fig.update_xaxes(
                showline=True,
                linecolor='rgba(0,0,0,0.3)',
                mirror=True,
            )

            fig.update_yaxes(
                showline=True,
                linecolor='rgba(0,0,0,0.3)',
                mirror=True,
            )

            out_path = figs_dir / "feature_importance.html"
            fig.write_html(str(out_path), include_plotlyjs='cdn')
            plots_map['feature_importance'] = os.path.relpath(out_path, output_dir)
            if verbose:
                print("  ✅ feature_importance saved")
    except Exception as e:
        if verbose:
            print(f"  ⚠️  feature_importance error: {e}")

    # 2. Correlation Analysis (highly correlated pairs)
    try:
        corr_info = report.get('correlation_analysis', {})
        pairs = corr_info.get('highly_correlated_pairs', [])[:30]
        if pairs:
            pair_labels = []
            correlations = []
            abs_correlations = []
            for p in pairs:
                f1 = p.get('feature1', '')
                f2 = p.get('feature2', '')
                corr = float(p.get('correlation', 0.0))
                # Truncate labels if too long
                def trunc(s): return s if len(s) <= 40 else s[:37] + '...'
                pair_labels.append(f"{trunc(f1)} ↔ {trunc(f2)}")
                correlations.append(corr)
                abs_correlations.append(abs(corr))

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=correlations,
                y=pair_labels,
                orientation='h',
                marker=dict(color=abs_correlations, colorscale='RdYlBu_r', showscale=True),
                hovertemplate='<b>%{y}</b><br>Correlation: %{x:.3f}<extra></extra>'
            ))
            fig.update_layout(
                height=max(400, len(pair_labels) * 28),
                margin=dict(t=20, b=20, l=20, r=20),
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family="Segoe UI, Tahoma, Geneva, Verdana, sans-serif"),
                xaxis_title="Correlation Coefficient (r)",
                yaxis_title="Feature Pairs"
            )

            fig.update_xaxes(
                showline=True,
                linecolor='rgba(0,0,0,0.3)',
                mirror=True,
                showgrid=True,
                gridwidth=0.5,
                gridcolor='rgba(0,0,0,0.15)',
            )

            fig.update_yaxes(
                showline=True,
                linecolor='rgba(0,0,0,0.3)',
                mirror=True,
            )

            out_path = figs_dir / "correlation_plot.html"
            fig.write_html(str(out_path), include_plotlyjs='cdn')
            plots_map['correlation_plot'] = os.path.relpath(out_path, output_dir)
            if verbose:
                print("  ✅ correlation_plot saved")
    except Exception as e:
        if verbose:
            print(f"  ⚠️  correlation_plot error: {e}")

    # 3. PCA Explained Variance
    try:
        pca_data = report.get('pca_analysis', {})
        explained_variance = pca_data.get('explained_variance', [])
        cumulative_variance = pca_data.get('cumulative_variance', [])
        if explained_variance:
            pcs = [f"PC{i+1}" for i in range(len(explained_variance))]
            fig = go.Figure()
            fig.add_trace(go.Bar(x=pcs, y=explained_variance, marker_color='#3498db', name='Individual'))
            if cumulative_variance:
                fig.add_trace(go.Scatter(x=pcs, y=cumulative_variance, name='Cumulative', line=dict(color='#e74c3c', width=3), marker=dict(size=6)))
            fig.update_layout(
                height=520,
                margin=dict(t=20, b=100, l=60, r=20),
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family="Segoe UI, Tahoma, Geneva, Verdana, sans-serif"),
                xaxis_title="Principal Components",
                yaxis_title="Explained Variance Ratio"
            )

            fig.update_xaxes(
                showline=True,
                linecolor='rgba(0,0,0,0.3)',
                mirror=True,
            )

            fig.update_yaxes(
                showline=True,
                linecolor='rgba(0,0,0,0.3)',
                mirror=True,
                showgrid=True,
                gridwidth=0.5,
                gridcolor='rgba(0,0,0,0.15)',
            )

            out_path = figs_dir / "pca_variance.html"
            fig.write_html(str(out_path), include_plotlyjs='cdn')
            plots_map['pca_variance'] = os.path.relpath(out_path, output_dir)
            if verbose:
                print("  ✅ pca_variance saved")
    except Exception as e:
        if verbose:
            print(f"  ⚠️  pca_variance error: {e}")

    # 4. ANOVA Feature-Target Relationship (scatter) - fix: allow p == 0
    try:
        ft_analysis = report.get('feature_target_analysis', {})
        all_scores = ft_analysis.get('all_scores', {})
        features_data = []

        if all_scores:
            for feature, scores in all_scores.items():
                anova_f = scores.get('anova_f')
                anova_p = scores.get('anova_p')
                mutual_info = scores.get('mutual_info', 0)

                # Skip if F is None or NaN or <= 0. ANOVA F <= 0 is not meaningful.
                if anova_f is None:
                    continue
                try:
                    anova_f_val = float(np.array(anova_f).tolist()[0]) if isinstance(anova_f, (list, tuple, np.ndarray)) else float(anova_f)
                except Exception:
                    continue
                if math.isnan(anova_f_val) or anova_f_val <= 0:
                    continue

                # mutual_info safe
                if mutual_info is None:
                    mutual_info_val = 0.0
                else:
                    try:
                        mutual_info_val = float(np.array(mutual_info).tolist()[0]) if isinstance(mutual_info, (list, tuple, np.ndarray)) else float(mutual_info)
                        if math.isnan(mutual_info_val):
                            mutual_info_val = 0.0
                    except Exception:
                        mutual_info_val = 0.0

                # p-value handling: allow p==0; safe_log10_p will cap it
                try:
                    if anova_p is None:
                        anova_p_val = float('nan')
                    else:
                        anova_p_val = float(np.array(anova_p).tolist()[0]) if isinstance(anova_p, (list, tuple, np.ndarray)) else float(anova_p)
                except Exception:
                    anova_p_val = float('nan')

                log10_p = safe_log10_p(anova_p_val)

                features_data.append({
                    'Feature': feature,
                    'anova_f': anova_f_val,
                    'anova_p': anova_p_val,
                    'mutual_info': mutual_info_val,
                    'log10_p': log10_p,
                    'significant': bool(scores.get('significant_anova', False))
                })

            if features_data:
                df_features = pd.DataFrame(features_data)

                # Create scatter plot - use anova_f vs log10_p
                fig = px.scatter(
                    df_features,
                    x='anova_f',
                    y='log10_p',
                    color='significant',
                    size='mutual_info',
                    hover_name='Feature',
                    color_discrete_map={True: '#2ecc71', False: '#e74c3c'}
                )
                # Add axis labels
                fig.update_layout(
                    height=620,
                    margin=dict(t=20, b=80, l=60, r=20),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(family="Segoe UI, Tahoma, Geneva, Verdana, sans-serif"),
                    xaxis_title="ANOVA F-statistic",
                    yaxis_title="-log₁₀(p-value)",
                    legend_title="Significant"
                )

                fig.update_xaxes(
                    showline=True,
                    linecolor='rgba(0,0,0,0.3)',
                    mirror=True,
                    showgrid=True,
                    gridwidth=0.5,
                    gridcolor='rgba(0,0,0,0.15)',
                )

                fig.update_yaxes(
                    showline=True,
                    linecolor='rgba(0,0,0,0.3)',
                    mirror=True,
                    showgrid=True,
                    gridwidth=0.5,
                    gridcolor='rgba(0,0,0,0.15)',
                )

                out_path = figs_dir / "feature_target_relationship.html"
                fig.write_html(str(out_path), include_plotlyjs='cdn')
                plots_map['feature_target_relationship'] = os.path.relpath(out_path, output_dir)
                if verbose:
                    print("  ✅ feature_target_relationship saved")
            else:
                if verbose:
                    print("  ⚠️  No valid ANOVA data after filtering (all NaN or invalid F-values)")
    except Exception as e:
        if verbose:
            print(f"  ⚠️  feature_target_relationship error: {e}")

    # 5. ANOVA Top (bar) and Mutual Info Top (bar)
    try:
        all_scores = report.get('feature_target_analysis', {}).get('all_scores', {})
        # Build DataFrame with anova_f and mutual_info where present
        rows = []
        for feat, s in all_scores.items():
            try:
                fval = s.get('anova_f')
                pval = s.get('anova_p')
                mi = s.get('mutual_info', 0)
                if fval is None or (isinstance(fval, float) and math.isnan(fval)):
                    continue
                rows.append({'Feature': feat, 'anova_f': float(fval), 'mutual_info': float(mi)})
            except Exception:
                continue

        if rows:
            df_rows = pd.DataFrame(rows)
            top_n = min(20, len(df_rows))

            # ANOVA Top Features
            df_top_anova = df_rows.nlargest(top_n, 'anova_f')
            fig_anova = px.bar(df_top_anova, x='anova_f', y='Feature', orientation='h', color='anova_f', color_continuous_scale='Blues')
            fig_anova.update_traces(marker=
                    dict(line =
                        dict (color='rgba(0,0,0,0.3)',width=0.5)
                        )
            )

            fig_anova.update_layout(
                height=max(400, top_n * 28),
                margin=dict(t=20, b=80, l=20, r=20),
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family="Segoe UI, Tahoma, Geneva, Verdana, sans-serif"),
                xaxis_title="ANOVA F-statistic",
                yaxis_title="Features",
                coloraxis_colorbar_title="F-value"
            )

            fig_anova.update_xaxes(
                showline=True,
                linecolor='rgba(0,0,0,0.3)',
                mirror=True,
                showgrid=True,
                gridwidth=0.5,
                gridcolor='rgba(0,0,0,0.15)',
            )

            fig_anova.update_yaxes(
                showline=True,
                linecolor='rgba(0,0,0,0.3)',
                mirror=True,
            )

            out_path = figs_dir / "feature_top_anova.html"
            fig_anova.write_html(str(out_path), include_plotlyjs='cdn')
            plots_map['feature_top_anova'] = os.path.relpath(out_path, output_dir)
            if verbose:
                print("  ✅ feature_top_anova saved")

            # Mutual Information Top Features
            df_top_mi = df_rows.nlargest(top_n, 'mutual_info')
            fig_mi = px.bar(df_top_mi, x='mutual_info', y='Feature', orientation='h', color='mutual_info', color_continuous_scale='Viridis')
            fig_mi.update_layout(
                height=max(400, top_n * 28),
                margin=dict(t=20, b=80, l=20, r=20),
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family="Segoe UI, Tahoma, Geneva, Verdana, sans-serif"),
                xaxis_title="Mutual Information Score",
                yaxis_title="Features",
                coloraxis_colorbar_title="MI Score"
            )

            fig_mi.update_xaxes(
                showline=True,
                linecolor='rgba(0,0,0,0.3)',
                mirror=True,
                showgrid=True,
                gridwidth=0.5,
                gridcolor='rgba(0,0,0,0.15)',
            )

            fig_mi.update_yaxes(
                showline=True,
                linecolor='rgba(0,0,0,0.3)',
                mirror=True,
            )

            out_path = figs_dir / "feature_top_mi.html"
            fig_mi.write_html(str(out_path), include_plotlyjs='cdn')
            plots_map['feature_top_mi'] = os.path.relpath(out_path, output_dir)
            if verbose:
                print("  ✅ feature_top_mi saved")
    except Exception as e:
        if verbose:
            print(f"  ⚠️  top feature bars error: {e}")

    # 6. Feature Statistics Heatmap (first 20 features)
    '''
    try:
        basic_stats = report.get('basic_statistics', {})
        dataset_info = report.get('dataset_info', {})
        feature_names = dataset_info.get('feature_names', [])[:20]
        stats_data = []
        for feat in feature_names:
            if feat in basic_stats:
                st = basic_stats[feat]
                stats_data.append({
                    'Feature': feat,
                    'Mean': st.get('mean', 0),
                    'Std': st.get('std', 0),
                    'Min': st.get('min', 0),
                    'Max': st.get('max', 0),
                    'Median': st.get('median', 0)
                })
        if stats_data:
            df_stats = pd.DataFrame(stats_data).set_index('Feature').T
            fig = go.Figure(data=go.Heatmap(z=df_stats.values, x=df_stats.columns, y=df_stats.index, colorscale='Viridis'))
            fig.update_layout(
                height=520,
                margin=dict(t=20, b=120, l=60, r=20),
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family="Segoe UI, Tahoma, Geneva, Verdana, sans-serif"),
                xaxis_title="Features",
                yaxis_title="Statistics"
            )
            out_path = figs_dir / "feature_stats_heatmap.html"
            fig.write_html(str(out_path), include_plotlyjs='cdn')
            plots_map['feature_stats_heatmap'] = os.path.relpath(out_path, output_dir)
            if verbose:
                print("  ✅ feature_stats_heatmap saved")
    except Exception as e:
        if verbose:
            print(f"  ⚠️  feature_stats_heatmap error: {e}")
    '''

    if verbose:
        print(f"✅ Created {len(plots_map)} feature HTML figures in {figs_dir}")

    return plots_map


def generate_feature_dashboard(report: dict, plots_map: Dict[str, str], output_path: str = "feature_analysis_dashboard.html", verbose: bool = False):
    """
    Generate an HTML dashboard page that embeds separate figure HTML files (one per plot).
    Updated styling to match FAIR Interactive Dashboard.
    """

    out_dir = Path(output_path).parent or Path('.')
    rel = lambda p: os.path.relpath(p, start=out_dir)

    current_date = datetime.now().strftime("%Y-%m-%d")

    # Extract basic metrics for key metrics section
    dataset_info = report.get('dataset_info', {})
    num_features = len(dataset_info.get('feature_names', []))
    num_samples = dataset_info.get('num_samples', 0)

    basic_stats = report.get('basic_statistics', {})
    avg_mean = np.mean([stats.get('mean', 0) for stats in basic_stats.values()]) if basic_stats else 0
    avg_std = np.mean([stats.get('std', 0) for stats in basic_stats.values()]) if basic_stats else 0

    # Build cards in the order we expect (removed test_plot)
    cards = [
        ('Figure 1: Feature Importance Comparison', plots_map.get('feature_importance')),
        ('Figure 2: Top Features by ANOVA F-value', plots_map.get('feature_top_anova')),
        ('Figure 3: Top Features by Mutual Information', plots_map.get('feature_top_mi')),
        ('Figure 4: Feature-Target Relationship (ANOVA vs -log10(p))', plots_map.get('feature_target_relationship')),
        ('Figure 5: Highly Correlated Feature Pairs', plots_map.get('correlation_plot')),
        ('Figure 6: PCA Explained Variance', plots_map.get('pca_variance')),
        ('Figure 7: Feature Statistics Heatmap', plots_map.get('feature_stats_heatmap')),
    ]

    # HTML header
    html_parts: List[str] = []
    html_parts.append("<!DOCTYPE html>")
    html_parts.append("<html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>")
    html_parts.append(f"<title>Feature Analysis Report - {current_date}</title>")
    html_parts.append('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">')
    html_parts.append('<script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>')
    html_parts.append(f"<style>{PAGE_STYLE}</style>")
    html_parts.append("</head><body><div class='container'>")

    # Header matching FAIR dashboard with updated subtitle
    html_parts.append(f"""
    <header>
        <div class="header-content">
            <h1><strong>Feature Analysis Report</strong></h1>
            <h4><strong>ELIXIR Protein-Protein Interaction Benchmark</strong></h4>
            <div>
                <a href="../index.html" class="dashboard-link">
                    <i class="fas fa-home"></i> PPI Benchmark FAIR
                </a>
                <a href="./prediction-report.html" class="dashboard-link">
                    <i class="fa-solid fa-brain"></i> Machine Learning
                </a>
                <a href="./fair-analysis.html" class="dashboard-link">
                    <i class="fas fa-bullseye"></i> FAIRfication
                </a>
                <a href="https://github.com/biofold/ppi-benchmark-fair" class="dashboard-link" target="_blank">
                    <i class="fab fa-github"></i> Repository
                </a>
            </div>
        </div>
    </header>
    """)

    # Main content
    html_parts.append("<main class='container'>")

    # Key Metrics Section (removed green box)
    html_parts.append(f"""
    <div class="section" id="key-metrics">
        <h2><i class="fas fa-clipboard-check"></i> Key Statistics</h2>
        <p>Comprehensive analysis of dataset features including importance scores, correlations, and statistical properties.</p>

        <div class="key-metrics">
            <div class="metric-card">
                <div class="metric-value">{num_features}</div>
                <div class="metric-label">Features</div>
            </div>

            <div class="metric-card">
                <div class="metric-value">{num_samples}</div>
                <div class="metric-label">Samples</div>
            </div>

            <div class="metric-card">
                <div class="metric-value">{avg_mean:.1f}</div>
                <div class="metric-label">Avg Mean</div>
            </div>

            <div class="metric-card">
                <div class="metric-value">{avg_std:.2f}</div>
                <div class="metric-label">Avg Std Dev</div>
            </div>
        </div>
    </div>
    """)

    # Navigation Bar (commented out but available)
    html_parts.append("""
    <!--
    <div class="nav-bar">
        <ul>
            <li><a href="#figure1">Feature Importance</a></li>
            <li><a href="#figure2">ANOVA Top</a></li>
            <li><a href="#figure3">Mutual Info</a></li>
            <li><a href="#figure4">Target Relationship</a></li>
            <li><a href="#figure5">Correlations</a></li>
            <li><a href="#figure6">PCA Variance</a></li>
            <li><a href="#figure7">Statistics</a></li>
        </ul>
    </div>
    -->
    """)

    # Control Buttons (commented out but available)
    html_parts.append("""
    <!--
    <div class="controls">
        <p style="margin-bottom: 15px; color: var(--primary-color); font-weight: 600;">Dashboard Controls:</p>
        <button class="toggle-button" onclick="toggleAllFigures('show')">
            <i class="fas fa-eye"></i> Show All Figures
        </button>
        <button class="toggle-button hidden" onclick="toggleAllFigures('hide')">
            <i class="fas fa-eye-slash"></i> Hide All Figures
        </button>
        <button class="toggle-button" onclick="expandAllFigures()">
            <i class="fas fa-expand"></i> Expand All
        </button>
        <button class="toggle-button" onclick="collapseAllFigures()">
            <i class="fas fa-compress"></i> Collapse All
        </button>
    </div>
    -->
    """)

    # Figures Section
    html_parts.append(f'<div id="figures">')

    # Add figure cards
    figure_counter = 1
    for title, rel_path in cards:
        if not rel_path:
            continue

        embed_src = rel(Path(output_path).parent / rel_path)

        # Get description based on figure type
        descriptions = {
            'Feature Importance Comparison': 'Shows feature importance scores from multiple methods (Random Forest, Logistic Regression, Mutual Information).',
            'Top Features by ANOVA F-value': 'Displays top features ranked by ANOVA F-statistic, indicating strong relationship with target variable.',
            'Top Features by Mutual Information': 'Shows top features ranked by mutual information score, measuring dependency between feature and target.',
            'Feature-Target Relationship (ANOVA vs -log10(p))': 'Scatter plot comparing ANOVA F-statistic vs -log10(p-value) with mutual information as bubble size.',
            'Highly Correlated Feature Pairs': 'Horizontal bar chart showing the most highly correlated feature pairs in the dataset.',
            'PCA Explained Variance': 'Bar chart showing explained variance ratio for principal components with cumulative variance line.',
            'Feature Statistics Heatmap': 'Heatmap visualization of basic statistics (mean, std, min, max, median) for the top features.',
        }

        description = descriptions.get(title.split(': ')[1] if ': ' in title else title, 'Feature analysis visualization.')

        html_parts.append(f"""
        <div class="figure-container" id="figure{figure_counter}">
            <h2 class="figure-title">{title}</h2>
            <p class="figure-description">{description}</p>
            <div style="padding: 50px 20px; background: white; border-radius: 8px; margin: 10px 0;">
            <iframe class="figure-embed" src="{embed_src}" loading="lazy" title="{title}"></iframe>
        </div>
        </div>
        """)
        figure_counter += 1

    html_parts.append('</div>')
    html_parts.append("</main>")

    # Footer with updated badges
    html_parts.append(f"""
    <footer id="footer">
        <div class="container">
            <div class="footer-content">
                Evaluation of machine learning features and their properties on the ELIXIR PPI Benchmark
            </div>
            <div class="footer-content" style="margin-top:25px;">
                <a href="https://github.com/biofold/ppi-benchmark-fair">
                <img src="https://img.shields.io/badge/FAIR_Score-87.5%2F100-brightgreen" alt="FAIR Score: 87.5/100"></a>
                <a href="https://creativecommons.org/licenses/by/4.0/">
                <img src="https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg" alt="License: CC BY 4.0"></a>
                <a href="https://doi.org/10.5281/zenodo.XXXXXXX">
                <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg" alt="DOI"></a>
                <a href="https://www.python.org/">
                <img src="https://img.shields.io/badge/Python-3.8%2B-blue.svg" alt="Python 3.8+"></a>
                <a href="https://schema.org/">
                <img src="https://img.shields.io/badge/Metadata-Schema.org%2BBioschemas-blue" alt="Schema.org+Bioschemas"></a>
                <a href="https://mlcommons.org/croissant/">
                <img src="https://img.shields.io/badge/ML-Croissant_1.0-yellow" alt="MLCommons Croissant"></a>
            </div>
            <div class="copyright">
                <p>Feature Analysis Report • Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>
        </div>
    </footer>
    """)

    # Add JavaScript
    html_parts.append(JS_SCRIPT)

    html_parts.append("</div></body></html>")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(html_parts))

    if verbose:
        print(f"✅ Feature analysis dashboard saved: {output_path}")


def create_prediction_plots(results: dict, output_dir: str, verbose: bool = False) -> Dict[str, str]:
    """
    Create prediction result plots and write them as separate HTML files.
    Returns mapping of plot keys to relative HTML paths (relative to output_dir).
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    figs_dir = Path(output_dir) / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    plots_map: Dict[str, str] = {}

    # FIXED: Extract model names correctly
    metadata_keys = {'cross_validation_settings', 'metadata', 'dataset_info', 'timestamp',
                     'model_settings', 'experiment_info', 'feature_info'}

    # Identify actual model results
    model_names = []
    for key in results.keys():
        if key in metadata_keys:
            continue
        if isinstance(results[key], dict):
            if 'cv_metrics' in results[key] or 'fold_metrics' in results[key]:
                model_names.append(key)

    if not model_names:
        # Fallback: look for common model names
        potential_models = ['RandomForest', 'LogisticRegression', 'SVM', 'XGBoost',
                           'GradientBoosting', 'DecisionTree', 'KNN', 'NeuralNetwork',
                           'AdaBoost', 'GaussianNB', 'MLP', 'LGBM', 'CatBoost']

        for key in results.keys():
            if any(model_name.lower() in key.lower() for model_name in potential_models):
                model_names.append(key)

    # Final fallback
    if not model_names:
        model_names = [k for k in results.keys() if k not in metadata_keys and k != 'cross_validation_settings']

    if not model_names:
        if verbose:
            print("⚠️  No valid model results found in JSON")
        return plots_map

    # Basic performance comparison bar chart (accuracy/precision/recall/f1) with error bars
    try:
        metrics = ['accuracy', 'precision', 'recall', 'f1']

        # Prepare data for error bars
        error_data = {}
        for metric in metrics:
            error_data[metric] = {'values': [], 'errors': []}

        # Calculate standard errors for each model and metric
        for name in model_names:
            fold_metrics = results[name].get('fold_metrics', {})
            cv_metrics = results[name].get('cv_metrics', {})

            for metric in metrics:
                # Get the mean value
                mean_value = cv_metrics.get(metric, 0)

                # Get fold scores for this metric
                if metric == 'accuracy':
                    fold_scores = fold_metrics.get('accuracies', [])
                elif metric == 'precision':
                    fold_scores = fold_metrics.get('precisions', [])
                elif metric == 'recall':
                    fold_scores = fold_metrics.get('recalls', [])
                elif metric == 'f1':
                    fold_scores = fold_metrics.get('f1_scores', [])
                else:
                    fold_scores = []

                # Calculate standard error
                if len(fold_scores) > 1:
                    std_err = np.std(fold_scores) / np.sqrt(len(fold_scores))
                else:
                    std_err = 0

                error_data[metric]['values'].append(mean_value)
                error_data[metric]['errors'].append(std_err)

        # Create the bar chart with error bars
        fig = go.Figure()

        colors = ['#4689a3', '#e74c3c', '#27ae60', '#f39c12']  # Different colors for each metric

        for i, metric in enumerate(metrics):
            values = error_data[metric]['values']
            errors = error_data[metric]['errors']

            # Add bar trace with error bars
            fig.add_trace(go.Bar(
                name=metric.capitalize(),
                x=model_names,
                y=values,
                error_y=dict(
                    type='data',
                    array=errors,
                    visible=True,
                    thickness=2,
                    width=6,
                    color='rgba(0,0,0,0.6)'
                ),
                marker_color=colors[i],
                hovertemplate='<b>%{x}</b><br>' +
                            f'{metric.capitalize()}: %{{y:.3f}}<br>' +
                            'Standard Error: ±%{customdata[0]:.3f}<extra></extra>',
                customdata=np.array([errors]).T  # For hover display
            ))

        # Customize the layout with axis labels
        fig.update_layout(
            barmode='group',
            height=520,
            margin=dict(t=20, b=120, l=60, r=20),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Segoe UI, Tahoma, Geneva, Verdana, sans-serif"),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            xaxis_title="Models",
            yaxis_title="Score"
        )

        # Update x-axis and y-axis with better formatting
        fig.update_xaxes(
            showline=True,
            mirror=True,
            linecolor='rgba(0,0,0,0.3)',
            tickformat='.2f'
        )

        fig.update_yaxes(
            range=[0, 1.05],  # For metrics normalized between 0-1
            showline=True,
            mirror=True,
            linecolor='rgba(0,0,0,0.3)',
            tickformat='.2f'
        )

        out_path = figs_dir / "performance_comparison.html"
        fig.write_html(str(out_path), include_plotlyjs='cdn')
        plots_map['performance_comparison'] = os.path.relpath(out_path, output_dir)
        if verbose:
            print("  ✅ performance_comparison with error bars saved")

    except Exception as e:
        if verbose:
            print(f"  ⚠️ performance_comparison error: {e}")

    # F1 per fold line plot with axis labels
    try:
        fig = go.Figure()
        for name in model_names:
            fold_metrics = results[name].get('fold_metrics', {})
            fold_f1s = fold_metrics.get('f1_scores', [])
            if fold_f1s:
                fig.add_trace(go.Scatter(
                    x=list(range(1, len(fold_f1s) + 1)),
                    y=fold_f1s,
                    mode='lines+markers',
                    name=name,
                    line=dict(width=3),
                    marker=dict(size=8),
                    hovertemplate=f'<b>{name}</b><br>' +
                            f'F1: %{{y:.3f}}<br>'
                ))

        # Add axis labels
        fig.update_layout(
            height=520,
            margin=dict(t=20, b=120, l=60, r=20),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Segoe UI, Tahoma, Geneva, Verdana, sans-serif"),
            xaxis_title="Cross-Validation Fold",
            yaxis_title="F1 Score",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        # Add grid lines
        fig.update_xaxes(
            dtick=1,
            showline=True,
            linecolor='rgba(0,0,0,0.3)',
            mirror=True,
            showgrid=True,
            gridwidth=0.5,
            gridcolor='rgba(0,0,0,0.15)',
            tickmode='linear'
        )

        fig.update_yaxes(
            showline=True,
            linecolor='rgba(0,0,0,0.3)',
            mirror=True,
            showgrid=True,
            gridwidth=0.5,
            gridcolor='rgba(0,0,0,0.15)',
            range=[0, 1.0],
            tickformat='.2f'
        )

        out_path = figs_dir / "f1_per_fold.html"
        fig.write_html(str(out_path), include_plotlyjs='cdn')
        plots_map['f1_per_fold'] = os.path.relpath(out_path, output_dir)
        if verbose:
            print("  ✅ f1_per_fold saved")
    except Exception as e:
        if verbose:
            print(f"  ⚠️ f1_per_fold error: {e}")

    # Radar chart with axis labels
    try:
        categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        fig = go.Figure()

        colors = ['#4689a3', '#e74c3c', '#27ae60', '#f39c12', '#9b59b6']

        for idx, name in enumerate(model_names):
            cv = results[name].get('cv_metrics', {})
            values = [
                cv.get('accuracy', 0),
                cv.get('precision', 0),
                cv.get('recall', 0),
                cv.get('f1', 0),
                cv.get('roc_auc', 0) or 0
            ]
            vals = values + [values[0]]
            cats = categories + [categories[0]]
            fig.add_trace(go.Scatterpolar(
                r=vals,
                theta=cats,
                fill='toself',
                name=name,
                line=dict(color=colors[idx % len(colors)], width=2)
            ))

        # Add axis labels and formatting
        fig.update_layout(
            height=620,
            margin=dict(t=20, b=120, l=20, r=20),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Segoe UI, Tahoma, Geneva, Verdana, sans-serif"),
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickformat='.2f',
                    gridcolor='rgba(0,0,0,0.1)',
                    linecolor='rgba(0,0,0,0.3)'
                ),
                angularaxis=dict(
                    gridcolor='rgba(0,0,0,0.1)',
                    linecolor='rgba(0,0,0,0.3)'
                )
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        out_path = figs_dir / "radar_chart.html"
        fig.write_html(str(out_path), include_plotlyjs='cdn')
        plots_map['radar_chart'] = os.path.relpath(out_path, output_dir)
        if verbose:
            print("  ✅ radar_chart saved")
    except Exception as e:
        if verbose:
            print(f"  ⚠️ radar_chart error: {e}")

    return plots_map


def generate_prediction_dashboard(results: dict, plots_map: Dict[str, str], output_path: str = "prediction_dashboard.html", verbose: bool = False):
    """
    Create a prediction-dashboard HTML that embeds the separate plot HTML files (stacked).
    Updated styling to match FAIR Interactive Dashboard.
    Includes performance summary table with standard errors.
    """
    out_dir = Path(output_path).parent or Path('.')
    current_date = datetime.now().strftime("%Y-%m-%d")

    # FIXED: Only consider keys that have 'cv_metrics' as actual model results
    # Common metadata keys to exclude
    metadata_keys = {'cross_validation_settings', 'metadata', 'dataset_info', 'timestamp',
                     'model_settings', 'experiment_info', 'feature_info'}

    # Identify actual model results by checking for cv_metrics or fold_metrics
    model_names = []
    for key in results.keys():
        if key in metadata_keys:
            continue
        # Check if this key contains model results (has cv_metrics or fold_metrics)
        if isinstance(results[key], dict):
            if 'cv_metrics' in results[key] or 'fold_metrics' in results[key]:
                model_names.append(key)

    # If the above doesn't work, fall back to explicit list of known model names
    if not model_names:
        # Common ML model names to look for
        potential_models = ['RandomForest', 'LogisticRegression', 'SVM', 'XGBoost',
                           'GradientBoosting', 'DecisionTree', 'KNN', 'NeuralNetwork',
                           'AdaBoost', 'GaussianNB', 'MLP', 'LGBM', 'CatBoost']

        for key in results.keys():
            if any(model_name.lower() in key.lower() for model_name in potential_models):
                model_names.append(key)

    # If still no models found, use all keys except known metadata
    if not model_names:
        model_names = [k for k in results.keys() if k not in metadata_keys and k != 'cross_validation_settings']

    best_model = None
    best_f1 = 0
    avg_f1 = 0

    # Prepare data for performance table with standard errors
    performance_data = []
    for name in model_names:
        if name not in results:
            continue

        model_result = results[name]
        cv_metrics = model_result.get('cv_metrics', {})
        f1 = cv_metrics.get('f1', 0)
        accuracy = cv_metrics.get('accuracy', 0)
        precision = cv_metrics.get('precision', 0)
        recall = cv_metrics.get('recall', 0)
        roc_auc = cv_metrics.get('roc_auc', 0) or 0

        # Get standard errors from fold metrics if available
        fold_metrics = model_result.get('fold_metrics', {})

        # Calculate standard error for F1
        f1_scores = fold_metrics.get('f1_scores', [])
        f1_std_err = np.std(f1_scores) / np.sqrt(len(f1_scores)) if len(f1_scores) > 1 else 0

        # Calculate standard error for accuracy
        accuracy_scores = fold_metrics.get('accuracies', [])
        accuracy_std_err = np.std(accuracy_scores) / np.sqrt(len(accuracy_scores)) if len(accuracy_scores) > 1 else 0

        # Calculate standard error for precision
        precision_scores = fold_metrics.get('precisions', [])
        precision_std_err = np.std(precision_scores) / np.sqrt(len(precision_scores)) if len(precision_scores) > 1 else 0

        # Calculate standard error for recall
        recall_scores = fold_metrics.get('recalls', [])
        recall_std_err = np.std(recall_scores) / np.sqrt(len(recall_scores)) if len(recall_scores) > 1 else 0

        # Calculate standard error for ROC-AUC
        roc_auc_scores = fold_metrics.get('roc_aucs', [])
        roc_auc_std_err = np.std(roc_auc_scores) / np.sqrt(len(roc_auc_scores)) if len(roc_auc_scores) > 1 else 0

        avg_f1 += f1
        if f1 > best_f1:
            best_f1 = f1
            best_model = name

        performance_data.append({
            'Model': name,
            'Accuracy': accuracy,
            'Accuracy_SE': accuracy_std_err,
            'Precision': precision,
            'Precision_SE': precision_std_err,
            'Recall': recall,
            'Recall_SE': recall_std_err,
            'F1-Score': f1,
            'F1-Score_SE': f1_std_err,
            'ROC-AUC': roc_auc,
            'ROC-AUC_SE': roc_auc_std_err
        })

    if model_names:
        avg_f1 /= len(model_names)

    # Find best metrics for highlighting (ignoring standard errors for comparison)
    best_accuracy = max([d['Accuracy'] for d in performance_data]) if performance_data else 0
    best_precision = max([d['Precision'] for d in performance_data]) if performance_data else 0
    best_recall = max([d['Recall'] for d in performance_data]) if performance_data else 0
    best_roc_auc = max([d['ROC-AUC'] for d in performance_data]) if performance_data else 0

    # Build HTML
    html_parts = []
    html_parts.append("<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>")
    html_parts.append(f"<title>Prediction Results Report - {current_date}</title>")
    html_parts.append('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">')
    html_parts.append('<script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>')
    html_parts.append(f"<style>{PAGE_STYLE}</style></head><body><div class='container'>")

    # Header with updated subtitle
    html_parts.append(f"""
    <header>
        <div class="header-content">
            <h1><strong>Prediction Results Report</strong></h1>
            <h4><strong>ELIXIR Protein-Protein Interaction Benchmark</strong></h4>
            <div>
                <a href="../index.html" class="dashboard-link">
                    <i class="fas fa-home"></i> PPI Benchmark FAIR
                </a>
                <a href="./feature-report.html" class="dashboard-link">
                    <i class="fas fa-chart-bar"></i> ML Features
                </a>
                <a href="./fair-analysis.html" class="dashboard-link">
                    <i class="fas fa-bullseye"></i> FAIRfication
                </a>
                <a href="https://github.com/biofold/ppi-benchmark-fair" class="dashboard-link" target="_blank">
                    <i class="fab fa-github"></i> Repository
                </a>
            </div>
        </div>
    </header>
    """)

    html_parts.append("<main class='container'>")

    # Key Metrics Section
    html_parts.append(f"""
    <div class="section" id="key-metrics">
        <h2><i class="fas fa-chart-bar"></i> Model Performance Statistics</h2>
        <p>Comprehensive evaluation of machine learning model performance across multiple metrics and cross-validation folds. Values shown as mean ± standard error.</p>

        <div class="key-metrics">
            <div class="metric-card">
                <div class="metric-value">{len(model_names)}</div>
                <div class="metric-label">Models</div>
            </div>

            <div class="metric-card">
                <div class="metric-value">{best_f1:.3f}</div>
                <div class="metric-label">Best F1 Score</div>
            </div>

            <div class="metric-card">
                <div class="metric-value">{avg_f1:.3f}</div>
                <div class="metric-label">Avg F1 Score</div>
            </div>

            <div class="metric-card">
                <div class="metric-value">{best_model if best_model else 'N/A'}</div>
                <div class="metric-label">Best Model</div>
            </div>
        </div>
    </div>
    """)

    # Performance Table Section
    html_parts.append(f"""
    <div class="section" id="performance-table">
        <h2><i class="fas fa-table"></i> Model Performance Summary Table</h2>
        <p>Detailed performance metrics for all evaluated models. Values shown as mean ± standard error (SE). Best performing values in each metric category are highlighted.</p>

        <table class="performance-table">
            <thead>
                <tr>
                    <th style="text-align:left !important;">Model</th>
                    <th>Accuracy</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1-Score</th>
                    <th>ROC-AUC</th>
                </tr>
            </thead>
            <tbody>
    """)

    # Add table rows
    for data in performance_data:
        # Determine which metrics are best for highlighting
        accuracy_class = 'best-metric' if abs(data['Accuracy'] - best_accuracy) < 0.001 else ''
        precision_class = 'best-metric' if abs(data['Precision'] - best_precision) < 0.001 else ''
        recall_class = 'best-metric' if abs(data['Recall'] - best_recall) < 0.001 else ''
        f1_class = 'best-metric' if abs(data['F1-Score'] - best_f1) < 0.001 else ''
        roc_auc_class = 'best-metric' if abs(data['ROC-AUC'] - best_roc_auc) < 0.001 else ''

        # Format values with standard errors
        accuracy_display = f"{data['Accuracy']:.3f} ± {data['Accuracy_SE']:.3f}"
        precision_display = f"{data['Precision']:.3f} ± {data['Precision_SE']:.3f}"
        recall_display = f"{data['Recall']:.3f} ± {data['Recall_SE']:.3f}"
        f1_display = f"{data['F1-Score']:.3f} ± {data['F1-Score_SE']:.3f}"

        # Format ROC-AUC - handle None or 0 values
        roc_auc_value = data['ROC-AUC']
        if roc_auc_value == 0 or roc_auc_value is None:
            roc_auc_display = 'N/A'
        else:
            roc_auc_display = f"{roc_auc_value:.3f} ± {data['ROC-AUC_SE']:.3f}"

        html_parts.append(f"""
                <tr>
                    <td style="text-align:left !important;"><strong>{data['Model']}</strong></td>
                    <td class="{accuracy_class}">{accuracy_display}</td>
                    <td class="{precision_class}">{precision_display}</td>
                    <td class="{recall_class}">{recall_display}</td>
                    <td class="{f1_class}">{f1_display}</td>
                    <td class="{roc_auc_class}">{roc_auc_display}</td>
                </tr>
        """)

    html_parts.append("""
            </tbody>
        </table>
        <div style="margin-top: 15px; font-size: 0.9rem; color: #666;">
            <p><strong>Note:</strong> Standard error (SE) represents the variability of the metric across cross-validation folds. Smaller SE indicates more consistent performance.</p>
        </div>
    </div>
    """)

    # Navigation Bar (commented out but available)
    html_parts.append("""
    <!--
    <div class="nav-bar">
        <ul>
            <li><a href="#key-metrics">Key Metrics</a></li>
            <li><a href="#performance-table">Performance Table</a></li>
            <li><a href="#figure1">Performance Comparison</a></li>
            <li><a href="#figure2">F1 per Fold</a></li>
            <li><a href="#figure3">Radar Chart</a></li>
        </ul>
    </div>
    -->
    """)

    # Control Buttons (commented out but available)
    html_parts.append("""
    <!--
    <div class="controls">
        <p style="margin-bottom: 15px; color: var(--primary-color); font-weight: 600;">Dashboard Controls:</p>
        <button class="toggle-button" onclick="toggleAllFigures('show')">
            <i class="fas fa-eye"></i> Show All Figures
        </button>
        <button class="toggle-button hidden" onclick="toggleAllFigures('hide')">
            <i class="fas fa-eye-slash"></i> Hide All Figures
        </button>
        <button class="toggle-button" onclick="expandAllFigures()">
            <i class="fas fa-expand"></i> Expand All
        </button>
        <button class="toggle-button" onclick="collapseAllFigures()">
            <i class="fas fa-compress"></i> Collapse All
        </button>
    </div>
    -->
    """)

    # Figures Section
    html_parts.append(f'<div id="figures">')

    # Cards in sensible order
    items = [
        ('Figure 1: Model Performance Comparison', plots_map.get('performance_comparison'), 'Bar chart comparing accuracy, precision, recall, and F1 scores across different models.'),
        ('Figure 2: F1 Score per Cross-Validation Fold', plots_map.get('f1_per_fold'), 'Line plot showing F1 score across each cross-validation fold for all models.'),
        ('Figure 3: Performance Radar Chart', plots_map.get('radar_chart'), 'Radar chart visualizing multiple performance metrics (Accuracy, Precision, Recall, F1, ROC-AUC) for model comparison.')
    ]

    figure_counter = 1
    for title, rel_path, description in items:
        if not rel_path:
            continue
        embed_src = os.path.relpath(Path(output_path).parent / rel_path, start=Path(output_path).parent)
        html_parts.append(f"""
        <div class="figure-container" id="figure{figure_counter}">
            <h2 class="figure-title">{title}</h2>
            <p class="figure-description">{description}</p>
            <div style="padding: 50px 20px; background: white; border-radius: 8px; margin: 10px 0;">
               <iframe class="figure-embed" src="{embed_src}" loading="lazy" title="{title}"></iframe>
            </div>
        </div>
        """)
        figure_counter += 1

    html_parts.append('</div>')
    html_parts.append("</main>")

    # Footer with updated badges
    html_parts.append(f"""
    <footer id="footer">
        <div class="container">
            <div class="footer-content">
                Evaluation of machine learning model performance on the ELIXIR PPI Benchmark
            </div>
            <div class="footer-content" style="margin-top:25px;">
                <a href="https://github.com/biofold/ppi-benchmark-fair">
                <img src="https://img.shields.io/badge/FAIR_Score-87.5%2F100-brightgreen" alt="FAIR Score: 87.5/100"></a>
                <a href="https://creativecommons.org/licenses/by/4.0/">
                <img src="https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg" alt="License: CC BY 4.0"></a>
                <a href="https://doi.org/10.5281/zenodo.XXXXXXX">
                <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg" alt="DOI"></a>
                <a href="https://www.python.org/">
                <img src="https://img.shields.io/badge/Python-3.8%2B-blue.svg" alt="Python 3.8+"></a>
                <a href="https://schema.org/">
                <img src="https://img.shields.io/badge/Metadata-Schema.org%2BBioschemas-blue" alt="Schema.org+Bioschemas"></a>
                <a href="https://mlcommons.org/croissant/">
                <img src="https://img.shields.io/badge/ML-Croissant_1.0-yellow" alt="MLCommons Croissant"></a>
            </div>
            <div class="copyright">
                <p>Prediction Results Report • Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>
        </div>
    </footer>
    """)

    # Add JavaScript
    html_parts.append(JS_SCRIPT)

    html_parts.append("</div></body></html>")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(html_parts))

    if verbose:
        print(f"✅ Prediction dashboard saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate HTML dashboards for ML results or feature analysis (output as separate figure files).')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prediction-results', type=str, help='Path to JSON with prediction results')
    group.add_argument('--feature-report', type=str, help='Path to feature evaluation JSON file')
    parser.add_argument('--output-dir', type=str, default='dashboard_output', help='Directory to write outputs (figures and dashboard)')
    parser.add_argument('--output-file', type=str, default=None, help='Optional output HTML file name for the dashboard')
    parser.add_argument('--verbose', action='store_true', help='Print verbose progress messages')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.feature_report:
        report_path = Path(args.feature_report)
        if not report_path.exists():
            print(f"Feature report not found: {report_path}")
            return
        with open(report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
        plots_map = create_feature_plots(report, str(output_dir), verbose=args.verbose)
        out_file = args.output_file or str(output_dir / "feature_analysis_dashboard.html")
        generate_feature_dashboard(report, plots_map, output_path=out_file, verbose=args.verbose)
        if args.verbose:
            print(f"Dashboard created at: {out_file}")
    elif args.prediction_results:
        results_path = Path(args.prediction_results)
        if not results_path.exists():
            print(f"Prediction results not found: {results_path}")
            return
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        plots_map = create_prediction_plots(results, str(output_dir), verbose=args.verbose)
        out_file = args.output_file or str(output_dir / "prediction_dashboard.html")
        generate_prediction_dashboard(results, plots_map, output_path=out_file, verbose=args.verbose)
        if args.verbose:
            print(f"Dashboard created at: {out_file}")

if __name__ == "__main__":
    main()
