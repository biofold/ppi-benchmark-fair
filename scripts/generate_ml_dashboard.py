#!/usr/bin/env python3
"""
Script to generate an interactive HTML dashboard for displaying results from
ppi_ml_croissant.py with interactive Plotly.js visualizations.
"""

import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

# Set Plotly template
pio.templates.default = "plotly_white"

def load_results_json(results_path):
    """Load results JSON file."""
    with open(results_path, 'r') as f:
        results = json.load(f)
    return results

def create_interactive_performance_plots(results):
    """Create interactive Plotly plots from results."""
    plots_html = {}
    
    # Get model names
    model_names = [name for name in results.keys() if name not in ['cross_validation_settings']]
    
    if not model_names:
        return plots_html
    
    # 1. Interactive metrics comparison bar chart
    try:
        fig = go.Figure()
        
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            values = []
            for name in model_names:
                if name in results:
                    values.append(results[name]['cv_metrics'].get(metric, 0))
            
            if values:
                fig.add_trace(go.Bar(
                    name=label,
                    x=model_names,
                    y=values,
                    text=[f'{v:.3f}' for v in values],
                    textposition='auto',
                    marker_color=px.colors.qualitative.Set1[i],
                    hovertemplate=f'<b>{label}</b>: %{{y:.3f}}<extra></extra>'
                ))
        
        fig.update_layout(
            title='Average Cross-Validation Metrics by Model',
            xaxis_title='Models',
            yaxis_title='Score',
            yaxis=dict(range=[0, 1.1]),
            barmode='group',
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            plot_bgcolor='rgba(248, 249, 250, 0.5)',
            paper_bgcolor='rgba(248, 249, 250, 0.1)',
        )
        
        plots_html['metrics_comparison'] = pio.to_html(fig, full_html=False, include_plotlyjs=False)
        print(f"✅ Created metrics comparison plot with {len(model_names)} models")
    except Exception as e:
        print(f"⚠️  Error creating metrics comparison plot: {e}")
    
    # 2. Interactive F1-Score per fold (line plot)
    try:
        fig = go.Figure()
        has_f1_data = False
        
        for name in model_names:
            if name in results and 'fold_metrics' in results[name]:
                fold_f1s = results[name]['fold_metrics'].get('f1_scores', [])
                if fold_f1s:
                    fig.add_trace(go.Scatter(
                        name=name,
                        x=list(range(1, len(fold_f1s) + 1)),
                        y=fold_f1s,
                        mode='lines+markers',
                        line=dict(width=2),
                        marker=dict(size=8),
                        hovertemplate='<b>%{text}</b><br>Fold: %{x}<br>F1-Score: %{y:.3f}<extra></extra>',
                        text=[name] * len(fold_f1s)
                    ))
                    has_f1_data = True
                    print(f"   Added {name} with {len(fold_f1s)} F1 scores")
        
        if has_f1_data:
            fig.update_layout(
                title='F1-Score per Fold by Model',
                xaxis_title='Fold',
                yaxis_title='F1-Score',
                yaxis=dict(range=[0, 1.1]),
                hovermode='closest',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                plot_bgcolor='rgba(248, 249, 250, 0.5)',
                paper_bgcolor='rgba(248, 249, 250, 0.1)',
            )
            
            plots_html['f1_per_fold'] = pio.to_html(fig, full_html=False, include_plotlyjs=False)
            print(f"✅ Created F1 per fold plot")
        else:
            print("⚠️  No F1 score data available for fold line plot")
    except Exception as e:
        print(f"⚠️  Error creating F1 per fold plot: {e}")
    
    # 3. Interactive radar chart for overall metrics
    try:
        fig = go.Figure()
        has_radar_data = False
        
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
                categories_complete = categories + [categories[0]]
                
                fig.add_trace(go.Scatterpolar(
                    name=name,
                    r=values,
                    theta=categories_complete,
                    fill='toself',
                    line=dict(width=2),
                    hovertemplate='<b>%{theta}</b>: %{r:.3f}<extra></extra>'
                ))
                has_radar_data = True
        
        if has_radar_data:
            fig.update_layout(
                title='Overall Model Performance (Radar Chart)',
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=1.1
                ),
                hovermode='closest',
            )
            
            plots_html['radar_chart'] = pio.to_html(fig, full_html=False, include_plotlyjs=False)
            print(f"✅ Created radar chart")
        else:
            print("⚠️  No overall metrics data available for radar chart")
    except Exception as e:
        print(f"⚠️  Error creating radar chart: {e}")
    
    # 4. Interactive box plot for metric distributions - FIXED VERSION
    try:
        print("📊 Creating metric distribution box plots...")
        
        # First, let's check what data we have
        data_available = []
        for name in model_names:
            if name in results and 'fold_metrics' in results[name]:
                fold_metrics = results[name]['fold_metrics']
                print(f"   Model '{name}' has fold metrics keys: {list(fold_metrics.keys())}")
                for metric_name, values in fold_metrics.items():
                    if isinstance(values, list) and len(values) > 0:
                        data_available.append(metric_name)
        
        # Get unique metric types
        unique_metrics = list(set(data_available))
        print(f"   Found metrics with data: {unique_metrics}")
        
        if unique_metrics:
            # Map metric names to display names
            metric_display_names = {
                'accuracies': 'Accuracy',
                'precisions': 'Precision', 
                'recalls': 'Recall',
                'f1_scores': 'F1-Score',
                'roc_aucs': 'ROC-AUC'
            }
            
            # Create subplots - use only metrics that have data
            metrics_to_plot = []
            for metric in unique_metrics:
                display_name = metric_display_names.get(metric, metric.replace('_', ' ').title())
                metrics_to_plot.append(display_name)
            
            # Limit to 4 metrics for a 2x2 grid
            metrics_to_plot = metrics_to_plot[:4]
            
            if metrics_to_plot:
                rows = 2
                cols = 2
                
                # Adjust grid if we have fewer metrics
                if len(metrics_to_plot) <= 2:
                    rows = 1
                    cols = len(metrics_to_plot)
                
                fig = make_subplots(
                    rows=rows, cols=cols,
                    subplot_titles=metrics_to_plot,
                    vertical_spacing=0.2 if rows > 1 else 0.1,
                    horizontal_spacing=0.15 if cols > 1 else 0.1
                )
                
                plot_index = 0
                for metric_display in metrics_to_plot:
                    # Find the original metric name
                    metric_original = None
                    for orig_metric, display in metric_display_names.items():
                        if display == metric_display:
                            metric_original = orig_metric
                            break
                    
                    if not metric_original:
                        # Try to find by partial match
                        for orig_metric in unique_metrics:
                            if metric_display.lower() in orig_metric.lower() or orig_metric.lower() in metric_display.lower():
                                metric_original = orig_metric
                                break
                    
                    if metric_original:
                        row = plot_index // cols + 1
                        col = plot_index % cols + 1
                        
                        # Collect data for this metric
                        metric_data = []
                        for name in model_names:
                            if name in results and 'fold_metrics' in results[name]:
                                values = results[name]['fold_metrics'].get(metric_original, [])
                                if isinstance(values, list) and values:
                                    # Clean the values - remove None/NaN
                                    clean_values = []
                                    for v in values:
                                        if v is not None:
                                            try:
                                                # Try to convert to float
                                                clean_values.append(float(v))
                                            except (ValueError, TypeError):
                                                # Skip non-numeric values
                                                pass
                                    
                                    if clean_values:
                                        metric_data.append({
                                            'Model': name,
                                            'Values': clean_values
                                        })
                        
                        # Add box plot for each model with data
                        if metric_data:
                            for i, data in enumerate(metric_data):
                                fig.add_trace(go.Box(
                                    y=data['Values'],
                                    name=data['Model'],
                                    boxpoints='all',  # Show all points
                                    jitter=0.3,       # Spread points out
                                    pointpos=-1.8,    # Position points
                                    marker=dict(
                                        size=6,
                                        opacity=0.6,
                                        color=px.colors.qualitative.Set1[i % len(px.colors.qualitative.Set1)]
                                    ),
                                    line=dict(width=1),
                                    showlegend=(plot_index == 0)  # Only show legend in first subplot
                                ), row=row, col=col)
                            
                            # Update subplot layout
                            fig.update_yaxes(
                                title_text='Score',
                                range=[0, 1.1],
                                row=row,
                                col=col
                            )
                        else:
                            # Add placeholder text if no data
                            fig.add_annotation(
                                x=0.5, y=0.5,
                                text=f"No {metric_display} data",
                                showarrow=False,
                                font=dict(size=14),
                                xref=f"x{plot_index + 1}",
                                yref=f"y{plot_index + 1}"
                            )
                        
                        plot_index += 1
                
                # Update overall layout
                fig.update_layout(
                    title='Metric Distributions Across Folds',
                    height=400 * rows,
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    plot_bgcolor='rgba(248, 249, 250, 0.5)',
                    paper_bgcolor='rgba(248, 249, 250, 0.1)',
                )
                
                plots_html['metric_distribution'] = pio.to_html(fig, full_html=False, include_plotlyjs=False)
                print(f"✅ Created metric distribution plot with {len(metrics_to_plot)} metrics")
            else:
                print("⚠️  No metrics to plot for distribution")
        else:
            print("⚠️  No fold metrics data available for distribution plots")
            
            # Create a placeholder plot
            fig = go.Figure()
            fig.add_annotation(
                x=0.5, y=0.5,
                text="No fold metrics data available<br>for distribution plots",
                showarrow=False,
                font=dict(size=16),
                xref="paper",
                yref="paper"
            )
            fig.update_layout(
                title='Metric Distributions Across Folds',
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=400
            )
            plots_html['metric_distribution'] = pio.to_html(fig, full_html=False, include_plotlyjs=False)
            
    except Exception as e:
        print(f"⚠️  Error creating box plots: {e}")
        import traceback
        traceback.print_exc()
        
        # Create error placeholder
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5,
            text=f"Error creating distribution plot:<br>{str(e)[:100]}...",
            showarrow=False,
            font=dict(size=14, color="red"),
            xref="paper",
            yref="paper"
        )
        fig.update_layout(
            title='Metric Distributions Across Folds',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=400
        )
        plots_html['metric_distribution'] = pio.to_html(fig, full_html=False, include_plotlyjs=False)
    
    return plots_html

def create_interactive_feature_plots(feature_eval_path):
    """Create interactive feature evaluation plots."""
    plots_html = {}
    
    if not feature_eval_path or not Path(feature_eval_path).exists():
        return plots_html
    
    try:
        with open(feature_eval_path, 'r') as f:
            feature_data = json.load(f)
        
        print("📊 Creating interactive feature plots...")
        
        # 1. Feature Correlation Interactive Plot
        if 'correlation_analysis' in feature_data:
            try:
                corr_info = feature_data['correlation_analysis']
                
                if 'highly_correlated_pairs' in corr_info and corr_info['highly_correlated_pairs']:
                    pairs = corr_info['highly_correlated_pairs'][:20]  # Top 20 pairs
                    
                    # Prepare data
                    pair_labels = []
                    correlations = []
                    feature1_list = []
                    feature2_list = []
                    
                    for pair in pairs:
                        f1 = pair.get('feature1', 'Unknown')
                        f2 = pair.get('feature2', 'Unknown')
                        corr = pair.get('correlation', 0)
                        
                        # Create a safe pair label without backslashes in the f-string
                        pair_label = f"{f1} ↔ {f2}"
                        pair_labels.append(pair_label)
                        correlations.append(abs(corr))
                        feature1_list.append(f1)
                        feature2_list.append(f2)
                    
                    # Create DataFrame
                    df_corr = pd.DataFrame({
                        'Feature Pair': pair_labels,
                        'Correlation': correlations,
                        'Feature1': feature1_list,
                        'Feature2': feature2_list,
                        'AbsCorrelation': [abs(c) for c in correlations]
                    })
                    
                    # Create interactive bar chart
                    fig = px.bar(
                        df_corr,
                        y='Feature Pair',
                        x='Correlation',
                        color='AbsCorrelation',
                        color_continuous_scale='RdYlBu_r',
                        orientation='h',
                        hover_data=['Feature1', 'Feature2'],
                        title=f'Top {len(pairs)} Highly Correlated Feature Pairs (|r| > 0.8)'
                    )
                    
                    fig.update_layout(
                        yaxis={'categoryorder': 'total ascending'},
                        xaxis_title='Absolute Correlation',
                        yaxis_title='Feature Pair',
                        coloraxis_colorbar=dict(title='|r|'),
                        height=max(400, len(pairs) * 25),
                        hovermode='y unified'
                    )
                    
                    # Add correlation value annotations
                    fig.update_traces(
                        texttemplate='%{x:.3f}',
                        textposition='outside',
                        hovertemplate='<b>%{customdata[0]} ↔ %{customdata[1]}</b><br>Correlation: %{x:.3f}<extra></extra>'
                    )
                    
                    plots_html['feature_correlation'] = pio.to_html(fig, full_html=False, include_plotlyjs=False)
            except Exception as e:
                print(f"⚠️  Error creating correlation plot: {e}")
        
        # 2. Feature-Target Relationship Interactive Plot
        if 'feature_target_analysis' in feature_data:
            try:
                ft_info = feature_data['feature_target_analysis']
                
                if 'all_scores' in ft_info:
                    # Prepare data
                    features_data = []
                    for feature, scores in ft_info['all_scores'].items():
                        features_data.append({
                            'Feature': feature,
                            'ANOVA_F': scores.get('anova_f', 0),
                            'ANOVA_p': scores.get('anova_p', 1),
                            'Mutual_Info': scores.get('mutual_info', 0),
                            'Significant': scores.get('significant_anova', False)
                        })
                    
                    df_features = pd.DataFrame(features_data)
                    
                    # Create interactive scatter plot
                    fig = px.scatter(
                        df_features,
                        x='ANOVA_F',
                        y='Mutual_Info',
                        color='Significant',
                        size='ANOVA_F',
                        hover_name='Feature',
                        hover_data=['ANOVA_p'],
                        title='Feature-Target Relationship: ANOVA vs Mutual Information',
                        color_discrete_map={True: '#2ecc71', False: '#e74c3c'},
                        labels={
                            'ANOVA_F': 'ANOVA F-value',
                            'Mutual_Info': 'Mutual Information',
                            'Significant': 'Significant (p < 0.05)',
                            'ANOVA_p': 'p-value'
                        }
                    )
                    
                    # Add trend line
                    fig.update_traces(
                        marker=dict(opacity=0.7),
                        hovertemplate='<b>%{hovertext}</b><br>ANOVA F: %{x:.3f}<br>Mutual Info: %{y:.3f}<br>p-value: %{customdata[0]:.2e}<extra></extra>'
                    )
                    
                    fig.update_layout(
                        hovermode='closest',
                        legend_title_text='Significance',
                        plot_bgcolor='rgba(248, 249, 250, 0.5)',
                        height=600
                    )
                    
                    plots_html['feature_target_relationship'] = pio.to_html(fig, full_html=False, include_plotlyjs=False)
                    
                    # Also create top features bar chart
                    top_n = min(15, len(df_features))
                    
                    # Top by ANOVA
                    df_top_anova = df_features.nlargest(top_n, 'ANOVA_F')
                    fig_anova = px.bar(
                        df_top_anova,
                        x='ANOVA_F',
                        y='Feature',
                        color='Significant',
                        orientation='h',
                        title=f'Top {top_n} Features by ANOVA F-value',
                        color_discrete_map={True: '#2ecc71', False: '#e74c3c'},
                        hover_data=['ANOVA_p']
                    )
                    
                    fig_anova.update_layout(
                        yaxis={'categoryorder': 'total ascending'},
                        xaxis_title='ANOVA F-value',
                        yaxis_title='Feature',
                        height=max(400, top_n * 25),
                        hovermode='y unified'
                    )
                    
                    plots_html['feature_top_anova'] = pio.to_html(fig_anova, full_html=False, include_plotlyjs=False)
                    
                    # Top by Mutual Information
                    df_top_mi = df_features.nlargest(top_n, 'Mutual_Info')
                    fig_mi = px.bar(
                        df_top_mi,
                        x='Mutual_Info',
                        y='Feature',
                        orientation='h',
                        title=f'Top {top_n} Features by Mutual Information',
                        color='Mutual_Info',
                        color_continuous_scale='Viridis',
                        hover_data=['ANOVA_p', 'Significant']
                    )
                    
                    fig_mi.update_layout(
                        yaxis={'categoryorder': 'total ascending'},
                        xaxis_title='Mutual Information Score',
                        yaxis_title='Feature',
                        height=max(400, top_n * 25),
                        hovermode='y unified'
                    )
                    
                    plots_html['feature_top_mi'] = pio.to_html(fig_mi, full_html=False, include_plotlyjs=False)
            except Exception as e:
                print(f"⚠️  Error creating feature-target plots: {e}")
        
        # 3. Feature Importance Interactive Plots
        if 'feature_importance' in feature_data:
            try:
                # Create subplot for different importance methods
                methods_data = []
                
                for method_name, method_data in feature_data['feature_importance'].items():
                    if 'top_features' in method_data and method_data['top_features']:
                        for feature, importance in method_data['top_features'][:15]:  # Top 15 per method
                            methods_data.append({
                                'Method': method_name.replace('_', ' ').title(),
                                'Feature': feature,
                                'Importance': importance
                            })
                
                if methods_data:
                    df_methods = pd.DataFrame(methods_data)
                    
                    # Interactive grouped bar chart
                    fig = px.bar(
                        df_methods,
                        x='Feature',
                        y='Importance',
                        color='Method',
                        barmode='group',
                        title='Feature Importance Across Different Methods (Top 15 per method)',
                        height=600
                    )
                    
                    fig.update_layout(
                        xaxis_title='Feature',
                        yaxis_title='Importance Score',
                        xaxis_tickangle=-45,
                        hovermode='x unified',
                        legend_title_text='Method'
                    )
                    
                    plots_html['feature_importance_comparison'] = pio.to_html(fig, full_html=False, include_plotlyjs=False)
                    
                    # Also create a heatmap of rankings
                    pivot_methods = df_methods.pivot_table(
                        index='Feature',
                        columns='Method',
                        values='Importance',
                        aggfunc='first'
                    ).fillna(0)
                    
                    fig_heatmap = go.Figure(data=go.Heatmap(
                        z=pivot_methods.values,
                        x=pivot_methods.columns.tolist(),
                        y=pivot_methods.index.tolist(),
                        colorscale='Viridis',
                        hoverongaps=False,
                        hovertemplate='<b>%{y}</b><br>Method: %{x}<br>Importance: %{z:.3f}<extra></extra>'
                    ))
                    
                    fig_heatmap.update_layout(
                        title='Feature Importance Heatmap',
                        xaxis_title='Method',
                        yaxis_title='Feature',
                        height=max(400, len(pivot_methods) * 20)
                    )
                    
                    plots_html['feature_importance_heatmap'] = pio.to_html(fig_heatmap, full_html=False, include_plotlyjs=False)
            except Exception as e:
                print(f"⚠️  Error creating feature importance plots: {e}")
        
        # 4. PCA Analysis Interactive Plot
        if 'pca_analysis' in feature_data:
            try:
                pca_info = feature_data['pca_analysis']
                
                if 'explained_variance' in pca_info and pca_info['explained_variance']:
                    explained_var = pca_info['explained_variance']
                    cumulative_var = pca_info['cumulative_variance']
                    
                    # Create interactive line chart
                    fig = go.Figure()
                    
                    # Individual explained variance
                    fig.add_trace(go.Bar(
                        name='Individual',
                        x=list(range(1, len(explained_var) + 1)),
                        y=explained_var,
                        marker_color='#3498db',
                        opacity=0.6,
                        hovertemplate='PC%{x}: %{y:.3f}<extra></extra>'
                    ))
                    
                    # Cumulative explained variance
                    fig.add_trace(go.Scatter(
                        name='Cumulative',
                        x=list(range(1, len(cumulative_var) + 1)),
                        y=cumulative_var,
                        mode='lines+markers',
                        line=dict(color='#e74c3c', width=3),
                        marker=dict(size=8),
                        hovertemplate='PC%{x}: %{y:.3f}<extra></extra>'
                    ))
                    
                    # Add 95% threshold line
                    fig.add_hline(
                        y=0.95,
                        line_dash="dash",
                        line_color="green",
                        annotation_text="95% threshold",
                        annotation_position="bottom right"
                    )
                    
                    # Find components needed for 95% variance
                    if 'components_needed_95' in pca_info:
                        n_components = pca_info['components_needed_95']
                        fig.add_vline(
                            x=n_components,
                            line_dash="dot",
                            line_color="green",
                            annotation_text=f"{n_components} PCs for 95%",
                            annotation_position="top right"
                        )
                    
                    fig.update_layout(
                        title='PCA Explained Variance',
                        xaxis_title='Principal Component',
                        yaxis_title='Explained Variance Ratio',
                        hovermode='x unified',
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        ),
                        height=500
                    )
                    
                    plots_html['pca_variance'] = pio.to_html(fig, full_html=False, include_plotlyjs=False)
                    
                    # Also create PC loadings visualization if available
                    if 'component_loadings' in pca_info:
                        try:
                            pc1_loadings = pca_info['component_loadings'].get('PC1', {})
                            if 'top_positive' in pc1_loadings and 'top_negative' in pc1_loadings:
                                # Prepare data for PC1 loadings
                                loadings_data = []
                                
                                for feature, loading in pc1_loadings.get('top_positive', []):
                                    loadings_data.append({
                                        'Feature': feature,
                                        'Loading': loading,
                                        'Type': 'Positive'
                                    })
                                
                                for feature, loading in pc1_loadings.get('top_negative', []):
                                    loadings_data.append({
                                        'Feature': feature,
                                        'Loading': loading,
                                        'Type': 'Negative'
                                    })
                                
                                if loadings_data:
                                    df_loadings = pd.DataFrame(loadings_data)
                                    
                                    fig_loadings = px.bar(
                                        df_loadings,
                                        x='Loading',
                                        y='Feature',
                                        color='Type',
                                        orientation='h',
                                        title='PC1 Feature Loadings (Most Influential Features)',
                                        color_discrete_map={'Positive': '#2ecc71', 'Negative': '#e74c3c'},
                                        height=max(300, len(df_loadings) * 20)
                                    )
                                
                                    fig_loadings.update_layout(
                                        yaxis={'categoryorder': 'total ascending'},
                                        xaxis_title='Loading Value',
                                        yaxis_title='Feature',
                                        hovermode='y unified'
                                    )
                                
                                    plots_html['pc1_loadings'] = pio.to_html(fig_loadings, full_html=False, include_plotlyjs=False)
                        except:
                            pass  # Skip if we can't create loadings plot
            except Exception as e:
                print(f"⚠️  Error creating PCA plots: {e}")
        
        print(f"✅ Created {len(plots_html)} interactive feature plots")
        
    except Exception as e:
        print(f"❌ Error creating interactive feature plots: {e}")
        import traceback
        traceback.print_exc()
    
    return plots_html

def generate_model_details_table(results):
    """Generate HTML table with detailed model results."""
    model_names = [name for name in results.keys() if name not in ['cross_validation_settings']]
    
    if not model_names:
        return "<p>No model results found.</p>"
    
    html = """
    <div class="table-container">
        <table class="interactive-table">
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
            
            # Determine row class based on best model
            best_model = results.get('cross_validation_settings', {}).get('best_model', '')
            row_class = 'best-model-row' if name == best_model else ''
            
            html += f"""
                <tr class="{row_class}">
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
        <table class="interactive-table">
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
                
                # Color code based on F1-score
                f1_score = f1_scores[fold_idx] if fold_idx < len(f1_scores) else 0
                f1_class = ''
                if f1_score > 0.8:
                    f1_class = 'excellent-score'
                elif f1_score > 0.6:
                    f1_class = 'good-score'
                elif f1_score > 0.4:
                    f1_class = 'fair-score'
                
                html += f"""
                    <tr>
                        <td>{'→' if fold_idx > 0 else name}</td>
                        <td>{fold_idx + 1}</td>
                        <td>{accuracy_val}</td>
                        <td>{precision_val}</td>
                        <td>{recall_val}</td>
                        <td class="{f1_class}"><strong>{f1_val}</strong></td>
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
    <section class="section">
    <div>
        <h3>Cross-Validation Settings</h3>
        <ul class="fair-checklist">
            <li><strong>Method:</strong> {settings.get('method', 'ClusterID-aware stratified cross-validation')}</li>
            <li><strong>Number of Folds:</strong> {settings.get('n_splits', 5)}</li>
            <li><strong>Random State:</strong> {settings.get('random_state', 42)}</li>
            <li><strong>Best Model:</strong> <span class="badge fair">{settings.get('best_model', 'Not determined')}</span></li>
        </ul>
    </div>
    </section>
    """
    
    return html

def generate_html_dashboard(results, performance_plots_html, feature_plots_html=None, feature_eval_path=None, output_path="ml_results_dashboard.html"):
    """Generate the HTML dashboard with interactive plots."""
    
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
    
    # Include Plotly.js from CDN
    plotly_js = '<script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>'
    
    # Determine if we have feature plots for conditional display
    has_feature_plots = bool(feature_plots_html)
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ELIXIR - PPI Benchmark ML Results Dashboard</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    {plotly_js}
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
            margin-bottom: 5px;
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
        
        .badge.feature {{
            background: linear-gradient(135deg, #3498db, #2980b9);
            color: white;
            box-shadow: 0 4px 10px rgba(52, 152, 219, 0.3);
        }}
        
        .dashboard-link {{
            display: inline-flex;
            align-items: center;
            gap: 10px;
            background-color: rgba(255,255,255,0.15);
            color: white;
            padding: 6px 6px;
            border-radius: 10px;
            text-decoration: none;
            font-weight: 400;
            transition: all 0.3s ease;
            border: 2px solid rgba(255,255,255,0.3);
            margin-top: 15px;
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
            margin: 5px 0 5px;
            font-size: 1.1rem;
        }}
        
        /* Tables */
        .table-container {{
            overflow-x: auto;
            margin: 20px 0;
        }}
        
        .interactive-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 5px 15px rgba(0,0,0,0.05);
        }}
        
        .interactive-table th, .interactive-table td {{
            padding: 15px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }}
        
        .interactive-table th {{
            background-color: #f8f9fa;
            color: var(--primary-color);
            font-weight: 600;
            position: sticky;
            top: 0;
        }}
        
        .interactive-table tr:hover {{
            background-color: #f8f9fa;
            transform: scale(1.005);
            transition: transform 0.1s ease;
        }}
        
        .interactive-table .best-model-row {{
            background-color: rgba(39, 174, 96, 0.1);
            border-left: 4px solid var(--success-color);
        }}
        
        .interactive-table .excellent-score {{
            color: #27ae60;
            font-weight: bold;
        }}
        
        .interactive-table .good-score {{
            color: #f39c12;
            font-weight: bold;
        }}
        
        .interactive-table .fair-score {{
            color: #e74c3c;
            font-weight: bold;
        }}
        
        /* Interactive Plot Containers */
        .plot-container {{
            background-color: white;
            border-radius: 8px;
            padding: 20px;
            margin: 30px 0;
            box-shadow: 0 5px 15px rgba(0,0,0,0.05);
            border: 1px solid #e0e0e0;
        }}
        
        .plot-title {{
            text-align: center;
            margin-bottom: 20px;
            color: var(--primary-color);
            font-weight: 600;
            font-size: 1.2rem;
        }}
        
        .plot-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
            gap: 30px;
            margin: 30px 0;
        }}
        
        /* Plotly plot styling */
        .js-plotly-plot {{
            width: 100%;
            height: 500px;
        }}
        
        .plot-description {{
            margin-top: 15px;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 6px;
            font-size: 0.95rem;
            color: #555;
            border-left: 3px solid var(--secondary-color);
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
            
            .plot-grid {{
                grid-template-columns: 1fr;
            }}
            
            .js-plotly-plot {{
                height: 400px;
            }}
            
            .badges {{
                flex-direction: column;
                align-items: center;
            }}
            
            .badge {{
                width: 90%;
                text-align: center;
            }}
            
            .footer-links {{
                flex-direction: column;
                gap: 15px;
            }}
        }}
        
        /* Model performance highlights */
        .performance-highlights {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}

                .metric-card {{
                    background: linear-gradient(135deg, var(--primary-color), #4689a3);
                    color: white;
                    padding: 25px;
                    border-radius: 12px;
                    text-align: center;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
                    transition: transform 0.3s ease;
                }}

                .metric-card:hover {{
                    transform: translateY(-5px);
                }}

                .metric-value {{
                    font-size: 2.5rem;
                    font-weight: bold;
                    margin-bottom: 10px;
                }}

                .metric-label {{
                    font-size: 1rem;
                    opacity: 0.9;
                }}
 
        .highlight-card {{
            background: linear-gradient(135deg, #f8f9fa, #e9ecef);
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            border-left: 5px solid var(--secondary-color);
            transition: transform 0.3s ease;
        }}
        
        .highlight-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.1);
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
            flex-wrap: wrap;
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
            flex: 1;
            min-width: 120px;
            text-align: center;
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
            animation: fadeIn 0.5s ease;
        }}
        
        .tab-content.active {{
            display: block;
        }}
        
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        /* Plot controls */
        .plot-controls {{
            display: flex;
            justify-content: flex-end;
            gap: 10px;
            margin-bottom: 10px;
        }}
        
        .plot-btn {{
            background: var(--secondary-color);
            color: white;
            border: none;
            padding: 8px 15px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.9rem;
            transition: background 0.3s ease;
        }}
        
        .plot-btn:hover {{
            background: var(--primary-color);
        }}
    </style>
    
    <script>
        // Tab switching functionality
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
            
            // Trigger resize for Plotly plots
            window.dispatchEvent(new Event('resize'));
            
            // Save active tab to localStorage
            localStorage.setItem('activeTab', tabName);
        }}
        
        // Plot control functions
        function downloadPlot(plotId, filename) {{
            const plotDiv = document.getElementById(plotId);
            Plotly.downloadImage(plotDiv, {{
                format: 'png',
                filename: filename,
                height: 600,
                width: 800,
                scale: 2
            }});
        }}
        
        function resetPlot(plotId) {{
            const plotDiv = document.getElementById(plotId);
            Plotly.relayout(plotDiv, {{}});
        }}
        
        // Initialize on page load
        document.addEventListener('DOMContentLoaded', function() {{
            // Restore active tab from localStorage
            const savedTab = localStorage.getItem('activeTab') || 'summary';
            showTab(savedTab);
            
            // Make tables sortable
            const tables = document.querySelectorAll('.interactive-table');
            tables.forEach(table => {{
                const headers = table.querySelectorAll('th');
                headers.forEach((header, index) => {{
                    if (index > 0) {{ // Don't make first column (Model name) sortable
                        header.style.cursor = 'pointer';
                        header.addEventListener('click', () => {{
                            sortTable(table, index);
                        }});
                    }}
                }});
            }});
            
            // Add hover effects to performance highlights
            const highlightCards = document.querySelectorAll('.metric-card');
            highlightCards.forEach(card => {{
                card.addEventListener('mouseenter', function() {{
                    this.style.transform = 'translateY(-5px)';
                }});
                card.addEventListener('mouseleave', function() {{
                    this.style.transform = 'translateY(0)';
                }});
            }});
        }});
        
        // Table sorting function
        function sortTable(table, column) {{
            const tbody = table.querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('tr'));
            
            // Determine sort direction
            const isAscending = table.getAttribute('data-sort-dir') !== 'asc';
            table.setAttribute('data-sort-dir', isAscending ? 'asc' : 'desc');
            
            rows.sort((a, b) => {{
                const aVal = a.children[column].textContent.trim();
                const bVal = b.children[column].textContent.trim();
                
                // Try to parse as number
                const aNum = parseFloat(aVal);
                const bNum = parseFloat(bVal);
                
                if (!isNaN(aNum) && !isNaN(bNum)) {{
                    return isAscending ? aNum - bNum : bNum - aNum;
                }} else {{
                    return isAscending ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
                }}
            }});
            
            // Reorder rows
            rows.forEach(row => tbody.appendChild(row));
            
            // Update sort indicators
            const headers = table.querySelectorAll('th');
            headers.forEach(header => {{
                header.textContent = header.textContent.replace(' ↑', '').replace(' ↓', '');
            }});
            
            const currentHeader = headers[column];
            currentHeader.textContent += isAscending ? ' ↑' : ' ↓';
        }}
    </script>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <header>
            <div class="header-content">
                <h1><strong>ML Prediction Dashboard</strong></h1>
                <h4 style="color:white;"><strong>ELIXIR Protein-Protein Interaction Benchmark</strong></h4>
                 <div>
                    <a href="./index.html" class="dashboard-link">
                        <i class="fas fa-home"></i> FAIR Metadata
                    </a>
                    <a href="./ml_croissant.html" class="dashboard-link">
                        <i class="fas fa-brain"></i> Machine Learning
                    </a>
                    <a href="https://github.com/biofold/ppi-benchmark-fair" class="dashboard-link" target="_blank">
                        <i class="fab fa-github"></i> Repository
                    </a>
                
    """
    
    html_content += f"""                </div>
                
    """
    
    html_content += f"""                </div>
            </div>
        </header>

        <main>
            <!-- Tab Container -->
            <div class="tab-container">
                <div class="tab-buttons">
                    <button class="tab-button active" onclick="showTab('summary')">Summary</button>
                    <button class="tab-button" onclick="showTab('performance')">Performance Plots</button>
                    <button class="tab-button" onclick="showTab('models')">Model Details</button>
    """
    
    # Add feature tab button conditionally
    if has_feature_plots:
        html_content += """                    <button class="tab-button" onclick="showTab('features')">Feature Analysis</button>
    """
    
    html_content += f"""                    <button class="tab-button" onclick="showTab('data')">Dataset Info</button>
                </div>
                
                <!-- Summary Tab -->
                <div id="summary" class="tab-content active">
                    <section class="section">
                        <h2><i class="fas fa-home"></i> Analysis Summary</h2>
                        
                        <div class="performance-highlights">
                            <div class="metric-card">
                                <div class="metric-value">{len(model_names)}</div>
                                <div class="metric-label">Models Trained</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-value">{n_splits}</div>
                                <div class="metric-label">CV Folds</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-value">{avg_f1:.3f}</div>
                                <div class="metric-label">Average F1-Score</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-value">{best_model}</div>
                                <div class="metric-label">Best Model</div>
                            </div>
                        </div>
                    </section>
                    <section class="section"> 
                        <h3>Key Findings</h3>
                        <ul class="fair-checklist">
                            <li>All models trained using <strong>ClusterID-aware cross-validation</strong> to prevent data leakage</li>
                            <li>Performance evaluated across {n_splits} folds</li>
                            <li><strong>{best_model}</strong> achieved the highest average F1-score</li>
                            <li>Models show consistent performance across different folds</li>
    """
    
    # Add feature analysis note conditionally
    if has_feature_plots:
        html_content += """                            <li><strong>Interactive feature analysis</strong> available with detailed visualizations</li>
    """
    
    html_content += f"""                        </ul>
                                       </section>
                        
                        {cv_settings_section}
                        <!-- 
                        <div class="plot-container">
                            <div class="plot-title">Interactive Dashboard Features</div>
                            <p style="text-align: center; margin-bottom: 15px;">This dashboard includes interactive visualizations with:</p>
                            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px;">
                                <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #3498db;">
                                    <strong>Hover Interactions</strong><br>Hover over plots to see detailed values
                                </div>
                                <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #2ecc71;">
                                    <strong>Zoom & Pan</strong><br>Click and drag to zoom, double-click to reset
                                </div>
                                <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #e74c3c;">
                                    <strong>Sortable Tables</strong><br>Click table headers to sort columns
                                </div>
                                <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #9b59b6;">
                                    <strong>Download Options</strong><br>Save plots as PNG images
                                </div>
                            </div>
                        </div>
                        -->
                    </section>
                </div>
                
                <!-- Performance Plots Tab -->
                <div id="performance" class="tab-content">
                    <section class="section">
                        <h2><i class="fas fa-chart-bar"></i> Interactive Performance Visualizations</h2>
                        
                        <div class="plot-grid">
    """
    
    # Add interactive performance plots
    plot_counter = 1
    
    if 'metrics_comparison' in performance_plots_html:
        html_content += f"""
                            <div class="plot-container">
                                <div class="plot-controls">
                                    <button class="plot-btn" onclick="downloadPlot('plot{plot_counter}', 'metrics_comparison')">
                                        <i class="fas fa-download"></i> Download
                                    </button>
                                    <button class="plot-btn" onclick="resetPlot('plot{plot_counter}')">
                                        <i class="fas fa-undo"></i> Reset View
                                    </button>
                                </div>
                                <div class="plot-title">Metrics Comparison Across Models</div>
                                <div id="plot{plot_counter}">
                                    {performance_plots_html['metrics_comparison']}
                                </div>
                                <div class="plot-description">
                                    Interactive comparison of average cross-validation metrics across all trained models. 
                                    Hover over bars to see exact values. Click legend items to show/hide metrics.
                                </div>
                            </div>
        """
        plot_counter += 1
    
    if 'f1_per_fold' in performance_plots_html:
        html_content += f"""
                            <div class="plot-container">
                                <div class="plot-controls">
                                    <button class="plot-btn" onclick="downloadPlot('plot{plot_counter}', 'f1_per_fold')">
                                        <i class="fas fa-download"></i> Download
                                    </button>
                                    <button class="plot-btn" onclick="resetPlot('plot{plot_counter}')">
                                        <i class="fas fa-undo"></i> Reset View
                                    </button>
                                </div>
                                <div class="plot-title">F1-Score Per Fold</div>
                                <div id="plot{plot_counter}">
                                    {performance_plots_html['f1_per_fold']}
                                </div>
                                <div class="plot-description">
                                    F1-Score for each fold across different models. Click and drag to zoom in on specific folds.
                                    Consistency across folds indicates robust model performance.
                                </div>
                            </div>
        """
        plot_counter += 1
    
    if 'radar_chart' in performance_plots_html:
        html_content += f"""
                            <div class="plot-container">
                                <div class="plot-controls">
                                    <button class="plot-btn" onclick="downloadPlot('plot{plot_counter}', 'radar_chart')">
                                        <i class="fas fa-download"></i> Download
                                    </button>
                                    <button class="plot-btn" onclick="resetPlot('plot{plot_counter}')">
                                        <i class="fas fa-undo"></i> Reset View
                                    </button>
                                </div>
                                <div class="plot-title">Overall Model Performance (Radar Chart)</div>
                                <div id="plot{plot_counter}">
                                    {performance_plots_html['radar_chart']}
                                </div>
                                <div class="plot-description">
                                    Radar chart comparing overall performance across multiple metrics. 
                                    Hover over points to see exact values. Larger areas indicate better overall performance.
                                </div>
                            </div>
        """
        plot_counter += 1
    
    if 'metric_distribution' in performance_plots_html:
        html_content += f"""
                            <div class="plot-container">
                                <div class="plot-controls">
                                    <button class="plot-btn" onclick="downloadPlot('plot{plot_counter}', 'metric_distribution')">
                                        <i class="fas fa-download"></i> Download
                                    </button>
                                    <button class="plot-btn" onclick="resetPlot('plot{plot_counter}')">
                                        <i class="fas fa-undo"></i> Reset View
                                    </button>
                                </div>
                                <div class="plot-title">Metric Distributions Across Folds</div>
                                <div id="plot{plot_counter}">
                                    {performance_plots_html['metric_distribution']}
                                </div>
                                <div class="plot-description">
                                    Box plots showing distribution of metrics across folds for each model. 
                                    Hover over points to see individual fold values. Tighter distributions indicate more consistent performance.
                                </div>
                            </div>
        """
        plot_counter += 1
    
    html_content += f"""
                        </div>
                    </section>
                </div>
                
                <!-- Model Details Tab -->
                <div id="models" class="tab-content">
                    <section class="section">
                        <h2><i class="fas fa-cogs"></i> Model Performance Details</h2>
                        
                        <h3>Average Cross-Validation Metrics</h3>
                        <p>The following table shows average performance metrics across all cross-validation folds. Click column headers to sort.</p>
                        {model_details_table}
                    </section> 
                     <section class="section">
                        <h3>Fold-by-Fold Performance</h3>
                        <p>Detailed performance metrics for each fold of cross-validation. F1-scores are color-coded for quick assessment.</p>
                        {fold_details_table}
                        
                        <section class="section">
                          <div >
                            <h3>Performance Insights</h3>
                            <ul class="fair-checklist">
                                <li>F1-Score is used as the primary metric for model comparison</li>
                                <li>Precision and Recall provide insight into error types</li>
                                <li>ROC-AUC indicates overall ranking performance (when available)</li>
                                <li>Consistency across folds indicates model robustness</li>
                            </ul>
                        </div>
                    </section>
                </div>
    """
    
    # Feature Analysis Tab (only if we have feature plots)
    if has_feature_plots:
        html_content += """
                <!-- Feature Analysis Tab -->
                <div id="features" class="tab-content">
                    <section class="section">
                        <h2><i class="fas fa-chart-line"></i> Interactive Feature Analysis</h2>
                        
                        <div class="plot-grid">
        """
        
        # Add interactive feature plots
        feature_plot_counter = 10  # Start plot IDs from 10
        
        if 'feature_correlation' in feature_plots_html:
            html_content += f"""
                            <div class="plot-container">
                                <div class="plot-controls">
                                    <button class="plot-btn" onclick="downloadPlot('plot{feature_plot_counter}', 'feature_correlation')">
                                        <i class="fas fa-download"></i> Download
                                    </button>
                                    <button class="plot-btn" onclick="resetPlot('plot{feature_plot_counter}')">
                                        <i class="fas fa-undo"></i> Reset View
                                    </button>
                                </div>
                                <div class="plot-title">Feature Correlation Analysis</div>
                                <div id="plot{feature_plot_counter}">
                                    {feature_plots_html['feature_correlation']}
                                </div>
                                <div class="plot-description">
                                    Top highly correlated feature pairs (|r| > 0.8). Color indicates correlation strength.
                                    Highly correlated features may be redundant and candidates for removal.
                                </div>
                            </div>
            """
            feature_plot_counter += 1
        
        if 'feature_target_relationship' in feature_plots_html:
            html_content += f"""
                            <div class="plot-container">
                                <div class="plot-controls">
                                    <button class="plot-btn" onclick="downloadPlot('plot{feature_plot_counter}', 'feature_target_relationship')">
                                        <i class="fas fa-download"></i> Download
                                    </button>
                                    <button class="plot-btn" onclick="resetPlot('plot{feature_plot_counter}')">
                                        <i class="fas fa-undo"></i> Reset View
                                    </button>
                                </div>
                                <div class="plot-title">Feature-Target Relationship</div>
                                <div id="plot{feature_plot_counter}">
                                    {feature_plots_html['feature_target_relationship']}
                                </div>
                                <div class="plot-description">
                                    Relationship between features and target variable. 
                                    Green points indicate features significantly related to target (ANOVA p < 0.05).
                                    Larger points have higher ANOVA F-values.
                                </div>
                            </div>
            """
            feature_plot_counter += 1
        
        if 'feature_top_anova' in feature_plots_html:
            html_content += f"""
                            <div class="plot-container">
                                <div class="plot-controls">
                                    <button class="plot-btn" onclick="downloadPlot('plot{feature_plot_counter}', 'feature_top_anova')">
                                        <i class="fas fa-download"></i> Download
                                    </button>
                                    <button class="plot-btn" onclick="resetPlot('plot{feature_plot_counter}')">
                                        <i class="fas fa-undo"></i> Reset View
                                    </button>
                                </div>
                                <div class="plot-title">Top Features by ANOVA F-value</div>
                                <div id="plot{feature_plot_counter}">
                                    {feature_plots_html['feature_top_anova']}
                                </div>
                                <div class="plot-description">
                                    Features with strongest linear relationships to target (higher F-values = stronger relationship).
                                    Green bars indicate statistically significant relationships (p < 0.05).
                                </div>
                            </div>
            """
            feature_plot_counter += 1
        
        if 'feature_top_mi' in feature_plots_html:
            html_content += f"""
                            <div class="plot-container">
                                <div class="plot-controls">
                                    <button class="plot-btn" onclick="downloadPlot('plot{feature_plot_counter}', 'feature_top_mi')">
                                        <i class="fas fa-download"></i> Download
                                    </button>
                                    <button class="plot-btn" onclick="resetPlot('plot{feature_plot_counter}')">
                                        <i class="fas fa-undo"></i> Reset View
                                    </button>
                                </div>
                                <div class="plot-title">Top Features by Mutual Information</div>
                                <div id="plot{feature_plot_counter}">
                                    {feature_plots_html['feature_top_mi']}
                                </div>
                                <div class="plot-description">
                                    Features with highest mutual information scores (capture non-linear relationships).
                                    Darker colors indicate higher mutual information scores.
                                </div>
                            </div>
            """
            feature_plot_counter += 1
        
        if 'feature_importance_comparison' in feature_plots_html:
            html_content += f"""
                            <div class="plot-container">
                                <div class="plot-controls">
                                    <button class="plot-btn" onclick="downloadPlot('plot{feature_plot_counter}', 'feature_importance_comparison')">
                                        <i class="fas fa-download"></i> Download
                                    </button>
                                    <button class="plot-btn" onclick="resetPlot('plot{feature_plot_counter}')">
                                        <i class="fas fa-undo"></i> Reset View
                                    </button>
                                </div>
                                <div class="plot-title">Feature Importance Comparison</div>
                                <div id="plot{feature_plot_counter}">
                                    {feature_plots_html['feature_importance_comparison']}
                                </div>
                                <div class="plot-description">
                                    Comparison of feature importance across different evaluation methods.
                                    Shows how different algorithms rank feature importance.
                                </div>
                            </div>
            """
            feature_plot_counter += 1
        
        if 'pca_variance' in feature_plots_html:
            html_content += f"""
                            <div class="plot-container">
                                <div class="plot-controls">
                                    <button class="plot-btn" onclick="downloadPlot('plot{feature_plot_counter}', 'pca_variance')">
                                        <i class="fas fa-download"></i> Download
                                    </button>
                                    <button class="plot-btn" onclick="resetPlot('plot{feature_plot_counter}')">
                                        <i class="fas fa-undo"></i> Reset View
                                    </button>
                                </div>
                                <div class="plot-title">PCA Explained Variance</div>
                                <div id="plot{feature_plot_counter}">
                                    {feature_plots_html['pca_variance']}
                                </div>
                                <div class="plot-description">
                                    Principal Component Analysis showing variance explained by each component.
                                    Shows how many components are needed to capture 95% of variance (green dashed line).
                                </div>
                            </div>
            """
            feature_plot_counter += 1
        
        html_content += """
                        </div>
                        </section>
                        <section class="section"> 
                        <div >
                            <h3>Feature Analysis Insights</h3>
                            <ul class="fair-checklist">
                                <li><strong>Correlation Analysis:</strong> Identifies redundant features that can be removed</li>
                                <li><strong>ANOVA F-values:</strong> Measures linear relationships between features and target</li>
                                <li><strong>Mutual Information:</strong> Captures non-linear relationships and dependencies</li>
                                <li><strong>Feature Importance:</strong> Shows which features contribute most to model predictions</li>
                                <li><strong>PCA Analysis:</strong> Helps with dimensionality reduction and feature engineering</li>
                            </ul>
                        </div>
                    </section>
                </div>
        """
    
    html_content += f"""
                <!-- Dataset Info Tab -->
                <div id="data" class="tab-content">
                    <section class="section">
                        <h2><i class="fas fa-database"></i> Dataset Information</h2>
                        
                        <h3>ML Pipeline Overview</h3>
                        <pre><code>Croissant Dataset → Feature Extraction → Preprocessing → 
ClusterID-aware CV → Model Training → Evaluation → Visualization</code></pre>
                        
                        <h3>Key Pipeline Components</h3>
                        <table class="interactive-table">
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
    """
    
    # Add feature evaluator row conditionally
    if has_feature_plots:
        html_content += """                                <tr>
                                    <td>FeatureEvaluator</td>
                                    <td>Analyze feature importance and relationships</td>
                                    <td>Interactive correlation, ANOVA, mutual information, PCA</td>
                                </tr>
    """
    
    html_content += f"""                                <tr>
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
                                    <td>Interactive visualizations, statistical tests</td>
                                </tr>
                            </tbody>
                        </table>
                   </section> 
                   <section class="section">
                        <h3>Methodology Notes</h3>
                        <ul class="fair-checklist">
                            <li><strong>ClusterID-aware cross-validation:</strong> Prevents data leakage by ensuring all interfaces from the same sequence cluster stay together in training or testing sets</li>
    """
    
    # Add feature evaluation note conditionally
    if has_feature_plots:
        html_content += """                            <li><strong>Interactive feature evaluation:</strong> Comprehensive analysis with zoomable, hoverable plots for correlation, ANOVA, mutual information, and PCA</li>
    """
    
    html_content += f"""                            <li><strong>Stratified sampling:</strong> Maintains class balance across folds</li>
                            <li><strong>Multiple metrics:</strong> Evaluates models using Accuracy, Precision, Recall, F1-Score, and ROC-AUC</li>
                            <li><strong>Feature scaling:</strong> All features standardized for distance-based algorithms</li>
                            <li><strong>Model diversity:</strong> Includes tree-based, linear, and kernel-based models</li>
                        </ul>
                       
                        <!-- 
                        <div class="fair-section">
                            <h3>Interactive Dashboard Features</h3>
                            <ul class="fair-checklist">
                                <li><strong>Hover for Details:</strong> Hover over any plot point to see exact values</li>
                                <li><strong>Zoom & Pan:</strong> Click and drag to zoom, double-click to reset view</li>
                                <li><strong>Sortable Tables:</strong> Click column headers to sort table data</li>
                                <li><strong>Download Plots:</strong> Save any plot as PNG image</li>
                                <li><strong>Tab Persistence:</strong> Your selected tab is saved between visits</li>
                                <li><strong>Responsive Design:</strong> Works on desktop, tablet, and mobile</li>
                            </ul>
                        </div>
                        -->
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
                        <img src="https://img.shields.io/badge/Interactive-Dashboard-blue" alt="Interactive Dashboard"></a>
                        <a href="https://www.python.org/">
                        <img src="https://img.shields.io/badge/Python-3.8%2B-blue.svg" alt="Python 3.8+"></a>
                        <a href="https://plotly.com/">
                        <img src="https://img.shields.io/badge/Plotly.js-Interactive-orange" alt="Plotly.js Interactive"></a>
                        <a href="https://mlcommons.org/croissant/">
                        <img src="https://img.shields.io/badge/ML-Croissant_1.0-yellow" alt="MLCommons Croissant"></a>
    """
    
    # Add feature analysis badge conditionally
    if has_feature_plots:
        html_content += """                        <a href="https://scikit-learn.org/stable/modules/feature_selection.html">
                        <img src="https://img.shields.io/badge/Interactive-Feature_Analysis-purple" alt="Interactive Feature Analysis"></a>
    """
    
    html_content += f"""                    </div>
                    
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
                        <a href="https://plotly.com/" class="footer-link" target="_blank">
                            <i class="fas fa-chart-line"></i> Plotly.js
                        </a>
                    </div>
                    
                    <div class="copyright">
                        <p>ELIXIR PPI Benchmark Interactive ML Dashboard • Generated on: {current_date}</p>
                        <p>Analysis Method: ClusterID-aware Cross-Validation with Interactive Feature Evaluation</p>
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
    
    print(f"✅ Interactive HTML dashboard generated: {output_path}")
    
    return html_content

def main():
    """Main function to generate the interactive HTML dashboard."""
    parser = argparse.ArgumentParser(
        description='Generate interactive HTML dashboard with Plotly.js for ML results from ppi_ml_croissant.py'
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
    
    args = parser.parse_args()
    
    print("""
╔══════════════════════════════════════════════════════════╗
║  Interactive ML Results Dashboard Generator              ║
║  with Plotly.js Visualizations                           ║
║  for ELIXIR PPI Benchmark                                ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    # Load results
    print(f"📊 Loading results from: {args.results}")
    results = load_results_json(args.results)
    
    # Create interactive performance plots
    print("🎨 Creating interactive performance plots...")
    performance_plots_html = create_interactive_performance_plots(results)
    
    # Create interactive feature plots
    feature_plots_html = {}
    if args.feature_eval:
        print("📈 Creating interactive feature plots...")
        feature_plots_html = create_interactive_feature_plots(args.feature_eval)
    
    # Generate HTML dashboard
    print("📄 Generating interactive HTML dashboard...")
    html_content = generate_html_dashboard(
        results=results,
        performance_plots_html=performance_plots_html,
        feature_plots_html=feature_plots_html,
        feature_eval_path=args.feature_eval,
        output_path=args.output
    )
    
    print(f"\n✅ Interactive dashboard generation complete!")
    print(f"   HTML file: {args.output}")
    print(f"   Models analyzed: {len([k for k in results.keys() if k != 'cross_validation_settings'])}")
    
    if args.feature_eval:
        print(f"   Interactive feature plots included: {len(feature_plots_html)}")
    
    print(f"\n🎯 Interactive Dashboard Features:")
    print("   • Hover-over tooltips with detailed values")
    print("   • Zoom and pan on all plots")
    print("   • Click-and-drag to select regions")
    print("   • Double-click to reset plot views")
    print("   • Sortable tables (click column headers)")
    print("   • Download plots as PNG images")
    print("   • Tab persistence (remembers your selection)")
    print("   • Responsive design for all devices")
    
    print(f"\n📊 Interactive Plots Generated:")
    print(f"   • Performance metrics comparison")
    print(f"   • F1-Score per fold trends")
    print(f"   • Metric distribution box plots")
    print(f"   • Radar chart for overall performance")
    if feature_plots_html:
        print(f"   • Feature correlation analysis")
        print(f"   • Feature-target relationship scatter plots")
        print(f"   • ANOVA and mutual information rankings")
        print(f"   • Feature importance comparisons")
        print(f"   • PCA explained variance")
    
    return html_content

if __name__ == "__main__":
    main()
