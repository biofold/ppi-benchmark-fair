#!/usr/bin/env python3
"""
Visualization Script for GitHub Repository FAIRness Analysis
Creates various plots and charts from the analysis results
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
matplotlib.rcParams['figure.figsize'] = [12, 8]
matplotlib.rcParams['font.size'] = 12

class FAIRVisualizer:
    """Visualize FAIR analysis results"""
    
    def __init__(self, report_file: str = None):
        """
        Initialize visualizer with report data
        
        Args:
            report_file: Path to JSON report file
        """
        self.report_file = report_file
        self.report_data = None
        self.df_scores = None
        self.df_improvements = None
        self.output_dir = "fair_visualizations"
        
        # Create output directory
        Path(self.output_dir).mkdir(exist_ok=True)
        
        if report_file:
            self.load_report(report_file)
    
    def load_report(self, report_file: str):
        """Load report data from JSON file"""
        try:
            with open(report_file, 'r') as f:
                self.report_data = json.load(f)
            
            # Create DataFrames
            if self.report_data.get('scores'):
                self.df_scores = pd.DataFrame(self.report_data['scores'])
            
            if self.report_data.get('improvements'):
                self.df_improvements = pd.DataFrame(self.report_data['improvements'])
            
            print(f"✓ Loaded report: {len(self.df_scores)} repositories")
            
        except Exception as e:
            print(f"Error loading report: {e}")
            raise
    
    def safe_correlation(self, df, columns):
        """Calculate correlation matrix safely handling zero variance"""
        valid_cols = []
        
        for col in columns:
            if col in df.columns:
                # Check if column has variance or if it's a single value
                if df[col].nunique() > 1 or len(df) == 1:
                    valid_cols.append(col)
        
        if len(valid_cols) < 2:
            # Return identity matrix if not enough columns
            return pd.DataFrame(np.eye(len(valid_cols)), 
                              index=valid_cols, 
                              columns=valid_cols)
        
        corr_matrix = df[valid_cols].corr(min_periods=1)
        
        # Fill diagonal with 1
        for col in valid_cols:
            if col in corr_matrix.index:
                corr_matrix.loc[col, col] = 1
        
        # Fill remaining NaN with 0
        return corr_matrix.fillna(0)
    
    def create_all_visualizations(self, output_format: str = 'html'):
        """
        Create all visualizations
        
        Args:
            output_format: 'html' for interactive, 'png' for static
        """
        if self.df_scores is None:
            print("No data to visualize. Load a report first.")
            return
        
        print(f"📊 Creating visualizations in {self.output_dir}/...")
        
        # 1. Create individual figures with explanations
        self.create_individual_figures(output_format)
        
        # 2. Individual charts
        self.create_radar_chart(output_format)
        
        # Only create parallel categories if we have enough data
        if len(self.df_scores) >= 3:
            self.create_parallel_categories(output_format)
        
        self.create_score_distribution(output_format)
        
        if self.df_improvements is not None and not self.df_improvements.empty:
            self.create_improvement_heatmap(output_format)
            self.create_improvement_priority_chart(output_format)
        
        self.create_principle_comparison(output_format)
        
        if 'metadata_files_count' in self.df_scores.columns:
            self.create_metadata_analysis(output_format)
        
        self.create_repository_ranking(output_format)
        
        # 3. Combined report
        self.create_combined_report()
        
        # 4. Create overview dashboard
        self.create_overview_dashboard(output_format)
        
        # 5. Create matplotlib visualizations
        self.create_matplotlib_visualizations()
        
        print(f"✅ All visualizations saved to {self.output_dir}/")
    
    def create_individual_figures(self, output_format: str = 'html'):
        """Create separate figures with explanations for each visualization"""
        if self.df_scores is None or len(self.df_scores) == 0:
            return
        
        principles = ['findable', 'accessible', 'interoperable', 'reusable']
        
        # ============================================
        # Figure 1: FAIR Score Ranking / Single Repository Gauge
        # ============================================
        fig1 = go.Figure()
        
        if len(self.df_scores) == 1:
            # Single repository - Gauge chart
            repo_name = self.df_scores['repository'].iloc[0].split('/')[-1][:20]
            total_score = self.df_scores['total'].iloc[0]
            
            fig1.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=total_score,
                    title={'text': f"{repo_name}<br>FAIR Score"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "red"},
                            {'range': [30, 60], 'color': "orange"},
                            {'range': [60, 80], 'color': "yellow"},
                            {'range': [80, 100], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': total_score
                        }
                    }
                )
            )
            
            fig1.update_layout(
                title={
                    'text': "Figure 1: Repository FAIR Score Gauge<br><sup>Shows overall FAIR compliance with color-coded ranges</sup>",
                    'y': 0.9
                },
                height=400,
                width=600,
                margin=dict(t=100, b=50, l=50, r=50)
            )
        else:
            # Multiple repositories - Ranking bar chart
            df_sorted = self.df_scores.sort_values('total', ascending=True)
            
            fig1.add_trace(
                go.Bar(
                    y=df_sorted['repository'].apply(lambda x: x.split('/')[-1][:20]),
                    x=df_sorted['total'],
                    orientation='h',
                    marker=dict(
                        color=df_sorted['total'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(
                            title="Score",
                            x=0.95,
                            len=0.8,
                            thickness=15
                        )
                    ),
                    text=[f"{score:.1f}" for score in df_sorted['total']],
                    textposition='auto',
                    name='Total Score'
                )
            )
            
            fig1.update_layout(
                title={
                    'text': "Figure 1: Repository FAIR Score Ranking<br><sup>Repositories sorted by total FAIR score</sup>",
                    'y': 0.9
                },
                xaxis_title="FAIR Score (0-100)",
                yaxis_title="Repository",
                height=max(400, len(df_sorted) * 30),
                width=800,
                margin=dict(t=100, b=50, l=150, r=100)
            )
        
        # Save Figure 1
        if output_format == 'html':
            fig1.write_html(f"{self.output_dir}/figure1_score_ranking.html")
        else:
            fig1.write_image(f"{self.output_dir}/figure1_score_ranking.png", width=1000, height=600)
        
        # ============================================
        # Figure 2: Score Distribution
        # ============================================
        fig2 = go.Figure()
        
        fig2.add_trace(
            go.Histogram(
                x=self.df_scores['total'],
                nbinsx=10 if len(self.df_scores) < 10 else 20,
                marker_color='lightblue',
                marker_line_color='darkblue',
                marker_line_width=1,
                name='Score Distribution'
            )
        )
        
        # Add mean and median lines
        mean_score = self.df_scores['total'].mean()
        median_score = self.df_scores['total'].median()
        
        fig2.add_vline(x=mean_score, line_dash="dash", line_color="red", 
                      annotation_text=f"Mean: {mean_score:.1f}", 
                      annotation_position="top right")
        
        fig2.add_vline(x=median_score, line_dash="dot", line_color="green", 
                      annotation_text=f"Median: {median_score:.1f}", 
                      annotation_position="top left")
        
        fig2.update_layout(
            title={
                'text': "Figure 2: FAIR Score Distribution<br><sup>Histogram showing frequency of scores across repositories</sup>",
                'y': 0.9
            },
            xaxis_title="FAIR Score",
            yaxis_title="Number of Repositories",
            height=400,
            width=700,
            showlegend=False,
            margin=dict(t=100, b=50, l=50, r=50)
        )
        
        # Save Figure 2
        if output_format == 'html':
            fig2.write_html(f"{self.output_dir}/figure2_score_distribution.html")
        else:
            fig2.write_image(f"{self.output_dir}/figure2_score_distribution.png", width=900, height=500)
        
        # ============================================
        # Figure 3: Radar Chart for Top Repository
        # ============================================
        fig3 = go.Figure()
        
        if len(self.df_scores) > 0:
            if len(self.df_scores) == 1:
                top_repo = self.df_scores.iloc[0]
            else:
                top_repo = self.df_scores.loc[self.df_scores['total'].idxmax()]
            
            repo_name = top_repo['repository'].split('/')[-1][:20]
            scores = [top_repo[p] for p in principles]
            
            fig3.add_trace(
                go.Scatterpolar(
                    r=scores + [scores[0]],  # Close the loop
                    theta=[p.capitalize() for p in principles] + [principles[0].capitalize()],
                    fill='toself',
                    name=f"{repo_name}",
                    line_color='green',
                    fillcolor='rgba(0, 128, 0, 0.3)'
                )
            )
            
            # Add annotations for each score
            for i, (principle, score) in enumerate(zip(principles, scores)):
                angle = i * (360 / len(principles))
                fig3.add_annotation(
                    text=f"{score:.1f}",
                    x=angle,
                    y=score + 5,
                    showarrow=False,
                    font=dict(size=10, color="black")
                )
        
        fig3.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    tickfont=dict(size=10),
                    gridcolor='lightgray'
                ),
                angularaxis=dict(
                    tickfont=dict(size=12),
                    rotation=90,
                    direction='clockwise'
                ),
                bgcolor='white'
            ),
            title={
                'text': f"Figure 3: FAIR Principles Radar Chart<br><sup>Shows performance across all FAIR principles for {repo_name if len(self.df_scores) > 0 else 'repository'}</sup>",
                'y': 0.95
            },
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            height=500,
            width=600,
            margin=dict(t=100, b=50, l=50, r=50)
        )
        
        # Save Figure 3
        if output_format == 'html':
            fig3.write_html(f"{self.output_dir}/figure3_radar_chart.html")
        else:
            fig3.write_image(f"{self.output_dir}/figure3_radar_chart.png", width=800, height=600)
        
        # ============================================
        # Figure 4: Improvement Priority
        # ============================================
        fig4 = go.Figure()
        
        if self.df_improvements is not None and not self.df_improvements.empty:
            priority_counts = self.df_improvements['priority'].value_counts()
            # Ensure all priorities are present
            for priority in ['High', 'Medium', 'Low']:
                if priority not in priority_counts:
                    priority_counts[priority] = 0
            
            priority_counts = priority_counts.reindex(['High', 'Medium', 'Low'])
            
            fig4.add_trace(
                go.Bar(
                    x=priority_counts.index,
                    y=priority_counts.values,
                    marker_color=['red', 'orange', 'green'],
                    text=[f"{count}" for count in priority_counts.values],
                    textposition='outside',
                    name='Improvement Priority'
                )
            )
            
            fig4.update_layout(
                title={
                    'text': "Figure 4: Improvement Priority Distribution<br><sup>Number of improvements needed by priority level</sup>",
                    'y': 0.9
                },
                xaxis_title="Priority Level",
                yaxis_title="Number of Improvements",
                height=400,
                width=600,
                showlegend=False,
                margin=dict(t=100, b=50, l=50, r=50)
            )
        else:
            fig4.add_annotation(
                text="No improvements needed or improvement data not available",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=16)
            )
            
            fig4.update_layout(
                title={
                    'text': "Figure 4: Improvement Priority Distribution<br><sup>No improvement data available</sup>",
                    'y': 0.9
                },
                height=300,
                width=500,
                showlegend=False,
                margin=dict(t=100, b=50, l=50, r=50)
            )
        
        # Save Figure 4
        if output_format == 'html':
            fig4.write_html(f"{self.output_dir}/figure4_improvement_priority.html")
        else:
            fig4.write_image(f"{self.output_dir}/figure4_improvement_priority.png", width=700, height=500)
        
        # ============================================
        # Figure 5: Metadata Analysis
        # ============================================
        fig5 = go.Figure()
        
        if 'metadata_files_count' in self.df_scores.columns:
            fig5.add_trace(
                go.Scatter(
                    x=self.df_scores['metadata_files_count'],
                    y=self.df_scores['total'],
                    mode='markers',
                    marker=dict(
                        size=15,
                        color=self.df_scores['total'],
                        colorscale='Plasma',
                        showscale=True,
                        colorbar=dict(
                            title="FAIR Score",
                            x=1.02,
                            len=0.8,
                            thickness=15
                        ),
                        line=dict(width=1, color='black')
                    ),
                    text=self.df_scores['repository'].apply(lambda x: x.split('/')[-1]),
                    hovertemplate=(
                        '<b>Repository:</b> %{text}<br>' +
                        '<b>Metadata Files:</b> %{x}<br>' +
                        '<b>FAIR Score:</b> %{y:.1f}<br>' +
                        '<extra></extra>'
                    ),
                    name='Repositories'
                )
            )
            
            # Add trendline if enough data points
            if len(self.df_scores) > 1:
                z = np.polyfit(self.df_scores['metadata_files_count'], self.df_scores['total'], 1)
                p = np.poly1d(z)
                
                x_range = np.linspace(
                    self.df_scores['metadata_files_count'].min(),
                    self.df_scores['metadata_files_count'].max(),
                    100
                )
                
                fig5.add_trace(
                    go.Scatter(
                        x=x_range,
                        y=p(x_range),
                        mode='lines',
                        line=dict(color='red', dash='dash', width=2),
                        name='Trendline',
                        hovertemplate='Trendline<extra></extra>'
                    )
                )
            
            # Calculate correlation
            if len(self.df_scores) > 1:
                correlation = self.df_scores['metadata_files_count'].corr(self.df_scores['total'])
                fig5.add_annotation(
                    text=f"Correlation: {correlation:.3f}",
                    xref="paper",
                    yref="paper",
                    x=0.02,
                    y=0.98,
                    showarrow=False,
                    font=dict(size=12, color="black"),
                    bgcolor="white",
                    bordercolor="black",
                    borderwidth=1
                )
            
            fig5.update_layout(
                title={
                    'text': "Figure 5: Metadata Files vs FAIR Score<br><sup>Relationship between metadata quantity and overall FAIR compliance</sup>",
                    'y': 0.9
                },
                xaxis_title="Number of Metadata Files",
                yaxis_title="FAIR Score",
                height=500,
                width=800,
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                ),
                margin=dict(t=100, b=50, l=50, r=100)
            )
        else:
            fig5.add_annotation(
                text="Metadata analysis not available",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=16)
            )
            
            fig5.update_layout(
                title={
                    'text': "Figure 5: Metadata Analysis<br><sup>Metadata data not available</sup>",
                    'y': 0.9
                },
                height=300,
                width=500,
                showlegend=False,
                margin=dict(t=100, b=50, l=50, r=50)
            )
        
        # Save Figure 5
        if output_format == 'html':
            fig5.write_html(f"{self.output_dir}/figure5_metadata_analysis.html")
        else:
            fig5.write_image(f"{self.output_dir}/figure5_metadata_analysis.png", width=900, height=600)
        
        # ============================================
        # Figure 6: Principle Comparison
        # ============================================
        fig6 = go.Figure()
        
        if len(self.df_scores) == 1:
            # Single repository principle scores
            repo_scores = self.df_scores.iloc[0]
            scores = [repo_scores[p] for p in principles]
            repo_name = repo_scores['repository'].split('/')[-1][:20]
            
            fig6.add_trace(
                go.Bar(
                    x=[p.capitalize() for p in principles],
                    y=scores,
                    marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
                    text=[f"{score:.1f}" for score in scores],
                    textposition='outside',
                    name=repo_name
                )
            )
            
            title_suffix = f"for {repo_name}"
        else:
            # Multiple repositories - average scores
            avg_scores = [self.df_scores[p].mean() for p in principles]
            
            fig6.add_trace(
                go.Bar(
                    x=[p.capitalize() for p in principles],
                    y=avg_scores,
                    marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
                    text=[f"{score:.1f}" for score in avg_scores],
                    textposition='outside',
                    name='Average Score'
                )
            )
            
            # Add standard deviation as error bars
            std_devs = [self.df_scores[p].std() for p in principles]
            
            fig6.add_trace(
                go.Scatter(
                    x=[p.capitalize() for p in principles],
                    y=avg_scores,
                    mode='markers',
                    marker=dict(
                        color='black',
                        size=8,
                        symbol='diamond'
                    ),
                    error_y=dict(
                        type='data',
                        array=std_devs,
                        visible=True,
                        thickness=1.5,
                        width=3
                    ),
                    name='± Std Dev',
                    showlegend=True
                )
            )
            
            title_suffix = f"across {len(self.df_scores)} repositories"
        
        fig6.update_layout(
            title={
                'text': f"Figure 6: FAIR Principles Comparison<br><sup>Performance breakdown by FAIR principle {title_suffix}</sup>",
                'y': 0.9
            },
            xaxis_title="FAIR Principle",
            yaxis_title="Score",
            yaxis_range=[0, 100],
            height=500,
            width=700,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            margin=dict(t=100, b=50, l=50, r=50)
        )
        
        # Save Figure 6
        if output_format == 'html':
            fig6.write_html(f"{self.output_dir}/figure6_principle_comparison.html")
        else:
            fig6.write_image(f"{self.output_dir}/figure6_principle_comparison.png", width=900, height=600)
        
        # ============================================
        # Figure 7: Repository Performance Breakdown
        # ============================================
        fig7 = go.Figure()
        
        if len(self.df_scores) == 1:
            # For single repository: Detailed principle breakdown
            repo_scores = self.df_scores.iloc[0]
            scores = [repo_scores[p] for p in principles]
            
            # Create a more detailed breakdown
            fig7.add_trace(
                go.Bar(
                    x=[p.capitalize() for p in principles],
                    y=scores,
                    marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
                    text=[f"{s:.1f}" for s in scores],
                    textposition='auto',
                    name='Principle Scores'
                )
            )
            
            fig7.update_layout(
                title={
                    'text': "Figure 7: FAIR Principles Detailed Breakdown<br><sup>Individual scores for each FAIR principle</sup>",
                    'y': 0.9
                },
                xaxis_title="Principle",
                yaxis_title="Score",
                yaxis_range=[0, 100],
                height=400,
                width=600,
                margin=dict(t=100, b=50, l=50, r=50)
            )
        else:
            # For multiple repositories: Grouped bar chart
            repo_names = self.df_scores['repository'].apply(lambda x: x.split('/')[-1][:15])
            
            # Show only top 10 repositories for readability
            if len(self.df_scores) > 10:
                top_repos = self.df_scores.nlargest(10, 'total')
                repo_names = top_repos['repository'].apply(lambda x: x.split('/')[-1][:15])
                df_display = top_repos
            else:
                df_display = self.df_scores
            
            colors = px.colors.qualitative.Set3[:4]
            
            for idx, principle in enumerate(principles):
                fig7.add_trace(
                    go.Bar(
                        name=principle.capitalize(),
                        x=repo_names,
                        y=df_display[principle],
                        marker_color=colors[idx],
                        opacity=0.8
                    )
                )
            
            fig7.update_layout(
                title={
                    'text': f"Figure 7: Repository Performance Breakdown<br><sup>Comparison of FAIR principles across repositories</sup>",
                    'y': 0.9
                },
                xaxis_title="Repository",
                yaxis_title="Score",
                yaxis_range=[0, 100],
                barmode='group',
                height=500,
                width=max(800, len(df_display) * 60),
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5
                ),
                margin=dict(t=100, b=100, l=50, r=50)
            )
        
        # Save Figure 7
        if output_format == 'html':
            fig7.write_html(f"{self.output_dir}/figure7_performance_breakdown.html")
        else:
            fig7.write_image(f"{self.output_dir}/figure7_performance_breakdown.png", width=1000, height=600)
        
        # ============================================
        # Figure 8: Missing Elements Heatmap
        # ============================================
        fig8 = go.Figure()
        
        if self.df_improvements is not None and not self.df_improvements.empty and len(self.df_improvements) > 0:
            missing_matrix = pd.crosstab(
                self.df_improvements['repository'],
                self.df_improvements['missing'],
                values=self.df_improvements['potential_points'],
                aggfunc='sum'
            ).fillna(0)
            
            if len(missing_matrix) > 0 and len(missing_matrix.columns) > 0:
                # Sort by total missing points
                missing_matrix['total'] = missing_matrix.sum(axis=1)
                missing_matrix = missing_matrix.sort_values('total', ascending=True)
                missing_matrix = missing_matrix.drop('total', axis=1)
                
                # Limit columns for readability
                if len(missing_matrix.columns) > 15:
                    col_sums = missing_matrix.sum().sort_values(ascending=False)
                    top_cols = col_sums.head(15).index
                    missing_matrix = missing_matrix[top_cols]
                
                y_labels = [str(x).split('/')[-1][:15] for x in missing_matrix.index]
                
                fig8.add_trace(
                    go.Heatmap(
                        z=missing_matrix.values,
                        x=missing_matrix.columns,
                        y=y_labels,
                        colorscale='Reds',
                        hoverongaps=False,
                        hovertemplate=(
                            '<b>Repository:</b> %{y}<br>' +
                            '<b>Missing Element:</b> %{x}<br>' +
                            '<b>Potential Points:</b> %{z}<br>' +
                            '<extra></extra>'
                        ),
                        colorbar=dict(
                            title="Potential<br>Points",
                            x=1.02,
                            len=0.8,
                            thickness=15
                        )
                    )
                )
                
                fig8.update_layout(
                    title={
                        'text': "Figure 8: Missing Elements Heatmap<br><sup>Potential points gain by addressing missing FAIR elements</sup>",
                        'y': 0.9
                    },
                    xaxis_title="Missing FAIR Element",
                    yaxis_title="Repository",
                    height=max(500, len(missing_matrix) * 30),
                    width=max(800, len(missing_matrix.columns) * 40),
                    margin=dict(t=100, b=100, l=150, r=100)
                )
                
                fig8.update_xaxes(tickangle=45)
            else:
                fig8.add_annotation(
                    text="No missing elements data available",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=16)
                )
                
                fig8.update_layout(
                    title={
                        'text': "Figure 8: Missing Elements Heatmap<br><sup>No missing elements data available</sup>",
                        'y': 0.9
                    },
                    height=300,
                    width=500,
                    margin=dict(t=100, b=50, l=50, r=50)
                )
        else:
            fig8.add_annotation(
                text="No improvements data available",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=16)
            )
            
            fig8.update_layout(
                title={
                    'text': "Figure 8: Missing Elements Heatmap<br><sup>No improvement data available</sup>",
                    'y': 0.9
                },
                height=300,
                width=500,
                margin=dict(t=100, b=50, l=50, r=50)
            )
        
        # Save Figure 8
        if output_format == 'html':
            fig8.write_html(f"{self.output_dir}/figure8_missing_elements.html")
        else:
            fig8.write_image(f"{self.output_dir}/figure8_missing_elements.png", width=900, height=600)
        
        # ============================================
        # Figure 9: Score Correlation Heatmap
        # ============================================
        fig9 = go.Figure()
        
        correlation_cols = ['total', 'findable', 'accessible', 'interoperable', 'reusable']
        if 'metadata_files_count' in self.df_scores.columns:
            correlation_cols.append('metadata_files_count')
        
        # Use safe correlation calculation
        corr_matrix = self.safe_correlation(self.df_scores, correlation_cols)
        
        # Prepare labels
        x_labels = []
        for col in correlation_cols:
            if col == 'metadata_files_count':
                x_labels.append('Metadata<br>Files')
            else:
                x_labels.append(col.capitalize()[:10])
        
        y_labels = x_labels.copy()
        
        fig9.add_trace(
            go.Heatmap(
                z=corr_matrix.values,
                x=x_labels,
                y=y_labels,
                colorscale='RdBu',
                zmid=0,
                zmin=-1,
                zmax=1,
                text=np.round(corr_matrix.values, 2),
                texttemplate='%{text}',
                textfont=dict(size=12, color='black'),
                hoverongaps=False,
                hovertemplate=(
                    '<b>Variable 1:</b> %{y}<br>' +
                    '<b>Variable 2:</b> %{x}<br>' +
                    '<b>Correlation:</b> %{z:.3f}<br>' +
                    '<extra></extra>'
                ),
                colorbar=dict(
                    title="Correlation<br>Coefficient",
                    x=1.02,
                    len=0.8,
                    thickness=15,
                    tickvals=[-1, -0.5, 0, 0.5, 1],
                    ticktext=['-1.0', '-0.5', '0.0', '0.5', '1.0']
                )
            )
        )
        
        # Add annotation about correlation interpretation
        fig9.add_annotation(
            text="Values near +1: Strong positive correlation<br>Values near -1: Strong negative correlation<br>Values near 0: Weak or no correlation",
            xref="paper",
            yref="paper",
            x=1.05,
            y=0.5,
            showarrow=False,
            align="left",
            font=dict(size=10, color="gray"),
            bgcolor="white",
            bordercolor="gray",
            borderwidth=1
        )
        
        fig9.update_layout(
            title={
                'text': "Figure 9: FAIR Score Correlation Matrix<br><sup>Relationships between different FAIR metrics</sup>",
                'y': 0.9
            },
            xaxis_title="Metric",
            yaxis_title="Metric",
            height=500,
            width=700,
            margin=dict(t=100, b=50, l=150, r=200)
        )
        
        # Save Figure 9
        if output_format == 'html':
            fig9.write_html(f"{self.output_dir}/figure9_correlation_matrix.html")
        else:
            fig9.write_image(f"{self.output_dir}/figure9_correlation_matrix.png", width=900, height=600)
        
        print(f"✓ Created 9 individual figures with explanations in {self.output_dir}/")
    
    def create_overview_dashboard(self, output_format: str = 'html'):
        """Create an overview dashboard with all figures (simpler version)"""
        # Create an HTML file that references all individual figures
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>FAIR Analysis Dashboard</title>
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f8f9fa;
                    color: #333;
                }
                .container {
                    max-width: 1400px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 0 20px rgba(0,0,0,0.1);
                }
                .header {
                    text-align: center;
                    margin-bottom: 40px;
                    padding-bottom: 20px;
                    border-bottom: 2px solid #007bff;
                }
                .header h1 {
                    color: #007bff;
                    margin-bottom: 10px;
                }
                .header p {
                    color: #666;
                    font-size: 16px;
                    max-width: 800px;
                    margin: 0 auto;
                    line-height: 1.6;
                }
                .figure-container {
                    margin: 40px 0;
                    padding: 20px;
                    background-color: #f8f9fa;
                    border-radius: 8px;
                    border-left: 4px solid #007bff;
                }
                .figure-title {
                    color: #495057;
                    margin-bottom: 15px;
                    font-size: 20px;
                }
                .figure-description {
                    color: #6c757d;
                    margin-bottom: 20px;
                    font-size: 14px;
                    line-height: 1.5;
                }
                .figure-embed {
                    width: 100%;
                    height: 500px;
                    border: none;
                    border-radius: 5px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }
                .nav-bar {
                    position: sticky;
                    top: 0;
                    background-color: white;
                    padding: 15px;
                    border-radius: 5px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    margin-bottom: 30px;
                    z-index: 1000;
                }
                .nav-bar ul {
                    list-style-type: none;
                    padding: 0;
                    margin: 0;
                    display: flex;
                    flex-wrap: wrap;
                    gap: 10px;
                    justify-content: center;
                }
                .nav-bar li {
                    display: inline;
                }
                .nav-bar a {
                    text-decoration: none;
                    color: #007bff;
                    padding: 8px 15px;
                    border-radius: 20px;
                    border: 1px solid #007bff;
                    transition: all 0.3s;
                    font-size: 14px;
                }
                .nav-bar a:hover {
                    background-color: #007bff;
                    color: white;
                }
                .footer {
                    text-align: center;
                    margin-top: 40px;
                    padding-top: 20px;
                    border-top: 1px solid #dee2e6;
                    color: #6c757d;
                    font-size: 14px;
                }
                .key-metrics {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin: 30px 0;
                }
                .metric-card {
                    background: linear-gradient(135deg, #007bff, #0056b3);
                    color: white;
                    padding: 20px;
                    border-radius: 8px;
                    text-align: center;
                }
                .metric-value {
                    font-size: 32px;
                    font-weight: bold;
                    margin-bottom: 5px;
                }
                .metric-label {
                    font-size: 14px;
                    opacity: 0.9;
                }
                .toggle-button {
                    background-color: #28a745;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 5px;
                    cursor: pointer;
                    font-size: 14px;
                    margin: 10px 5px;
                    transition: background-color 0.3s;
                }
                .toggle-button:hover {
                    background-color: #218838;
                }
                .toggle-button.hidden {
                    background-color: #dc3545;
                }
                @media (max-width: 768px) {
                    .figure-embed {
                        height: 400px;
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
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📊 FAIR Analysis Dashboard</h1>
                    <p>Comprehensive visualization of FAIR (Findable, Accessible, Interoperable, Reusable) principles compliance across GitHub repositories</p>
                </div>
                
                <div class="nav-bar">
                    <ul>
                        <li><a href="#figure1">Score Ranking</a></li>
                        <li><a href="#figure2">Distribution</a></li>
                        <li><a href="#figure3">Radar Chart</a></li>
                        <li><a href="#figure4">Improvements</a></li>
                        <li><a href="#figure5">Metadata</a></li>
                        <li><a href="#figure6">Principles</a></li>
                        <li><a href="#figure7">Performance</a></li>
                        <li><a href="#figure8">Missing Elements</a></li>
                        <li><a href="#figure9">Correlations</a></li>
                    </ul>
                </div>
                
                <!-- Key Metrics Section -->
                <div class="key-metrics" id="metrics">
        """
        
        # Add key metrics if report data exists
        if self.report_data and 'statistics' in self.report_data:
            stats = self.report_data['statistics']
            metrics = [
                ('Repositories', f"{len(self.df_scores)}"),
                ('Average Score', f"{stats.get('average_total', 0):.1f}"),
                ('Highest Score', f"{stats.get('highest_total', 0):.1f}"),
                ('Lowest Score', f"{stats.get('lowest_total', 0):.1f}"),
            ]
            
            if 'average_metadata_files' in stats:
                metrics.append(('Avg Metadata', f"{stats.get('average_metadata_files', 0):.1f}"))
            
            for label, value in metrics:
                html_content += f"""
                    <div class="metric-card">
                        <div class="metric-value">{value}</div>
                        <div class="metric-label">{label}</div>
                    </div>
                """
        
        html_content += """
                </div>
                
                <!-- Control Buttons -->
                <div style="text-align: center; margin: 20px 0;">
                    <button class="toggle-button" onclick="toggleAllFigures('show')">Show All Figures</button>
                    <button class="toggle-button hidden" onclick="toggleAllFigures('hide')">Hide All Figures</button>
                    <button class="toggle-button" onclick="expandAllFigures()">Expand All</button>
                    <button class="toggle-button" onclick="collapseAllFigures()">Collapse All</button>
                </div>
        """
        
        # List of figures with descriptions
        figures = [
            {
                'id': 'figure1',
                'title': 'Figure 1: FAIR Score Ranking / Gauge',
                'description': 'Shows overall FAIR compliance score for each repository. For single repositories, a gauge chart indicates the score level with color-coded ranges (red: 0-30, orange: 30-60, yellow: 60-80, green: 80-100).',
                'file': 'figure1_score_ranking.html'
            },
            {
                'id': 'figure2',
                'title': 'Figure 2: Score Distribution',
                'description': 'Histogram displaying the frequency distribution of FAIR scores across all analyzed repositories. Dashed lines show mean (red) and median (green) scores.',
                'file': 'figure2_score_distribution.html'
            },
            {
                'id': 'figure3',
                'title': 'Figure 3: FAIR Principles Radar Chart',
                'description': 'Visualizes performance across all four FAIR principles (Findable, Accessible, Interoperable, Reusable) for the top-performing or single repository. Each axis represents a FAIR principle.',
                'file': 'figure3_radar_chart.html'
            },
            {
                'id': 'figure4',
                'title': 'Figure 4: Improvement Priority Distribution',
                'description': 'Bar chart showing the number of improvements needed categorized by priority level (High, Medium, Low). Helps identify urgent action items.',
                'file': 'figure4_improvement_priority.html'
            },
            {
                'id': 'figure5',
                'title': 'Figure 5: Metadata Files vs FAIR Score',
                'description': 'Scatter plot examining the relationship between the number of metadata files and overall FAIR score. Includes trendline and correlation coefficient.',
                'file': 'figure5_metadata_analysis.html'
            },
            {
                'id': 'figure6',
                'title': 'Figure 6: FAIR Principles Comparison',
                'description': 'Bar chart comparing average scores across the four FAIR principles. For multiple repositories, includes error bars showing standard deviation.',
                'file': 'figure6_principle_comparison.html'
            },
            {
                'id': 'figure7',
                'title': 'Figure 7: Repository Performance Breakdown',
                'description': 'Detailed comparison of FAIR principle scores across repositories. For single repositories, shows individual scores; for multiple repositories, grouped bars display all principles.',
                'file': 'figure7_performance_breakdown.html'
            },
            {
                'id': 'figure8',
                'title': 'Figure 8: Missing Elements Heatmap',
                'description': 'Heatmap showing which FAIR elements are missing across repositories and their potential point value. Redder cells indicate more valuable improvements.',
                'file': 'figure8_missing_elements.html'
            },
            {
                'id': 'figure9',
                'title': 'Figure 9: FAIR Score Correlation Matrix',
                'description': 'Heatmap showing correlations between different FAIR metrics. Helps identify relationships between principles (e.g., if repositories scoring high on Findable also score high on Reusable).',
                'file': 'figure9_correlation_matrix.html'
            }
        ]
        
        # Add each figure section
        for fig in figures:
            html_content += f"""
                <div class="figure-container" id="{fig['id']}">
                    <h2 class="figure-title">{fig['title']}</h2>
                    <p class="figure-description">{fig['description']}</p>
                    <iframe class="figure-embed" src="{fig['file']}" title="{fig['title']}"></iframe>
                </div>
            """

        html_content += f"""
                <div class="footer">
                    <p>FAIR Analysis Dashboard • Generated on: {self.report_data.get('timestamp', 'N/A') if self.report_data else 'N/A'}</p>
                    <p>FAIR Principles: Findable, Accessible, Interoperable, Reusable</p>
                    <p><a href="index.html">View Complete Report</a> | <a href="#metrics">Back to Top</a></p>
                </div>
            </div>
            
            <script>
                // JavaScript for interactive controls
                function toggleAllFigures(action) {{
                    const figures = document.querySelectorAll('.figure-container iframe');
                    const buttons = document.querySelectorAll('.toggle-button');
                    
                    if (action === 'show') {{
                        figures.forEach(fig => {{
                            fig.style.display = 'block';
                            fig.parentElement.parentElement.style.display = 'block';
                        }});
                        buttons[0].classList.add('hidden');
                        buttons[1].classList.remove('hidden');
                    }} else {{
                        figures.forEach(fig => {{
                            fig.style.display = 'none';
                        }});
                        buttons[0].classList.remove('hidden');
                        buttons[1].classList.add('hidden');
                    }}
                }}
                
                function expandAllFigures() {{
                    const figures = document.querySelectorAll('.figure-embed');
                    figures.forEach(fig => {{
                        fig.style.height = '600px';
                    }});
                }}
                
                function collapseAllFigures() {{
                    const figures = document.querySelectorAll('.figure-embed');
                    figures.forEach(fig => {{
                        fig.style.height = '400px';
                    }});
                }}
                
                // Smooth scrolling for navigation
                document.querySelectorAll('.nav-bar a').forEach(anchor => {{
                    anchor.addEventListener('click', function(e) {{
                        e.preventDefault();
                        const targetId = this.getAttribute('href');
                        if (targetId !== '#') {{
                            const targetElement = document.querySelector(targetId);
                            if (targetElement) {{
                                window.scrollTo({{
                                    top: targetElement.offsetTop - 80,
                                    behavior: 'smooth'
                                }});
                            }}
                        }}
                    }});
                }});
                
                // Lazy loading for iframes
                document.addEventListener("DOMContentLoaded", function() {{
                    const iframes = document.querySelectorAll('.figure-embed');
                    
                    const observer = new IntersectionObserver((entries) => {{
                        entries.forEach(entry => {{
                            if (entry.isIntersecting) {{
                                const iframe = entry.target;
                                if (!iframe.dataset.loaded) {{
                                    iframe.dataset.loaded = true;
                                    // Iframes load on src attribute, so no additional action needed
                                }}
                            }}
                        }});
                    }});
                    
                    iframes.forEach(iframe => observer.observe(iframe));
                }});
            </script>
        </body>
        </html>
        """
        
        # Write dashboard HTML file
        with open(f"{self.output_dir}/fair_dashboard.html", 'w') as f:
            f.write(html_content)
        
        print(f"✓ Dashboard saved to {self.output_dir}/fair_dashboard.html")
    
    def create_radar_chart(self, output_format: str = 'html'):
        """Create radar chart showing FAIR principles for each repository"""
        if self.df_scores is None or len(self.df_scores) == 0:
            return
        
        principles = ['findable', 'accessible', 'interoperable', 'reusable']
        
        fig = go.Figure()
        
        # Add traces for each repository (limit to top 10 for readability)
        display_repos = min(10, len(self.df_scores))
        if len(self.df_scores) > 10:
            top_repos = self.df_scores.nlargest(display_repos, 'total')
        else:
            top_repos = self.df_scores
        
        colors = px.colors.qualitative.Set3[:display_repos]
        
        for idx, row in top_repos.iterrows():
            repo_name = row['repository'].split('/')[-1][:20]
            scores = [row[p] for p in principles]
            
            fig.add_trace(go.Scatterpolar(
                r=scores + [scores[0]],  # Close the loop
                theta=[p.capitalize() for p in principles] + [principles[0].capitalize()],
                fill='toself',
                name=f"{repo_name} ({row['total']:.1f})",
                opacity=0.7,
                line_color=colors[idx % len(colors)]
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    tickfont=dict(size=10)
                ),
                angularaxis=dict(
                    tickfont=dict(size=12)
                )
            ),
            title=f"FAIR Principles Radar Chart ({display_repos} Repositories)",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            width=1000,
            height=700
        )
        
        if output_format == 'html':
            fig.write_html(f"{self.output_dir}/radar_chart_legacy.html")
        else:
            fig.write_image(f"{self.output_dir}/radar_chart_legacy.png")
    
    def create_parallel_categories(self, output_format: str = 'html'):
        """Create parallel categories plot - only works with 3+ repositories"""
        if self.df_scores is None or len(self.df_scores) < 3:
            print("Note: Parallel categories plot requires at least 3 repositories")
            # Create a simple placeholder instead
            fig = go.Figure()
            fig.add_annotation(
                text="Parallel categories plot requires at least 3 repositories",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(
                title="Parallel Categories (Not enough data)",
                width=800,
                height=400
            )
            
            if output_format == 'html':
                fig.write_html(f"{self.output_dir}/parallel_categories.html")
            else:
                fig.write_image(f"{self.output_dir}/parallel_categories.png")
            return
        
        try:
            # Sort by total score for consistent ordering
            df_viz = self.df_scores.sort_values('total', ascending=False).copy()
            df_viz['repo_short'] = df_viz['repository'].apply(lambda x: x.split('/')[-1][:15])
            
            # Create performance categories
            for principle in ['findable', 'accessible', 'interoperable', 'reusable']:
                # Use custom bins to ensure all categories have data
                df_viz[f'{principle}_cat'] = pd.cut(
                    df_viz[principle],
                    bins=[0, 40, 70, 90, 100],
                    labels=['Low (0-40)', 'Medium (41-70)', 'High (71-90)', 'Excellent (91-100)']
                )
            
            # Add total category
            df_viz['total_cat'] = pd.cut(
                df_viz['total'],
                bins=[0, 40, 70, 90, 100],
                labels=['Low (0-40)', 'Medium (41-70)', 'High (71-90)', 'Excellent (91-100)']
            )
            
            # Create dimensions - ensure all values are strings
            dimensions = [
                dict(label='Repository', 
                     values=df_viz['repo_short'].astype(str).tolist()),
                dict(label='Findable', 
                     values=df_viz['findable_cat'].astype(str).tolist()),
                dict(label='Accessible', 
                     values=df_viz['accessible_cat'].astype(str).tolist()),
                dict(label='Interoperable', 
                     values=df_viz['interoperable_cat'].astype(str).tolist()),
                dict(label='Reusable', 
                     values=df_viz['reusable_cat'].astype(str).tolist()),
                dict(label='Overall', 
                     values=df_viz['total_cat'].astype(str).tolist())
            ]
            
            # Create the parallel categories plot with better layout
            fig = go.Figure(data=go.Parcoords(
                line=dict(
                    color=df_viz['total'].tolist(),
                    colorscale='Viridis',
                    showscale=True,
                    cmin=0,
                    cmax=100,
                    colorbar=dict(
                        title="Total Score",
                        x=1.02,  # Position outside
                        len=0.8,
                        thickness=15
                    )
                ),
                dimensions=dimensions,
                labelfont=dict(size=10),
                tickfont=dict(size=8),
                rangefont=dict(size=8)
            ))
            
            fig.update_layout(
                title="FAIR Principles Parallel Categories",
                width=1400,  # Wider for better readability
                height=600,
                margin=dict(l=80, r=120, t=80, b=80)  # Adjust margins
            )
            
            if output_format == 'html':
                fig.write_html(f"{self.output_dir}/parallel_categories.html")
            else:
                fig.write_image(f"{self.output_dir}/parallel_categories.png", width=1600, height=700)
                
        except Exception as e:
            print(f"Error creating parallel categories plot: {e}")
            # Create a simple placeholder instead
            fig = go.Figure()
            fig.add_annotation(
                text="Could not create parallel categories plot",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(
                title="Parallel Categories (Error)",
                width=800,
                height=400
            )
            
            if output_format == 'html':
                fig.write_html(f"{self.output_dir}/parallel_categories.html")
            else:
                fig.write_image(f"{self.output_dir}/parallel_categories.png")
    
    def create_score_distribution(self, output_format: str = 'html'):
        """Create distribution plots for FAIR scores"""
        if self.df_scores is None:
            return
        
        principles = ['findable', 'accessible', 'interoperable', 'reusable']
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('Total Score Distribution', 'Findable', 'Accessible',
                          'Interoperable', 'Reusable', 'All Principles'),
            specs=[
                [{'type': 'histogram'}, {'type': 'box'}, {'type': 'box'}],
                [{'type': 'box'}, {'type': 'box'}, {'type': 'violin'}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.12
        )
        
        # 1. Total Score Histogram
        fig.add_trace(
            go.Histogram(
                x=self.df_scores['total'],
                nbinsx=10 if len(self.df_scores) < 10 else 20,
                name='Total Score',
                marker_color='lightblue',
                showlegend=False
            ),
            row=1, col=1
        )
        fig.update_xaxes(title_text="Score", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        
        # 2-5. Box plots for each principle
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for idx, principle in enumerate(principles):
            row = 1 if idx < 2 else 2
            col = (idx % 2) + 2 if idx < 2 else (idx - 2) + 1
            
            fig.add_trace(
                go.Box(
                    y=self.df_scores[principle],
                    name=principle.capitalize(),
                    marker_color=colors[idx],
                    boxpoints='outliers',
                    showlegend=False
                ),
                row=row, col=col
            )
            fig.update_yaxes(title_text="Score", range=[0, 100], row=row, col=col)
        
        # 6. Violin plot for all principles
        for idx, principle in enumerate(principles):
            fig.add_trace(
                go.Violin(
                    y=self.df_scores[principle],
                    name=principle.capitalize(),
                    box_visible=True,
                    meanline_visible=True,
                    marker_color=colors[idx],
                    showlegend=True
                ),
                row=2, col=3
            )
        
        fig.update_layout(
            title="FAIR Score Distributions",
            height=800,
            width=1200,
            showlegend=True,
            template='plotly_white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )
        
        if output_format == 'html':
            fig.write_html(f"{self.output_dir}/score_distribution.html")
        else:
            fig.write_image(f"{self.output_dir}/score_distribution.png")
    
    def create_improvement_heatmap(self, output_format: str = 'html'):
        """Create heatmap of improvements needed"""
        if self.df_improvements is None or self.df_improvements.empty:
            print("Note: No improvements data for heatmap")
            return
        
        # Create improvement matrix
        improvement_matrix = pd.crosstab(
            self.df_improvements['repository'].apply(lambda x: x.split('/')[-1][:15]),
            self.df_improvements['missing'],
            values=self.df_improvements['potential_points'],
            aggfunc='sum'
        ).fillna(0)
        
        # Sort by total improvement points
        improvement_matrix['total'] = improvement_matrix.sum(axis=1)
        improvement_matrix = improvement_matrix.sort_values('total', ascending=False)
        improvement_matrix = improvement_matrix.drop('total', axis=1)
        
        # Limit to top 20 improvements for readability
        if len(improvement_matrix.columns) > 20:
            col_totals = improvement_matrix.sum().sort_values(ascending=False)
            top_cols = col_totals.head(20).index
            improvement_matrix = improvement_matrix[top_cols]
        
        fig = go.Figure(data=go.Heatmap(
            z=improvement_matrix.values,
            x=improvement_matrix.columns,
            y=improvement_matrix.index,
            colorscale='Reds',
            colorbar=dict(
                title="Potential Points",
                x=1.02,  # Position outside
                len=0.8,
                thickness=15
            ),
            hovertemplate=(
                'Repository: %{y}<br>' +
                'Improvement: %{x}<br>' +
                'Potential Points: %{z}<extra></extra>'
            )
        ))
        
        fig.update_layout(
            title="Improvement Heatmap - Potential Score Gains",
            xaxis_title="Improvement Needed",
            yaxis_title="Repository",
            height=600,
            width=max(1000, len(improvement_matrix.columns) * 40),
            margin=dict(l=100, r=150, t=80, b=100)  # Adjust margins for colorbar
        )
        
        if output_format == 'html':
            fig.write_html(f"{self.output_dir}/improvement_heatmap.html")
        else:
            fig.write_image(f"{self.output_dir}/improvement_heatmap.png")
    
    def create_principle_comparison(self, output_format: str = 'html'):
        """Create comparison of FAIR principles"""
        if self.df_scores is None:
            return
        
        principles = ['findable', 'accessible', 'interoperable', 'reusable']
        
        # Create grouped bar chart
        repo_names = self.df_scores['repository'].apply(lambda x: x.split('/')[-1][:15])
        
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set3[:4]
        
        for idx, principle in enumerate(principles):
            fig.add_trace(go.Bar(
                name=principle.capitalize(),
                x=repo_names,
                y=self.df_scores[principle],
                marker_color=colors[idx],
                opacity=0.8
            ))
        
        # Change the bar mode
        fig.update_layout(
            barmode='group',
            title="FAIR Principles Comparison by Repository",
            xaxis_title="Repository",
            yaxis_title="Score",
            yaxis_range=[0, 100],
            height=600,
            width=max(1000, len(self.df_scores) * 40),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )
        
        if output_format == 'html':
            fig.write_html(f"{self.output_dir}/principle_comparison.html")
        else:
            fig.write_image(f"{self.output_dir}/principle_comparison.png")
    
    def create_metadata_analysis(self, output_format: str = 'html'):
        """Create visualizations for metadata analysis"""
        if self.df_scores is None or 'metadata_files_count' not in self.df_scores.columns:
            print("Note: No metadata analysis data available")
            return
        
        # Create a single figure instead of subplots to avoid legend issues
        fig = go.Figure()
        
        # 1. Metadata Files vs FAIR Score (Scatter)
        fig.add_trace(
            go.Scatter(
                x=self.df_scores['metadata_files_count'],
                y=self.df_scores['total'],
                mode='markers',
                marker=dict(
                    size=15,
                    color=self.df_scores['total'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(
                        title="FAIR Score",
                        x=1.02,  # Position outside
                        len=0.8,
                        thickness=15
                    )
                ),
                text=self.df_scores['repository'].apply(lambda x: x.split('/')[-1][:10]),
                name='Repositories',
                hovertemplate=(
                    'Repository: %{text}<br>' +
                    'Metadata Files: %{x}<br>' +
                    'FAIR Score: %{y:.1f}<extra></extra>'
                )
            )
        )
        
        # Add trendline if enough data points
        if len(self.df_scores) > 1:
            z = np.polyfit(self.df_scores['metadata_files_count'], self.df_scores['total'], 1)
            p = np.poly1d(z)
            
            x_range = np.linspace(
                self.df_scores['metadata_files_count'].min(),
                self.df_scores['metadata_files_count'].max(),
                100
            )
            
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=p(x_range),
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    name='Trendline',
                    showlegend=True
                )
            )
        
        # Update layout
        fig.update_layout(
            title="Metadata Files vs FAIR Score",
            xaxis_title="Number of Metadata Files",
            yaxis_title="FAIR Score",
            height=600,
            width=1000,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            margin=dict(l=80, r=150, t=80, b=80)  # Adjust margins for colorbar
        )
        
        if output_format == 'html':
            fig.write_html(f"{self.output_dir}/metadata_analysis.html")
        else:
            fig.write_image(f"{self.output_dir}/metadata_analysis.png")
    
    def create_repository_ranking(self, output_format: str = 'html'):
        """Create detailed repository ranking visualization"""
        if self.df_scores is None:
            return
        
        # Sort by total score
        df_sorted = self.df_scores.sort_values('total', ascending=True)
        
        # Create horizontal bar chart
        fig = go.Figure()
        
        # Add bars for each principle
        principles = ['findable', 'accessible', 'interoperable', 'reusable']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        repo_names = df_sorted['repository'].apply(lambda x: x.split('/')[-1][:20])
        
        for idx, principle in enumerate(principles):
            fig.add_trace(go.Bar(
                y=repo_names,
                x=df_sorted[principle],
                name=principle.capitalize(),
                orientation='h',
                marker_color=colors[idx],
                opacity=0.7
            ))
        
        # Add total score as line
        fig.add_trace(go.Scatter(
            y=repo_names,
            x=df_sorted['total'],
            mode='markers',
            name='Total Score',
            marker=dict(
                size=12,
                color='black',
                symbol='diamond'
            ),
            hovertemplate=(
                'Repository: %{y}<br>' +
                'Total Score: %{x:.1f}<extra></extra>'
            )
        ))
        
        fig.update_layout(
            title="Repository FAIR Score Ranking",
            xaxis_title="Score",
            yaxis_title="Repository",
            barmode='stack',
            height=max(600, len(df_sorted) * 25),
            width=1000,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )
        
        if output_format == 'html':
            fig.write_html(f"{self.output_dir}/repository_ranking.html")
        else:
            fig.write_image(f"{self.output_dir}/repository_ranking.png")
    
    def create_improvement_priority_chart(self, output_format: str = 'html'):
        """Create chart showing improvement priorities"""
        if self.df_improvements is None or self.df_improvements.empty:
            print("Note: No improvements data for priority chart")
            return
        
        # Group by principle and priority
        priority_data = self.df_improvements.groupby(
            ['principle', 'priority']
        )['potential_points'].sum().reset_index()
        
        # Ensure all priorities are present for each principle
        principles = self.df_improvements['principle'].unique()
        all_priorities = ['High', 'Medium', 'Low']
        
        complete_data = []
        for principle in principles:
            for priority in all_priorities:
                existing = priority_data[
                    (priority_data['principle'] == principle) & 
                    (priority_data['priority'] == priority)
                ]
                if len(existing) > 0:
                    complete_data.append(existing.iloc[0].to_dict())
                else:
                    complete_data.append({
                        'principle': principle,
                        'priority': priority,
                        'potential_points': 0
                    })
        
        priority_data = pd.DataFrame(complete_data)
        
        # Create stacked bar chart
        fig = px.bar(
            priority_data,
            x='principle',
            y='potential_points',
            color='priority',
            color_discrete_map={
                'High': 'red',
                'Medium': 'orange',
                'Low': 'green'
            },
            title="Improvement Priority by FAIR Principle",
            labels={
                'principle': 'FAIR Principle',
                'potential_points': 'Total Potential Points',
                'priority': 'Priority'
            }
        )
        
        fig.update_layout(
            height=500,
            width=800,
            xaxis_title="FAIR Principle",
            yaxis_title="Total Potential Points Gain",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )
        
        # Add value labels on bars
        fig.update_traces(texttemplate='%{y:.0f}', textposition='outside')
        
        if output_format == 'html':
            fig.write_html(f"{self.output_dir}/improvement_priority.html")
        else:
            fig.write_image(f"{self.output_dir}/improvement_priority.png")
    
    def create_matplotlib_visualizations(self):
        """Create static visualizations using matplotlib"""
        if self.df_scores is None:
            return
        
        # Set up the figure
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Score Distribution (Histogram)
        ax1 = plt.subplot(3, 3, 1)
        ax1.hist(self.df_scores['total'], bins=15, color='skyblue', edgecolor='black', alpha=0.7)
        ax1.set_title('Total FAIR Score Distribution')
        ax1.set_xlabel('Score')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # 2. Principle Comparison (Boxplot)
        ax2 = plt.subplot(3, 3, 2)
        principles = ['findable', 'accessible', 'interoperable', 'reusable']
        data = [self.df_scores[p] for p in principles]
        
        box = ax2.boxplot(data, patch_artist=True, labels=[p.capitalize() for p in principles])
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
        
        ax2.set_title('FAIR Principles Distribution')
        ax2.set_ylabel('Score')
        ax2.grid(True, alpha=0.3)
        
        # 3. Repository Ranking (Horizontal Bar)
        ax3 = plt.subplot(3, 3, 3)
        df_sorted = self.df_scores.sort_values('total', ascending=True)
        y_pos = range(len(df_sorted))
        
        ax3.barh(y_pos, df_sorted['total'], color='teal', alpha=0.7)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(df_sorted['repository'].apply(lambda x: x.split('/')[-1][:15]))
        ax3.set_title('Repository Ranking')
        ax3.set_xlabel('Total FAIR Score')
        ax3.grid(True, alpha=0.3)
        
        # 4. Metadata Analysis (if available)
        if 'metadata_files_count' in self.df_scores.columns:
            ax4 = plt.subplot(3, 3, 4)
            ax4.scatter(
                self.df_scores['metadata_files_count'],
                self.df_scores['total'],
                c=self.df_scores['total'],
                cmap='viridis',
                s=100,
                alpha=0.7
            )
            
            # Add trendline
            if len(self.df_scores) > 1:
                z = np.polyfit(self.df_scores['metadata_files_count'], self.df_scores['total'], 1)
                p = np.poly1d(z)
                x_range = np.linspace(
                    self.df_scores['metadata_files_count'].min(),
                    self.df_scores['metadata_files_count'].max(),
                    100
                )
                ax4.plot(x_range, p(x_range), "r--", alpha=0.8)
            
            ax4.set_title('Metadata Files vs FAIR Score')
            ax4.set_xlabel('Number of Metadata Files')
            ax4.set_ylabel('FAIR Score')
            ax4.grid(True, alpha=0.3)
            
            # Add colorbar
            sm = plt.cm.ScalarMappable(cmap='viridis', 
                                     norm=plt.Normalize(vmin=self.df_scores['total'].min(), 
                                                       vmax=self.df_scores['total'].max()))
            sm.set_array([])
            plt.colorbar(sm, ax=ax4, label='FAIR Score')
        
        # 5. Improvement Priority (if available)
        if self.df_improvements is not None and not self.df_improvements.empty:
            ax5 = plt.subplot(3, 3, 5)
            priority_counts = self.df_improvements['priority'].value_counts()
            
            colors = {'High': 'red', 'Medium': 'orange', 'Low': 'green'}
            bar_colors = [colors.get(p, 'gray') for p in priority_counts.index]
            
            ax5.bar(priority_counts.index, priority_counts.values, color=bar_colors, alpha=0.7)
            ax5.set_title('Improvement Priority Distribution')
            ax5.set_xlabel('Priority Level')
            ax5.set_ylabel('Count')
            ax5.grid(True, alpha=0.3)
            
            # Add value labels
            for i, v in enumerate(priority_counts.values):
                ax5.text(i, v + 0.5, str(v), ha='center', va='bottom')
        
        # 6. Correlation Heatmap
        ax6 = plt.subplot(3, 3, 6)
        corr_cols = ['total', 'findable', 'accessible', 'interoperable', 'reusable']
        if 'metadata_files_count' in self.df_scores.columns:
            corr_cols.append('metadata_files_count')
        
        corr_matrix = self.df_scores[corr_cols].corr()
        
        im = ax6.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax6.set_xticks(range(len(corr_cols)))
        ax6.set_yticks(range(len(corr_cols)))
        ax6.set_xticklabels([c.capitalize()[:10] for c in corr_cols], rotation=45, ha='right')
        ax6.set_yticklabels([c.capitalize()[:10] for c in corr_cols])
        ax6.set_title('Correlation Matrix')
        
        # Add text annotations
        for i in range(len(corr_cols)):
            for j in range(len(corr_cols)):
                text = ax6.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                              ha="center", va="center", color="black", fontsize=9)
        
        plt.colorbar(im, ax=ax6)
        
        # 7. Radar Chart for Top Repository
        ax7 = plt.subplot(3, 3, 7, projection='polar')
        if len(self.df_scores) > 0:
            top_repo = self.df_scores.loc[self.df_scores['total'].idxmax()]
            
            angles = np.linspace(0, 2 * np.pi, len(principles), endpoint=False).tolist()
            scores = [top_repo[p] for p in principles]
            
            # Close the loop
            angles += angles[:1]
            scores += scores[:1]
            
            ax7.plot(angles, scores, 'o-', linewidth=2, label=f"Top: {top_repo['repository'].split('/')[-1][:15]}")
            ax7.fill(angles, scores, alpha=0.25)
            
            ax7.set_xticks(angles[:-1])
            ax7.set_xticklabels([p.capitalize() for p in principles])
            ax7.set_ylim(0, 100)
            ax7.set_title('Top Repository - FAIR Principles')
            ax7.grid(True)
            ax7.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        # 8. Stacked Bar Chart for Principles
        ax8 = plt.subplot(3, 3, 8)
        
        # Get average scores for each principle
        avg_scores = [self.df_scores[p].mean() for p in principles]
        
        bars = ax8.bar(range(len(principles)), avg_scores, 
                      color=['lightblue', 'lightgreen', 'lightcoral', 'lightyellow'])
        
        ax8.set_xticks(range(len(principles)))
        ax8.set_xticklabels([p.capitalize() for p in principles])
        ax8.set_ylabel('Average Score')
        ax8.set_title('Average Scores by Principle')
        ax8.set_ylim(0, 100)
        ax8.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax8.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}', ha='center', va='bottom')
        
        # Adjust layout
        plt.tight_layout()
        plt.suptitle('FAIR Analysis Visualizations', fontsize=16, y=1.02)
        
        # Save figure
        plt.savefig(f'{self.output_dir}/matplotlib_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Matplotlib visualizations saved to {self.output_dir}/matplotlib_summary.png")
    
    def create_combined_report(self):
        """Create a combined HTML report with all visualizations"""
        if self.df_scores is None:
            return
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>FAIR Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                .header {{ background-color: #2c3e50; color: white; padding: 20px; border-radius: 10px; }}
                .section {{ background-color: white; padding: 20px; margin: 20px 0; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                h1, h2, h3 {{ color: #2c3e50; }}
                .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }}
                .stat-card {{ background-color: #ecf0f1; padding: 15px; border-radius: 5px; text-align: center; }}
                .stat-value {{ font-size: 24px; font-weight: bold; color: #3498db; }}
                .stat-label {{ font-size: 14px; color: #7f8c8d; }}
                .visualization {{ margin: 20px 0; }}
                .repo-list {{ max-height: 300px; overflow-y: auto; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                .priority-high {{ color: red; font-weight: bold; }}
                .priority-medium {{ color: orange; }}
                .priority-low {{ color: green; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📊 FAIR Analysis Report</h1>
                <p>Generated on: {self.report_data.get('timestamp', 'N/A') if self.report_data else 'N/A'}</p>
                <p>Repositories analyzed: {len(self.df_scores)}</p>
            </div>
            
            <div class="section">
                <h2>📈 Key Statistics</h2>
                <div class="stats-grid">
        """
        
        # Add statistics
        stats = self.report_data.get('statistics', {}) if self.report_data else {}
        
        stat_items = [
            ('Average Score', f"{stats.get('average_total', 0):.1f}/100"),
            ('Median Score', f"{stats.get('median_total', 0):.1f}/100"),
            ('Highest Score', f"{stats.get('highest_total', 0):.1f}/100"),
            ('Lowest Score', f"{stats.get('lowest_total', 0):.1f}/100"),
            ('Score Range', f"{stats.get('lowest_total', 0):.1f}-{stats.get('highest_total', 0):.1f}"),
        ]
        
        if 'average_metadata_files' in stats:
            stat_items.append(('Avg Metadata Files', f"{stats.get('average_metadata_files', 0):.1f}"))
        
        for label, value in stat_items:
            html_content += f"""
                    <div class="stat-card">
                        <div class="stat-label">{label}</div>
                        <div class="stat-value">{value}</div>
                    </div>
            """
        
        html_content += """
                </div>
            </div>
            
            <div class="section">
                <h2>🏆 Top Repositories</h2>
                <div class="repo-list">
                    <table>
                        <tr>
                            <th>Rank</th>
                            <th>Repository</th>
                            <th>Total Score</th>
                            <th>Findable</th>
                            <th>Accessible</th>
                            <th>Interoperable</th>
                            <th>Reusable</th>
                        </tr>
        """
        
        # Add top repositories
        top_n = min(5, len(self.df_scores))
        top_repos = self.df_scores.nlargest(top_n, 'total')
        for idx, (_, row) in enumerate(top_repos.iterrows(), 1):
            repo_name = row['repository'].split('/')[-1]
            html_content += f"""
                        <tr>
                            <td>{idx}</td>
                            <td><a href="{row['repository']}" target="_blank">{repo_name}</a></td>
                            <td><strong>{row['total']:.1f}</strong></td>
                            <td>{row['findable']:.1f}</td>
                            <td>{row['accessible']:.1f}</td>
                            <td>{row['interoperable']:.1f}</td>
                            <td>{row['reusable']:.1f}</td>
                        </tr>
            """
        
        html_content += """
                    </table>
                </div>
            </div>
            
            <div class="section">
                <h2>📊 Interactive Dashboard</h2>
                <p>Explore all visualizations in the interactive dashboard:</p>
                <div style="text-align: center; margin: 30px 0;">
                    <a href="fair_dashboard.html" style="background-color: #3498db; color: white; padding: 15px 30px; text-decoration: none; border-radius: 5px; font-size: 18px; display: inline-block;">
                        🚀 Open Interactive Dashboard
                    </a>
                </div>
                <p>The dashboard includes 9 detailed visualizations with explanations:</p>
                <ol>
                    <li><strong>FAIR Score Ranking/Gauge</strong> - Overall compliance score</li>
                    <li><strong>Score Distribution</strong> - Frequency histogram</li>
                    <li><strong>FAIR Principles Radar Chart</strong> - Multi-dimensional view</li>
                    <li><strong>Improvement Priority</strong> - Action items by urgency</li>
                    <li><strong>Metadata Analysis</strong> - Metadata impact on scores</li>
                    <li><strong>Principles Comparison</strong> - Breakdown by FAIR principle</li>
                    <li><strong>Performance Breakdown</strong> - Detailed repository comparison</li>
                    <li><strong>Missing Elements Heatmap</strong> - Improvement opportunities</li>
                    <li><strong>Correlation Matrix</strong> - Relationships between metrics</li>
                </ol>
            </div>
        """
        
        # Add improvement summary if available
        if self.df_improvements is not None and not self.df_improvements.empty:
            html_content += """
            <div class="section">
                <h2>🔧 Improvement Summary</h2>
                <p>Total improvements identified: """ + str(len(self.df_improvements)) + """</p>
                <table>
                    <tr>
                        <th>Priority</th>
                        <th>Count</th>
                        <th>Potential Points</th>
                    </tr>
            """
            
            for priority in ['High', 'Medium', 'Low']:
                priority_df = self.df_improvements[self.df_improvements['priority'] == priority]
                count = len(priority_df)
                points = priority_df['potential_points'].sum()
                
                priority_class = f'priority-{priority.lower()}'
                
                html_content += f"""
                    <tr>
                        <td class="{priority_class}">{priority}</td>
                        <td>{count}</td>
                        <td>{points}</td>
                    </tr>
                """
            
            html_content += """
                </table>
            </div>
            """
        
        # Add recommendations
        html_content += """
            <div class="section">
                <h2>📋 Recommendations</h2>
                <h3>Immediate Actions (High Priority):</h3>
                <ul>
                    <li>Add missing README files with clear documentation</li>
                    <li>Include proper LICENSE files with open-source licenses</li>
                    <li>Add structured metadata files (JSON/YAML) following standards</li>
                    <li>Include DOI or persistent identifiers for datasets</li>
                </ul>
                
                <h3>Medium-term Improvements:</h3>
                <ul>
                    <li>Use standard metadata schemas (Schema.org, DataCite, Bioschemas)</li>
                    <li>Add usage examples and tutorials</li>
                    <li>Include contact information in metadata</li>
                    <li>Add data schemas and documentation</li>
                </ul>
                
                <h3>Best Practices:</h3>
                <ul>
                    <li>Store metadata in dedicated directories (e.g., bioschema/, metadata/)</li>
                    <li>Use controlled vocabularies and ontologies</li>
                    <li>Include provenance information (creation date, version)</li>
                    <li>Add citation files (CITATION.cff) for proper attribution</li>
                </ul>
            </div>
            
            <div class="header">
                <p>Report generated by FAIR Analysis Tool</p>
                <p>For more information about FAIR principles, visit: 
                <a href="https://www.go-fair.org/fair-principles/" style="color: white;">GO FAIR</a></p>
            </div>
        </body>
        </html>
        """
        
        # Write HTML file
        with open(f"{self.output_dir}/index.html", 'w') as f:
            f.write(html_content)
        
        print(f"✓ HTML report saved to {self.output_dir}/index.html")

def main():
    """Main function to run visualizations"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize FAIR analysis results')
    parser.add_argument('report_file', help='Path to JSON report file')
    parser.add_argument('--format', default='html', choices=['html', 'png'],
                       help='Output format (html for interactive, png for static)')
    parser.add_argument('--all', action='store_true',
                       help='Create all visualizations')
    
    args = parser.parse_args()
    
    print("🚀 FAIR Analysis Visualizer")
    print("="*80)
    
    # Create visualizer
    visualizer = FAIRVisualizer(args.report_file)
    
    if args.all:
        # Create all visualizations
        visualizer.create_all_visualizations(args.format)
    else:
        # Create just the main visualizations
        visualizer.create_individual_figures(args.format)
        visualizer.create_combined_report()
        print(f"✓ Main figures and report saved to {visualizer.output_dir}/")
    
    print(f"\n📁 Open {visualizer.output_dir}/index.html for the complete report")
    print(f"📊 Open {visualizer.output_dir}/fair_dashboard.html for the interactive dashboard")

if __name__ == "__main__":
    main()
