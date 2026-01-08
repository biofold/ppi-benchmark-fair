#!/usr/bin/env python3
"""
Visualization Script for GitHub Repository FAIRness Analysis
Creates various plots and charts from the analysis results
Updated to match generate_ml_report.py style and behavior
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
import os
from datetime import datetime
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
matplotlib.rcParams['figure.figsize'] = [12, 8]
matplotlib.rcParams['font.size'] = 12

# CSS / page style matching FAIR Interactive Dashboard (same as generate_ml_report.py)
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
    height: 560px;
    border: none;
    border-radius: 8px;
    box-shadow: 0 5px 15px rgba(0,0,0,0.05);
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

class FAIRVisualizer:
    """Visualize FAIR analysis results - Updated to match generate_ml_report.py style"""
    
    def __init__(self, report_file: str = None, output_dir: str = "dashboard_output"):
        """
        Initialize visualizer with report data
        
        Args:
            report_file: Path to JSON report file
            output_dir: Directory to write outputs (figures and dashboard)
        """
        self.report_file = report_file
        self.report_data = None
        self.df_scores = None
        self.df_improvements = None
        self.output_dir = output_dir
        
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
    
    def create_fair_plots(self, verbose: bool = False) -> Dict[str, str]:
        """
        Create FAIR analysis plots and save each plot as an independent HTML file.
        Returns mapping of plot keys to relative HTML file paths.
        """
        if self.df_scores is None or len(self.df_scores) == 0:
            return {}
        
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        figs_dir = Path(self.output_dir) / "figures"
        figs_dir.mkdir(parents=True, exist_ok=True)
        
        plots_map: Dict[str, str] = {}
        
        if verbose:
            print("📊 Creating FAIR analysis visualizations...")
        
        principles = ['findable', 'accessible', 'interoperable', 'reusable']
        
        # 1. FAIR Score Ranking / Single Repository Gauge
        try:
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
                    height=400,
                    width=600,
                    margin=dict(t=20, b=50, l=50, r=50),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(family="Segoe UI, Tahoma, Geneva, Verdana, sans-serif"),
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
                                x=1.15,
                                len=0.8,
                                thickness=15,
                                xpad=20
                            )
                        ),
                        text=[f"{score:.1f}" for score in df_sorted['total']],
                        textposition='auto',
                        name='Total Score'
                    )
                )
                
                fig1.update_layout(
                    xaxis_title="FAIR Score (0-100)",
                    yaxis_title="Repository",
                    height=max(400, len(df_sorted) * 30),
                    width=900,
                    margin=dict(t=20, b=50, l=150, r=200),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(family="Segoe UI, Tahoma, Geneva, Verdana, sans-serif"),
                )
             
            fig1.update_xaxes(
                showline=True,
                linecolor='rgba(0,0,0,0.3)',
                mirror=True,
                showgrid=True,
                gridwidth=0.5,
                gridcolor='rgba(0,0,0,0.15)',
            )

            fig1.update_yaxes(
                showline=True,
                linecolor='rgba(0,0,0,0.3)',
                mirror=True,
            )           
 
            out_path = figs_dir / "score_ranking.html"
            fig1.write_html(str(out_path), include_plotlyjs='cdn')
            plots_map['score_ranking'] = os.path.relpath(out_path, self.output_dir)
            if verbose:
                print("  ✅ score_ranking saved")
        except Exception as e:
            if verbose:
                print(f"  ⚠️  score_ranking error: {e}")
        
        # 2. Score Distribution
        '''
        try:
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
                xaxis_title="FAIR Score",
                yaxis_title="Number of Repositories",
                height=400,
                width=700,
                showlegend=False,
                margin=dict(t=20, b=50, l=50, r=50),
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family="Segoe UI, Tahoma, Geneva, Verdana, sans-serif"),
            )
            fig2.update_xaxes(
                showline=True,
                linecolor='rgba(0,0,0,0.3)',
                mirror=True,
                showgrid=True,
                gridwidth=0.5,
                gridcolor='rgba(0,0,0,0.15)',
            )

            fig2.update_yaxes(
                showline=True,
                linecolor='rgba(0,0,0,0.3)',
                mirror=True,
                showgrid=True,
                gridwidth=0.5,
                gridcolor='rgba(0,0,0,0.15)',
            )            
 
            out_path = figs_dir / "score_distribution.html"
            fig2.write_html(str(out_path), include_plotlyjs='cdn')
            plots_map['score_distribution'] = os.path.relpath(out_path, self.output_dir)
            if verbose:
                print("  ✅ score_distribution saved")
        except Exception as e:
            if verbose:
                print(f"  ⚠️  score_distribution error: {e}")
        '''
        
        # 3. Radar Chart for ALL Repositories
        try:
            fig3 = go.Figure()

            if len(self.df_scores) > 0:
                # Use a color palette that works well for multiple repositories
                if len(self.df_scores) <= 10:
                    # Use qualitative colors for up to 10 repositories
                    colors = px.colors.qualitative.Plotly[:len(self.df_scores)]
                else:
                    # Use sequential colors for many repositories
                    colors = px.colors.sequential.Viridis[:len(self.df_scores)]
                
                for idx, (_, row) in enumerate(self.df_scores.iterrows()):
                    repo_name = row['repository'].split('/')[-1][:20]
                    scores = [row[p] for p in principles]
                    
                    # Get the color for this repository
                    color = colors[idx % len(colors)]
                    
                    # Convert color to rgba for fill with transparency
                    try:
                        if color.startswith('#'):
                            hex_color = color.lstrip('#')
                            r = int(hex_color[0:2], 16)
                            g = int(hex_color[2:4], 16)
                            b = int(hex_color[4:6], 16)
                            rgba_fill = f'rgba({r}, {g}, {b}, 0.3)'
                            rgba_line = f'rgba({r}, {g}, {b}, 0.8)'
                        elif color.startswith('rgb('):
                            rgb_values = color[4:-1].split(',')
                            r = int(rgb_values[0].strip())
                            g = int(rgb_values[1].strip())
                            b = int(rgb_values[2].strip())
                            rgba_fill = f'rgba({r}, {g}, {b}, 0.3)'
                            rgba_line = f'rgba({r}, {g}, {b}, 0.8)'
                        else:
                            rgba_fill = f'rgba({(idx * 50) % 255}, {(idx * 100) % 255}, {(idx * 150) % 255}, 0.3)'
                            rgba_line = f'rgba({(idx * 50) % 255}, {(idx * 100) % 255}, {(idx * 150) % 255}, 0.8)'
                    except:
                        rgba_fill = 'rgba(100, 149, 237, 0.3)'
                        rgba_line = 'rgba(100, 149, 237, 0.8)'
                    
                    fig3.add_trace(
                        go.Scatterpolar(
                            r=scores + [scores[0]],  # Close the loop
                            theta=[p.capitalize() for p in principles] + [principles[0].capitalize()],
                            fill='toself',
                            name=f"{repo_name} ({row['total']:.1f})",
                            line=dict(color=rgba_line, width=2),
                            fillcolor=rgba_fill,
                            opacity=0.7
                        )
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
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                ),
                height=600,
                width=800,
                margin=dict(t=20, b=50, l=50, r=50),
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family="Segoe UI, Tahoma, Geneva, Verdana, sans-serif"),
            )
            
            out_path = figs_dir / "radar_chart.html"
            fig3.write_html(str(out_path), include_plotlyjs='cdn')
            plots_map['radar_chart'] = os.path.relpath(out_path, self.output_dir)
            if verbose:
                print("  ✅ radar_chart saved")
        except Exception as e:
            if verbose:
                print(f"  ⚠️  radar_chart error: {e}")

        '''
        # 4. Improvement Priority Distribution
        try:
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
                    xaxis_title="Priority Level",
                    yaxis_title="Number of Improvements",
                    height=400,
                    width=600,
                    showlegend=False,
                    margin=dict(t=20, b=50, l=50, r=50),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(family="Segoe UI, Tahoma, Geneva, Verdana, sans-serif"),
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
                    height=300,
                    width=500,
                    showlegend=False,
                    margin=dict(t=20, b=50, l=50, r=50),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(family="Segoe UI, Tahoma, Geneva, Verdana, sans-serif"),
                )
           
            fig4.update_xaxes(
                showline=True,
                linecolor='rgba(0,0,0,0.3)',
                mirror=True,
            )

            fig4.update_yaxes(
                showline=True,
                linecolor='rgba(0,0,0,0.3)',
                mirror=True,
                showgrid=True,
                gridwidth=0.5,
                gridcolor='rgba(0,0,0,0.15)',
            ) 
            out_path = figs_dir / "improvement_priority.html"
            fig4.write_html(str(out_path), include_plotlyjs='cdn')
            plots_map['improvement_priority'] = os.path.relpath(out_path, self.output_dir)
            if verbose:
                print("  ✅ improvement_priority saved")
        except Exception as e:
            if verbose:
                print(f"  ⚠️  improvement_priority error: {e}")
        '''

        # 5. Metadata Analysis
        '''
        try:
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
                
                fig5.update_layout(
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
                    margin=dict(t=20, b=50, l=50, r=100),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(family="Segoe UI, Tahoma, Geneva, Verdana, sans-serif"),
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
                    height=300,
                    width=500,
                    showlegend=False,
                    margin=dict(t=20, b=50, l=50, r=50),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(family="Segoe UI, Tahoma, Geneva, Verdana, sans-serif"),
                )
            fig5.update_xaxes(
                showline=True,
                linecolor='rgba(0,0,0,0.3)',
                mirror=True,
                showgrid=True,
                gridwidth=0.5,
                gridcolor='rgba(0,0,0,0.15)',
            )

            fig5.update_yaxes(
                showline=True,
                linecolor='rgba(0,0,0,0.3)',
                mirror=True,
                showgrid=True,
                gridwidth=0.5,
                gridcolor='rgba(0,0,0,0.15)',
            )
 
            out_path = figs_dir / "metadata_analysis.html"
            fig5.write_html(str(out_path), include_plotlyjs='cdn')
            plots_map['metadata_analysis'] = os.path.relpath(out_path, self.output_dir)
            if verbose:
                print("  ✅ metadata_analysis saved")
        except Exception as e:
            if verbose:
                print(f"  ⚠️  metadata_analysis error: {e}")
        '''        

        # 6. Principle Comparison
        try:
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
            
            fig6.update_layout(
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
                margin=dict(t=20, b=50, l=50, r=50),
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family="Segoe UI, Tahoma, Geneva, Verdana, sans-serif"),
            )
            fig6.update_xaxes(
                showline=True,
                linecolor='rgba(0,0,0,0.3)',
                mirror=True,
            )

            fig6.update_yaxes(
                showline=True,
                linecolor='rgba(0,0,0,0.3)',
                mirror=True,
                showgrid=True,
                gridwidth=0.5,
                gridcolor='rgba(0,0,0,0.15)',
            )            
 
            out_path = figs_dir / "principle_comparison.html"
            fig6.write_html(str(out_path), include_plotlyjs='cdn')
            plots_map['principle_comparison'] = os.path.relpath(out_path, self.output_dir)
            if verbose:
                print("  ✅ principle_comparison saved")
        except Exception as e:
            if verbose:
                print(f"  ⚠️  principle_comparison error: {e}")
        
        # 7. Repository Performance Breakdown
        try:
            fig7 = go.Figure()
            
            if len(self.df_scores) == 1:
                # For single repository: Detailed principle breakdown
                repo_scores = self.df_scores.iloc[0]
                scores = [repo_scores[p] for p in principles]
                
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
                    xaxis_title="Principle",
                    yaxis_title="Score",
                    yaxis_range=[0, 100],
                    height=400,
                    width=600,
                    margin=dict(t=20, b=50, l=50, r=50),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(family="Segoe UI, Tahoma, Geneva, Verdana, sans-serif"),
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
                    margin=dict(t=20, b=100, l=50, r=50),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(family="Segoe UI, Tahoma, Geneva, Verdana, sans-serif"),
                )
            fig7.update_xaxes(
                showline=True,
                linecolor='rgba(0,0,0,0.3)',
                mirror=True,
            )

            fig7.update_yaxes(
                showline=True,
                linecolor='rgba(0,0,0,0.3)',
                mirror=True,
                showgrid=True,
                gridwidth=0.5,
                gridcolor='rgba(0,0,0,0.15)',
            )
              
            out_path = figs_dir / "performance_breakdown.html"
            fig7.write_html(str(out_path), include_plotlyjs='cdn')
            plots_map['performance_breakdown'] = os.path.relpath(out_path, self.output_dir)
            if verbose:
                print("  ✅ performance_breakdown saved")
        except Exception as e:
            if verbose:
                print(f"  ⚠️  performance_breakdown error: {e}")
        
        # 8. Missing Elements Heatmap
        try:
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
                        xaxis_title="Missing FAIR Element",
                        yaxis_title="Repository",
                        height=max(500, len(missing_matrix) * 30),
                        width=max(800, len(missing_matrix.columns) * 40),
                        margin=dict(t=20, b=100, l=150, r=100),
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font=dict(family="Segoe UI, Tahoma, Geneva, Verdana, sans-serif"),
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
                        height=300,
                        width=500,
                        margin=dict(t=20, b=50, l=50, r=50),
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font=dict(family="Segoe UI, Tahoma, Geneva, Verdana, sans-serif"),
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
                    height=300,
                    width=500,
                    margin=dict(t=20, b=50, l=50, r=50),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(family="Segoe UI, Tahoma, Geneva, Verdana, sans-serif"),
                )
            fig8.update_xaxes(
                showline=True,
                linecolor='rgba(0,0,0,0.3)',
                mirror=True,
            )

            fig8.update_yaxes(
                showline=True,
                linecolor='rgba(0,0,0,0.3)',
                mirror=True,
            )
 
            out_path = figs_dir / "missing_elements.html"
            fig8.write_html(str(out_path), include_plotlyjs='cdn')
            plots_map['missing_elements'] = os.path.relpath(out_path, self.output_dir)
            if verbose:
                print("  ✅ missing_elements saved")
        except Exception as e:
            if verbose:
                print(f"  ⚠️  missing_elements error: {e}")
        
        # 9. Score Correlation Heatmap
        '''
        try:
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
            
            fig9.update_layout(
                xaxis_title="Metric",
                yaxis_title="Metric",
                height=500,
                width=850,
                margin=dict(t=20, b=50, l=150, r=150),
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family="Segoe UI, Tahoma, Geneva, Verdana, sans-serif"),
            )
            
            out_path = figs_dir / "correlation_matrix.html"
            fig9.write_html(str(out_path), include_plotlyjs='cdn')
            plots_map['correlation_matrix'] = os.path.relpath(out_path, self.output_dir)
            if verbose:
                print("  ✅ correlation_matrix saved")
        except Exception as e:
            if verbose:
                print(f"  ⚠️  correlation_matrix error: {e}")
        '''        
 
        if verbose:
            print(f"✅ Created {len(plots_map)} FAIR HTML figures in {figs_dir}")
          
        return plots_map
    
    def generate_fair_dashboard(self, plots_map: Dict[str, str], output_path: str = "fair_dashboard.html", verbose: bool = False):
        """
        Generate an HTML dashboard page that embeds separate figure HTML files (one per plot).
        Updated styling to match FAIR Interactive Dashboard.
        """
        out_dir = Path(output_path).parent or Path('.')
        rel = lambda p: os.path.relpath(p, start=out_dir)
        
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Extract basic metrics for key metrics section
        stats = self.report_data.get('statistics', {}) if self.report_data else {}
        num_repositories = len(self.df_scores)
        avg_score = stats.get('average_total', 0)
        highest_score = stats.get('highest_total', 0)
        lowest_score = stats.get('lowest_total', 0)
        
        # Build cards in order
        cards = [
            ('Figure 1: FAIR Score Ranking ', plots_map.get('score_ranking'), 
             'Shows overall FAIR compliance score for each repository. For single repositories, a gauge chart indicates the score level with color-coded ranges.'),
            #('Figure 2: Score Distribution', plots_map.get('score_distribution'),
            # 'Histogram displaying the frequency distribution of FAIR scores across all analyzed repositories. Dashed lines show mean (red) and median (green) scores.'),
            ('Figure 2: FAIR Principles Radar Chart', plots_map.get('radar_chart'),
             'Visualizes performance across all four FAIR principles (Findable, Accessible, Interoperable, Reusable) for all repositories. Each axis represents a FAIR principle.'),
            #('Figure 4: Improvement Priority Distribution', plots_map.get('improvement_priority'),
            # 'Bar chart showing the number of improvements needed categorized by priority level (High, Medium, Low). Helps identify urgent action items.'),
            #('Figure 3: Metadata Files vs FAIR Score', plots_map.get('metadata_analysis'),
            # 'Scatter plot examining the relationship between the number of metadata files and overall FAIR score. Includes trendline and correlation coefficient.'),
            ('Figure 3: FAIR Principles Comparison', plots_map.get('principle_comparison'),
             'Bar chart comparing average scores across the four FAIR principles. For multiple repositories, includes error bars showing standard deviation.'),
            ('Figure 4: Repository Performance Breakdown', plots_map.get('performance_breakdown'),
             'Detailed comparison of FAIR principle scores across repositories. For single repositories, shows individual scores; for multiple repositories, grouped bars display all principles.'),
            ('Figure 5: Missing Elements Heatmap', plots_map.get('missing_elements'),
             'Heatmap showing which FAIR elements are missing across repositories and their potential point value. Redder cells indicate more valuable improvements.'),
            #('Figure 9: FAIR Score Correlation Matrix', plots_map.get('correlation_matrix'),
            # 'Heatmap showing correlations between different FAIR metrics. Helps identify relationships between principles.'),
        ]
        
        # HTML header
        html_parts: List[str] = []
        html_parts.append("<!DOCTYPE html>")
        html_parts.append("<html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>")
        html_parts.append(f"<title>FAIR Analysis Report - {current_date}</title>")
        html_parts.append('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">')
        html_parts.append('<script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>')
        html_parts.append(f"<style>{PAGE_STYLE}</style>")
        html_parts.append("</head><body><div class='container'>")
        
        # Header matching FAIR dashboard
        html_parts.append(f"""
        <header>
            <div class="header-content">
                <h1><strong>FAIR Analysis Report</strong></h1>
                <h4><strong>Transforming ELIXIR 3D-Bioinfo PPI dataset into FAIR-compliant, machine learning-ready resource</strong></h4>
                <div>
                    <a href="../index.html" class="dashboard-link">
                        <i class="fas fa-home"></i> PPI Benchmark FAIR
                    </a>
                    <a href="/ppi-benchmark-fair/fair-report/index.html" class="dashboard-link">
                        <i class="fas fa-bullseye"></i> FAIRification 
                    </a>
                    <a href="/ppi-benchmark-fair/ml-report/index.html" class="dashboard-link">
                        <i class="fa-solid fa-brain"></i> Machine Learning
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
        
        # Key Metrics Section
        html_parts.append(f"""
        <div class="section" id="key-metrics">
            <h2><i class="fas fa-clipboard-check"></i> Key Statistics</h2>
            <p>Comprehensive analysis of FAIR principles compliance across GitHub repositories.</p>
            
            <div class="key-metrics">
                <div class="metric-card">
                    <div class="metric-value">{num_repositories}</div>
                    <div class="metric-label">Repositories</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-value">{avg_score:.1f}</div>
                    <div class="metric-label">Average Score</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-value">{highest_score:.1f}</div>
                    <div class="metric-label">Highest Score</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-value">{lowest_score:.1f}</div>
                    <div class="metric-label">Lowest Score</div>
                </div>
            </div>
        </div>
        """)
        
        # Navigation Bar (commented out but available)
        html_parts.append("""
        <!--
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
        for title, rel_path, description in cards:
            if not rel_path:
                continue
            
            embed_src = rel(Path(output_path).parent / rel_path)
            
            html_parts.append(f"""
            <div class="figure-container" id="figure{figure_counter}">
                <h2 class="figure-title">{title}</h2>
                <p class="figure-description">{description}</p>
                <iframe class="figure-embed" src="{embed_src}" loading="lazy" title="{title}"></iframe>
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
                    Evaluation of FAIR principles compliance for scientific data repositories
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
                    <p>FAIR Analysis Report • Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
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
            print(f"✅ FAIR analysis dashboard saved: {output_path}")


def main():
    """Main function to run visualizations"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize FAIR analysis results - Updated to match generate_ml_report.py style')
    parser.add_argument('report_file', help='Path to JSON report file')
    parser.add_argument('--output-dir', type=str, default='fair_dashboard_output', 
                       help='Directory to write outputs (figures and dashboard)')
    parser.add_argument('--output-file', type=str, default=None, 
                       help='Optional output HTML file name for the dashboard')
    parser.add_argument('--verbose', action='store_true', 
                       help='Print verbose progress messages')
    
    args = parser.parse_args()
    
    print("🚀 FAIR Analysis Visualizer")
    print("="*80)
    
    # Create visualizer
    visualizer = FAIRVisualizer(args.report_file, args.output_dir)
    
    # Create plots and dashboard (matching generate_ml_report.py behavior)
    plots_map = visualizer.create_fair_plots(verbose=args.verbose)
    
    # Generate dashboard
    out_file = args.output_file or str(Path(args.output_dir) / "fair_dashboard.html")
    visualizer.generate_fair_dashboard(plots_map, output_path=out_file, verbose=args.verbose)
    
    print(f"✅ Dashboard created at: {out_file}")
    print(f"📁 Figures saved in: {args.output_dir}/figures/")

if __name__ == "__main__":
    main()
