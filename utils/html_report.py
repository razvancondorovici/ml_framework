"""HTML report generator for training runs."""

import base64
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import json
import pandas as pd
from datetime import datetime


def generate_html_report(run_folder: Union[str, Path], 
                        title: str = "Training Report",
                        include_plots: bool = True,
                        include_samples: bool = True,
                        include_metrics: bool = True) -> str:
    """Generate HTML report for a training run.
    
    Args:
        run_folder: Path to training run folder
        title: Report title
        include_plots: Whether to include plot images
        include_samples: Whether to include sample visualizations
        include_metrics: Whether to include metrics tables
        
    Returns:
        HTML content as string
    """
    run_folder = Path(run_folder)
    
    # Load run data
    run_data = _load_run_data(run_folder)
    
    # Generate HTML content
    html_content = _generate_html_structure(title, run_data, run_folder, include_plots, include_samples, include_metrics)
    
    return html_content


def save_html_report(run_folder: Union[str, Path], 
                    output_path: Optional[Union[str, Path]] = None,
                    title: str = "Training Report",
                    include_plots: bool = True,
                    include_samples: bool = True,
                    include_metrics: bool = True) -> Path:
    """Save HTML report to file.
    
    Args:
        run_folder: Path to training run folder
        output_path: Path to save HTML report (default: run_folder/index.html)
        title: Report title
        include_plots: Whether to include plot images
        include_samples: Whether to include sample visualizations
        include_metrics: Whether to include metrics tables
        
    Returns:
        Path to saved HTML file
    """
    run_folder = Path(run_folder)
    
    if output_path is None:
        output_path = run_folder / 'index.html'
    else:
        output_path = Path(output_path)
    
    # Generate HTML content
    html_content = generate_html_report(run_folder, title, include_plots, include_samples, include_metrics)
    
    # Save to file
    output_path.write_text(html_content, encoding='utf-8')
    
    return output_path


def _load_run_data(run_folder: Path) -> Dict[str, Any]:
    """Load data from a training run.
    
    Args:
        run_folder: Path to run folder
        
    Returns:
        Dictionary containing run data
    """
    run_data = {}
    
    # Load config
    config_path = run_folder / 'config.yaml'
    if config_path.exists():
        from utils.config import load_config
        run_data['config'] = load_config(config_path)
    
    # Load training history
    history_path = run_folder / 'training_history.json'
    if history_path.exists():
        with open(history_path, 'r') as f:
            run_data['history'] = json.load(f)
    
    # Load metrics
    metrics_path = run_folder / 'metrics.json'
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            run_data['metrics'] = json.load(f)
    
    # Load scalars
    scalars_path = run_folder / 'scalars.csv'
    if scalars_path.exists():
        run_data['scalars'] = pd.read_csv(scalars_path)
    
    # Find plot files
    plot_dir = run_folder / 'plots'
    if plot_dir.exists():
        run_data['plots'] = list(plot_dir.glob('*.png'))
    
    # Find sample files
    sample_dir = run_folder / 'samples'
    if sample_dir.exists():
        run_data['samples'] = list(sample_dir.glob('*.png'))
    
    return run_data


def _generate_html_structure(title: str, 
                           run_data: Dict[str, Any], 
                           run_folder: Path,
                           include_plots: bool,
                           include_samples: bool,
                           include_metrics: bool) -> str:
    """Generate HTML structure.
    
    Args:
        title: Report title
        run_data: Run data dictionary
        run_folder: Run folder path
        include_plots: Whether to include plots
        include_samples: Whether to include samples
        include_metrics: Whether to include metrics
        
    Returns:
        HTML content as string
    """
    html_parts = []
    
    # HTML header
    html_parts.append(_generate_html_header(title))
    
    # Report header
    html_parts.append(_generate_report_header(run_data, run_folder))
    
    # Configuration section
    if 'config' in run_data:
        html_parts.append(_generate_config_section(run_data['config']))
    
    # Metrics section
    if include_metrics and 'metrics' in run_data:
        html_parts.append(_generate_metrics_section(run_data['metrics']))
    
    # Training history section
    if 'history' in run_data:
        html_parts.append(_generate_history_section(run_data['history']))
    
    # Plots section
    if include_plots and 'plots' in run_data:
        html_parts.append(_generate_plots_section(run_data['plots'], run_folder))
    
    # Samples section
    if include_samples and 'samples' in run_data:
        html_parts.append(_generate_samples_section(run_data['samples'], run_folder))
    
    # HTML footer
    html_parts.append(_generate_html_footer())
    
    return '\n'.join(html_parts)


def _generate_html_header(title: str) -> str:
    """Generate HTML header with CSS styles."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            text-align: center;
            margin-bottom: 30px;
            border-bottom: 3px solid #007acc;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #007acc;
            margin-top: 30px;
            margin-bottom: 15px;
            border-left: 4px solid #007acc;
            padding-left: 15px;
        }}
        h3 {{
            color: #555;
            margin-top: 20px;
            margin-bottom: 10px;
        }}
        .section {{
            margin-bottom: 30px;
            padding: 20px;
            background-color: #f9f9f9;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
        }}
        .config-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .config-item {{
            background-color: white;
            padding: 15px;
            border-radius: 5px;
            border: 1px solid #ddd;
        }}
        .config-key {{
            font-weight: bold;
            color: #007acc;
        }}
        .config-value {{
            color: #666;
            margin-left: 10px;
        }}
        .metrics-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        .metrics-table th,
        .metrics-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        .metrics-table th {{
            background-color: #007acc;
            color: white;
        }}
        .metrics-table tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .plot-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-top: 15px;
        }}
        .plot-item {{
            text-align: center;
            background-color: white;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #ddd;
        }}
        .plot-item img {{
            max-width: 100%;
            height: auto;
            border-radius: 5px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .plot-title {{
            font-weight: bold;
            margin-top: 10px;
            color: #333;
        }}
        .sample-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .sample-item {{
            text-align: center;
            background-color: white;
            padding: 10px;
            border-radius: 8px;
            border: 1px solid #ddd;
        }}
        .sample-item img {{
            max-width: 100%;
            height: auto;
            border-radius: 5px;
        }}
        .timestamp {{
            color: #666;
            font-size: 0.9em;
            text-align: center;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
        }}
        .no-data {{
            text-align: center;
            color: #666;
            font-style: italic;
            padding: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">"""


def _generate_report_header(run_data: Dict[str, Any], run_folder: Path) -> str:
    """Generate report header section."""
    experiment_name = "Unknown"
    if 'config' in run_data and 'experiment' in run_data['config']:
        experiment_name = run_data['config']['experiment'].get('name', 'Unknown')
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return f"""
        <h1>{experiment_name} - Training Report</h1>
        <div class="section">
            <h2>Run Information</h2>
            <p><strong>Experiment:</strong> {experiment_name}</p>
            <p><strong>Run Folder:</strong> {run_folder}</p>
            <p><strong>Generated:</strong> {timestamp}</p>
        </div>"""


def _generate_config_section(config: Dict[str, Any]) -> str:
    """Generate configuration section."""
    html_parts = ['<div class="section">', '<h2>Configuration</h2>', '<div class="config-grid">']
    
    def add_config_items(config_dict, prefix=""):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                add_config_items(value, f"{prefix}{key}.")
            else:
                html_parts.append(f'''
                    <div class="config-item">
                        <span class="config-key">{prefix}{key}:</span>
                        <span class="config-value">{value}</span>
                    </div>
                ''')
    
    add_config_items(config)
    html_parts.extend(['</div>', '</div>'])
    
    return '\n'.join(html_parts)


def _generate_metrics_section(metrics: Dict[str, Any]) -> str:
    """Generate metrics section."""
    html_parts = ['<div class="section">', '<h2>Final Metrics</h2>', '<table class="metrics-table">']
    html_parts.append('<tr><th>Metric</th><th>Value</th></tr>')
    
    for metric, value in metrics.items():
        if isinstance(value, (int, float)):
            html_parts.append(f'<tr><td>{metric}</td><td>{value:.4f}</td></tr>')
        else:
            html_parts.append(f'<tr><td>{metric}</td><td>{value}</td></tr>')
    
    html_parts.extend(['</table>', '</div>'])
    
    return '\n'.join(html_parts)


def _generate_history_section(history: Dict[str, Any]) -> str:
    """Generate training history section."""
    html_parts = ['<div class="section">', '<h2>Training History</h2>']
    
    if 'train_loss' in history and history['train_loss']:
        final_train_loss = history['train_loss'][-1]
        html_parts.append(f'<p><strong>Final Train Loss:</strong> {final_train_loss:.4f}</p>')
    
    if 'val_loss' in history and history['val_loss']:
        final_val_loss = history['val_loss'][-1]
        best_val_loss = min(history['val_loss'])
        html_parts.append(f'<p><strong>Final Val Loss:</strong> {final_val_loss:.4f}</p>')
        html_parts.append(f'<p><strong>Best Val Loss:</strong> {best_val_loss:.4f}</p>')
    
    if 'learning_rates' in history and history['learning_rates']:
        final_lr = history['learning_rates'][-1]
        html_parts.append(f'<p><strong>Final Learning Rate:</strong> {final_lr:.2e}</p>')
    
    html_parts.append('</div>')
    
    return '\n'.join(html_parts)


def _generate_plots_section(plots: List[Path], run_folder: Path) -> str:
    """Generate plots section."""
    html_parts = ['<div class="section">', '<h2>Training Plots</h2>', '<div class="plot-grid">']
    
    for plot_path in sorted(plots):
        # Convert image to base64 for embedding
        try:
            with open(plot_path, 'rb') as f:
                img_data = base64.b64encode(f.read()).decode('utf-8')
            
            plot_name = plot_path.stem.replace('_', ' ').title()
            html_parts.append(f'''
                <div class="plot-item">
                    <img src="data:image/png;base64,{img_data}" alt="{plot_name}">
                    <div class="plot-title">{plot_name}</div>
                </div>
            ''')
        except Exception as e:
            print(f"Error loading plot {plot_path}: {e}")
    
    if not plots:
        html_parts.append('<div class="no-data">No plots available</div>')
    
    html_parts.extend(['</div>', '</div>'])
    
    return '\n'.join(html_parts)


def _generate_samples_section(samples: List[Path], run_folder: Path) -> str:
    """Generate samples section."""
    html_parts = ['<div class="section">', '<h2>Sample Visualizations</h2>', '<div class="sample-grid">']
    
    for sample_path in sorted(samples):
        # Convert image to base64 for embedding
        try:
            with open(sample_path, 'rb') as f:
                img_data = base64.b64encode(f.read()).decode('utf-8')
            
            sample_name = sample_path.stem.replace('_', ' ').title()
            html_parts.append(f'''
                <div class="sample-item">
                    <img src="data:image/png;base64,{img_data}" alt="{sample_name}">
                    <div class="plot-title">{sample_name}</div>
                </div>
            ''')
        except Exception as e:
            print(f"Error loading sample {sample_path}: {e}")
    
    if not samples:
        html_parts.append('<div class="no-data">No sample visualizations available</div>')
    
    html_parts.extend(['</div>', '</div>'])
    
    return '\n'.join(html_parts)


def _generate_html_footer() -> str:
    """Generate HTML footer."""
    return f"""
        <div class="timestamp">
            Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        </div>
    </div>
</body>
</html>"""


def create_html_report_for_run(run_folder: Union[str, Path]) -> Path:
    """Create HTML report for a specific run.
    
    Args:
        run_folder: Path to training run folder
        
    Returns:
        Path to generated HTML report
    """
    run_folder = Path(run_folder)
    
    if not run_folder.exists():
        raise ValueError(f"Run folder not found: {run_folder}")
    
    # Generate and save HTML report
    html_path = save_html_report(run_folder)
    
    print(f"HTML report generated: {html_path}")
    
    return html_path


def create_html_reports_for_runs(runs_dir: Union[str, Path]) -> List[Path]:
    """Create HTML reports for all runs in a directory.
    
    Args:
        runs_dir: Path to directory containing runs
        
    Returns:
        List of paths to generated HTML reports
    """
    runs_dir = Path(runs_dir)
    
    if not runs_dir.exists():
        raise ValueError(f"Runs directory not found: {runs_dir}")
    
    # Find all run folders
    run_folders = []
    for item in runs_dir.iterdir():
        if item.is_dir() and (item / 'config.yaml').exists():
            run_folders.append(item)
    
    if not run_folders:
        print(f"No run folders found in {runs_dir}")
        return []
    
    # Generate HTML reports for all runs
    html_paths = []
    for run_folder in run_folders:
        try:
            html_path = create_html_report_for_run(run_folder)
            html_paths.append(html_path)
        except Exception as e:
            print(f"Error creating report for {run_folder}: {e}")
    
    print(f"Generated {len(html_paths)} HTML reports")
    
    return html_paths
