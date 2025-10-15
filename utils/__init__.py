"""Utilities package for the PyTorch training framework."""

from .seed import set_seed
from .device import get_device
from .config import load_config, create_run_folder, save_config
from .logger import MetricLogger, ScalarLogger
from .plotting import (
    plot_loss_curves,
    plot_metric_curves,
    plot_lr_curve,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_pr_curves,
    plot_sample_grid,
    plot_segmentation_overlay
)
from .checkpoint import save_checkpoint, load_checkpoint
from .html_report import (
    generate_html_report,
    save_html_report,
    create_html_report_for_run,
    create_html_reports_for_runs
)

__all__ = [
    'set_seed',
    'get_device', 
    'load_config',
    'create_run_folder',
    'save_config',
    'MetricLogger',
    'ScalarLogger',
    'plot_loss_curves',
    'plot_metric_curves', 
    'plot_lr_curve',
    'plot_confusion_matrix',
    'plot_roc_curves',
    'plot_pr_curves',
    'plot_sample_grid',
    'plot_segmentation_overlay',
    'save_checkpoint',
    'load_checkpoint',
    'generate_html_report',
    'save_html_report',
    'create_html_report_for_run',
    'create_html_reports_for_runs'
]
