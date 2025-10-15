#!/usr/bin/env python3
"""Generate HTML reports for training runs."""

import argparse
import sys
from pathlib import Path
from utils.html_report import create_html_report_for_run, create_html_reports_for_runs


def main():
    parser = argparse.ArgumentParser(description='Generate HTML reports for training runs')
    parser.add_argument('--run-folder', type=str, help='Path to specific run folder')
    parser.add_argument('--runs-dir', type=str, default='runs', help='Path to runs directory (default: runs)')
    parser.add_argument('--output', type=str, help='Output path for HTML report (only for single run)')
    parser.add_argument('--title', type=str, default='Training Report', help='Report title')
    parser.add_argument('--no-plots', action='store_true', help='Exclude plots from report')
    parser.add_argument('--no-samples', action='store_true', help='Exclude samples from report')
    parser.add_argument('--no-metrics', action='store_true', help='Exclude metrics from report')
    
    args = parser.parse_args()
    
    try:
        if args.run_folder:
            # Generate report for specific run
            run_folder = Path(args.run_folder)
            if not run_folder.exists():
                print(f"Error: Run folder not found: {run_folder}")
                sys.exit(1)
            
            html_path = create_html_report_for_run(run_folder)
            print(f"HTML report generated: {html_path}")
            
        else:
            # Generate reports for all runs in directory
            runs_dir = Path(args.runs_dir)
            if not runs_dir.exists():
                print(f"Error: Runs directory not found: {runs_dir}")
                sys.exit(1)
            
            html_paths = create_html_reports_for_runs(runs_dir)
            if html_paths:
                print(f"Generated {len(html_paths)} HTML reports:")
                for path in html_paths:
                    print(f"  - {path}")
            else:
                print("No runs found to generate reports for")
                
    except Exception as e:
        print(f"Error generating HTML report: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
