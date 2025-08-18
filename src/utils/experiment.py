"""
Experiment Management Utilities

Tools for tracking and managing ML experiments.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Simple experiment tracking utility."""

    def __init__(self, experiment_dir: str = "experiments"):
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(exist_ok=True)
        self.current_experiment = None
        self.start_time = None

    def start_experiment(self, name: str, config: dict[str, Any]) -> str:
        """Start a new experiment."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_id = f"{name}_{timestamp}"

        self.current_experiment = {
            "id": experiment_id,
            "name": name,
            "start_time": datetime.now().isoformat(),
            "config": config,
            "metrics": {},
            "logs": [],
        }

        self.start_time = time.time()

        # Create experiment directory
        exp_dir = self.experiment_dir / experiment_id
        exp_dir.mkdir(exist_ok=True)

        logger.info(f"Started experiment: {experiment_id}")
        return experiment_id

    def log_metric(self, name: str, value: float, step: int | None = None) -> None:
        """Log a metric value."""
        if self.current_experiment is None:
            logger.warning("No active experiment. Call start_experiment() first.")
            return

        if name not in self.current_experiment["metrics"]:
            self.current_experiment["metrics"][name] = []

        metric_entry = {"value": value, "timestamp": datetime.now().isoformat()}

        if step is not None:
            metric_entry["step"] = step

        self.current_experiment["metrics"][name].append(metric_entry)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log multiple metrics at once."""
        for name, value in metrics.items():
            self.log_metric(name, value, step)

    def log_message(self, message: str, level: str = "INFO") -> None:
        """Log a message."""
        if self.current_experiment is None:
            logger.warning("No active experiment. Call start_experiment() first.")
            return

        log_entry = {"message": message, "level": level, "timestamp": datetime.now().isoformat()}

        self.current_experiment["logs"].append(log_entry)
        logger.log(getattr(logging, level.upper(), logging.INFO), message)

    def end_experiment(self, final_metrics: dict[str, float] | None = None) -> None:
        """End the current experiment."""
        if self.current_experiment is None:
            logger.warning("No active experiment to end.")
            return

        # Add final metrics
        if final_metrics:
            self.log_metrics(final_metrics)

        # Calculate duration
        if self.start_time:
            duration = time.time() - self.start_time
            self.current_experiment["duration_seconds"] = duration

        self.current_experiment["end_time"] = datetime.now().isoformat()

        # Save experiment
        self._save_experiment()

        logger.info(f"Ended experiment: {self.current_experiment['id']}")
        self.current_experiment = None
        self.start_time = None

    def _save_experiment(self) -> None:
        """Save experiment data to file."""
        if self.current_experiment is None:
            return

        exp_id = self.current_experiment["id"]
        exp_file = self.experiment_dir / exp_id / "experiment.json"

        with open(exp_file, "w") as f:
            json.dump(self.current_experiment, f, indent=2)

    def list_experiments(self) -> list[dict[str, Any]]:
        """List all experiments."""
        experiments = []

        for exp_dir in self.experiment_dir.iterdir():
            if exp_dir.is_dir():
                exp_file = exp_dir / "experiment.json"
                if exp_file.exists():
                    with open(exp_file) as f:
                        experiments.append(json.load(f))

        return sorted(experiments, key=lambda x: x["start_time"], reverse=True)

    def get_experiment(self, experiment_id: str) -> dict[str, Any] | None:
        """Get experiment by ID."""
        exp_file = self.experiment_dir / experiment_id / "experiment.json"

        if exp_file.exists():
            with open(exp_file) as f:
                return json.load(f)

        return None


class MetricsAggregator:
    """Utility for aggregating metrics across experiments."""

    @staticmethod
    def get_best_experiment(
        experiments: list[dict[str, Any]], metric_name: str, maximize: bool = True
    ) -> dict[str, Any] | None:
        """Find the best experiment based on a metric."""
        valid_experiments = []

        for exp in experiments:
            if metric_name in exp.get("metrics", {}):
                metric_values = exp["metrics"][metric_name]
                if metric_values:
                    # Get the last (most recent) value
                    last_value = metric_values[-1]["value"]
                    valid_experiments.append((exp, last_value))

        if not valid_experiments:
            return None

        # Sort by metric value
        valid_experiments.sort(key=lambda x: x[1], reverse=maximize)
        return valid_experiments[0][0]

    @staticmethod
    def compare_experiments(experiments: list[dict[str, Any]], metric_names: list[str]) -> dict[str, Any]:
        """Compare experiments across multiple metrics."""
        comparison = {"experiments": [], "summary": {}}

        for exp in experiments:
            exp_summary = {"id": exp["id"], "name": exp["name"], "metrics": {}}

            for metric_name in metric_names:
                if metric_name in exp.get("metrics", {}):
                    metric_values = exp["metrics"][metric_name]
                    if metric_values:
                        exp_summary["metrics"][metric_name] = metric_values[-1]["value"]
                else:
                    exp_summary["metrics"][metric_name] = None

            comparison["experiments"].append(exp_summary)

        # Calculate summary statistics
        for metric_name in metric_names:
            values = [
                exp["metrics"][metric_name]
                for exp in comparison["experiments"]
                if exp["metrics"][metric_name] is not None
            ]

            if values:
                comparison["summary"][metric_name] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values),
                }

        return comparison
