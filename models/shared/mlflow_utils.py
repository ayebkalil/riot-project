"""
MLflow Utilities - Centralized logging and experiment management
"""

import mlflow
import mlflow.sklearn
import mlflow.xgboost
from typing import Dict, Any, Optional
import json


class MLflowTracker:
    """Centralized MLflow logging utility"""
    
    def __init__(self, experiment_name: str, mlflow_uri: str = "http://localhost:5000"):
        """
        Initialize MLflow tracker
        
        Args:
            experiment_name: Name of the experiment (e.g., "rank-tier-classification")
            mlflow_uri: MLflow tracking server URI (defaults to local if server not running)
        """
        self.experiment_name = experiment_name
        self.mlflow_available = True
        
        try:
            mlflow.set_tracking_uri(mlflow_uri)
            self._create_experiment_if_not_exists()
        except Exception as e:
            print(f"⚠ MLflow server not available at {mlflow_uri}, using local tracking")
            print(f"  (Start with: mlflow ui --port 5000)")
            # Fall back to local tracking with SQLite
            mlflow.set_tracking_uri("sqlite:///mlruns.db")
            self.mlflow_available = False
            try:
                self._create_experiment_if_not_exists()
            except Exception as e2:
                print(f"⚠ Could not initialize MLflow: {e2}")
                self.mlflow_available = False
    
    def _create_experiment_if_not_exists(self):
        """Create experiment if it doesn't exist"""
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                mlflow.create_experiment(self.experiment_name)
                print(f"✓ Created MLflow experiment: {self.experiment_name}")
            else:
                print(f"✓ Using existing MLflow experiment: {self.experiment_name}")
        except Exception as e:
            print(f"Warning: Could not create experiment - {e}")
    
    def start_run(self, run_name: str):
        """
        Start a new MLflow run
        
        Args:
            run_name: Name of the run (e.g., "RandomForest-v1")
        """
        try:
            mlflow.set_experiment(self.experiment_name)
            mlflow.start_run(run_name=run_name)
            print(f"✓ Started MLflow run: {run_name}")
        except Exception as e:
            if self.mlflow_available:
                print(f"⚠ Could not start MLflow run: {e}")
            # Continue without MLflow - just log to stdout
    
    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters"""
        try:
            mlflow.log_params(params)
            print(f"✓ Logged {len(params)} hyperparameters")
        except Exception as e:
            print(f"⚠ Could not log params to MLflow: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics"""
        try:
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value, step=step)
            print(f"✓ Logged {len(metrics)} metrics")
        except Exception as e:
            print(f"⚠ Could not log metrics to MLflow: {e}")
    
    def log_model(self, model, model_name: str, flavor: str = "sklearn"):
        """
        Log trained model
        
        Args:
            model: Trained model object
            model_name: Name for the model
            flavor: "sklearn", "xgboost", "keras", etc.
        """
        try:
            # Skip MLflow model logging (can hang on Windows)
            # Models are saved locally instead
            print(f"✓ Skipping MLflow model logging (saved locally)")
            return
            
        except Exception as e:
            print(f"⚠ Could not log model to MLflow: {e}")
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log artifact (file or directory)"""
        try:
            mlflow.log_artifact(local_path, artifact_path)
            print(f"✓ Logged artifact: {local_path}")
        except Exception as e:
            print(f"⚠ Could not log artifact to MLflow: {e}")
    
    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None):
        """Log multiple artifacts (directory)"""
        try:
            mlflow.log_artifacts(local_dir, artifact_path)
            print(f"✓ Logged artifacts from: {local_dir}")
        except Exception as e:
            print(f"⚠ Could not log artifacts to MLflow: {e}")
    
    def log_dict(self, dictionary: Dict, file_name: str = "metadata.json"):
        """Log dictionary as JSON artifact"""
        try:
            import tempfile
            import os
            
            with tempfile.TemporaryDirectory() as tmpdir:
                file_path = os.path.join(tmpdir, file_name)
                with open(file_path, "w") as f:
                    json.dump(dictionary, f, indent=2)
                mlflow.log_artifact(file_path)
            print(f"✓ Logged dictionary as: {file_name}")
        except Exception as e:
            print(f"⚠ Could not log dict to MLflow: {e}")
    
    def end_run(self):
        """End the current run"""
        try:
            mlflow.end_run()
            print("✓ Ended MLflow run")
        except Exception as e:
            print(f"⚠ Could not end MLflow run: {e}")
    
    @staticmethod
    def get_best_run(experiment_name: str, metric: str = "accuracy", mode: str = "max") -> Dict[str, Any]:
        """
        Get best run from an experiment
        
        Args:
            experiment_name: Experiment name
            metric: Metric to optimize
            mode: "max" or "min"
        
        Returns:
            Best run info dict
        """
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                return {}
            
            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
            
            if len(runs) == 0:
                return {}
            
            metric_col = f"metrics.{metric}"
            if metric_col not in runs.columns:
                return {}
            
            if mode == "max":
                best_run = runs.loc[runs[metric_col].idxmax()]
            else:
                best_run = runs.loc[runs[metric_col].idxmin()]
            
            return best_run.to_dict()
        
        except Exception as e:
            print(f"Error getting best run: {e}")
            return {}
