"""
Visualization Utilities - Plotting for model comparisons and analysis
"""

import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid threading issues
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional
from sklearn.metrics import confusion_matrix, roc_curve, auc
import pandas as pd


class ModelVisualizations:
    """Visualization utilities for model analysis"""
    
    @staticmethod
    def plot_hyperparameter_comparison(
        runs_data: List[Dict],
        metric_name: str = "accuracy",
        save_path: Optional[str] = None
    ):
        """
        Plot hyperparameter comparison across multiple runs
        
        Args:
            runs_data: List of dicts with 'run_name', 'params', 'metrics'
            metric_name: Metric to plot on y-axis
            save_path: Path to save plot
        """
        plt.figure(figsize=(12, 6))
        
        run_names = [run['run_name'] for run in runs_data]
        metrics = [run['metrics'].get(metric_name, 0) for run in runs_data]
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(run_names)))
        bars = plt.bar(run_names, metrics, color=colors, edgecolor='black', alpha=0.7)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.xlabel('Run Name', fontsize=12, fontweight='bold')
        plt.ylabel(metric_name.capitalize(), fontsize=12, fontweight='bold')
        plt.title(f'Hyperparameter Comparison - {metric_name}', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved plot: {save_path}")
        plt.show()
    
    @staticmethod
    def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None,
        title: str = "Confusion Matrix",
        save_path: Optional[str] = None
    ):
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Class names for labels
            title: Plot title
            save_path: Path to save plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved plot: {save_path}")
        plt.show()
    
    @staticmethod
    def plot_roc_curves(
        y_true: np.ndarray,
        y_pred_proba: Dict[str, np.ndarray],
        title: str = "ROC Curves Comparison",
        save_path: Optional[str] = None
    ):
        """
        Plot ROC curves for multiple models
        
        Args:
            y_true: True labels
            y_pred_proba: Dict of {model_name: probabilities}
            title: Plot title
            save_path: Path to save plot
        """
        plt.figure(figsize=(10, 8))
        
        for model_name, proba in y_pred_proba.items():
            fpr, tpr, _ = roc_curve(y_true, proba)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.4f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved plot: {save_path}")
        plt.show()
    
    @staticmethod
    def plot_feature_importance(
        feature_names: List[str],
        importance_scores: np.ndarray,
        top_n: int = 15,
        title: str = "Feature Importance",
        save_path: Optional[str] = None
    ):
        """
        Plot feature importance
        
        Args:
            feature_names: List of feature names
            importance_scores: Array of importance scores
            top_n: Number of top features to display
            title: Plot title
            save_path: Path to save plot
        """
        # Sort by importance
        indices = np.argsort(importance_scores)[-top_n:]
        top_features = [feature_names[i] for i in indices]
        top_scores = importance_scores[indices]
        
        plt.figure(figsize=(10, 8))
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
        plt.barh(top_features, top_scores, color=colors, edgecolor='black', alpha=0.7)
        
        plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
        plt.title(title, fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved plot: {save_path}")
        plt.show()
    
    @staticmethod
    def plot_learning_curve(
        train_sizes: List[int],
        train_scores: np.ndarray,
        val_scores: np.ndarray,
        title: str = "Learning Curve",
        save_path: Optional[str] = None
    ):
        """
        Plot learning curve (train vs validation scores)
        
        Args:
            train_sizes: Training set sizes
            train_scores: Training scores
            val_scores: Validation scores
            title: Plot title
            save_path: Path to save plot
        """
        plt.figure(figsize=(10, 6))
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.plot(train_sizes, train_mean, 'o-', label='Training Score', linewidth=2, markersize=8)
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
        
        plt.plot(train_sizes, val_mean, 'o-', label='Validation Score', linewidth=2, markersize=8)
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
        
        plt.xlabel('Training Set Size', fontsize=12, fontweight='bold')
        plt.ylabel('Score', fontsize=12, fontweight='bold')
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved plot: {save_path}")
        plt.show()
    
    @staticmethod
    def plot_metrics_comparison(
        models_data: Dict[str, Dict[str, float]],
        metrics_to_plot: List[str],
        save_path: Optional[str] = None
    ):
        """
        Plot metrics comparison across models
        
        Args:
            models_data: Dict of {model_name: {metric_name: value}}
            metrics_to_plot: List of metrics to display
            save_path: Path to save plot
        """
        df = pd.DataFrame(models_data).T
        df = df[metrics_to_plot]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        df.plot(kind='bar', ax=ax, edgecolor='black', alpha=0.7)
        
        plt.ylabel('Score', fontsize=12, fontweight='bold')
        plt.xlabel('Model', fontsize=12, fontweight='bold')
        plt.title('Model Metrics Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved plot: {save_path}")
        plt.show()
