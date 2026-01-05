"""
Model evaluation module for speech emotion recognition.
Generates metrics, confusion matrix, and classification reports.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_recall_fscore_support,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
import tensorflow as tf
from tensorflow import keras

from config import (
    EMOTION_NAMES, NUM_CLASSES,
    METRICS_DIR, PLOTS_DIR, FINAL_MODELS_DIR
)


class ModelEvaluator:
    """
    Evaluator for emotion recognition model.
    Computes metrics and generates visualizations.
    """
    
    def __init__(self, model: keras.Model):
        """
        Initialize evaluator.
        
        Args:
            model: Trained Keras model
        """
        self.model = model
        self.y_true = None
        self.y_pred = None
        self.y_pred_proba = None
        
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on data.
        
        Args:
            X: Input features
            
        Returns:
            Tuple of (predicted_classes, prediction_probabilities)
        """
        self.y_pred_proba = self.model.predict(X, verbose=0)
        self.y_pred = np.argmax(self.y_pred_proba, axis=1)
        return self.y_pred, self.y_pred_proba
    
    def evaluate(self, 
                X: np.ndarray, 
                y: np.ndarray,
                verbose: bool = True) -> Dict:
        """
        Evaluate model on test data.
        
        Args:
            X: Test features
            y: Test labels
            verbose: Whether to print metrics
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.y_true = y
        self.predict(X)
        
        # Overall accuracy
        accuracy = accuracy_score(self.y_true, self.y_pred)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            self.y_true, self.y_pred, average=None
        )
        
        # Macro averages
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            self.y_true, self.y_pred, average='macro'
        )
        
        # Weighted averages
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            self.y_true, self.y_pred, average='weighted'
        )
        
        metrics = {
            'accuracy': float(accuracy),
            'macro_precision': float(macro_precision),
            'macro_recall': float(macro_recall),
            'macro_f1': float(macro_f1),
            'weighted_precision': float(weighted_precision),
            'weighted_recall': float(weighted_recall),
            'weighted_f1': float(weighted_f1),
            'per_class': {}
        }
        
        # Per-class metrics
        for i, emotion in enumerate(EMOTION_NAMES):
            metrics['per_class'][emotion] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1_score': float(f1[i]),
                'support': int(support[i])
            }
        
        if verbose:
            self.print_metrics(metrics)
        
        return metrics
    
    def print_metrics(self, metrics: Dict) -> None:
        """Print evaluation metrics."""
        print("\n" + "=" * 70)
        print("EVALUATION METRICS")
        print("=" * 70)
        print(f"\nOverall Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"\nMacro Averages:")
        print(f"  Precision: {metrics['macro_precision']:.4f}")
        print(f"  Recall:    {metrics['macro_recall']:.4f}")
        print(f"  F1-Score:  {metrics['macro_f1']:.4f}")
        print(f"\nWeighted Averages:")
        print(f"  Precision: {metrics['weighted_precision']:.4f}")
        print(f"  Recall:    {metrics['weighted_recall']:.4f}")
        print(f"  F1-Score:  {metrics['weighted_f1']:.4f}")
        print(f"\nPer-Class Metrics:")
        print(f"{'Emotion':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print("-" * 70)
        for emotion, values in metrics['per_class'].items():
            print(f"{emotion:<10} {values['precision']:<12.4f} {values['recall']:<12.4f} "
                  f"{values['f1_score']:<12.4f} {values['support']:<10}")
        print("=" * 70)
    
    def plot_confusion_matrix(self,
                             normalize: bool = True,
                             save_path: Optional[Path] = None,
                             figsize: Tuple[int, int] = (10, 8)) -> None:
        """
        Plot confusion matrix.
        
        Args:
            normalize: Whether to normalize values
            save_path: Path to save figure
            figsize: Figure size
        """
        if self.y_true is None or self.y_pred is None:
            raise ValueError("Must call evaluate() first")
        
        cm = confusion_matrix(self.y_true, self.y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=EMOTION_NAMES,
                   yticklabels=EMOTION_NAMES,
                   cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curves(self,
                       save_path: Optional[Path] = None,
                       figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot ROC curves for multi-class classification.
        
        Args:
            save_path: Path to save figure
            figsize: Figure size
        """
        if self.y_true is None or self.y_pred_proba is None:
            raise ValueError("Must call evaluate() first")
        
        # Binarize labels
        y_true_bin = label_binarize(self.y_true, classes=range(NUM_CLASSES))
        
        # Compute ROC curve and AUC for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(NUM_CLASSES):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], self.y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Plot
        plt.figure(figsize=figsize)
        colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, NUM_CLASSES))
        
        for i, color in zip(range(NUM_CLASSES), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'{EMOTION_NAMES[i]} (AUC = {roc_auc[i]:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.50)')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Multi-Class', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curves saved to {save_path}")
        
        plt.show()
    
    def generate_classification_report(self,
                                      save_path: Optional[Path] = None) -> str:
        """
        Generate classification report.
        
        Args:
            save_path: Path to save report
            
        Returns:
            Classification report string
        """
        if self.y_true is None or self.y_pred is None:
            raise ValueError("Must call evaluate() first")
        
        report = classification_report(
            self.y_true, self.y_pred,
            target_names=EMOTION_NAMES,
            digits=4
        )
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write("CLASSIFICATION REPORT\n")
                f.write("=" * 70 + "\n\n")
                f.write(report)
            print(f"Classification report saved to {save_path}")
        
        return report
    
    def save_metrics(self, metrics: Dict, save_path: Path) -> None:
        """
        Save metrics to JSON file.
        
        Args:
            metrics: Metrics dictionary
            save_path: Path to save JSON
        """
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {save_path}")
    
    def full_evaluation(self,
                       X_test: np.ndarray,
                       y_test: np.ndarray,
                       save_results: bool = True) -> Dict:
        """
        Perform full evaluation with all metrics and plots.
        
        Args:
            X_test: Test features
            y_test: Test labels
            save_results: Whether to save results
            
        Returns:
            Dictionary with all metrics
        """
        print("\n" + "=" * 70)
        print("FULL MODEL EVALUATION")
        print("=" * 70)
        
        # Evaluate
        metrics = self.evaluate(X_test, y_test, verbose=True)
        
        if save_results:
            # Save metrics
            metrics_path = METRICS_DIR / 'test_metrics.json'
            self.save_metrics(metrics, metrics_path)
            
            # Save classification report
            report_path = METRICS_DIR / 'classification_report.txt'
            self.generate_classification_report(report_path)
            
            # Plot and save confusion matrix
            cm_path = PLOTS_DIR / 'confusion_matrix.png'
            self.plot_confusion_matrix(normalize=True, save_path=cm_path)
            
            # Plot and save ROC curves
            roc_path = PLOTS_DIR / 'roc_curves.png'
            self.plot_roc_curves(save_path=roc_path)
        
        return metrics


def evaluate_model(model_path: Path,
                  X_test: np.ndarray,
                  y_test: np.ndarray) -> Dict:
    """
    Convenience function to evaluate a saved model.
    
    Args:
        model_path: Path to saved model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Evaluation metrics
    """
    print(f"Loading model from {model_path}...")
    model = keras.models.load_model(model_path)
    
    evaluator = ModelEvaluator(model)
    metrics = evaluator.full_evaluation(X_test, y_test, save_results=True)
    
    return metrics


if __name__ == "__main__":
    """Test evaluation module."""
    print("Testing Model Evaluation Module...\n")
    
    print("Note: To evaluate a model, you need test data.")
    print("Please run training first: python src/train.py")
    print("\nOr load test data manually and call evaluate_model()")
    
    # Find latest model
    model_files = list(FINAL_MODELS_DIR.glob('*.keras'))
    if not model_files:
        print("\n❌ No trained model found. Please train a model first.")
        print("Run: python src/train.py")
    else:
        latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
        print(f"\n✓ Found model: {latest_model.name}")
        print(f"Model path: {latest_model}")
        print("\nTo evaluate, run training with evaluation enabled.")

