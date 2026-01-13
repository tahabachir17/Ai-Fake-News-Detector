import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import pandas as pd
import numpy as np
from typing import Dict, Union, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelEvaluator:
    """
    Class to evaluate model performance and generate reports.
    """
    def __init__(self):
        pass

    def evaluate(self, y_true: Union[pd.Series, np.ndarray, List], y_pred: Union[pd.Series, np.ndarray, List]) -> Dict[str, float]:
        """
        Calculates accuracy, precision, recall, and F1-score.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            
        Returns:
            Dict[str, float]: Dictionary of metrics.
        """
        logging.info("Calculating metrics...")
        
        # Calculate metrics (using 'weighted' average for multiclass/imbalanced handled gracefully, 
        # though binary is expected. 'binary' requires pos_label adjustment if not 0/1 or -1/1. 
        # using 'weighted' is generally safe for report or auto-detect based on labels?)
        # For simplicity in this fake news (Fake/Real) context, let's assume binary or string labels.
        # 'weighted' handles both binary (if treated as multiclass) and multiclass. 
        # But specifically for Fake/Real, we might want 'pos_label' specific metrics.
        # However, the requirement asks for generic "Precision, Recall". 
        # 'weighted' is a safe default for a general evaluator without knowing pos_label.
        
        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
            "Recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
            "F1-Score": f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        logging.info(f"Metrics calculated: {metrics}")
        return metrics

    def plot_confusion_matrix(self, y_true: Union[pd.Series, np.ndarray, List], y_pred: Union[pd.Series, np.ndarray, List], save_path: str = None) -> None:
        """
        Generates and saves a confusion matrix heatmap.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            save_path (str, optional): Path to save the plot image.
        """
        logging.info("Generating confusion matrix...")
        cm = confusion_matrix(y_true, y_pred)
        
        # Get unique labels from data to ensure correct axis labeling
        labels = sorted(list(set(y_true) | set(y_pred)))
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        if save_path:
            logging.info(f"Saving confusion matrix to {save_path}...")
            plt.savefig(save_path)
            plt.close() # Close plot to free memory
        else:
            plt.show()

    def save_classification_report(self, y_true: Union[pd.Series, np.ndarray, List], y_pred: Union[pd.Series, np.ndarray, List], save_path: str) -> None:
        """
        Generates and saves the classification report.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            save_path (str): Path to save the text report.
        """
        logging.info("Generating classification report...")
        report = classification_report(y_true, y_pred, zero_division=0)
        
        logging.info(f"Saving classification report to {save_path}...")
        with open(save_path, 'w') as f:
            f.write(report)
        
        logging.info("Report saved.")
