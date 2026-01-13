import sys
import os
import logging
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from fake_news_detector.models.evaluator import ModelEvaluator

def test_model_evaluator():
    logging.info("Starting ModelEvaluator Verification")
    
    # 1. Create Dummy Predictions
    y_true = ['Fake', 'Fake', 'Real', 'Real', 'Real', 'Fake', 'Real']
    y_pred = ['Fake', 'Real', 'Real', 'Real', 'Fake', 'Fake', 'Real']
    # True: F, F, R, R, R, F, R (3 Fake, 4 Real)
    # Pred: F, R, R, R, F, F, R (3 Fake, 4 Real) but some wrong
    # Correct: 1(F), 3(R), 4(R), 6(F), 7(R) -> 5 correct. 
    # Wrong: 2(F->R), 5(R->F) -> 2 wrong.
    
    evaluator = ModelEvaluator()
    
    # 2. Test evaluate()
    logging.info("\n--- Testing Metrics ---")
    metrics = evaluator.evaluate(y_true, y_pred)
    print("Metrics:", metrics)
    
    # Check Accuracy
    # 5/7 = 0.714
    expected_acc = 5/7
    assert abs(metrics['Accuracy'] - expected_acc) < 0.01, f"Accuracy mismatch. Expected {expected_acc}, got {metrics['Accuracy']}"
    assert all(k in metrics for k in ["Accuracy", "Precision", "Recall", "F1-Score"]), "Missing metrics keys"

    # 3. Test plot_confusion_matrix()
    logging.info("\n--- Testing Confusion Matrix Plot ---")
    cm_path = "temp_confusion_matrix.png"
    if os.path.exists(cm_path):
        os.remove(cm_path)
        
    evaluator.plot_confusion_matrix(y_true, y_pred, save_path=cm_path)
    assert os.path.exists(cm_path), "Confusion matrix image not created"
    
    # 4. Test save_classification_report()
    logging.info("\n--- Testing Classification Report ---")
    report_path = "temp_report.txt"
    if os.path.exists(report_path):
        os.remove(report_path)
        
    evaluator.save_classification_report(y_true, y_pred, save_path=report_path)
    assert os.path.exists(report_path), "Report file not created"
    
    with open(report_path, 'r') as f:
        content = f.read()
        print("Report Content:\n", content)
        assert "precision" in content
        assert "recall" in content
        assert "f1-score" in content
        
    # Clean up
    if os.path.exists(cm_path):
        os.remove(cm_path)
    if os.path.exists(report_path):
        os.remove(report_path)
        
    logging.info("\nVerification Successful!")

if __name__ == "__main__":
    test_model_evaluator()
