import pandas as pd
import sys
import os
import shutil
import logging
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from fake_news_detector.data.loader import DataLoader

def test_data_loader():
    logging.info("Starting DataLoader Verification")
    
    # 1. Create Dummy Data
    data = {
        'text': [f"News {i}" for i in range(100)],
        'label': (['Fake'] * 20) + (['Real'] * 80) # 20% Fake, 80% Real
    }
    df = pd.DataFrame(data)
    
    loader = DataLoader()
    
    # 2. Test Stratified Split
    logging.info("\n--- Testing Stratified Split ---")
    X_train, X_test, y_train, y_test = loader.get_train_test_split(df, test_size=0.2, target_column='label')
    
    train_counts = y_train.value_counts(normalize=True)
    test_counts = y_test.value_counts(normalize=True)
    
    print("\nTrain Class Distribution:")
    print(train_counts)
    print("\nTest Class Distribution:")
    print(test_counts)
    
    # Assert proportions are roughly equal (stratified)
    # Fake should be ~0.2, Real ~0.8
    assert abs(train_counts['Fake'] - 0.2) < 0.05, "Train 'Fake' ratio mismatch"
    assert abs(test_counts['Fake'] - 0.2) < 0.05, "Test 'Fake' ratio mismatch"
    assert len(X_test) == 20, "Test set size mismatch"
    
    # 3. Test Serialization (Parquet)
    logging.info("\n--- Testing Serialization (Parquet) ---")
    test_path = "temp_test_data.parquet"
    if os.path.exists(test_path):
        os.remove(test_path)
        
    loader.save_data(df, test_path)
    assert os.path.exists(test_path), "Parquet file not created"
    
    loaded_df = loader.load_data(test_path)
    pd.testing.assert_frame_equal(df, loaded_df)
    logging.info("Parquet Save/Load Verified")
    
    # Clean up
    if os.path.exists(test_path):
        os.remove(test_path)
        
    # 4. Test Serialization (Pickle Fallback/Explicit)
    logging.info("\n--- Testing Serialization (Pickle) ---")
    test_pkl_path = "temp_test_data.pkl"
    if os.path.exists(test_pkl_path):
        os.remove(test_pkl_path)
        
    loader.save_data(df, test_pkl_path)
    assert os.path.exists(test_pkl_path), "Pickle file not created"
    
    loaded_pkl_df = loader.load_data(test_pkl_path)
    pd.testing.assert_frame_equal(df, loaded_pkl_df)
    logging.info("Pickle Save/Load Verified")

    if os.path.exists(test_pkl_path):
        os.remove(test_pkl_path)

    logging.info("\nVerification Successful!")

if __name__ == "__main__":
    test_data_loader()
