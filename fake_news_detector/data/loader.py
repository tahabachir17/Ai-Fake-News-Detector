from fake_news_detector.data.collector import DataCollector
import logging
from typing import Optional, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataLoader:
    """
    Class to manage data loading, splitting, and serialization.
    """
    def __init__(self, raw_data_path: str = "fake_news_detector/data/raw"):
        self.collector = DataCollector(raw_data_path)

    def load_data(self) -> Optional[pd.DataFrame]:
        """
        Convenience method to load raw data.
        Alias for fetch_raw_data().
        
        Returns:
            pd.DataFrame: Merged dataframe with normalized columns.
        """
        return self.fetch_raw_data()

    def fetch_raw_data(self) -> Optional[pd.DataFrame]:
        """
        Fetches fresh data using DataCollector.
        Returns:
            pd.DataFrame: Merged dataframe with normalized columns.
        """
        logging.info("Delegating data collection to DataCollector...")
        return self.collector.collect_data()

    def get_train_test_split(self, df: pd.DataFrame, test_size: float = 0.2, target_column: str = 'label', random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Splits the dataframe into training and testing sets using stratified sampling.
        
        Args:
            df (pd.DataFrame): Input dataframe.
            test_size (float): Proportion of the dataset to include in the test split.
            target_column (str): Name of the target column to stratify by.
            random_state (int): Random state for reproducibility.
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: X_train, X_test, y_train, y_test
        """
        logging.info(f"Splitting data with test_size={test_size} and stratification on '{target_column}'...")
        
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
        
        logging.info(f"Split complete. Train size: {len(X_train)}, Test size: {len(X_test)}")
        return X_train, X_test, y_train, y_test

    def save_data(self, df: pd.DataFrame, path: str) -> None:
        """
        Saves the dataframe to disk. Uses Parquet if supported/extension matches, else Pickle.
        
        Args:
            df (pd.DataFrame): Dataframe to save.
            path (str): Destination path.
        """
        logging.info(f"Saving data to {path}...")
        
        # Create directory if it doesn't exist
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        
        if path.endswith('.parquet'):
            try:
                df.to_parquet(path, index=False)
            except ImportError:
                 logging.warning("pyarrow or fastparquet not installed. Falling back to pickle.")
                 df.to_pickle(path.replace('.parquet', '.pkl'))
        else:
            df.to_pickle(path)
            
        logging.info("Data saved successfully.")

    def load_data(self, path: str) -> pd.DataFrame:
        """
        Loads data from disk.
        
        Args:
            path (str): Path to the file.
            
        Returns:
            pd.DataFrame: Loaded dataframe.
        """
        logging.info(f"Loading data from {path}...")
        
        if not os.path.exists(path):
             raise FileNotFoundError(f"File not found: {path}")

        if path.endswith('.parquet'):
            try:
                df = pd.read_parquet(path)
            except ImportError:
                 # Try pickle if parquet fails (though extension says parquet, likely means library missing)
                 logging.warning("pyarrow/fastparquet missing. Cannot load parquet.")
                 raise
        else:
            df = pd.read_pickle(path)
            
        logging.info(f"Data loaded. Shape: {df.shape}")
        return df
