
import pandas as pd
import os
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataLoader:
    """
    Class to load and merge data from CSV files.
    """
    def __init__(self, raw_data_path: str = "fake_news_detector/data/raw"):
        self.raw_data_path = raw_data_path

    def load_data(self) -> Optional[pd.DataFrame]:
        """
        Loads True.csv and Fake.csv, adds labels, and merges them.
        Returns:
            pd.DataFrame: Merged dataframe with 'text', 'title', 'subject', 'date', and 'label' columns.
                          Returns None if files are not found.
        """
        true_path = os.path.join(self.raw_data_path, 'True.csv')
        fake_path = os.path.join(self.raw_data_path, 'Fake.csv')

        if not (os.path.exists(true_path) and os.path.exists(fake_path)):
             logging.error(f"Data files not found in {self.raw_data_path}")
             return None

        try:
            logging.info("Loading True.csv...")
            true_df = pd.read_csv(true_path)
            true_df['label'] = 0  # 0 for Real news

            logging.info("Loading Fake.csv...")
            fake_df = pd.read_csv(fake_path)
            fake_df['label'] = 1  # 1 for Fake news

            logging.info("Merging datasets...")
            df = pd.concat([true_df, fake_df], axis=0).reset_index(drop=True)
            
            # Shuffle the data
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            logging.info(f"Data loaded successfully. Shape: {df.shape}")
            return df
        
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            return None
