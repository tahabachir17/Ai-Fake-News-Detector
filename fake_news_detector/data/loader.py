
from fake_news_detector.data.collector import DataCollector
import logging
from typing import Optional
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataLoader:
    """
    Class to load data using DataCollector.
    """
    def __init__(self, raw_data_path: str = "fake_news_detector/data/raw"):
        self.collector = DataCollector(raw_data_path)

    def load_data(self) -> Optional[pd.DataFrame]:
        """
        Loads data using DataCollector.
        Returns:
            pd.DataFrame: Merged dataframe with normalized columns.
        """
        logging.info("Delegating data collection to DataCollector...")
        return self.collector.collect_data()
