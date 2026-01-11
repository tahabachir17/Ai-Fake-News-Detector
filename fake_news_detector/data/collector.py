
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataCollector:
    """
    Class to handle data collection.
    Currently focuses on validating the existence of local raw data.
    Future extensions could include scraping or API fetching.
    """
    def __init__(self, raw_data_path: str = "fake_news_detector/data/raw"):
        self.raw_data_path = raw_data_path

    def collect_data(self):
        """
        Validates the existence of True.csv and Fake.csv in the raw data directory.
        Returns:
            bool: True if data exists, False otherwise.
        """
        true_path = os.path.join(self.raw_data_path, 'True.csv')
        fake_path = os.path.join(self.raw_data_path, 'Fake.csv')

        if os.path.exists(true_path) and os.path.exists(fake_path):
            logging.info(f"Data found at {self.raw_data_path}")
            return True
        else:
            logging.error(f"Data not found at {self.raw_data_path}. Please fetch True.csv and Fake.csv.")
            return False
