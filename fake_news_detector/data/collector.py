
import os
import pandas as pd
import logging
import glob
from typing import List, Optional, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataCollector:
    """
    Production-ready DataCollector for ingesting data from various file formats.
    """
    def __init__(self, data_dir: str):
        """
        Initialize the DataCollector.
        Args:
            data_dir (str): Directory containing the data files.
        """
        self.data_dir = data_dir
        # Standard column mapping to 'text' and 'title'
        # We aim to keep 'title' separate if possible to allow rich features.
        self.column_mapping = {
            'text': 'text',
            'body': 'text',
            'content': 'text',
            'article_content': 'text',
            
            'title': 'title',
            'headline': 'title',
            'header': 'title',
            'article_title': 'title',
            
            # Target
            'label': 'label',
            'target': 'label',
            'class': 'label',
            'type': 'label'
        }

    def collect_data(self) -> pd.DataFrame:
        """
        Collects and merges data from all supported files in the data directory.
        Returns:
            pd.DataFrame: Consolidated dataframe.
        """
        all_files = []
        extensions = ['*.csv', '*.json', '*.xlsx', '*.xls']
        
        for ext in extensions:
            all_files.extend(glob.glob(os.path.join(self.data_dir, ext)))
        
        logging.info(f"Found {len(all_files)} files in {self.data_dir}")
        
        dfs = []
        for file_path in all_files:
            try:
                df = self._load_file(file_path)
                if df is not None:
                    # In this specific project context, we have True.csv and Fake.csv without labels inside.
                    filename = os.path.basename(file_path).lower()
                    
                    # Normalize first to see if we have label
                    df = self._normalize_columns(df)
                    
                    if 'label' not in df.columns:
                        if 'true' in filename:
                            df['label'] = 0
                        elif 'fake' in filename:
                            df['label'] = 1
                    
                    df = self._validate_data(df)
                    
                    if not df.empty:
                        dfs.append(df)
                        logging.info(f"Successfully loaded {file_path} with {len(df)} rows.")
            except Exception as e:
                logging.error(f"Error processing {file_path}: {e}")

        if not dfs:
            logging.warning("No valid data found.")
            return pd.DataFrame()

        consolidated_df = pd.concat(dfs, ignore_index=True)
        logging.info(f"Total consolidated rows: {len(consolidated_df)}")
        return consolidated_df

    def _load_file(self, file_path: str) -> Optional[pd.DataFrame]:
        """Loads a single file based on extension."""
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.csv':
            return pd.read_csv(file_path)
        elif ext == '.json':
            return pd.read_json(file_path)
        elif ext in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
        return None

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Renames columns to standard names and handles duplicates."""
        # 1. Lowercase all columns
        df.columns = [c.lower() for c in df.columns]
        
        # 2. Rename based on mapping
        rename_map = {}
        for col in df.columns:
            if col in self.column_mapping:
                rename_map[col] = self.column_mapping[col]
        
        if rename_map:
            df.rename(columns=rename_map, inplace=True)
            
        # 3. Handle duplicates (e.g. if we had 'text' and 'body' both mapped to 'text')
        # We want to keep the content. Ideally concat them?
        # For simplicity in this baseline:
        # If we have multiple 'text' columns, we keep the first one or concat?
        
        # Check for duplicates
        if df.columns.duplicated().any():
            logging.info("Duplicate columns found after normalization. resolving...")
            # We want to keep 'text' and 'title'.
            # If we have multiple 'text' columns, let's keep the one that was originally 'text' if possible, or just the first one.
            # Or better: drop duplicates.
            df = df.loc[:, ~df.columns.duplicated()]
            
        return df

    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validates and cleans the dataframe."""
        # Ensure we have 'text' column.
        if 'text' not in df.columns:
             # If we only have title, map title to text as fallback?
             if 'title' in df.columns:
                 logging.warning("Missing 'text' column, using 'title' as 'text'.")
                 df['text'] = df['title']
             else:
                 logging.warning("Dropped dataframe due to missing 'text' column.")
                 return pd.DataFrame()

        # Drop empty text
        df = df.dropna(subset=['text'])
        df = df[df['text'].str.strip().astype(bool)]

        # Ensure label exists
        if 'label' not in df.columns:
             logging.warning("Dropped dataframe due to missing 'label' column.")
             return pd.DataFrame()
             
        return df
