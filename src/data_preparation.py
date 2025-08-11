"""
GOQII Health Data Exploratory Data Analysis Pipeline
Phase 1: Data Preparation Module

This module handles file discovery, data loading, standardization, and cleaning
for various health metrics (BP, Sleep, Steps, HR, SpO2, Temperature).
"""

import os
import glob
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional, Any
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreparation:
    """Handles data discovery, loading, cleaning, and standardization."""
    
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.cleaned_dir = os.path.join(output_dir, 'cleaned')
        
        # Create output directories
        os.makedirs(self.cleaned_dir, exist_ok=True)
        
        # Metric type patterns for filename detection
        self.metric_patterns = {
            'bp': r'bp_\d+\.csv',
            'sleep': r'sleep_\d+\.csv', 
            'steps': r'steps_\d+\.csv',
            'hr': r'hr_\d+\.json',
            'spo2': r'spo2_\d+\.json',
            'temp': r'temp_\d+\.json',
            'ecg': r'ecg_\d+\.(csv|json)'
        }
        
        # Expected data frequencies (seconds)
        self.expected_frequencies = {
            'bp': 86400,  # Daily
            'sleep': 86400,  # Daily
            'steps': 86400,  # Daily
            'hr': 300,  # 5 minutes
            'spo2': 3600,  # Hourly
            'temp': 3600,  # Hourly
            'ecg': 60  # Minute
        }
        
    def discover_files(self) -> Dict[str, List[str]]:
        """
        Recursively discover all CSV and JSON files in input directory.
        
        Returns:
            Dict mapping metric types to lists of file paths
        """
        discovered_files = {metric: [] for metric in self.metric_patterns.keys()}
        
        # Search for all CSV and JSON files
        all_files = []
        for ext in ['*.csv', '*.json']:
            all_files.extend(glob.glob(os.path.join(self.input_dir, '**', ext), recursive=True))
        
        logger.info(f"Found {len(all_files)} total files")
        
        # Classify files by metric type
        for file_path in all_files:
            filename = os.path.basename(file_path)
            
            for metric_type, pattern in self.metric_patterns.items():
                if re.match(pattern, filename):
                    discovered_files[metric_type].append(file_path)
                    logger.info(f"Classified {filename} as {metric_type}")
                    break
            else:
                logger.warning(f"Could not classify file: {filename}")
        
        return discovered_files
    
    def extract_participant_id(self, file_path: str) -> str:
        """Extract participant ID from filename."""
        filename = os.path.basename(file_path)
        # Extract number after underscore and before extension
        match = re.search(r'_(\d+)', filename)
        return match.group(1) if match else 'unknown'
    
    def load_csv_data(self, file_path: str, metric_type: str) -> pd.DataFrame:
        """Load CSV data and standardize format."""
        try:
            df = pd.read_csv(file_path)
            participant_id = self.extract_participant_id(file_path)
            df['participant_id'] = participant_id
            df['metric_type'] = metric_type
            
            logger.info(f"Loaded CSV {file_path}: {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Error loading CSV {file_path}: {e}")
            return pd.DataFrame()
    
    def load_json_data(self, file_path: str, metric_type: str) -> pd.DataFrame:
        """Load JSON data and convert to DataFrame."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            participant_id = self.extract_participant_id(file_path)
            
            # Handle different JSON structures
            if metric_type == 'hr' and 'jsonData' in data:
                records = data['jsonData']
            elif metric_type == 'spo2' and 'data' in data:
                records = data['data']
            elif metric_type == 'temp' and 'readings' in data:
                records = data['readings']
            else:
                # Assume direct array or try to find data array
                records = data if isinstance(data, list) else data.get('data', [])
            
            df = pd.DataFrame(records)
            df['participant_id'] = participant_id
            df['metric_type'] = metric_type
            
            logger.info(f"Loaded JSON {file_path}: {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Error loading JSON {file_path}: {e}")
            return pd.DataFrame()
    
    def convert_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert Unix timestamps to UTC datetime."""
        if 'timestamp' in df.columns:
            df['timestamp_unix'] = df['timestamp']
            df['datetime_utc'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
        return df
    
    def clean_bp_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean blood pressure data."""
        if df.empty:
            return df
            
        # Drop readings outside normal ranges
        before_count = len(df)
        df = df[
            (df['systolic'].between(80, 200)) & 
            (df['diastolic'].between(50, 130)) &
            (df['pulse'].between(40, 150))
        ].copy()
        
        # Add quality flags
        df['quality_flag'] = 'normal'
        df.loc[df['systolic'] > 140, 'quality_flag'] = 'high_systolic'
        df.loc[df['diastolic'] > 90, 'quality_flag'] = 'high_diastolic'
        
        logger.info(f"BP cleaning: {before_count} -> {len(df)} rows")
        return df
    
    def clean_sleep_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean sleep data."""
        if df.empty:
            return df
            
        required_cols = ['light_minutes', 'deep_minutes', 'rem_minutes', 'total_minutes']
        
        # Drop incomplete rows
        before_count = len(df)
        df = df.dropna(subset=required_cols).copy()
        
        # Validate sleep stages don't exceed total
        sleep_sum = df['light_minutes'] + df['deep_minutes'] + df['rem_minutes']
        df = df[sleep_sum <= df['total_minutes'] * 1.1].copy()  # Allow 10% tolerance
        
        df['quality_flag'] = 'normal'
        df.loc[df['total_minutes'] < 180, 'quality_flag'] = 'short_sleep'
        df.loc[df['total_minutes'] > 600, 'quality_flag'] = 'long_sleep'
        
        logger.info(f"Sleep cleaning: {before_count} -> {len(df)} rows")
        return df
    
    def clean_steps_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean steps data."""
        if df.empty:
            return df
            
        # Remove negative values
        before_count = len(df)
        df = df[
            (df['steps'] >= 0) & 
            (df['distance_km'] >= 0) & 
            (df['calories'] >= 0)
        ].copy()
        
        # Remove unrealistic values
        df = df[df['steps'] <= 50000].copy()  # Max realistic daily steps
        
        df['quality_flag'] = 'normal'
        df.loc[df['steps'] > 20000, 'quality_flag'] = 'high_activity'
        df.loc[df['steps'] < 1000, 'quality_flag'] = 'low_activity'
        
        logger.info(f"Steps cleaning: {before_count} -> {len(df)} rows")
        return df
    
    def clean_hr_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean heart rate data."""
        if df.empty:
            return df
            
        # Remove impossible heart rates
        before_count = len(df)
        df = df[df['value'].between(30, 220)].copy()
        
        df['quality_flag'] = 'normal'
        df.loc[df['value'] < 60, 'quality_flag'] = 'low_hr'
        df.loc[df['value'] > 100, 'quality_flag'] = 'high_hr'
        
        logger.info(f"HR cleaning: {before_count} -> {len(df)} rows")
        return df
    
    def clean_spo2_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean SpO2 data."""
        if df.empty:
            return df
            
        # Keep only realistic SpO2 values
        before_count = len(df)
        df = df[df['value'].between(80, 100)].copy()
        
        df['quality_flag'] = 'normal'
        df.loc[df['value'] < 95, 'quality_flag'] = 'low_oxygen'
        
        logger.info(f"SpO2 cleaning: {before_count} -> {len(df)} rows")
        return df
    
    def clean_temp_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean temperature data."""
        if df.empty:
            return df
            
        # Keep realistic temperature range (Fahrenheit)
        before_count = len(df)
        df = df[df['value'].between(90, 105)].copy()
        
        df['quality_flag'] = 'normal'
        df.loc[df['value'] > 100.4, 'quality_flag'] = 'fever'
        df.loc[df['value'] < 97.0, 'quality_flag'] = 'low_temp'
        
        logger.info(f"Temperature cleaning: {before_count} -> {len(df)} rows")
        return df
    
    def create_unified_schema(self, df: pd.DataFrame, metric_type: str) -> pd.DataFrame:
        """Convert data to unified schema format."""
        if df.empty:
            return pd.DataFrame(columns=[
                'participant_id', 'metric_type', 'timestamp_unix', 'datetime_utc',
                'value_1', 'value_2', 'unit', 'metadata', 'quality_flag'
            ])
        
        unified_df = pd.DataFrame()
        unified_df['participant_id'] = df['participant_id']
        unified_df['metric_type'] = df['metric_type']
        unified_df['timestamp_unix'] = df['timestamp_unix']
        unified_df['datetime_utc'] = df['datetime_utc']
        unified_df['quality_flag'] = df.get('quality_flag', 'normal')
        
        # Map metric-specific columns to unified schema
        if metric_type == 'bp':
            unified_df['value_1'] = df['systolic']
            unified_df['value_2'] = df['diastolic']
            unified_df['unit'] = 'mmHg'
            unified_df['metadata'] = df['pulse'].astype(str)
            
        elif metric_type == 'sleep':
            unified_df['value_1'] = df['total_minutes']
            unified_df['value_2'] = None
            unified_df['unit'] = 'minutes'
            unified_df['metadata'] = df.apply(
                lambda x: f"light:{x['light_minutes']},deep:{x['deep_minutes']},rem:{x['rem_minutes']}", 
                axis=1
            )
            
        elif metric_type == 'steps':
            unified_df['value_1'] = df['steps']
            unified_df['value_2'] = df['distance_km']
            unified_df['unit'] = 'steps'
            unified_df['metadata'] = df['calories'].astype(str)
            
        elif metric_type in ['hr', 'spo2', 'temp']:
            unified_df['value_1'] = df['value']
            unified_df['value_2'] = None
            
            if metric_type == 'hr':
                unified_df['unit'] = 'bpm'
            elif metric_type == 'spo2':
                unified_df['unit'] = '%'
                unified_df['metadata'] = df.get('status', 'unknown')
            elif metric_type == 'temp':
                unified_df['unit'] = '°F'
            
            if 'metadata' not in unified_df.columns:
                unified_df['metadata'] = ''
        
        return unified_df
    
    def process_metric_files(self, files: List[str], metric_type: str) -> pd.DataFrame:
        """Process all files for a specific metric type."""
        all_data = []
        
        for file_path in files:
            # Load data
            if file_path.endswith('.csv'):
                df = self.load_csv_data(file_path, metric_type)
            else:
                df = self.load_json_data(file_path, metric_type)
            
            if df.empty:
                continue
                
            # Convert timestamps
            df = self.convert_timestamps(df)
            
            # Apply metric-specific cleaning
            if metric_type == 'bp':
                df = self.clean_bp_data(df)
            elif metric_type == 'sleep':
                df = self.clean_sleep_data(df)
            elif metric_type == 'steps':
                df = self.clean_steps_data(df)
            elif metric_type == 'hr':
                df = self.clean_hr_data(df)
            elif metric_type == 'spo2':
                df = self.clean_spo2_data(df)
            elif metric_type == 'temp':
                df = self.clean_temp_data(df)
            
            # Convert to unified schema
            unified_df = self.create_unified_schema(df, metric_type)
            all_data.append(unified_df)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df = combined_df.sort_values(['participant_id', 'datetime_utc'])
            return combined_df
        else:
            return pd.DataFrame()
    
    def run_data_preparation(self) -> Dict[str, pd.DataFrame]:
        """Run the complete data preparation pipeline."""
        logger.info("Starting data preparation phase...")
        
        # Discover files
        discovered_files = self.discover_files()
        
        # Process each metric type
        processed_data = {}
        
        for metric_type, files in discovered_files.items():
            if not files:
                logger.info(f"No files found for metric: {metric_type}")
                continue
                
            logger.info(f"Processing {len(files)} files for metric: {metric_type}")
            
            # Process files for this metric
            df = self.process_metric_files(files, metric_type)
            
            if not df.empty:
                # Save cleaned data
                output_path = os.path.join(self.cleaned_dir, f"{metric_type}.csv")
                df.to_csv(output_path, index=False)
                logger.info(f"Saved cleaned {metric_type} data: {len(df)} rows to {output_path}")
                
                processed_data[metric_type] = df
            else:
                logger.warning(f"No valid data for metric: {metric_type}")
        
        logger.info("Data preparation phase completed!")
        return processed_data


if __name__ == "__main__":
    # Test the data preparation module
    input_dir = "../data/input"
    output_dir = "../results"
    
    prep = DataPreparation(input_dir, output_dir)
    processed_data = prep.run_data_preparation()
    
    # Print summary
    for metric_type, df in processed_data.items():
        print(f"{metric_type}: {len(df)} rows, {df['participant_id'].nunique()} participants")
