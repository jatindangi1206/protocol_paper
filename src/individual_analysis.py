"""
GOQII Health Data Exploratory Data Analysis Pipeline
Phase 2: Individual-Level Analysis Module

This module handles missing data analysis, compliance analysis, 
noise detection, and trigger frequency analysis for individual participants.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
from typing import Dict, List, Tuple, Optional
import logging

# Optional import for missingno
try:
    import missingno as msno
    HAS_MISSINGNO = True
except ImportError:
    HAS_MISSINGNO = False
    print("Warning: missingno not available. Some missing data visualizations will be skipped.")

logger = logging.getLogger(__name__)


class IndividualAnalysis:
    """Handles individual-level analysis for health metrics."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.summary_dir = os.path.join(output_dir, 'summary')
        self.plots_dir = os.path.join(output_dir, 'plots')
        
        # Create output directories
        os.makedirs(self.summary_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Expected frequencies (seconds) for compliance calculation
        self.expected_frequencies = {
            'bp': 86400,    # Daily
            'sleep': 86400, # Daily
            'steps': 86400, # Daily
            'hr': 300,      # 5 minutes
            'spo2': 3600,   # Hourly
            'temp': 3600,   # Hourly
            'ecg': 60       # Minute
        }
        
        # Metric types for trigger analysis (user-initiated vs automatic)
        self.trigger_metrics = {'bp', 'ecg', 'spo2'}  # User-initiated
        self.automatic_metrics = {'hr', 'steps', 'sleep', 'temp'}  # Automatic
    
    def calculate_missing_data_ratio(self, df: pd.DataFrame, metric_type: str, 
                                   participant_id: str) -> Dict[str, float]:
        """Calculate missing data ratio vs expected frequency."""
        
        if df.empty:
            return {'missing_ratio': 1.0, 'temporal_gaps': 0, 'max_gap_hours': 0}
        
        # Get participant data
        participant_data = df[df['participant_id'] == participant_id].copy()
        
        if participant_data.empty:
            return {'missing_ratio': 1.0, 'temporal_gaps': 0, 'max_gap_hours': 0}
        
        # Calculate expected vs actual readings
        start_time = participant_data['datetime_utc'].min()
        end_time = participant_data['datetime_utc'].max()
        duration_seconds = (end_time - start_time).total_seconds()
        
        expected_frequency = self.expected_frequencies.get(metric_type, 3600)
        expected_readings = max(1, int(duration_seconds / expected_frequency))
        actual_readings = len(participant_data)
        
        missing_ratio = max(0, (expected_readings - actual_readings) / expected_readings)
        
        # Calculate temporal gaps
        participant_data = participant_data.sort_values('datetime_utc')
        time_diffs = participant_data['datetime_utc'].diff()
        expected_interval = timedelta(seconds=expected_frequency)
        
        # Count gaps larger than 2x expected interval
        large_gaps = time_diffs > (expected_interval * 2)
        temporal_gaps = large_gaps.sum()
        
        # Maximum gap in hours
        max_gap_hours = time_diffs.max().total_seconds() / 3600 if not time_diffs.empty else 0
        
        return {
            'missing_ratio': missing_ratio,
            'temporal_gaps': temporal_gaps,
            'max_gap_hours': max_gap_hours,
            'expected_readings': expected_readings,
            'actual_readings': actual_readings
        }
    
    def analyze_missing_data_all_participants(self, processed_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Analyze missing data for all participants and metrics."""
        
        missing_analysis = []
        
        for metric_type, df in processed_data.items():
            if df.empty:
                continue
                
            participants = df['participant_id'].unique()
            
            for participant_id in participants:
                missing_stats = self.calculate_missing_data_ratio(df, metric_type, participant_id)
                
                missing_analysis.append({
                    'participant_id': participant_id,
                    'metric_type': metric_type,
                    **missing_stats
                })
        
        missing_df = pd.DataFrame(missing_analysis)
        
        # Save missing data analysis
        output_path = os.path.join(self.summary_dir, 'missing_data_analysis.csv')
        missing_df.to_csv(output_path, index=False)
        logger.info(f"Saved missing data analysis to {output_path}")
        
        return missing_df
    
    def calculate_compliance_score(self, df: pd.DataFrame, metric_type: str, 
                                 participant_id: str) -> Dict[str, float]:
        """Calculate compliance score for a participant and metric."""
        
        participant_data = df[df['participant_id'] == participant_id].copy()
        
        if participant_data.empty:
            return {'compliance_score': 0.0, 'days_active': 0, 'total_readings': 0}
        
        # Calculate compliance based on metric type
        if metric_type in self.trigger_metrics:
            # For trigger-based metrics, compliance is based on daily usage
            participant_data['date'] = participant_data['datetime_utc'].dt.date
            days_with_readings = participant_data['date'].nunique()
            
            # Get total days in study period
            start_date = participant_data['date'].min()
            end_date = participant_data['date'].max()
            total_days = (end_date - start_date).days + 1
            
            compliance_score = days_with_readings / total_days if total_days > 0 else 0
            
        else:
            # For automatic metrics, compliance is based on expected frequency
            missing_stats = self.calculate_missing_data_ratio(df, metric_type, participant_id)
            compliance_score = 1 - missing_stats['missing_ratio']
        
        return {
            'compliance_score': compliance_score,
            'days_active': days_with_readings if metric_type in self.trigger_metrics else None,
            'total_readings': len(participant_data)
        }
    
    def analyze_compliance_all_participants(self, processed_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Analyze compliance for all participants and metrics."""
        
        compliance_analysis = []
        
        for metric_type, df in processed_data.items():
            if df.empty:
                continue
                
            participants = df['participant_id'].unique()
            
            for participant_id in participants:
                compliance_stats = self.calculate_compliance_score(df, metric_type, participant_id)
                
                compliance_analysis.append({
                    'participant_id': participant_id,
                    'metric_type': metric_type,
                    'trigger_type': 'user_initiated' if metric_type in self.trigger_metrics else 'automatic',
                    **compliance_stats
                })
        
        compliance_df = pd.DataFrame(compliance_analysis)
        
        # Save compliance analysis
        output_path = os.path.join(self.summary_dir, 'compliance_analysis.csv')
        compliance_df.to_csv(output_path, index=False)
        logger.info(f"Saved compliance analysis to {output_path}")
        
        return compliance_df
    
    def detect_noise_and_outliers(self, df: pd.DataFrame, metric_type: str) -> pd.DataFrame:
        """Detect noise and add quality flags for outliers."""
        
        if df.empty:
            return df
        
        df = df.copy()
        
        # Define noise thresholds by metric type
        if metric_type == 'hr':
            # HR variability - flag rapid changes
            for participant in df['participant_id'].unique():
                participant_mask = df['participant_id'] == participant
                participant_data = df[participant_mask].sort_values('datetime_utc')
                
                if len(participant_data) > 1:
                    hr_values = participant_data['value_1'].values
                    hr_diff = np.abs(np.diff(hr_values))
                    
                    # Flag if HR changes >30 bpm between consecutive readings
                    rapid_changes = np.where(hr_diff > 30)[0]
                    for idx in rapid_changes:
                        original_idx = participant_data.iloc[idx + 1].name
                        df.loc[original_idx, 'quality_flag'] = 'hr_variability'
        
        elif metric_type == 'temp':
            # Temperature spikes - flag rapid changes
            for participant in df['participant_id'].unique():
                participant_mask = df['participant_id'] == participant
                participant_data = df[participant_mask].sort_values('datetime_utc')
                
                if len(participant_data) > 1:
                    temp_values = participant_data['value_1'].values
                    temp_diff = np.abs(np.diff(temp_values))
                    
                    # Flag if temperature changes >2°F between consecutive readings
                    rapid_changes = np.where(temp_diff > 2.0)[0]
                    for idx in rapid_changes:
                        original_idx = participant_data.iloc[idx + 1].name
                        df.loc[original_idx, 'quality_flag'] = 'temp_spike'
        
        elif metric_type == 'steps':
            # False step detection - flag unrealistic patterns
            for participant in df['participant_id'].unique():
                participant_mask = df['participant_id'] == participant
                participant_data = df[participant_mask].sort_values('datetime_utc')
                
                # Flag days with >40k steps as potential false readings
                high_steps = participant_data['value_1'] > 40000
                df.loc[participant_data[high_steps].index, 'quality_flag'] = 'false_steps'
        
        # Statistical outlier detection using IQR method for all metrics
        for participant in df['participant_id'].unique():
            participant_mask = df['participant_id'] == participant
            participant_data = df[participant_mask]
            
            if len(participant_data) >= 4:  # Need at least 4 points for IQR
                Q1 = participant_data['value_1'].quantile(0.25)
                Q3 = participant_data['value_1'].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = (participant_data['value_1'] < lower_bound) | \
                          (participant_data['value_1'] > upper_bound)
                
                df.loc[participant_data[outliers].index, 'quality_flag'] = 'statistical_outlier'
        
        return df
    
    def analyze_trigger_frequency(self, df: pd.DataFrame, metric_type: str) -> pd.DataFrame:
        """Analyze trigger frequency for user-initiated metrics."""
        
        if metric_type not in self.trigger_metrics or df.empty:
            return pd.DataFrame()
        
        trigger_analysis = []
        
        for participant in df['participant_id'].unique():
            participant_data = df[df['participant_id'] == participant].copy()
            participant_data['date'] = participant_data['datetime_utc'].dt.date
            participant_data['hour'] = participant_data['datetime_utc'].dt.hour
            
            # Daily trigger counts
            daily_triggers = participant_data.groupby('date').size()
            
            # Peak trigger times
            hourly_triggers = participant_data.groupby('hour').size()
            peak_hour = hourly_triggers.idxmax() if not hourly_triggers.empty else None
            
            # Calculate trigger statistics
            trigger_stats = {
                'participant_id': participant,
                'metric_type': metric_type,
                'total_triggers': len(participant_data),
                'days_with_triggers': len(daily_triggers),
                'avg_daily_triggers': daily_triggers.mean(),
                'max_daily_triggers': daily_triggers.max(),
                'peak_hour': peak_hour,
                'triggers_at_peak': hourly_triggers.max() if not hourly_triggers.empty else 0
            }
            
            trigger_analysis.append(trigger_stats)
        
        trigger_df = pd.DataFrame(trigger_analysis)
        
        # Save trigger analysis
        if not trigger_df.empty:
            output_path = os.path.join(self.summary_dir, f'trigger_analysis_{metric_type}.csv')
            trigger_df.to_csv(output_path, index=False)
            logger.info(f"Saved trigger analysis for {metric_type} to {output_path}")
        
        return trigger_df
    
    def create_missingness_heatmap(self, missing_df: pd.DataFrame):
        """Create missingness heatmap for participants × metrics."""
        
        if missing_df.empty:
            return
        
        # Pivot data for heatmap
        heatmap_data = missing_df.pivot(index='participant_id', 
                                      columns='metric_type', 
                                      values='missing_ratio')
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_data, annot=True, cmap='Reds', fmt='.2f',
                   cbar_kws={'label': 'Missing Data Ratio'})
        plt.title('Missing Data Heatmap by Participant and Metric')
        plt.xlabel('Metric Type')
        plt.ylabel('Participant ID')
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.plots_dir, 'missing_data_heatmap.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved missing data heatmap to {plot_path}")
    
    def create_compliance_heatmap(self, compliance_df: pd.DataFrame):
        """Create compliance heatmap for participants × metrics."""
        
        if compliance_df.empty:
            return
        
        # Pivot data for heatmap
        heatmap_data = compliance_df.pivot(index='participant_id', 
                                         columns='metric_type', 
                                         values='compliance_score')
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_data, annot=True, cmap='Greens', fmt='.2f',
                   cbar_kws={'label': 'Compliance Score'})
        plt.title('Compliance Score Heatmap by Participant and Metric')
        plt.xlabel('Metric Type')
        plt.ylabel('Participant ID')
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.plots_dir, 'compliance_heatmap.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved compliance heatmap to {plot_path}")
    
    def run_individual_analysis(self, processed_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Run the complete individual-level analysis pipeline."""
        
        logger.info("Starting individual-level analysis...")
        
        # Analyze missing data
        missing_df = self.analyze_missing_data_all_participants(processed_data)
        
        # Analyze compliance
        compliance_df = self.analyze_compliance_all_participants(processed_data)
        
        # Detect noise and outliers for each metric
        processed_data_with_flags = {}
        for metric_type, df in processed_data.items():
            df_with_flags = self.detect_noise_and_outliers(df, metric_type)
            processed_data_with_flags[metric_type] = df_with_flags
            
            # Update cleaned data with quality flags
            output_path = os.path.join(os.path.dirname(self.summary_dir), 'cleaned', f"{metric_type}.csv")
            df_with_flags.to_csv(output_path, index=False)
        
        # Analyze trigger frequency for user-initiated metrics
        trigger_analyses = {}
        for metric_type in self.trigger_metrics:
            if metric_type in processed_data:
                trigger_df = self.analyze_trigger_frequency(processed_data[metric_type], metric_type)
                if not trigger_df.empty:
                    trigger_analyses[metric_type] = trigger_df
        
        # Create visualizations
        self.create_missingness_heatmap(missing_df)
        self.create_compliance_heatmap(compliance_df)
        
        logger.info("Individual-level analysis completed!")
        
        return {
            'missing_data': missing_df,
            'compliance': compliance_df,
            'trigger_analyses': trigger_analyses,
            'processed_data_with_flags': processed_data_with_flags
        }


if __name__ == "__main__":
    # Test the individual analysis module
    import sys
    sys.path.append('.')
    from data_preparation import DataPreparation
    
    input_dir = "../data/input"
    output_dir = "../results"
    
    # Run data preparation first
    prep = DataPreparation(input_dir, output_dir)
    processed_data = prep.run_data_preparation()
    
    # Run individual analysis
    individual_analysis = IndividualAnalysis(output_dir)
    analysis_results = individual_analysis.run_individual_analysis(processed_data)
    
    # Print summary
    print("Individual Analysis Results:")
    print(f"Missing data analysis: {len(analysis_results['missing_data'])} records")
    print(f"Compliance analysis: {len(analysis_results['compliance'])} records")
    print(f"Trigger analyses: {len(analysis_results['trigger_analyses'])} metrics")
