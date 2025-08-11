"""
GOQII Health Data Exploratory Data Analysis Pipeline
Phase 3: Cohort-Level Aggregation Module

This module handles cohort-level statistics, cross-metric correlations,
and population trends analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from scipy import stats
import os
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class CohortAnalysis:
    """Handles cohort-level aggregation and analysis."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.summary_dir = os.path.join(output_dir, 'summary')
        self.plots_dir = os.path.join(output_dir, 'plots')
        self.merged_dir = os.path.join(output_dir, 'merged')
        
        # Create output directories
        os.makedirs(self.summary_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.merged_dir, exist_ok=True)
    
    def calculate_cohort_statistics(self, processed_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate cohort-level statistics for each metric."""
        
        cohort_stats = []
        
        for metric_type, df in processed_data.items():
            if df.empty:
                continue
            
            # Overall statistics
            total_participants = df['participant_id'].nunique()
            total_readings = len(df)
            
            # Value statistics (for primary value)
            value_stats = df['value_1'].describe()
            
            # Quality statistics
            quality_counts = df['quality_flag'].value_counts()
            normal_ratio = quality_counts.get('normal', 0) / len(df) if len(df) > 0 else 0
            
            # Time range
            date_range = df['datetime_utc'].max() - df['datetime_utc'].min()
            
            stats_record = {
                'metric_type': metric_type,
                'total_participants': total_participants,
                'total_readings': total_readings,
                'mean_value': value_stats['mean'],
                'median_value': value_stats['50%'],
                'std_value': value_stats['std'],
                'q25_value': value_stats['25%'],
                'q75_value': value_stats['75%'],
                'min_value': value_stats['min'],
                'max_value': value_stats['max'],
                'normal_quality_ratio': normal_ratio,
                'date_range_days': date_range.days,
                'readings_per_participant': total_readings / total_participants if total_participants > 0 else 0
            }
            
            # Metric-specific statistics
            if metric_type == 'bp':
                # Additional stats for systolic/diastolic
                stats_record['mean_diastolic'] = df['value_2'].mean()
                stats_record['median_diastolic'] = df['value_2'].median()
                
            elif metric_type == 'sleep':
                # Parse sleep stage data from metadata
                sleep_data = []
                for metadata in df['metadata'].dropna():
                    try:
                        stages = {}
                        for stage_info in metadata.split(','):
                            stage, minutes = stage_info.split(':')
                            stages[stage] = float(minutes)
                        sleep_data.append(stages)
                    except:
                        continue
                
                if sleep_data:
                    sleep_df = pd.DataFrame(sleep_data)
                    stats_record['mean_light_sleep'] = sleep_df.get('light', pd.Series()).mean()
                    stats_record['mean_deep_sleep'] = sleep_df.get('deep', pd.Series()).mean()
                    stats_record['mean_rem_sleep'] = sleep_df.get('rem', pd.Series()).mean()
            
            cohort_stats.append(stats_record)
        
        cohort_df = pd.DataFrame(cohort_stats)
        
        # Save cohort statistics
        output_path = os.path.join(self.summary_dir, 'cohort_statistics.csv')
        cohort_df.to_csv(output_path, index=False)
        logger.info(f"Saved cohort statistics to {output_path}")
        
        return cohort_df
    
    def calculate_compliance_distribution(self, compliance_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate compliance distribution and quartiles."""
        
        if compliance_df.empty:
            return pd.DataFrame()
        
        compliance_summary = []
        
        for metric_type in compliance_df['metric_type'].unique():
            metric_compliance = compliance_df[compliance_df['metric_type'] == metric_type]['compliance_score']
            
            # Calculate quartiles and statistics
            summary = {
                'metric_type': metric_type,
                'mean_compliance': metric_compliance.mean(),
                'median_compliance': metric_compliance.median(),
                'std_compliance': metric_compliance.std(),
                'q25_compliance': metric_compliance.quantile(0.25),
                'q75_compliance': metric_compliance.quantile(0.75),
                'min_compliance': metric_compliance.min(),
                'max_compliance': metric_compliance.max(),
                'participants_count': len(metric_compliance)
            }
            
            # Identify quartile performers
            q25, q75 = metric_compliance.quantile(0.25), metric_compliance.quantile(0.75)
            
            top_performers = compliance_df[
                (compliance_df['metric_type'] == metric_type) & 
                (compliance_df['compliance_score'] >= q75)
            ]['participant_id'].tolist()
            
            improvement_targets = compliance_df[
                (compliance_df['metric_type'] == metric_type) & 
                (compliance_df['compliance_score'] <= q25)
            ]['participant_id'].tolist()
            
            summary['top_performers'] = ','.join(map(str, top_performers))
            summary['improvement_targets'] = ','.join(map(str, improvement_targets))
            
            compliance_summary.append(summary)
        
        compliance_summary_df = pd.DataFrame(compliance_summary)
        
        # Save compliance distribution
        output_path = os.path.join(self.summary_dir, 'compliance_distribution.csv')
        compliance_summary_df.to_csv(output_path, index=False)
        logger.info(f"Saved compliance distribution to {output_path}")
        
        return compliance_summary_df
    
    def create_aligned_dataset(self, processed_data: Dict[str, pd.DataFrame], 
                             interval_minutes: int = 5) -> pd.DataFrame:
        """Create time-aligned dataset on regular grid for correlation analysis."""
        
        # Find overall time range
        all_times = []
        for df in processed_data.values():
            if not df.empty:
                all_times.extend(df['datetime_utc'].tolist())
        
        if not all_times:
            return pd.DataFrame()
        
        start_time = min(all_times)
        end_time = max(all_times)
        
        # Create time grid
        time_grid = pd.date_range(start=start_time, end=end_time, 
                                freq=f'{interval_minutes}T')
        
        # Create aligned dataset
        aligned_data = []
        
        # Get all participants
        all_participants = set()
        for df in processed_data.values():
            if not df.empty:
                all_participants.update(df['participant_id'].unique())
        
        for participant_id in all_participants:
            for timestamp in time_grid:
                record = {
                    'participant_id': participant_id,
                    'datetime_utc': timestamp,
                    'timestamp_unix': timestamp.timestamp()
                }
                
                # For each metric, find the closest reading within a window
                window_minutes = interval_minutes * 2  # Allow 2x interval tolerance
                
                for metric_type, df in processed_data.items():
                    if df.empty:
                        record[f'{metric_type}_value'] = np.nan
                        continue
                    
                    participant_data = df[df['participant_id'] == participant_id]
                    
                    if participant_data.empty:
                        record[f'{metric_type}_value'] = np.nan
                        continue
                    
                    # Find closest reading within window
                    time_diffs = np.abs((participant_data['datetime_utc'] - timestamp).dt.total_seconds())
                    closest_idx = time_diffs.idxmin()
                    
                    if time_diffs.loc[closest_idx] <= (window_minutes * 60):
                        record[f'{metric_type}_value'] = participant_data.loc[closest_idx, 'value_1']
                    else:
                        record[f'{metric_type}_value'] = np.nan
                
                aligned_data.append(record)
        
        aligned_df = pd.DataFrame(aligned_data)
        
        # Save aligned dataset
        output_path = os.path.join(self.merged_dir, 'merged_metrics.csv')
        aligned_df.to_csv(output_path, index=False)
        logger.info(f"Saved aligned dataset to {output_path}")
        
        return aligned_df
    
    def calculate_cross_metric_correlations(self, aligned_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate cross-metric correlations."""
        
        if aligned_df.empty:
            return pd.DataFrame()
        
        # Get metric columns
        metric_columns = [col for col in aligned_df.columns if col.endswith('_value')]
        
        if len(metric_columns) < 2:
            logger.warning("Not enough metrics for correlation analysis")
            return pd.DataFrame()
        
        correlations = []
        
        # Calculate pairwise correlations
        for i, metric1 in enumerate(metric_columns):
            for j, metric2 in enumerate(metric_columns):
                if i < j:  # Avoid duplicate pairs
                    
                    # Get data for both metrics (only rows with both values)
                    valid_data = aligned_df[[metric1, metric2]].dropna()
                    
                    if len(valid_data) < 10:  # Need at least 10 points
                        continue
                    
                    # Calculate correlations
                    pearson_corr, pearson_p = pearsonr(valid_data[metric1], valid_data[metric2])
                    spearman_corr, spearman_p = spearmanr(valid_data[metric1], valid_data[metric2])
                    
                    correlations.append({
                        'metric1': metric1.replace('_value', ''),
                        'metric2': metric2.replace('_value', ''),
                        'pearson_correlation': pearson_corr,
                        'pearson_p_value': pearson_p,
                        'spearman_correlation': spearman_corr,
                        'spearman_p_value': spearman_p,
                        'sample_size': len(valid_data),
                        'significant_pearson': pearson_p < 0.05,
                        'significant_spearman': spearman_p < 0.05
                    })
        
        correlation_df = pd.DataFrame(correlations)
        
        # Save correlations
        output_path = os.path.join(self.summary_dir, 'cross_metric_correlations.csv')
        correlation_df.to_csv(output_path, index=False)
        logger.info(f"Saved cross-metric correlations to {output_path}")
        
        return correlation_df
    
    def analyze_population_trends(self, processed_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Analyze population trends over time."""
        
        trend_analyses = {}
        
        for metric_type, df in processed_data.items():
            if df.empty:
                continue
            
            # Weekly trends
            df_copy = df.copy()
            df_copy['week'] = df_copy['datetime_utc'].dt.to_period('W')
            df_copy['weekday'] = df_copy['datetime_utc'].dt.day_name()
            df_copy['hour'] = df_copy['datetime_utc'].dt.hour
            
            # Weekly averages
            weekly_trends = df_copy.groupby('week').agg({
                'value_1': ['mean', 'std', 'count'],
                'participant_id': 'nunique'
            }).round(2)
            
            weekly_trends.columns = ['mean_value', 'std_value', 'reading_count', 'participant_count']
            weekly_trends = weekly_trends.reset_index()
            
            # Weekday vs weekend comparison
            df_copy['is_weekend'] = df_copy['datetime_utc'].dt.weekday >= 5
            weekend_comparison = df_copy.groupby('is_weekend')['value_1'].agg(['mean', 'std', 'count'])
            
            # Seasonal trends (if data spans multiple months)
            df_copy['month'] = df_copy['datetime_utc'].dt.month
            monthly_trends = df_copy.groupby('month')['value_1'].agg(['mean', 'std', 'count'])
            
            # Save individual trend analyses
            trend_analyses[metric_type] = {
                'weekly': weekly_trends,
                'weekend_comparison': weekend_comparison,
                'monthly': monthly_trends
            }
            
            # Save weekly trends
            weekly_path = os.path.join(self.summary_dir, f'{metric_type}_weekly_trends.csv')
            weekly_trends.to_csv(weekly_path, index=False)
            
            # Save weekend comparison
            weekend_path = os.path.join(self.summary_dir, f'{metric_type}_weekend_comparison.csv')
            weekend_comparison.to_csv(weekend_path)
            
            # Save monthly trends
            monthly_path = os.path.join(self.summary_dir, f'{metric_type}_monthly_trends.csv')
            monthly_trends.to_csv(monthly_path)
            
            logger.info(f"Saved trend analyses for {metric_type}")
        
        return trend_analyses
    
    def create_correlation_heatmap(self, correlation_df: pd.DataFrame):
        """Create correlation heatmap."""
        
        if correlation_df.empty:
            return
        
        # Create correlation matrix for heatmap
        metrics = list(set(correlation_df['metric1'].tolist() + correlation_df['metric2'].tolist()))
        corr_matrix = np.zeros((len(metrics), len(metrics)))
        
        # Fill correlation matrix
        for _, row in correlation_df.iterrows():
            i = metrics.index(row['metric1'])
            j = metrics.index(row['metric2'])
            corr_value = row['pearson_correlation']
            corr_matrix[i, j] = corr_value
            corr_matrix[j, i] = corr_value
        
        # Set diagonal to 1
        np.fill_diagonal(corr_matrix, 1)
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, 
                   xticklabels=metrics, 
                   yticklabels=metrics,
                   annot=True, 
                   cmap='RdBu_r', 
                   center=0,
                   fmt='.2f')
        plt.title('Cross-Metric Correlations (Pearson)')
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.plots_dir, 'correlation_heatmap.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved correlation heatmap to {plot_path}")
    
    def create_weekly_trends_plot(self, trend_analyses: Dict[str, Dict]):
        """Create weekly trends line plot."""
        
        plt.figure(figsize=(15, 10))
        
        plot_count = len(trend_analyses)
        cols = 2
        rows = (plot_count + 1) // 2
        
        for i, (metric_type, trends) in enumerate(trend_analyses.items(), 1):
            plt.subplot(rows, cols, i)
            
            weekly_data = trends['weekly']
            if not weekly_data.empty:
                plt.plot(range(len(weekly_data)), weekly_data['mean_value'], 
                        marker='o', label=f'{metric_type.upper()} Mean')
                plt.fill_between(range(len(weekly_data)), 
                               weekly_data['mean_value'] - weekly_data['std_value'],
                               weekly_data['mean_value'] + weekly_data['std_value'],
                               alpha=0.3)
                
                plt.title(f'{metric_type.upper()} Weekly Trends')
                plt.xlabel('Week')
                plt.ylabel('Value')
                plt.xticks(range(0, len(weekly_data), max(1, len(weekly_data)//5)))
                plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.plots_dir, 'weekly_trends.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved weekly trends plot to {plot_path}")
    
    def run_cohort_analysis(self, processed_data: Dict[str, pd.DataFrame], 
                          compliance_df: pd.DataFrame) -> Dict[str, Any]:
        """Run the complete cohort-level analysis pipeline."""
        
        logger.info("Starting cohort-level analysis...")
        
        # Calculate cohort statistics
        cohort_stats = self.calculate_cohort_statistics(processed_data)
        
        # Calculate compliance distribution
        compliance_dist = self.calculate_compliance_distribution(compliance_df)
        
        # Create aligned dataset
        aligned_df = self.create_aligned_dataset(processed_data)
        
        # Calculate cross-metric correlations
        correlations = self.calculate_cross_metric_correlations(aligned_df)
        
        # Analyze population trends
        trend_analyses = self.analyze_population_trends(processed_data)
        
        # Create visualizations
        self.create_correlation_heatmap(correlations)
        self.create_weekly_trends_plot(trend_analyses)
        
        logger.info("Cohort-level analysis completed!")
        
        return {
            'cohort_statistics': cohort_stats,
            'compliance_distribution': compliance_dist,
            'aligned_dataset': aligned_df,
            'correlations': correlations,
            'trend_analyses': trend_analyses
        }


if __name__ == "__main__":
    # Test the cohort analysis module
    import sys
    sys.path.append('.')
    from data_preparation import DataPreparation
    from individual_analysis import IndividualAnalysis
    
    input_dir = "../data/input"
    output_dir = "../results"
    
    # Run data preparation
    prep = DataPreparation(input_dir, output_dir)
    processed_data = prep.run_data_preparation()
    
    # Run individual analysis
    individual_analysis = IndividualAnalysis(output_dir)
    individual_results = individual_analysis.run_individual_analysis(processed_data)
    
    # Run cohort analysis
    cohort_analysis = CohortAnalysis(output_dir)
    cohort_results = cohort_analysis.run_cohort_analysis(
        processed_data, individual_results['compliance']
    )
    
    # Print summary
    print("Cohort Analysis Results:")
    print(f"Cohort statistics: {len(cohort_results['cohort_statistics'])} metrics")
    print(f"Correlations found: {len(cohort_results['correlations'])} pairs")
    print(f"Aligned dataset: {len(cohort_results['aligned_dataset'])} records")
