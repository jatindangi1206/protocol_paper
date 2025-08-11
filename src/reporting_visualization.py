"""
GOQII Health Data Exploratory Data Analysis Pipeline
Phase 4: Reporting & Visualization Module

This module handles individual and cohort-level visualizations,
and creates comprehensive reports.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

logger = logging.getLogger(__name__)


class ReportingVisualization:
    """Handles reporting and visualization for the EDA pipeline."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.plots_dir = os.path.join(output_dir, 'plots')
        self.summary_dir = os.path.join(output_dir, 'summary')
        
        # Create output directories
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(os.path.join(self.plots_dir, 'individual'), exist_ok=True)
        os.makedirs(os.path.join(self.plots_dir, 'cohort'), exist_ok=True)
        
    def create_individual_hr_plot(self, df: pd.DataFrame, participant_id: str):
        """Create heart rate time series plot for individual participant."""
        
        participant_data = df[df['participant_id'] == participant_id].copy()
        
        if participant_data.empty:
            return
        
        participant_data = participant_data.sort_values('datetime_utc')
        
        plt.figure(figsize=(12, 6))
        plt.plot(participant_data['datetime_utc'], participant_data['value_1'], 
                linewidth=1, alpha=0.7, color='red')
        
        # Add quality flag indicators
        flagged_data = participant_data[participant_data['quality_flag'] != 'normal']
        if not flagged_data.empty:
            plt.scatter(flagged_data['datetime_utc'], flagged_data['value_1'], 
                       color='orange', s=20, alpha=0.8, label='Quality Issues')
        
        plt.title(f'Heart Rate Time Series - Participant {participant_id}')
        plt.xlabel('Date')
        plt.ylabel('Heart Rate (bpm)')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        if not flagged_data.empty:
            plt.legend()
        
        # Add statistical info
        mean_hr = participant_data['value_1'].mean()
        std_hr = participant_data['value_1'].std()
        plt.axhline(y=mean_hr, color='blue', linestyle='--', alpha=0.5, 
                   label=f'Mean: {mean_hr:.1f} ± {std_hr:.1f}')
        plt.legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.plots_dir, 'individual', f'hr_participant_{participant_id}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_individual_temp_plot(self, df: pd.DataFrame, participant_id: str):
        """Create temperature time series plot for individual participant."""
        
        participant_data = df[df['participant_id'] == participant_id].copy()
        
        if participant_data.empty:
            return
        
        participant_data = participant_data.sort_values('datetime_utc')
        
        plt.figure(figsize=(12, 6))
        plt.plot(participant_data['datetime_utc'], participant_data['value_1'], 
                linewidth=1, alpha=0.7, color='green')
        
        # Add fever threshold line
        plt.axhline(y=100.4, color='red', linestyle='--', alpha=0.7, 
                   label='Fever Threshold (100.4°F)')
        
        # Add quality flag indicators
        flagged_data = participant_data[participant_data['quality_flag'] != 'normal']
        if not flagged_data.empty:
            plt.scatter(flagged_data['datetime_utc'], flagged_data['value_1'], 
                       color='orange', s=20, alpha=0.8, label='Quality Issues')
        
        plt.title(f'Temperature Time Series - Participant {participant_id}')
        plt.xlabel('Date')
        plt.ylabel('Temperature (°F)')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.plots_dir, 'individual', f'temp_participant_{participant_id}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_individual_spo2_plot(self, df: pd.DataFrame, participant_id: str):
        """Create SpO2 time series plot for individual participant."""
        
        participant_data = df[df['participant_id'] == participant_id].copy()
        
        if participant_data.empty:
            return
        
        participant_data = participant_data.sort_values('datetime_utc')
        
        plt.figure(figsize=(12, 6))
        plt.plot(participant_data['datetime_utc'], participant_data['value_1'], 
                linewidth=1, alpha=0.7, color='blue')
        
        # Add normal range
        plt.axhline(y=95, color='green', linestyle='--', alpha=0.7, 
                   label='Normal Threshold (95%)')
        
        # Add quality flag indicators
        flagged_data = participant_data[participant_data['quality_flag'] != 'normal']
        if not flagged_data.empty:
            plt.scatter(flagged_data['datetime_utc'], flagged_data['value_1'], 
                       color='orange', s=20, alpha=0.8, label='Quality Issues')
        
        plt.title(f'SpO₂ Time Series - Participant {participant_id}')
        plt.xlabel('Date')
        plt.ylabel('SpO₂ (%)')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.plots_dir, 'individual', f'spo2_participant_{participant_id}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_individual_bp_plot(self, df: pd.DataFrame, participant_id: str):
        """Create blood pressure scatter plot for individual participant."""
        
        participant_data = df[df['participant_id'] == participant_id].copy()
        
        if participant_data.empty:
            return
        
        plt.figure(figsize=(10, 8))
        
        # Create scatter plot of systolic vs diastolic
        scatter = plt.scatter(participant_data['value_2'], participant_data['value_1'], 
                             c=participant_data.index, cmap='viridis', alpha=0.7, s=50)
        
        # Add BP categories
        plt.axhline(y=140, color='red', linestyle='--', alpha=0.5, label='High Systolic (140)')
        plt.axvline(x=90, color='red', linestyle='--', alpha=0.5, label='High Diastolic (90)')
        plt.axhline(y=120, color='orange', linestyle='--', alpha=0.5, label='Elevated Systolic (120)')
        
        plt.xlabel('Diastolic Pressure (mmHg)')
        plt.ylabel('Systolic Pressure (mmHg)')
        plt.title(f'Blood Pressure Pattern - Participant {participant_id}')
        plt.colorbar(scatter, label='Reading Order')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.plots_dir, 'individual', f'bp_participant_{participant_id}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_individual_sleep_plot(self, df: pd.DataFrame, participant_id: str):
        """Create sleep stages pie chart for individual participant."""
        
        participant_data = df[df['participant_id'] == participant_id].copy()
        
        if participant_data.empty:
            return
        
        # Parse sleep data from metadata
        sleep_totals = {'light': 0, 'deep': 0, 'rem': 0}
        
        for metadata in participant_data['metadata'].dropna():
            try:
                for stage_info in metadata.split(','):
                    stage, minutes = stage_info.split(':')
                    sleep_totals[stage] += float(minutes)
            except:
                continue
        
        if sum(sleep_totals.values()) == 0:
            return
        
        # Create pie chart
        plt.figure(figsize=(10, 8))
        
        colors = ['lightblue', 'darkblue', 'purple']
        wedges, texts, autotexts = plt.pie(sleep_totals.values(), 
                                          labels=sleep_totals.keys(), 
                                          autopct='%1.1f%%',
                                          colors=colors,
                                          startangle=90)
        
        plt.title(f'Sleep Stages Distribution - Participant {participant_id}')
        
        # Add total sleep time
        total_sleep = sum(sleep_totals.values())
        plt.figtext(0.5, 0.02, f'Total Sleep Time: {total_sleep:.0f} minutes ({total_sleep/60:.1f} hours)', 
                   ha='center', fontsize=10)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.plots_dir, 'individual', f'sleep_participant_{participant_id}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_individual_steps_plot(self, df: pd.DataFrame, participant_id: str):
        """Create daily steps bar chart for individual participant."""
        
        participant_data = df[df['participant_id'] == participant_id].copy()
        
        if participant_data.empty:
            return
        
        participant_data = participant_data.sort_values('datetime_utc')
        participant_data['date'] = participant_data['datetime_utc'].dt.date
        
        plt.figure(figsize=(12, 6))
        
        # Create bar chart
        bars = plt.bar(range(len(participant_data)), participant_data['value_1'], 
                      color='skyblue', alpha=0.7)
        
        # Color bars based on activity level
        for i, (_, row) in enumerate(participant_data.iterrows()):
            if row['value_1'] > 10000:
                bars[i].set_color('green')
            elif row['value_1'] < 5000:
                bars[i].set_color('orange')
        
        # Add goal line
        plt.axhline(y=10000, color='red', linestyle='--', alpha=0.7, 
                   label='Recommended Daily Steps (10,000)')
        
        plt.xlabel('Days')
        plt.ylabel('Steps')
        plt.title(f'Daily Steps - Participant {participant_id}')
        plt.xticks(range(0, len(participant_data), max(1, len(participant_data)//7)), 
                  [str(d) for d in participant_data['date'][::max(1, len(participant_data)//7)]], 
                  rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.plots_dir, 'individual', f'steps_participant_{participant_id}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_cohort_metric_distribution_plots(self, processed_data: Dict[str, pd.DataFrame]):
        """Create distribution plots for each metric across the cohort."""
        
        for metric_type, df in processed_data.items():
            if df.empty:
                continue
            
            plt.figure(figsize=(15, 10))
            
            # Distribution plot
            plt.subplot(2, 2, 1)
            plt.hist(df['value_1'], bins=30, alpha=0.7, edgecolor='black')
            plt.title(f'{metric_type.upper()} Value Distribution')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            
            # Box plot by participant
            plt.subplot(2, 2, 2)
            participants = df['participant_id'].unique()
            if len(participants) <= 10:  # Only if manageable number
                df.boxplot(column='value_1', by='participant_id', ax=plt.gca())
                plt.title(f'{metric_type.upper()} by Participant')
                plt.suptitle('')  # Remove default title
            else:
                # Overall box plot
                plt.boxplot(df['value_1'])
                plt.title(f'{metric_type.upper()} Overall Distribution')
            plt.ylabel('Value')
            
            # Quality flag distribution
            plt.subplot(2, 2, 3)
            quality_counts = df['quality_flag'].value_counts()
            plt.pie(quality_counts.values, labels=quality_counts.index, autopct='%1.1f%%')
            plt.title(f'{metric_type.upper()} Quality Flags')
            
            # Time series of daily averages
            plt.subplot(2, 2, 4)
            df_daily = df.copy()
            df_daily['date'] = df_daily['datetime_utc'].dt.date
            daily_avg = df_daily.groupby('date')['value_1'].mean()
            
            plt.plot(daily_avg.index, daily_avg.values, marker='o', alpha=0.7)
            plt.title(f'{metric_type.upper()} Daily Average Trend')
            plt.xlabel('Date')
            plt.ylabel('Average Value')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(self.plots_dir, 'cohort', f'{metric_type}_cohort_analysis.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Created cohort analysis plot for {metric_type}")
    
    def create_compliance_summary_plot(self, compliance_df: pd.DataFrame):
        """Create compliance summary visualization."""
        
        if compliance_df.empty:
            return
        
        plt.figure(figsize=(15, 10))
        
        # Compliance by metric type
        plt.subplot(2, 2, 1)
        compliance_by_metric = compliance_df.groupby('metric_type')['compliance_score'].mean()
        bars = plt.bar(compliance_by_metric.index, compliance_by_metric.values)
        plt.title('Average Compliance by Metric')
        plt.ylabel('Compliance Score')
        plt.xticks(rotation=45)
        
        # Color bars based on compliance level
        for i, bar in enumerate(bars):
            if compliance_by_metric.iloc[i] >= 0.8:
                bar.set_color('green')
            elif compliance_by_metric.iloc[i] >= 0.6:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        # Compliance distribution
        plt.subplot(2, 2, 2)
        plt.hist(compliance_df['compliance_score'], bins=20, alpha=0.7, edgecolor='black')
        plt.title('Compliance Score Distribution')
        plt.xlabel('Compliance Score')
        plt.ylabel('Count')
        plt.axvline(x=0.8, color='green', linestyle='--', label='High Compliance (0.8)')
        plt.axvline(x=0.6, color='orange', linestyle='--', label='Medium Compliance (0.6)')
        plt.legend()
        
        # Compliance by trigger type
        plt.subplot(2, 2, 3)
        trigger_compliance = compliance_df.groupby('trigger_type')['compliance_score'].mean()
        plt.bar(trigger_compliance.index, trigger_compliance.values, 
               color=['lightblue', 'lightcoral'])
        plt.title('Compliance by Trigger Type')
        plt.ylabel('Average Compliance Score')
        
        # Participant compliance heatmap
        plt.subplot(2, 2, 4)
        pivot_data = compliance_df.pivot(index='participant_id', 
                                       columns='metric_type', 
                                       values='compliance_score')
        sns.heatmap(pivot_data, annot=True, cmap='RdYlGn', fmt='.2f', cbar_kws={'label': 'Compliance'})
        plt.title('Compliance Heatmap')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.plots_dir, 'cohort', 'compliance_summary.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Created compliance summary plot")
        
    def generate_individual_reports(self, processed_data: Dict[str, pd.DataFrame]):
        """Generate individual plots for all participants."""
        
        # Get all participants
        all_participants = set()
        for df in processed_data.values():
            if not df.empty:
                all_participants.update(df['participant_id'].unique())
        
        logger.info(f"Generating individual reports for {len(all_participants)} participants")
        
        for participant_id in all_participants:
            # Create plots for each metric type
            for metric_type, df in processed_data.items():
                if df.empty:
                    continue
                    
                if metric_type == 'hr':
                    self.create_individual_hr_plot(df, participant_id)
                elif metric_type == 'temp':
                    self.create_individual_temp_plot(df, participant_id)
                elif metric_type == 'spo2':
                    self.create_individual_spo2_plot(df, participant_id)
                elif metric_type == 'bp':
                    self.create_individual_bp_plot(df, participant_id)
                elif metric_type == 'sleep':
                    self.create_individual_sleep_plot(df, participant_id)
                elif metric_type == 'steps':
                    self.create_individual_steps_plot(df, participant_id)
        
        logger.info("Individual reports generated successfully")
        
    def generate_summary_report(self, results: Dict[str, Any]):
        """Generate a comprehensive summary report."""
        
        report_path = os.path.join(self.summary_dir, 'eda_summary_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("GOQII Health Data Exploratory Data Analysis - Summary Report\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Data overview
            f.write("DATA OVERVIEW\n")
            f.write("-" * 20 + "\n")
            
            if 'cohort_statistics' in results:
                cohort_stats = results['cohort_statistics']
                for _, row in cohort_stats.iterrows():
                    f.write(f"\n{row['metric_type'].upper()}:\n")
                    f.write(f"  Participants: {row['total_participants']}\n")
                    f.write(f"  Total Readings: {row['total_readings']}\n")
                    f.write(f"  Mean Value: {row['mean_value']:.2f}\n")
                    f.write(f"  Normal Quality Ratio: {row['normal_quality_ratio']:.2%}\n")
                    f.write(f"  Date Range: {row['date_range_days']} days\n")
            
            # Compliance summary
            f.write("\n\nCOMPLIANCE SUMMARY\n")
            f.write("-" * 20 + "\n")
            
            if 'compliance_distribution' in results:
                compliance_dist = results['compliance_distribution']
                for _, row in compliance_dist.iterrows():
                    f.write(f"\n{row['metric_type'].upper()}:\n")
                    f.write(f"  Mean Compliance: {row['mean_compliance']:.2%}\n")
                    f.write(f"  Participants: {row['participants_count']}\n")
                    if row['top_performers']:
                        f.write(f"  Top Performers: {row['top_performers']}\n")
                    if row['improvement_targets']:
                        f.write(f"  Need Improvement: {row['improvement_targets']}\n")
            
            # Correlation findings
            f.write("\n\nCORRELATION ANALYSIS\n")
            f.write("-" * 20 + "\n")
            
            if 'correlations' in results:
                correlations = results['correlations']
                significant_corr = correlations[correlations['significant_pearson']]
                
                if not significant_corr.empty:
                    f.write("Significant correlations found:\n")
                    for _, row in significant_corr.iterrows():
                        f.write(f"  {row['metric1']} - {row['metric2']}: r={row['pearson_correlation']:.3f} (p={row['pearson_p_value']:.3f})\n")
                else:
                    f.write("No significant correlations found.\n")
            
            # Files generated
            f.write("\n\nOUTPUT FILES\n")
            f.write("-" * 20 + "\n")
            f.write("Cleaned Data: results/cleaned/{metric}.csv\n")
            f.write("Merged Dataset: results/merged/merged_metrics.csv\n")
            f.write("Summary Statistics: results/summary/*.csv\n")
            f.write("Individual Plots: results/plots/individual/\n")
            f.write("Cohort Plots: results/plots/cohort/\n")
        
        logger.info(f"Summary report saved to {report_path}")
        
    def run_reporting_visualization(self, processed_data: Dict[str, pd.DataFrame], 
                                  analysis_results: Dict[str, Any]) -> None:
        """Run the complete reporting and visualization pipeline."""
        
        logger.info("Starting reporting and visualization...")
        
        # Generate individual reports
        self.generate_individual_reports(processed_data)
        
        # Generate cohort visualizations
        self.create_cohort_metric_distribution_plots(processed_data)
        
        # Create compliance summary if available
        if 'compliance' in analysis_results:
            self.create_compliance_summary_plot(analysis_results['compliance'])
        
        # Generate comprehensive summary report
        self.generate_summary_report(analysis_results)
        
        logger.info("Reporting and visualization completed!")


if __name__ == "__main__":
    # Test the reporting module
    import sys
    sys.path.append('.')
    from data_preparation import DataPreparation
    from individual_analysis import IndividualAnalysis
    from cohort_analysis import CohortAnalysis
    
    input_dir = "../data/input"
    output_dir = "../results"
    
    # Run full pipeline
    prep = DataPreparation(input_dir, output_dir)
    processed_data = prep.run_data_preparation()
    
    individual_analysis = IndividualAnalysis(output_dir)
    individual_results = individual_analysis.run_individual_analysis(processed_data)
    
    cohort_analysis = CohortAnalysis(output_dir)
    cohort_results = cohort_analysis.run_cohort_analysis(
        processed_data, individual_results['compliance']
    )
    
    # Combine results
    all_results = {**individual_results, **cohort_results}
    
    # Generate reports
    reporting = ReportingVisualization(output_dir)
    reporting.run_reporting_visualization(processed_data, all_results)
    
    print("Reporting and visualization completed!")
