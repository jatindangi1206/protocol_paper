#!/usr/bin/env python3
"""
GOQII Health Data Exploratory Data Analysis Pipeline
Main Pipeline Script

This script orchestrates the complete EDA pipeline for GOQII health data,
implementing all four phases:
1. Data Preparation
2. Individual-Level Analysis
3. Cohort-Level Aggregation
4. Reporting & Visualization

Usage:
    python eda_pipeline.py --input_dir data/input --output_dir results/
"""

import argparse
import os
import sys
import logging
import time
from datetime import datetime
from typing import Dict, Any

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preparation import DataPreparation
from individual_analysis import IndividualAnalysis
from cohort_analysis import CohortAnalysis
from reporting_visualization import ReportingVisualization

# Configure logging
def setup_logging(output_dir: str, log_level: str = 'INFO'):
    """Setup logging configuration."""
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f'eda_pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


class EDAPipeline:
    """Main EDA Pipeline orchestrator."""
    
    def __init__(self, input_dir: str, output_dir: str, log_level: str = 'INFO'):
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # Setup logging
        self.logger = setup_logging(output_dir, log_level)
        
        # Validate directories
        self._validate_directories()
        
        # Initialize pipeline components
        self.data_prep = DataPreparation(input_dir, output_dir)
        self.individual_analysis = IndividualAnalysis(output_dir)
        self.cohort_analysis = CohortAnalysis(output_dir)
        self.reporting = ReportingVisualization(output_dir)
        
        self.logger.info("EDA Pipeline initialized successfully")
        
    def _validate_directories(self):
        """Validate input and output directories."""
        if not os.path.exists(self.input_dir):
            raise ValueError(f"Input directory does not exist: {self.input_dir}")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.logger.info(f"Input directory: {self.input_dir}")
        self.logger.info(f"Output directory: {self.output_dir}")
        
    def run_phase1_data_preparation(self) -> Dict[str, Any]:
        """Execute Phase 1: Data Preparation."""
        self.logger.info("="*60)
        self.logger.info("PHASE 1: DATA PREPARATION")
        self.logger.info("="*60)
        
        start_time = time.time()
        
        try:
            # Run data preparation
            processed_data = self.data_prep.run_data_preparation()
            
            # Log summary
            total_files = sum(len(df) for df in processed_data.values())
            total_participants = set()
            for df in processed_data.values():
                if not df.empty:
                    total_participants.update(df['participant_id'].unique())
            
            self.logger.info(f"Phase 1 completed in {time.time() - start_time:.2f} seconds")
            self.logger.info(f"Processed {len(processed_data)} metric types")
            self.logger.info(f"Total records: {total_files}")
            self.logger.info(f"Total participants: {len(total_participants)}")
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Phase 1 failed: {e}")
            raise
            
    def run_phase2_individual_analysis(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Phase 2: Individual-Level Analysis."""
        self.logger.info("="*60)
        self.logger.info("PHASE 2: INDIVIDUAL-LEVEL ANALYSIS")
        self.logger.info("="*60)
        
        start_time = time.time()
        
        try:
            # Run individual analysis
            individual_results = self.individual_analysis.run_individual_analysis(processed_data)
            
            # Log summary
            missing_records = len(individual_results.get('missing_data', []))
            compliance_records = len(individual_results.get('compliance', []))
            trigger_analyses = len(individual_results.get('trigger_analyses', {}))
            
            self.logger.info(f"Phase 2 completed in {time.time() - start_time:.2f} seconds")
            self.logger.info(f"Missing data analysis: {missing_records} records")
            self.logger.info(f"Compliance analysis: {compliance_records} records")
            self.logger.info(f"Trigger analyses: {trigger_analyses} metrics")
            
            return individual_results
            
        except Exception as e:
            self.logger.error(f"Phase 2 failed: {e}")
            raise
            
    def run_phase3_cohort_analysis(self, processed_data: Dict[str, Any], 
                                 individual_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Phase 3: Cohort-Level Aggregation."""
        self.logger.info("="*60)
        self.logger.info("PHASE 3: COHORT-LEVEL AGGREGATION")
        self.logger.info("="*60)
        
        start_time = time.time()
        
        try:
            # Run cohort analysis
            compliance_df = individual_results.get('compliance', None)
            cohort_results = self.cohort_analysis.run_cohort_analysis(processed_data, compliance_df)
            
            # Log summary
            cohort_stats = len(cohort_results.get('cohort_statistics', []))
            correlations = len(cohort_results.get('correlations', []))
            aligned_records = len(cohort_results.get('aligned_dataset', []))
            
            self.logger.info(f"Phase 3 completed in {time.time() - start_time:.2f} seconds")
            self.logger.info(f"Cohort statistics: {cohort_stats} metrics")
            self.logger.info(f"Correlations analyzed: {correlations} pairs")
            self.logger.info(f"Aligned dataset: {aligned_records} records")
            
            return cohort_results
            
        except Exception as e:
            self.logger.error(f"Phase 3 failed: {e}")
            raise
            
    def run_phase4_reporting(self, processed_data: Dict[str, Any], 
                           all_results: Dict[str, Any]) -> None:
        """Execute Phase 4: Reporting & Visualization."""
        self.logger.info("="*60)
        self.logger.info("PHASE 4: REPORTING & VISUALIZATION")
        self.logger.info("="*60)
        
        start_time = time.time()
        
        try:
            # Run reporting and visualization
            self.reporting.run_reporting_visualization(processed_data, all_results)
            
            self.logger.info(f"Phase 4 completed in {time.time() - start_time:.2f} seconds")
            self.logger.info("All visualizations and reports generated successfully")
            
        except Exception as e:
            self.logger.error(f"Phase 4 failed: {e}")
            raise
            
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """Execute the complete EDA pipeline."""
        
        self.logger.info("Starting GOQII Health Data EDA Pipeline")
        self.logger.info(f"Pipeline start time: {datetime.now()}")
        
        pipeline_start_time = time.time()
        
        try:
            # Phase 1: Data Preparation
            processed_data = self.run_phase1_data_preparation()
            
            if not processed_data:
                self.logger.warning("No data processed. Pipeline terminated.")
                return {}
            
            # Phase 2: Individual Analysis
            individual_results = self.run_phase2_individual_analysis(processed_data)
            
            # Phase 3: Cohort Analysis
            cohort_results = self.run_phase3_cohort_analysis(processed_data, individual_results)
            
            # Combine all results
            all_results = {
                **individual_results,
                **cohort_results,
                'processed_data': processed_data
            }
            
            # Phase 4: Reporting & Visualization
            self.run_phase4_reporting(processed_data, all_results)
            
            # Final summary
            total_time = time.time() - pipeline_start_time
            self.logger.info("="*60)
            self.logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("="*60)
            self.logger.info(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
            self.logger.info(f"Pipeline end time: {datetime.now()}")
            
            # Print output summary
            self._print_output_summary()
            
            return all_results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise
            
    def _print_output_summary(self):
        """Print summary of generated outputs."""
        self.logger.info("\nOUTPUT SUMMARY:")
        self.logger.info("-" * 30)
        
        # Count files in each directory
        directories = {
            'Cleaned Data': os.path.join(self.output_dir, 'cleaned'),
            'Merged Data': os.path.join(self.output_dir, 'merged'),
            'Summary Reports': os.path.join(self.output_dir, 'summary'),
            'Individual Plots': os.path.join(self.output_dir, 'plots', 'individual'),
            'Cohort Plots': os.path.join(self.output_dir, 'plots', 'cohort'),
            'Logs': os.path.join(self.output_dir, 'logs')
        }
        
        for name, directory in directories.items():
            if os.path.exists(directory):
                file_count = len([f for f in os.listdir(directory) 
                                if os.path.isfile(os.path.join(directory, f))])
                self.logger.info(f"{name}: {file_count} files in {directory}")
            else:
                self.logger.info(f"{name}: Directory not found")


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description='GOQII Health Data Exploratory Data Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python eda_pipeline.py --input_dir data/input --output_dir results/
    python eda_pipeline.py --input_dir /path/to/data --output_dir /path/to/results --log_level DEBUG
    
Output Structure:
    results/
    ├── cleaned/          # Cleaned data files per metric
    ├── merged/           # Time-aligned merged dataset
    ├── summary/          # Statistical summaries and reports
    ├── plots/
    │   ├── individual/   # Individual participant plots
    │   └── cohort/       # Cohort-level visualizations
    └── logs/             # Pipeline execution logs
        """
    )
    
    parser.add_argument('--input_dir', required=True,
                       help='Directory containing input CSV and JSON files')
    parser.add_argument('--output_dir', required=True,
                       help='Directory for output files and reports')
    parser.add_argument('--log_level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    parser.add_argument('--version', action='version', version='EDA Pipeline v1.0')
    
    args = parser.parse_args()
    
    try:
        # Initialize and run pipeline
        pipeline = EDAPipeline(args.input_dir, args.output_dir, args.log_level)
        results = pipeline.run_complete_pipeline()
        
        print("\n" + "="*60)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"📁 Check results in: {args.output_dir}")
        print("📊 Reports and visualizations have been generated")
        print("📋 Check the summary report for detailed findings")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n❌ Pipeline interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
