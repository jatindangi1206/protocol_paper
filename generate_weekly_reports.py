#!/usr/bin/env python3
"""
Weekly Health Reports Generator with LangChain Integration

This script integrates LangChain capabilities into the GOQII Health Data EDA Pipeline
to generate intelligent weekly health summaries, anomaly detection, and personalized
recommendations for individual participants and cohort-level insights.

Usage:
    python generate_weekly_reports.py --input_dir data/input --output_dir results/
    python generate_weekly_reports.py --help

Requirements:
    - Set up API keys in environment variables (see config.py)
    - Install LangChain: pip install langchain openai faiss-cpu
"""

import argparse
import os
import sys
import logging
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# Import existing pipeline modules
try:
    from data_preparation import DataPreparation
    from individual_analysis import IndividualAnalysis
    from cohort_analysis import CohortAnalysis
    from langchain_integration import LangChainHealthAnalyzer, LANGCHAIN_AVAILABLE
    import config
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WeeklyReportsGenerator:
    """Main class for generating weekly health reports with LangChain."""
    
    def __init__(self, input_dir: str, output_dir: str, enable_langchain: bool = True):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.enable_langchain = enable_langchain and LANGCHAIN_AVAILABLE
        
        # Initialize pipeline components
        self.data_prep = DataPreparation(input_dir, output_dir)
        self.individual_analysis = IndividualAnalysis(output_dir)
        self.cohort_analysis = CohortAnalysis(output_dir)
        
        # Initialize LangChain analyzer if available
        self.langchain_analyzer = None
        if self.enable_langchain:
            try:
                config_status = config.validate_config()
                if config_status['valid']:
                    self.langchain_analyzer = LangChainHealthAnalyzer()
                    logger.info("✅ LangChain integration enabled")
                else:
                    logger.warning(f"❌ LangChain configuration issues: {config_status['issues']}")
                    self.enable_langchain = False
            except Exception as e:
                logger.warning(f"❌ Failed to initialize LangChain: {e}")
                self.enable_langchain = False
        
        if not self.enable_langchain:
            logger.info("📊 Running without LangChain integration")
    
    def prepare_weekly_data(self) -> Dict[str, pd.DataFrame]:
        """Prepare and aggregate data by weeks."""
        logger.info("🔄 Preparing weekly aggregated data...")
        
        # Run data preparation pipeline
        processed_data = self.data_prep.run_data_preparation()
        
        if not processed_data:
            logger.error("No data processed from input directory")
            return {}
        
        # Aggregate data by participant and week
        weekly_aggregated = {}
        
        for metric_type, df in processed_data.items():
            if df.empty:
                continue
            
            # Convert datetime and create week grouping
            df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
            df['week_start'] = df['datetime_utc'].dt.to_period('W').dt.start_time
            
            # Group by participant and week
            weekly_data = []
            
            for participant_id in df['participant_id'].unique():
                participant_df = df[df['participant_id'] == participant_id]
                
                for week_start in participant_df['week_start'].unique():
                    week_df = participant_df[participant_df['week_start'] == week_start]
                    
                    # Calculate weekly aggregates
                    weekly_stats = {
                        'participant_id': participant_id,
                        'week_start': week_start,
                        'metric_type': metric_type,
                        'data_points': len(week_df),
                        'mean_value': week_df['value_1'].mean(),
                        'min_value': week_df['value_1'].min(),
                        'max_value': week_df['value_1'].max(),
                        'std_value': week_df['value_1'].std(),
                        'quality_normal_ratio': len(week_df[week_df['quality_flag'] == 'normal']) / len(week_df)
                    }
                    
                    # Add metric-specific aggregates
                    if metric_type == 'bp':
                        weekly_stats['mean_diastolic'] = week_df['value_2'].mean()
                        weekly_stats['pulse_avg'] = pd.to_numeric(week_df['metadata'], errors='coerce').mean()
                    elif metric_type == 'sleep':
                        # Parse sleep metadata for stage analysis
                        sleep_stages = self._parse_sleep_metadata(week_df['metadata'])
                        weekly_stats.update(sleep_stages)
                    elif metric_type == 'steps':
                        weekly_stats['distance_avg'] = week_df['value_2'].mean()
                        weekly_stats['calories_avg'] = pd.to_numeric(week_df['metadata'], errors='coerce').mean()
                    
                    weekly_data.append(weekly_stats)
            
            if weekly_data:
                weekly_aggregated[metric_type] = pd.DataFrame(weekly_data)
        
        logger.info(f"✅ Aggregated data for {len(weekly_aggregated)} metric types")
        return weekly_aggregated
    
    def create_participant_weekly_summaries(self, weekly_aggregated: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """Create weekly summaries for each participant."""
        logger.info("📝 Generating individual participant summaries...")
        
        # Get all participant-week combinations
        participant_weeks = set()
        for metric_df in weekly_aggregated.values():
            for _, row in metric_df.iterrows():
                participant_weeks.add((row['participant_id'], row['week_start']))
        
        individual_summaries = []
        
        for participant_id, week_start in participant_weeks:
            logger.info(f"Processing participant {participant_id}, week {week_start.strftime('%Y-%m-%d')}")
            
            # Collect all metrics for this participant-week
            participant_week_data = {}
            for metric_type, metric_df in weekly_aggregated.items():
                week_data = metric_df[
                    (metric_df['participant_id'] == participant_id) & 
                    (metric_df['week_start'] == week_start)
                ]
                if not week_data.empty:
                    participant_week_data[metric_type] = week_data.iloc[0].to_dict()
            
            if not participant_week_data:
                continue
            
            # Generate summary using LangChain if available
            if self.enable_langchain and self.langchain_analyzer:
                try:
                    # Create a DataFrame from the weekly data for formatting
                    week_df = self._create_week_dataframe(participant_week_data)
                    
                    summary = self.langchain_analyzer.generate_individual_summary(
                        week_df, participant_id, week_start.strftime('%Y-%m-%d')
                    )
                    individual_summaries.append(summary)
                    
                except Exception as e:
                    logger.error(f"LangChain summary failed for {participant_id}: {e}")
                    # Fallback to basic summary
                    basic_summary = self._create_basic_summary(participant_id, week_start, participant_week_data)
                    individual_summaries.append(basic_summary)
            else:
                # Create basic summary without LangChain
                basic_summary = self._create_basic_summary(participant_id, week_start, participant_week_data)
                individual_summaries.append(basic_summary)
        
        logger.info(f"✅ Generated {len(individual_summaries)} individual summaries")
        return individual_summaries
    
    def create_cohort_summary(self, weekly_aggregated: Dict[str, pd.DataFrame], 
                            individual_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create cohort-level summary."""
        logger.info("🏥 Generating cohort-level summary...")
        
        # Prepare cohort-level DataFrame
        cohort_data = []
        
        for metric_type, metric_df in weekly_aggregated.items():
            cohort_stats = {
                'metric_type': metric_type,
                'total_participants': metric_df['participant_id'].nunique(),
                'total_weeks': len(metric_df),
                'mean_value': metric_df['mean_value'].mean(),
                'std_value': metric_df['mean_value'].std(),
                'min_participant_avg': metric_df.groupby('participant_id')['mean_value'].mean().min(),
                'max_participant_avg': metric_df.groupby('participant_id')['mean_value'].mean().max(),
                'avg_data_quality': metric_df['quality_normal_ratio'].mean()
            }
            cohort_data.append(cohort_stats)
        
        cohort_df = pd.DataFrame(cohort_data)
        
        # Generate cohort summary using LangChain if available
        if self.enable_langchain and self.langchain_analyzer:
            try:
                cohort_summary = self.langchain_analyzer.generate_cohort_summary(
                    cohort_df, individual_summaries
                )
            except Exception as e:
                logger.error(f"LangChain cohort summary failed: {e}")
                cohort_summary = self._create_basic_cohort_summary(cohort_df, individual_summaries)
        else:
            cohort_summary = self._create_basic_cohort_summary(cohort_df, individual_summaries)
        
        logger.info("✅ Generated cohort summary")
        return cohort_summary
    
    def save_reports(self, individual_summaries: List[Dict[str, Any]], 
                    cohort_summary: Dict[str, Any]) -> Dict[str, List[str]]:
        """Save all reports in multiple formats."""
        logger.info("💾 Saving reports...")
        
        saved_files = {'markdown': [], 'html': [], 'json': []}
        
        if self.enable_langchain and self.langchain_analyzer:
            # Use LangChain's save functionality
            langchain_files = self.langchain_analyzer.save_reports(individual_summaries, cohort_summary)
            saved_files['markdown'].extend(langchain_files.get('individual_markdown', []))
            saved_files['markdown'].append(langchain_files.get('cohort_markdown', ''))
            saved_files['html'].extend(langchain_files.get('individual_html', []))
            saved_files['html'].append(langchain_files.get('cohort_html', ''))
            
            # Store in vector database if enabled
            if config.ENABLE_VECTOR_STORE:
                try:
                    all_reports = individual_summaries + [cohort_summary]
                    self.langchain_analyzer.store_reports_in_vector_db(all_reports)
                    logger.info("✅ Stored reports in vector database")
                except Exception as e:
                    logger.warning(f"Failed to store in vector database: {e}")
        else:
            # Basic file saving
            saved_files = self._save_basic_reports(individual_summaries, cohort_summary)
        
        # Save JSON versions for programmatic access
        json_files = self._save_json_reports(individual_summaries, cohort_summary)
        saved_files['json'] = json_files
        
        logger.info(f"✅ Saved reports: {len(saved_files['markdown'])} MD, {len(saved_files['html'])} HTML, {len(saved_files['json'])} JSON")
        return saved_files
    
    def generate_weekly_reports(self) -> Dict[str, Any]:
        """Main method to generate all weekly reports."""
        logger.info("🚀 Starting weekly health reports generation...")
        
        start_time = datetime.now()
        
        try:
            # Step 1: Prepare weekly data
            weekly_aggregated = self.prepare_weekly_data()
            
            if not weekly_aggregated:
                logger.error("No weekly data to process")
                return {'success': False, 'error': 'No data processed'}
            
            # Step 2: Generate individual summaries
            individual_summaries = self.create_participant_weekly_summaries(weekly_aggregated)
            
            # Step 3: Generate cohort summary
            cohort_summary = self.create_cohort_summary(weekly_aggregated, individual_summaries)
            
            # Step 4: Save reports
            saved_files = self.save_reports(individual_summaries, cohort_summary)
            
            # Summary statistics
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                'success': True,
                'execution_time_seconds': execution_time,
                'individual_summaries_count': len(individual_summaries),
                'cohort_summary_generated': bool(cohort_summary),
                'langchain_enabled': self.enable_langchain,
                'saved_files': saved_files,
                'metrics_processed': list(weekly_aggregated.keys()),
                'participants_count': len(set(s.get('participant_id', '') for s in individual_summaries))
            }
            
            logger.info("🎉 Weekly reports generation completed successfully!")
            logger.info(f"⏱️  Execution time: {execution_time:.2f} seconds")
            logger.info(f"👥 Processed {result['participants_count']} participants")
            logger.info(f"📊 Generated {result['individual_summaries_count']} individual summaries")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Weekly reports generation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'execution_time_seconds': (datetime.now() - start_time).total_seconds()
            }
    
    def _parse_sleep_metadata(self, metadata_series: pd.Series) -> Dict[str, float]:
        """Parse sleep metadata to extract stage information."""
        sleep_stages = {'light_avg': 0, 'deep_avg': 0, 'rem_avg': 0}
        
        for metadata in metadata_series.dropna():
            try:
                if isinstance(metadata, str) and ':' in metadata:
                    stages = {}
                    for stage_info in metadata.split(','):
                        if ':' in stage_info:
                            stage, minutes = stage_info.split(':')
                            stages[stage.strip()] = float(minutes)
                    
                    sleep_stages['light_avg'] += stages.get('light', 0)
                    sleep_stages['deep_avg'] += stages.get('deep', 0)
                    sleep_stages['rem_avg'] += stages.get('rem', 0)
            except:
                continue
        
        # Average over the number of sleep records
        count = len(metadata_series.dropna())
        if count > 0:
            for key in sleep_stages:
                sleep_stages[key] /= count
        
        return sleep_stages
    
    def _create_week_dataframe(self, participant_week_data: Dict[str, Dict]) -> pd.DataFrame:
        """Create a DataFrame from participant week data for LangChain formatting."""
        rows = []
        
        for metric_type, data in participant_week_data.items():
            row = {
                'participant_id': data['participant_id'],
                'datetime_utc': data['week_start'],
                'metric_type': metric_type,
                'value_1': data['mean_value'],
                'quality_flag': 'normal' if data['quality_normal_ratio'] > 0.8 else 'flagged'
            }
            
            # Add metric-specific data
            if metric_type == 'bp':
                row['value_2'] = data.get('mean_diastolic', 0)
                row['metadata'] = str(data.get('pulse_avg', 0))
            elif metric_type == 'sleep':
                row['metadata'] = f"light:{data.get('light_avg', 0)},deep:{data.get('deep_avg', 0)},rem:{data.get('rem_avg', 0)}"
            elif metric_type == 'steps':
                row['value_2'] = data.get('distance_avg', 0)
                row['metadata'] = str(data.get('calories_avg', 0))
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _create_basic_summary(self, participant_id: str, week_start: datetime, 
                            participant_week_data: Dict[str, Dict]) -> Dict[str, Any]:
        """Create basic summary without LangChain."""
        
        summary_parts = [f"# Weekly Health Summary - Participant {participant_id}"]
        summary_parts.append(f"**Week Starting:** {week_start.strftime('%Y-%m-%d')}")
        summary_parts.append("")
        
        # Summarize each metric
        for metric_type, data in participant_week_data.items():
            summary_parts.append(f"## {metric_type.upper()}")
            summary_parts.append(f"- Average: {data['mean_value']:.1f}")
            summary_parts.append(f"- Range: {data['min_value']:.1f} - {data['max_value']:.1f}")
            summary_parts.append(f"- Data quality: {data['quality_normal_ratio']:.1%}")
            summary_parts.append("")
        
        # Basic recommendations
        summary_parts.append("## Recommendations")
        summary_parts.append("- Continue regular monitoring")
        summary_parts.append("- Consult healthcare provider for any concerns")
        summary_parts.append("")
        
        return {
            'participant_id': participant_id,
            'week_start': week_start.strftime('%Y-%m-%d'),
            'summary': '\n'.join(summary_parts),
            'data_quality_rating': 'Acceptable',
            'generated_at': datetime.now().isoformat()
        }
    
    def _create_basic_cohort_summary(self, cohort_df: pd.DataFrame, 
                                   individual_summaries: List[Dict]) -> Dict[str, Any]:
        """Create basic cohort summary without LangChain."""
        
        summary_parts = ["# Cohort Health Analysis Summary"]
        summary_parts.append(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d')}")
        summary_parts.append(f"**Cohort Size:** {len(individual_summaries)} participants")
        summary_parts.append("")
        
        summary_parts.append("## Metric Overview")
        for _, row in cohort_df.iterrows():
            summary_parts.append(f"### {row['metric_type'].upper()}")
            summary_parts.append(f"- Participants: {row['total_participants']}")
            summary_parts.append(f"- Average value: {row['mean_value']:.1f}")
            summary_parts.append(f"- Data quality: {row['avg_data_quality']:.1%}")
            summary_parts.append("")
        
        return {
            'cohort_size': len(individual_summaries),
            'analysis_date': datetime.now().isoformat(),
            'summary': '\n'.join(summary_parts)
        }
    
    def _save_basic_reports(self, individual_summaries: List[Dict], 
                          cohort_summary: Dict) -> Dict[str, List[str]]:
        """Save reports in basic format."""
        
        # Create directories
        os.makedirs(config.MARKDOWN_DIR, exist_ok=True)
        os.makedirs(config.HTML_DIR, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        saved_files = {'markdown': [], 'html': []}
        
        # Save individual reports
        for summary in individual_summaries:
            participant_id = summary.get('participant_id', 'unknown')
            week_start = summary.get('week_start', 'unknown')
            
            # Markdown
            md_filename = f"individual_{participant_id}_{week_start}_{timestamp}.md"
            md_path = os.path.join(config.MARKDOWN_DIR, md_filename)
            
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(summary.get('summary', 'No summary available'))
            
            saved_files['markdown'].append(md_path)
        
        # Save cohort report
        cohort_md_filename = f"cohort_summary_{timestamp}.md"
        cohort_md_path = os.path.join(config.MARKDOWN_DIR, cohort_md_filename)
        
        with open(cohort_md_path, 'w', encoding='utf-8') as f:
            f.write(cohort_summary.get('summary', 'No summary available'))
        
        saved_files['markdown'].append(cohort_md_path)
        
        return saved_files
    
    def _save_json_reports(self, individual_summaries: List[Dict], 
                         cohort_summary: Dict) -> List[str]:
        """Save reports in JSON format for programmatic access."""
        
        json_dir = os.path.join(config.REPORTS_DIR, 'json')
        os.makedirs(json_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        saved_files = []
        
        # Save individual summaries
        individuals_path = os.path.join(json_dir, f'individual_summaries_{timestamp}.json')
        with open(individuals_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(individual_summaries, f, indent=2, default=str)
        saved_files.append(individuals_path)
        
        # Save cohort summary
        cohort_path = os.path.join(json_dir, f'cohort_summary_{timestamp}.json')
        with open(cohort_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(cohort_summary, f, indent=2, default=str)
        saved_files.append(cohort_path)
        
        return saved_files


def main():
    """Main function with command-line interface."""
    
    parser = argparse.ArgumentParser(
        description='Generate Weekly Health Reports with LangChain Integration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python generate_weekly_reports.py --input_dir data/input --output_dir results/
    python generate_weekly_reports.py --input_dir data/input --output_dir results/ --no-langchain
    
Environment Setup:
    export OPENAI_API_KEY="your_api_key_here"
    export LLM_PROVIDER="openai"
    
Output Structure:
    reports/
    ├── markdown/          # Markdown reports
    ├── html/              # HTML reports  
    ├── json/              # JSON data for programmatic access
    └── vector_store/      # FAISS vector store (if enabled)
        """
    )
    
    parser.add_argument('--input_dir', required=True,
                       help='Directory containing input health data files')
    parser.add_argument('--output_dir', required=True,
                       help='Directory for pipeline outputs')
    parser.add_argument('--no-langchain', action='store_true',
                       help='Disable LangChain integration (basic reports only)')
    parser.add_argument('--config-check', action='store_true',
                       help='Check configuration and exit')
    parser.add_argument('--test-langchain', action='store_true',
                       help='Test LangChain integration and exit')
    
    args = parser.parse_args()
    
    # Configuration check
    if args.config_check:
        print("🔧 Checking configuration...")
        config_status = config.validate_config()
        print(f"Valid: {config_status['valid']}")
        print(f"Provider: {config_status['provider']}")
        print(f"Issues: {config_status['issues'] if not config_status['valid'] else 'None'}")
        
        if not config_status['valid']:
            print("\n📝 Example configuration:")
            config.print_env_example()
        
        return 0 if config_status['valid'] else 1
    
    # LangChain test
    if args.test_langchain:
        print("🧪 Testing LangChain integration...")
        if LANGCHAIN_AVAILABLE:
            from langchain_integration import test_langchain_integration
            success = test_langchain_integration()
            return 0 if success else 1
        else:
            print("❌ LangChain not available")
            return 1
    
    # Main execution
    try:
        enable_langchain = not args.no_langchain
        
        generator = WeeklyReportsGenerator(
            args.input_dir, 
            args.output_dir, 
            enable_langchain=enable_langchain
        )
        
        result = generator.generate_weekly_reports()
        
        if result['success']:
            print("\n" + "="*60)
            print("✅ WEEKLY REPORTS GENERATED SUCCESSFULLY!")
            print("="*60)
            print(f"⏱️  Execution time: {result['execution_time_seconds']:.2f} seconds")
            print(f"👥 Participants processed: {result['participants_count']}")
            print(f"📊 Individual summaries: {result['individual_summaries_count']}")
            print(f"🤖 LangChain enabled: {result['langchain_enabled']}")
            print(f"📁 Output directory: {args.output_dir}")
            print("\n📋 Generated files:")
            
            for file_type, files in result['saved_files'].items():
                print(f"  {file_type.upper()}: {len(files)} files")
            
            return 0
        else:
            print(f"\n❌ Report generation failed: {result['error']}")
            return 1
            
    except KeyboardInterrupt:
        print("\n❌ Process interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        logger.exception("Unexpected error occurred")
        return 1


if __name__ == "__main__":
    sys.exit(main())
