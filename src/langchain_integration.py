"""
LangChain Integration Module for GOQII Health Data EDA Pipeline

This module integrates LangChain capabilities to generate intelligent health summaries,
anomaly detection, personalized recommendations, and cohort-level insights from 
weekly aggregated health metrics.
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime, timedelta
import json
import re
from pathlib import Path

# LangChain imports
try:
    from langchain.llms import OpenAI
    from langchain.chat_models import ChatOpenAI
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from langchain.schema import Document
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    LANGCHAIN_AVAILABLE = False
    print(f"Warning: LangChain not available: {e}")
    print("Install with: pip install langchain openai faiss-cpu")

# Import configuration
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

logger = logging.getLogger(__name__)


class HealthDataFormatter:
    """Formats health data into natural language context for LLMs."""
    
    def __init__(self):
        self.health_thresholds = config.HEALTH_THRESHOLDS
        
    def format_weekly_participant_data(self, participant_df: pd.DataFrame, 
                                     participant_id: str, week_start: str) -> str:
        """Format weekly participant data into natural language."""
        
        if participant_df.empty:
            return f"No data available for participant {participant_id} for week starting {week_start}."
        
        # Calculate weekly aggregates
        weekly_stats = self._calculate_weekly_stats(participant_df)
        
        # Format into readable text
        context = f"""
=== WEEKLY HEALTH SUMMARY ===
Participant ID: {participant_id}
Week Starting: {week_start}
Data Collection Period: {len(participant_df)} readings

=== VITAL SIGNS ==="""
        
        # Heart Rate
        if 'hr' in weekly_stats:
            hr_stats = weekly_stats['hr']
            context += f"""
Heart Rate:
  - Average: {hr_stats['mean']:.1f} bpm (Range: {hr_stats['min']:.0f}-{hr_stats['max']:.0f})
  - Resting periods: {hr_stats.get('below_60_count', 0)} readings below 60 bpm
  - Active periods: {hr_stats.get('above_100_count', 0)} readings above 100 bpm
  - Variability: {hr_stats['std']:.1f} bpm standard deviation"""
        
        # Blood Pressure
        if 'bp_systolic' in weekly_stats and 'bp_diastolic' in weekly_stats:
            sys_stats = weekly_stats['bp_systolic']
            dia_stats = weekly_stats['bp_diastolic']
            context += f"""
Blood Pressure:
  - Average: {sys_stats['mean']:.0f}/{dia_stats['mean']:.0f} mmHg
  - Systolic range: {sys_stats['min']:.0f}-{sys_stats['max']:.0f} mmHg
  - Diastolic range: {dia_stats['min']:.0f}-{dia_stats['max']:.0f} mmHg
  - High readings (>140/90): {sys_stats.get('high_count', 0)} occurrences"""
        
        # SpO2
        if 'spo2' in weekly_stats:
            spo2_stats = weekly_stats['spo2']
            context += f"""
Oxygen Saturation (SpO₂):
  - Average: {spo2_stats['mean']:.1f}% (Range: {spo2_stats['min']:.1f}-{spo2_stats['max']:.1f}%)
  - Low readings (<95%): {spo2_stats.get('low_count', 0)} occurrences"""
        
        # Temperature
        if 'temp' in weekly_stats:
            temp_stats = weekly_stats['temp']
            context += f"""
Body Temperature:
  - Average: {temp_stats['mean']:.1f}°F (Range: {temp_stats['min']:.1f}-{temp_stats['max']:.1f}°F)
  - Fever episodes (>100.4°F): {temp_stats.get('fever_count', 0)} occurrences"""
        
        context += "\n=== ACTIVITY & SLEEP ==="
        
        # Steps
        if 'steps' in weekly_stats:
            steps_stats = weekly_stats['steps']
            context += f"""
Daily Steps:
  - Average: {steps_stats['mean']:.0f} steps/day
  - Most active day: {steps_stats['max']:.0f} steps
  - Least active day: {steps_stats['min']:.0f} steps
  - Days meeting 10k goal: {steps_stats.get('goal_days', 0)} out of {steps_stats['count']} days"""
        
        # Sleep
        if 'sleep_total' in weekly_stats:
            sleep_stats = weekly_stats['sleep_total']
            context += f"""
Sleep Patterns:
  - Average sleep: {sleep_stats['mean']/60:.1f} hours/night (Range: {sleep_stats['min']/60:.1f}-{sleep_stats['max']/60:.1f} hours)
  - Sleep debt: {self._calculate_sleep_debt(sleep_stats)} hours this week
  - Sleep quality consistency: {self._assess_sleep_consistency(sleep_stats)}"""
        
        # Data Quality Assessment
        context += "\n=== DATA QUALITY ==="
        quality_assessment = self._assess_data_quality(participant_df, weekly_stats)
        context += f"""
Completeness: {quality_assessment['completeness']:.1%}
Missing days: {quality_assessment['missing_days']}
Anomalous readings: {quality_assessment['anomaly_count']} ({quality_assessment['anomaly_rate']:.1%})
Data reliability: {quality_assessment['reliability']}"""
        
        return context
    
    def format_cohort_data(self, cohort_summary: pd.DataFrame, 
                          individual_summaries: List[Dict[str, Any]]) -> str:
        """Format cohort-level data into natural language."""
        
        context = f"""
=== COHORT WEEKLY SUMMARY ===
Analysis Date: {datetime.now().strftime('%Y-%m-%d')}
Total Participants: {len(individual_summaries)}
Data Collection Period: Weekly aggregated metrics

=== COHORT VITAL SIGNS OVERVIEW ==="""
        
        # Aggregate cohort statistics
        for metric in ['hr', 'bp_systolic', 'bp_diastolic', 'spo2', 'temp']:
            if metric in cohort_summary.columns:
                metric_data = cohort_summary[metric].dropna()
                if not metric_data.empty:
                    context += f"""
{metric.upper().replace('_', ' ')}:
  - Cohort average: {metric_data.mean():.1f}
  - Range: {metric_data.min():.1f} - {metric_data.max():.1f}
  - Standard deviation: {metric_data.std():.1f}
  - Participants outside normal range: {self._count_outliers(metric_data, metric)}"""
        
        context += "\n=== ACTIVITY & LIFESTYLE PATTERNS ==="
        
        # Activity patterns
        if 'steps' in cohort_summary.columns:
            steps_data = cohort_summary['steps'].dropna()
            context += f"""
Physical Activity:
  - Average daily steps: {steps_data.mean():.0f}
  - Most active participant: {steps_data.max():.0f} steps/day
  - Least active participant: {steps_data.min():.0f} steps/day
  - Participants meeting 10k goal: {len(steps_data[steps_data >= 10000])}/{len(steps_data)}"""
        
        # Sleep patterns
        if 'sleep_total' in cohort_summary.columns:
            sleep_data = cohort_summary['sleep_total'].dropna() / 60  # Convert to hours
            context += f"""
Sleep Patterns:
  - Average sleep duration: {sleep_data.mean():.1f} hours/night
  - Sleep range: {sleep_data.min():.1f} - {sleep_data.max():.1f} hours
  - Participants with adequate sleep (7-9h): {len(sleep_data[(sleep_data >= 7) & (sleep_data <= 9)])}/{len(sleep_data)}"""
        
        # Participant classifications
        context += "\n=== PARTICIPANT CLASSIFICATIONS ==="
        classifications = self._classify_participants(individual_summaries)
        
        context += f"""
High Performers: {len(classifications['high_performers'])} participants
  - Excellent data quality and health metrics within normal ranges
  
Attention Needed: {len(classifications['attention_needed'])} participants
  - Some metrics outside normal ranges or data quality issues
  
High Risk: {len(classifications['high_risk'])} participants
  - Multiple concerning health indicators or poor data quality

Data Quality Distribution:
  - Good quality: {classifications['quality_distribution']['good']} participants
  - Acceptable quality: {classifications['quality_distribution']['acceptable']} participants
  - Poor quality: {classifications['quality_distribution']['poor']} participants"""
        
        return context
    
    def _calculate_weekly_stats(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate weekly statistics for each metric."""
        stats = {}
        
        for column in df.columns:
            if column in ['participant_id', 'metric_type', 'datetime_utc', 'timestamp_unix', 'quality_flag', 'metadata']:
                continue
                
            if df[column].dtype in ['float64', 'int64'] and not df[column].isna().all():
                series = df[column].dropna()
                
                metric_stats = {
                    'mean': series.mean(),
                    'min': series.min(),
                    'max': series.max(),
                    'std': series.std(),
                    'count': len(series)
                }
                
                # Add metric-specific counts
                if 'hr' in column.lower():
                    metric_stats['below_60_count'] = len(series[series < 60])
                    metric_stats['above_100_count'] = len(series[series > 100])
                elif 'bp_systolic' in column.lower():
                    metric_stats['high_count'] = len(series[series > 140])
                elif 'spo2' in column.lower():
                    metric_stats['low_count'] = len(series[series < 95])
                elif 'temp' in column.lower():
                    metric_stats['fever_count'] = len(series[series > 100.4])
                elif 'steps' in column.lower():
                    metric_stats['goal_days'] = len(series[series >= 10000])
                
                stats[column] = metric_stats
        
        return stats
    
    def _assess_data_quality(self, df: pd.DataFrame, weekly_stats: Dict) -> Dict[str, Any]:
        """Assess data quality for the week."""
        total_possible_readings = 7 * 24 * 12  # Assuming 5-minute intervals
        actual_readings = len(df)
        
        completeness = actual_readings / total_possible_readings if total_possible_readings > 0 else 0
        
        # Count anomalies based on quality flags
        anomaly_count = len(df[df.get('quality_flag', 'normal') != 'normal'])
        anomaly_rate = anomaly_count / len(df) if len(df) > 0 else 0
        
        # Calculate missing days
        if 'datetime_utc' in df.columns:
            df['date'] = pd.to_datetime(df['datetime_utc']).dt.date
            unique_days = df['date'].nunique()
            missing_days = 7 - unique_days
        else:
            missing_days = 0
        
        # Determine reliability
        if completeness >= 0.9 and anomaly_rate <= 0.05:
            reliability = "Excellent"
        elif completeness >= 0.7 and anomaly_rate <= 0.15:
            reliability = "Good"
        elif completeness >= 0.5 and anomaly_rate <= 0.3:
            reliability = "Acceptable"
        else:
            reliability = "Poor"
        
        return {
            'completeness': completeness,
            'missing_days': missing_days,
            'anomaly_count': anomaly_count,
            'anomaly_rate': anomaly_rate,
            'reliability': reliability
        }
    
    def _calculate_sleep_debt(self, sleep_stats: Dict) -> float:
        """Calculate sleep debt for the week."""
        target_sleep = 8 * 60  # 8 hours in minutes
        actual_sleep = sleep_stats['mean']
        return max(0, (target_sleep - actual_sleep) * sleep_stats['count'] / 60)
    
    def _assess_sleep_consistency(self, sleep_stats: Dict) -> str:
        """Assess sleep consistency."""
        cv = sleep_stats['std'] / sleep_stats['mean'] if sleep_stats['mean'] > 0 else 0
        
        if cv <= 0.15:
            return "Very Consistent"
        elif cv <= 0.25:
            return "Moderately Consistent"
        else:
            return "Inconsistent"
    
    def _count_outliers(self, data: pd.Series, metric: str) -> int:
        """Count participants outside normal range for a metric."""
        thresholds = self.health_thresholds.get(metric, {})
        if not thresholds:
            return 0
        
        low = thresholds.get('low', float('-inf'))
        high = thresholds.get('high', float('inf'))
        
        return len(data[(data < low) | (data > high)])
    
    def _classify_participants(self, individual_summaries: List[Dict]) -> Dict[str, Any]:
        """Classify participants based on health and data quality."""
        classifications = {
            'high_performers': [],
            'attention_needed': [],
            'high_risk': [],
            'quality_distribution': {'good': 0, 'acceptable': 0, 'poor': 0}
        }
        
        for summary in individual_summaries:
            participant_id = summary.get('participant_id', 'unknown')
            data_quality = summary.get('data_quality_rating', 'unknown').lower()
            
            # Count quality distribution
            if data_quality in classifications['quality_distribution']:
                classifications['quality_distribution'][data_quality] += 1
            
            # Classify based on summary content (simplified logic)
            if 'excellent' in summary.get('summary', '').lower() or data_quality == 'good':
                classifications['high_performers'].append(participant_id)
            elif 'concerning' in summary.get('summary', '').lower() or data_quality == 'poor':
                classifications['high_risk'].append(participant_id)
            else:
                classifications['attention_needed'].append(participant_id)
        
        return classifications


class LangChainHealthAnalyzer:
    """Main class for LangChain-powered health data analysis."""
    
    def __init__(self):
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain is required but not available")
        
        self.config = config.get_model_config()
        self.formatter = HealthDataFormatter()
        self.llm = self._initialize_llm()
        self.vector_store = None
        
        # Create output directories
        os.makedirs(config.MARKDOWN_DIR, exist_ok=True)
        os.makedirs(config.HTML_DIR, exist_ok=True)
        
        if config.ENABLE_VECTOR_STORE:
            self._initialize_vector_store()
    
    def _initialize_llm(self):
        """Initialize the LLM based on configuration."""
        if self.config['provider'] == 'openai':
            return ChatOpenAI(
                model=self.config['model'],
                temperature=self.config['temperature'],
                openai_api_key=self.config['api_key']
            )
        elif self.config['provider'] == 'google':
            # Note: This would require google-cloud-aiplatform
            from langchain.llms import VertexAI
            return VertexAI(
                model_name=self.config['model'],
                project=self.config['project_id'],
                location=self.config['location']
            )
        elif self.config['provider'] == 'anthropic':
            from langchain.chat_models import ChatAnthropic
            return ChatAnthropic(
                model=self.config['model'],
                anthropic_api_key=self.config['api_key']
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config['provider']}")
    
    def _initialize_vector_store(self):
        """Initialize FAISS vector store for historical insights."""
        try:
            if os.path.exists(config.VECTOR_STORE_PATH):
                embeddings = OpenAIEmbeddings()
                self.vector_store = FAISS.load_local(config.VECTOR_STORE_PATH, embeddings)
                logger.info("Loaded existing vector store")
            else:
                self.vector_store = None
                logger.info("Vector store will be created when first report is stored")
        except Exception as e:
            logger.warning(f"Failed to initialize vector store: {e}")
            self.vector_store = None
    
    def generate_individual_summary(self, participant_df: pd.DataFrame, 
                                  participant_id: str, week_start: str) -> Dict[str, Any]:
        """Generate individual health summary using LangChain."""
        
        # Format data for LLM
        weekly_data = self.formatter.format_weekly_participant_data(
            participant_df, participant_id, week_start
        )
        
        # Create prompt template
        individual_prompt = PromptTemplate(
            input_variables=["weekly_data"],
            template="""You are a health data analyst. Given this weekly wearable data:

{weekly_data}

Please provide a comprehensive analysis with the following structure:

1. **Health Status Summary**: Summarise the participant's health status in a concise but informative way.

2. **Pattern Analysis**: Identify any unusual patterns or anomalies and link them to possible causes.

3. **Personalized Recommendations**: Provide specific, actionable recommendations based on the data.

4. **Data Quality Rating**: Rate overall data quality as Good, Acceptable, or Poor with detailed reasons.

Please format your response clearly with headers and bullet points where appropriate. Focus on being helpful and actionable while maintaining a professional medical tone."""
        )
        
        # Create and run chain
        chain = LLMChain(llm=self.llm, prompt=individual_prompt)
        
        try:
            response = chain.run(weekly_data=weekly_data)
            
            # Extract data quality rating
            data_quality_rating = self._extract_data_quality_rating(response)
            
            result = {
                'participant_id': participant_id,
                'week_start': week_start,
                'summary': response,
                'data_quality_rating': data_quality_rating,
                'generated_at': datetime.now().isoformat(),
                'raw_data_length': len(weekly_data)
            }
            
            logger.info(f"Generated summary for participant {participant_id}, week {week_start}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate summary for participant {participant_id}: {e}")
            return {
                'participant_id': participant_id,
                'week_start': week_start,
                'summary': f"Error generating summary: {e}",
                'data_quality_rating': 'Unknown',
                'generated_at': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def generate_cohort_summary(self, cohort_df: pd.DataFrame, 
                              individual_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate cohort-level summary using LangChain."""
        
        # Format cohort data for LLM
        cohort_data = self.formatter.format_cohort_data(cohort_df, individual_summaries)
        
        # Create prompt template
        cohort_prompt = PromptTemplate(
            input_variables=["cohort_data"],
            template="""You are analysing a cohort of wearable health data participants. Given the aggregated weekly statistics:

{cohort_data}

Please provide a comprehensive cohort analysis with the following structure:

1. **Cohort Health Trends**: Summarise the overall trends and patterns in the group.

2. **Risk Stratification**: Identify outliers, high-risk participants, and clusters with similar patterns.

3. **Cohort-Wide Recommendations**: Suggest protocol improvements and population-level interventions.

4. **Data Quality Assessment**: Evaluate the overall data collection quality and suggest improvements.

5. **Research Insights**: Highlight any interesting findings that could inform future studies.

Please format your response clearly with headers and bullet points. Focus on population health insights and actionable recommendations for study coordinators."""
        )
        
        # Create and run chain
        chain = LLMChain(llm=self.llm, prompt=cohort_prompt)
        
        try:
            response = chain.run(cohort_data=cohort_data)
            
            result = {
                'cohort_size': len(individual_summaries),
                'analysis_date': datetime.now().isoformat(),
                'summary': response,
                'individual_summaries_count': len(individual_summaries),
                'data_quality_distribution': self._analyze_cohort_quality(individual_summaries)
            }
            
            logger.info(f"Generated cohort summary for {len(individual_summaries)} participants")
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate cohort summary: {e}")
            return {
                'cohort_size': len(individual_summaries),
                'analysis_date': datetime.now().isoformat(),
                'summary': f"Error generating cohort summary: {e}",
                'error': str(e)
            }
    
    def store_reports_in_vector_db(self, reports: List[Dict[str, Any]]):
        """Store reports in vector database for future querying."""
        if not config.ENABLE_VECTOR_STORE:
            logger.info("Vector store disabled, skipping storage")
            return
        
        try:
            documents = []
            
            for report in reports:
                if 'summary' in report:
                    # Create document with metadata
                    doc = Document(
                        page_content=report['summary'],
                        metadata={
                            'participant_id': report.get('participant_id', 'cohort'),
                            'week_start': report.get('week_start', report.get('analysis_date', '')),
                            'data_quality': report.get('data_quality_rating', 'unknown'),
                            'generated_at': report.get('generated_at', datetime.now().isoformat()),
                            'report_type': 'individual' if 'participant_id' in report else 'cohort'
                        }
                    )
                    documents.append(doc)
            
            if not documents:
                logger.warning("No valid documents to store in vector database")
                return
            
            # Initialize embeddings
            embeddings = OpenAIEmbeddings()
            
            if self.vector_store is None:
                # Create new vector store
                self.vector_store = FAISS.from_documents(documents, embeddings)
                os.makedirs(os.path.dirname(config.VECTOR_STORE_PATH), exist_ok=True)
                self.vector_store.save_local(config.VECTOR_STORE_PATH)
                logger.info(f"Created new vector store with {len(documents)} documents")
            else:
                # Add to existing vector store
                self.vector_store.add_documents(documents)
                self.vector_store.save_local(config.VECTOR_STORE_PATH)
                logger.info(f"Added {len(documents)} documents to existing vector store")
                
        except Exception as e:
            logger.error(f"Failed to store reports in vector database: {e}")
    
    def query_historical_insights(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Query historical insights from vector store."""
        if not self.vector_store:
            logger.warning("Vector store not available for querying")
            return []
        
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            
            results = []
            for doc in docs:
                results.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'relevance_score': 'N/A'  # FAISS doesn't return scores by default
                })
            
            logger.info(f"Retrieved {len(results)} historical insights for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to query historical insights: {e}")
            return []
    
    def save_reports(self, individual_reports: List[Dict[str, Any]], 
                    cohort_report: Dict[str, Any]):
        """Save reports in Markdown and HTML formats."""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save individual reports
        for report in individual_reports:
            participant_id = report.get('participant_id', 'unknown')
            week_start = report.get('week_start', 'unknown')
            
            # Markdown format
            md_filename = f"individual_{participant_id}_{week_start}_{timestamp}.md"
            md_path = os.path.join(config.MARKDOWN_DIR, md_filename)
            
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(f"# Health Summary - Participant {participant_id}\n\n")
                f.write(f"**Week Starting:** {week_start}\n")
                f.write(f"**Generated:** {report.get('generated_at', 'Unknown')}\n")
                f.write(f"**Data Quality:** {report.get('data_quality_rating', 'Unknown')}\n\n")
                f.write("---\n\n")
                f.write(report.get('summary', 'No summary available'))
            
            # HTML format
            html_filename = f"individual_{participant_id}_{week_start}_{timestamp}.html"
            html_path = os.path.join(config.HTML_DIR, html_filename)
            
            html_content = self._convert_to_html(
                f"Health Summary - Participant {participant_id}",
                report.get('summary', 'No summary available'),
                report
            )
            
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
        
        # Save cohort report
        # Markdown format
        cohort_md_filename = f"cohort_summary_{timestamp}.md"
        cohort_md_path = os.path.join(config.MARKDOWN_DIR, cohort_md_filename)
        
        with open(cohort_md_path, 'w', encoding='utf-8') as f:
            f.write("# Cohort Health Analysis Summary\n\n")
            f.write(f"**Analysis Date:** {cohort_report.get('analysis_date', 'Unknown')}\n")
            f.write(f"**Cohort Size:** {cohort_report.get('cohort_size', 'Unknown')}\n\n")
            f.write("---\n\n")
            f.write(cohort_report.get('summary', 'No summary available'))
        
        # HTML format
        cohort_html_filename = f"cohort_summary_{timestamp}.html"
        cohort_html_path = os.path.join(config.HTML_DIR, cohort_html_filename)
        
        cohort_html_content = self._convert_to_html(
            "Cohort Health Analysis Summary",
            cohort_report.get('summary', 'No summary available'),
            cohort_report
        )
        
        with open(cohort_html_path, 'w', encoding='utf-8') as f:
            f.write(cohort_html_content)
        
        logger.info(f"Saved {len(individual_reports)} individual reports and 1 cohort report")
        
        return {
            'individual_markdown': [os.path.join(config.MARKDOWN_DIR, f"individual_{r.get('participant_id', 'unknown')}_{r.get('week_start', 'unknown')}_{timestamp}.md") for r in individual_reports],
            'individual_html': [os.path.join(config.HTML_DIR, f"individual_{r.get('participant_id', 'unknown')}_{r.get('week_start', 'unknown')}_{timestamp}.html") for r in individual_reports],
            'cohort_markdown': cohort_md_path,
            'cohort_html': cohort_html_path
        }
    
    def _extract_data_quality_rating(self, summary_text: str) -> str:
        """Extract data quality rating from summary text."""
        summary_lower = summary_text.lower()
        
        if 'good' in summary_lower and 'quality' in summary_lower:
            return 'Good'
        elif 'acceptable' in summary_lower and 'quality' in summary_lower:
            return 'Acceptable'
        elif 'poor' in summary_lower and 'quality' in summary_lower:
            return 'Poor'
        else:
            return 'Unknown'
    
    def _analyze_cohort_quality(self, individual_summaries: List[Dict]) -> Dict[str, int]:
        """Analyze data quality distribution across cohort."""
        distribution = {'Good': 0, 'Acceptable': 0, 'Poor': 0, 'Unknown': 0}
        
        for summary in individual_summaries:
            quality = summary.get('data_quality_rating', 'Unknown')
            distribution[quality] = distribution.get(quality, 0) + 1
        
        return distribution
    
    def _convert_to_html(self, title: str, content: str, metadata: Dict) -> str:
        """Convert content to HTML format."""
        
        # Simple markdown-like conversion
        html_content = content
        html_content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html_content)
        html_content = re.sub(r'\*(.*?)\*', r'<em>\1</em>', html_content)
        html_content = re.sub(r'### (.*?)\n', r'<h3>\1</h3>\n', html_content)
        html_content = re.sub(r'## (.*?)\n', r'<h2>\1</h2>\n', html_content)
        html_content = re.sub(r'# (.*?)\n', r'<h1>\1</h1>\n', html_content)
        html_content = html_content.replace('\n', '<br>\n')
        
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
        h1, h2, h3 {{ color: #2c3e50; }}
        .metadata {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .content {{ line-height: 1.6; }}
        .footer {{ margin-top: 30px; padding-top: 20px; border-top: 1px solid #dee2e6; font-size: 0.9em; color: #6c757d; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    
    <div class="metadata">
        <strong>Metadata:</strong><br>
        {self._format_metadata_html(metadata)}
    </div>
    
    <div class="content">
        {html_content}
    </div>
    
    <div class="footer">
        Generated by GOQII Health Data EDA Pipeline with LangChain Integration
    </div>
</body>
</html>
        """
    
    def _format_metadata_html(self, metadata: Dict) -> str:
        """Format metadata as HTML."""
        html_parts = []
        for key, value in metadata.items():
            if key not in ['summary']:
                html_parts.append(f"<strong>{key.replace('_', ' ').title()}:</strong> {value}")
        return "<br>".join(html_parts)


# Example usage and testing functions
def test_langchain_integration():
    """Test the LangChain integration with sample data."""
    
    # Check configuration
    config_status = config.validate_config()
    if not config_status['valid']:
        print("Configuration issues:", config_status['issues'])
        config.print_env_example()
        return False
    
    print("✅ Configuration validated")
    
    # Create sample data
    sample_data = pd.DataFrame({
        'participant_id': ['001'] * 10,
        'datetime_utc': pd.date_range('2024-01-01', periods=10, freq='1H'),
        'hr': [70, 75, 72, 78, 65, 80, 85, 70, 68, 72],
        'steps': [1200, 800, 2000, 1500, 500, 3000, 2500, 1000, 800, 1200],
        'quality_flag': ['normal'] * 10
    })
    
    try:
        analyzer = LangChainHealthAnalyzer()
        print("✅ LangChain analyzer initialized")
        
        # Test individual summary
        individual_summary = analyzer.generate_individual_summary(
            sample_data, '001', '2024-01-01'
        )
        print("✅ Individual summary generated")
        
        # Test saving
        analyzer.save_reports([individual_summary], {
            'cohort_size': 1,
            'analysis_date': datetime.now().isoformat(),
            'summary': 'Test cohort summary'
        })
        print("✅ Reports saved")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    test_langchain_integration()
