# GOQII Health Data Exploratory Data Analysis Protocol Paper

## Overview

This repository implements a comprehensive **Exploratory Data Analysis (EDA) pipeline** for GOQII health data as described in the protocol paper. The pipeline processes multiple health metrics including blood pressure, sleep patterns, step counts, heart rate, SpO₂, and temperature readings from wearable devices.

## Features

### 🔍 **Phase 1: Data Preparation**
- **Automatic file discovery** for CSV and JSON health data files
- **Data loading and standardization** with unified schema
- **Metric-specific cleaning rules** for each health parameter
- **Quality flag assignment** for data validation

### 📊 **Phase 2: Individual-Level Analysis**
- **Missing data analysis** with temporal gap detection
- **Compliance analysis** for user engagement metrics
- **Noise detection** and outlier identification
- **Trigger frequency analysis** for user-initiated measurements

### 🏥 **Phase 3: Cohort-Level Aggregation**
- **Cohort statistics** with quartile analysis
- **Cross-metric correlations** using Pearson and Spearman methods
- **Population trends** including weekly and seasonal patterns
- **Time-aligned dataset** creation for correlation analysis

### 📈 **Phase 4: Reporting & Visualization**
- **Individual participant plots** for each health metric
- **Cohort-level visualizations** and distribution analysis
- **Compliance heatmaps** and correlation matrices
- **Comprehensive summary reports**

### 🤖 **LangChain AI Integration** (NEW!)
- **Intelligent health summaries** using Large Language Models
- **Automated anomaly detection** with natural language explanations
- **Personalized health recommendations** for each participant
- **Data quality assessments** with actionable insights
- **Conversational querying** of historical health data
- **Vector database storage** for efficient report retrieval

## Project Structure

```
protocol_paper/
├── src/                          # Source code modules
│   ├── data_preparation.py       # Phase 1: Data loading and cleaning
│   ├── individual_analysis.py    # Phase 2: Individual-level analysis
│   ├── cohort_analysis.py        # Phase 3: Cohort aggregation
│   ├── reporting_visualization.py # Phase 4: Reporting and plots
│   ├── langchain_integration.py  # LangChain AI integration
│   └── eda_pipeline.py           # Main pipeline orchestrator
├── config.py                     # LangChain configuration
├── generate_weekly_reports.py    # AI-powered weekly reports generator
├── data/
│   └── input/                    # Place your health data files here
├── results/                      # Pipeline outputs
│   ├── cleaned/                  # Cleaned data per metric
│   ├── merged/                   # Time-aligned merged dataset
│   ├── summary/                  # Statistical summaries
│   ├── plots/
│   │   ├── individual/           # Per-participant visualizations
│   │   └── cohort/              # Cohort-level plots
│   └── logs/                    # Execution logs
├── reports/                     # AI-generated reports (NEW!)
│   ├── markdown/                # Markdown health summaries
│   ├── html/                    # HTML reports
│   ├── json/                    # JSON data for programmatic access
│   └── vector_store/            # FAISS vector database
├── requirements.txt             # Python dependencies
└── README.md                   # This file
```

## Supported Data Formats

### CSV Files
- **Blood Pressure** (`bp_*.csv`): `timestamp, systolic, diastolic, pulse`
- **Sleep** (`sleep_*.csv`): `timestamp, light_minutes, deep_minutes, rem_minutes, total_minutes`
- **Steps** (`steps_*.csv`): `timestamp, steps, distance_km, calories`

### JSON Files
- **Heart Rate** (`hr_*.json`): `{"jsonData": [{"timestamp": unix_time, "value": bpm}]}`
- **SpO₂** (`spo2_*.json`): `{"data": [{"timestamp": unix_time, "value": percentage, "status": "normal"}]}`
- **Temperature** (`temp_*.json`): `{"readings": [{"timestamp": unix_time, "value": fahrenheit}]}`

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone or download this repository**
```bash
git clone <repository-url>
cd protocol_paper
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Prepare your data**
   - Place your health data files in `data/input/`
   - Files should follow the naming convention: `{metric}_{participant_id}.{csv|json}`
   - Example: `bp_001.csv`, `hr_001.json`, `sleep_001.csv`

## Usage

### Basic Usage

Run the complete EDA pipeline:

```bash
python src/eda_pipeline.py --input_dir data/input --output_dir results/
```

### Advanced Options

```bash
# With debug logging
python src/eda_pipeline.py --input_dir data/input --output_dir results/ --log_level DEBUG

# Custom directories
python src/eda_pipeline.py --input_dir /path/to/your/data --output_dir /path/to/results

# View help
python src/eda_pipeline.py --help
```

### 🤖 AI-Powered Weekly Reports (NEW!)

Generate intelligent health summaries using LangChain integration:

```bash
# Basic usage with AI summaries
python generate_weekly_reports.py --input_dir data/input --output_dir results/

# Without LangChain (basic reports only)
python generate_weekly_reports.py --input_dir data/input --output_dir results/ --no-langchain

# Check configuration
python generate_weekly_reports.py --config-check

# Test LangChain integration
python generate_weekly_reports.py --test-langchain
```

#### Setting up LangChain Integration

1. **Install LangChain dependencies:**
```bash
pip install langchain openai faiss-cpu google-cloud-aiplatform anthropic
```

2. **Configure API keys:**
```bash
# For OpenAI (recommended)
export OPENAI_API_KEY="your_api_key_here"
export LLM_PROVIDER="openai"

# For Google Vertex AI
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
export LLM_PROVIDER="google"

# For Anthropic Claude
export ANTHROPIC_API_KEY="your_api_key_here"
export LLM_PROVIDER="anthropic"
```

3. **Optional: Enable vector database storage**
```bash
export ENABLE_VECTOR_STORE="true"
```

#### AI Features

- **Intelligent Summaries**: Natural language interpretation of health metrics
- **Anomaly Detection**: AI identifies unusual patterns with explanations
- **Personalized Recommendations**: Tailored advice based on individual data
- **Data Quality Ratings**: Automated assessment of data completeness
- **Conversational Queries**: Ask questions about historical health trends
- **Multi-format Output**: Markdown, HTML, and JSON reports

### Example Data

The repository includes example data files to test the pipeline:
- `data/input/bp_001.csv` - Sample blood pressure data
- `data/input/hr_001.json` - Sample heart rate data
- `data/input/sleep_001.csv` - Sample sleep data
- `data/input/steps_001.csv` - Sample step count data
- `data/input/spo2_001.json` - Sample SpO₂ data
- `data/input/temp_001.json` - Sample temperature data

## Output Description

### Cleaned Data (`results/cleaned/`)
- `bp.csv` - Cleaned blood pressure readings
- `hr.csv` - Cleaned heart rate measurements
- `sleep.csv` - Cleaned sleep session data
- `steps.csv` - Cleaned daily step counts
- `spo2.csv` - Cleaned SpO₂ measurements
- `temp.csv` - Cleaned temperature readings

### Merged Dataset (`results/merged/`)
- `merged_metrics.csv` - Time-aligned dataset on 5-minute grid for correlation analysis

### Summary Reports (`results/summary/`)
- `cohort_statistics.csv` - Descriptive statistics per metric
- `compliance_analysis.csv` - Participant compliance scores
- `missing_data_analysis.csv` - Missing data patterns
- `cross_metric_correlations.csv` - Correlation analysis results
- `*_weekly_trends.csv` - Weekly trend analysis per metric
- `eda_summary_report.txt` - Comprehensive text summary

### Visualizations (`results/plots/`)

#### Individual Plots (`individual/`)
- `hr_participant_{id}.png` - Heart rate time series
- `temp_participant_{id}.png` - Temperature time series
- `spo2_participant_{id}.png` - SpO₂ time series
- `bp_participant_{id}.png` - Blood pressure scatter plot
- `sleep_participant_{id}.png` - Sleep stages pie chart
- `steps_participant_{id}.png` - Daily steps bar chart

#### Cohort Plots (`cohort/`)
- `{metric}_cohort_analysis.png` - Distribution and trend analysis
- `compliance_summary.png` - Compliance analysis visualizations
- `correlation_heatmap.png` - Cross-metric correlation matrix
- `weekly_trends.png` - Population-level weekly trends
- `missing_data_heatmap.png` - Missing data patterns
- `compliance_heatmap.png` - Participant compliance matrix

## Data Cleaning Rules

### Blood Pressure
- **Valid ranges**: Systolic 80-200 mmHg, Diastolic 50-130 mmHg, Pulse 40-150 bpm
- **Quality flags**: High systolic (>140), High diastolic (>90)

### Sleep
- **Requirements**: Complete sleep stage data (light, deep, REM, total)
- **Validation**: Sleep stages sum ≤ total time (10% tolerance)
- **Quality flags**: Short sleep (<3h), Long sleep (>10h)

### Steps
- **Valid ranges**: Non-negative values, <50,000 daily steps
- **Quality flags**: High activity (>20k steps), Low activity (<1k steps)

### Heart Rate
- **Valid range**: 30-220 bpm
- **Quality flags**: Low HR (<60), High HR (>100)

### SpO₂
- **Valid range**: 80-100%
- **Quality flags**: Low oxygen (<95%)

### Temperature
- **Valid range**: 90-105°F
- **Quality flags**: Fever (>100.4°F), Low temperature (<97°F)

## Analysis Methodology

### Missing Data Analysis
- Calculates missing data ratio vs. expected frequency per metric
- Identifies temporal gaps exceeding 2× expected interval
- Generates participant × metric missing data heatmap

### Compliance Analysis
- **User-initiated metrics** (BP, ECG, SpO₂): Daily usage compliance
- **Automatic metrics** (HR, Steps, Sleep, Temperature): Expected frequency compliance
- Identifies top performers and improvement targets by quartile

### Quality Assessment
- Statistical outlier detection using IQR method
- Metric-specific noise detection (HR variability, temperature spikes)
- Quality flag assignment for all data points

### Correlation Analysis
- Creates time-aligned dataset on 5-minute grid
- Calculates Pearson and Spearman correlations between metrics
- Tests for statistical significance (p < 0.05)

### Population Trends
- Weekly average trends with standard deviation
- Weekday vs. weekend comparison
- Monthly seasonal analysis (if data spans multiple months)

## Technical Requirements

### Dependencies
- **pandas** ≥1.5.0 - Data manipulation and analysis
- **numpy** ≥1.21.0 - Numerical computing
- **matplotlib** ≥3.5.0 - Basic plotting
- **seaborn** ≥0.11.0 - Statistical visualization
- **scipy** ≥1.7.0 - Statistical functions
- **statsmodels** ≥0.13.0 - Statistical modeling
- **missingno** ≥0.5.0 - Missing data visualization
- **plotly** ≥5.0.0 - Interactive visualizations (optional)

### LangChain Integration Dependencies (Optional)
- **langchain** ≥0.1.0 - LLM orchestration framework
- **openai** ≥1.0.0 - OpenAI API integration
- **google-cloud-aiplatform** ≥1.30.0 - Google Vertex AI (optional)
- **anthropic** ≥0.3.0 - Anthropic Claude API (optional)
- **faiss-cpu** ≥1.7.4 - Vector database for report storage
- **tiktoken** ≥0.5.0 - Token counting for LLM optimization
- **python-dotenv** ≥1.0.0 - Environment variable management
- **markdownify** ≥0.11.0 - HTML to Markdown conversion

### System Requirements
- **Memory**: 4GB+ RAM for medium datasets (recommended 8GB+)
- **Storage**: 1GB+ free space for outputs
- **Python**: 3.8+ (tested with 3.8, 3.9, 3.10, 3.11)

## Troubleshooting

### Common Issues

1. **No files found**
   - Verify files are in `data/input/` directory
   - Check file naming convention: `{metric}_{id}.{csv|json}`
   - Ensure file permissions allow reading

2. **Import errors**
   - Install all dependencies: `pip install -r requirements.txt`
   - Verify Python version ≥3.8

3. **Memory issues**
   - Process data in smaller batches
   - Increase system memory or use a machine with more RAM

4. **Missing plots**
   - Check that matplotlib backend supports file output
   - Verify write permissions in output directory

### Log Files
Detailed execution logs are saved in `results/logs/` with timestamps. Check these files for detailed error information.

## Customization

### Adding New Metrics
1. Add file pattern to `metric_patterns` in `data_preparation.py`
2. Implement cleaning function in `DataPreparation` class
3. Add visualization function in `ReportingVisualization` class
4. Update expected frequency in analysis modules

### Modifying Cleaning Rules
Edit the metric-specific cleaning functions in `data_preparation.py`:
- `clean_bp_data()`
- `clean_sleep_data()`
- `clean_steps_data()`
- `clean_hr_data()`
- `clean_spo2_data()`
- `clean_temp_data()`

### Custom Analysis
Add new analysis functions to the respective phase modules:
- **Phase 2**: `individual_analysis.py`
- **Phase 3**: `cohort_analysis.py`
- **Phase 4**: `reporting_visualization.py`

## Contributing

When contributing to this project:

1. Follow the existing code structure and naming conventions
2. Add comprehensive docstrings for new functions
3. Include error handling and logging
4. Update requirements.txt if adding new dependencies
5. Test with example data before submitting

## License

This project is part of the GOQII Health Data Exploratory Data Analysis Protocol Paper. Please refer to the associated research publication for usage guidelines and citation requirements.

## Citation

If you use this pipeline in your research, please cite:

```
GOQII Health Data Exploratory Data Analysis Protocol Paper
Jatin Dangi, 2025

```

## Support

For questions or issues:
1. Check the troubleshooting section above
2. Review log files in `results/logs/`
3. Create an issue with detailed error information and system specifications

---

**Last Updated**: November 2024  
**Pipeline Version**: 1.0  
**Compatible Python Versions**: 3.8+
