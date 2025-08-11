# 🚀 GOQII Health Data EDA Pipeline - Quick Start Guide

## � Interactive Menu System (NEW!)

This project now includes an **interactive menu system** that makes it super easy to run all pipeline operations! You have two options:

### Option 1: Makefile (Recommended)
```bash
make
```

### Option 2: Shell Script (Alternative)
```bash
./run.sh
```

Both will show you this beautiful interactive menu:

```
╔══════════════════════════════════════════════════════════════╗
║           GOQII Health Data EDA Pipeline Menu                ║
║                   🏥 Health Analytics Tool                   ║
╠══════════════════════════════════════════════════════════════╣
║  📊 ANALYSIS PIPELINES:                                     ║
║    1) Run complete EDA pipeline                             ║
║    2) Generate AI weekly reports                            ║
║    3) Generate basic weekly reports                         ║
║  🧪 TESTING & VALIDATION:                                   ║
║    4) Run integration tests                                 ║
║    5) Quick pipeline validation                             ║
║  🔧 SETUP & CONFIGURATION:                                  ║
║    7) Install all dependencies                              ║
║    8) Install with LangChain support                        ║
║  🧹 MAINTENANCE:                                            ║
║   11) Clean all outputs                                     ║
║  📊 DATA MANAGEMENT:                                        ║
║   15) Validate input data format                           ║
║   16) Show data file summary                               ║
╚══════════════════════════════════════════════════════════════╝
```

## 🎯 Quick Start Steps

### 1. First Time Setup
```bash
# Start the interactive menu
make

# Select option 7: Install all dependencies
# This installs pandas, numpy, matplotlib, seaborn, etc.
```

### 2. For AI Features (Optional)
```bash
# In the menu, select option 8: Install with LangChain support
# Then select option 10: Setup environment variables
# Follow the guide to set up your OpenAI/Google/Anthropic API keys
```

### 3. Prepare Your Data
```bash
# Select option 16: Show data file summary
# This shows you what data files are available

# Place your health data files in data/input/
# Supported formats:
#   - CSV: bp_*.csv, sleep_*.csv, steps_*.csv  
#   - JSON: hr_*.json, spo2_*.json, temp_*.json
```

### 4. Run Analysis
```bash
# Option 1: Complete statistical analysis
# Select option 1: Run complete EDA pipeline

# Option 2: AI-powered weekly reports  
# Select option 2: Generate AI weekly reports

# Option 3: Basic weekly reports (no AI)
# Select option 3: Generate basic weekly reports
```

### 5. View Results
The menu automatically shows you what was generated:
- **EDA Results**: `results/` directory with plots and statistics
- **AI Reports**: `reports/` directory with markdown and HTML summaries

## 🛠️ Menu Categories

### 📊 Analysis Pipelines (Options 1-3)
- **Complete EDA**: Full 4-phase statistical analysis
- **AI Weekly Reports**: LangChain-powered intelligent summaries  
- **Basic Weekly Reports**: Simple weekly aggregations

### 🧪 Testing & Validation (Options 4-6)
- **Integration Tests**: Full test suite
- **Quick Validation**: Fast pipeline check
- **LangChain Test**: AI integration verification

### 🔧 Setup & Configuration (Options 7-10)
- **Install Dependencies**: Basic Python packages
- **Install AI Support**: LangChain and LLM packages
- **Check Config**: Verify API key setup
- **Setup Environment**: Guide for API key configuration

### 🧹 Maintenance (Options 11-14)
- **Clean All**: Remove all outputs
- **Clean Results**: Remove EDA outputs only
- **Clean Reports**: Remove AI reports only  
- **Clean Cache**: Remove Python cache files

### 📊 Data Management (Options 15-17)
- **Validate Data**: Check input file formats
- **Show Data**: Display data file summary
- **Create Sample**: Generate example data files

### 📋 Documentation & Info (Options 18-20)
- **Help**: Detailed usage information
- **Status**: Project and dependency status
- **Docs**: Open documentation files

## 💡 Smart Features

### ✨ Auto-Detection
- **Python Version**: Automatically uses `python3` or `python`
- **Dependencies**: Shows what's installed vs missing
- **Data Files**: Automatically discovers CSV/JSON files

### 🎨 Beautiful Output
- **Color-coded**: Different colors for different types of operations
- **Progress Indicators**: Real-time feedback during operations
- **Result Summaries**: Shows what files were generated

### 🔄 Convenience Features
- **Return to Menu**: After each operation, automatically returns to menu
- **Error Handling**: Graceful handling of missing dependencies
- **File Counting**: Shows how many files were processed/generated

## 📁 Your Data Format

Place your files in `data/input/` following this naming pattern:

### CSV Files
- `bp_001.csv` → Blood pressure: `timestamp,systolic,diastolic,pulse`
- `sleep_001.csv` → Sleep: `timestamp,light_minutes,deep_minutes,rem_minutes,total_minutes`
- `steps_001.csv` → Steps: `timestamp,steps,distance_km,calories`

### JSON Files
- `hr_001.json` → Heart rate: `{"jsonData": [{"timestamp": unix_time, "value": bpm}]}`
- `spo2_001.json` → SpO₂: `{"data": [{"timestamp": unix_time, "value": percent, "status": "normal"}]}`
- `temp_001.json` → Temperature: `{"readings": [{"timestamp": unix_time, "value": fahrenheit}]}`

## 🔍 What You Get

### Individual Analysis
- Missing data patterns per participant
- Compliance scores for each metric
- Quality flags for outliers and noise
- Individual time-series plots

### Cohort Analysis  
- Descriptive statistics across all participants
- Cross-metric correlations
- Weekly and seasonal trends
- Population-level visualizations

### Reports
- `eda_summary_report.txt` - Comprehensive findings
- CSV files with detailed statistics
- PNG plots for presentations

## 🛠 Customization

### Change Data Cleaning Rules
Edit the cleaning functions in `src/data_preparation.py`:
- `clean_bp_data()` - Blood pressure thresholds
- `clean_hr_data()` - Heart rate ranges
- `clean_sleep_data()` - Sleep validation rules

### Add New Metrics
1. Add filename pattern to `metric_patterns` in `DataPreparation`
2. Implement cleaning function
3. Add visualization in `ReportingVisualization`

### Modify Analysis
- **Individual level**: Edit `src/individual_analysis.py`
- **Cohort level**: Edit `src/cohort_analysis.py`
- **Visualizations**: Edit `src/reporting_visualization.py`

## 📞 Support

- Check `results/logs/` for detailed execution logs
- See `README.md` for full documentation
- Example data included for testing

---

**Ready to analyze your health data?** Run the pipeline and explore the results! 🏥📊
