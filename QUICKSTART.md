# GOQII Health Data EDA Pipeline - Quick Start Guide

## 🚀 Quick Setup (5 minutes)

### 1. Prerequisites
- Python 3.8 or higher
- pip package manager

### 2. Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scipy statsmodels
```

### 3. Run with Example Data
```bash
python3 src/eda_pipeline.py --input_dir data/input --output_dir results
```

### 4. Check Results
Results will be generated in the `results/` directory:
- **Cleaned data**: `results/cleaned/*.csv`
- **Summary reports**: `results/summary/*.csv`
- **Visualizations**: `results/plots/`

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
