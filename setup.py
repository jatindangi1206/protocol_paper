#!/usr/bin/env python3
"""
Setup script for GOQII Health Data EDA Pipeline
This script helps set up the environment and run initial tests.
"""

import os
import sys
import subprocess
import pkg_resources
from pathlib import Path

def check_python_version():
    """Check if Python version is 3.8+"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        return False

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = [
        'pandas>=1.5.0',
        'numpy>=1.21.0', 
        'matplotlib>=3.5.0',
        'seaborn>=0.11.0',
        'scipy>=1.7.0',
        'statsmodels>=0.13.0'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            pkg_resources.require([package])
        except pkg_resources.DistributionNotFound:
            missing_packages.append(package)
        except pkg_resources.VersionConflict:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {missing_packages}")
        return False
    else:
        print("✅ All required packages are installed")
        return True

def verify_directory_structure():
    """Verify the project directory structure"""
    required_dirs = [
        'src',
        'data/input',
        'results'
    ]
    
    required_files = [
        'src/eda_pipeline.py',
        'src/data_preparation.py',
        'src/individual_analysis.py',
        'src/cohort_analysis.py',
        'src/reporting_visualization.py',
        'requirements.txt',
        'README.md'
    ]
    
    # Check directories
    for directory in required_dirs:
        if not os.path.exists(directory):
            print(f"❌ Missing directory: {directory}")
            return False
    
    # Check files
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ Missing file: {file_path}")
            return False
    
    print("✅ Project structure is correct")
    return True

def run_test_pipeline():
    """Run the pipeline with example data"""
    print("🧪 Running test pipeline with example data...")
    
    try:
        # Change to project directory
        os.chdir(Path(__file__).parent)
        
        # Run the pipeline
        result = subprocess.run([
            sys.executable, 
            "src/eda_pipeline.py", 
            "--input_dir", "data/input",
            "--output_dir", "results",
            "--log_level", "INFO"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Test pipeline completed successfully!")
            print("📁 Check the 'results' directory for outputs")
            return True
        else:
            print("❌ Test pipeline failed:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error running test pipeline: {e}")
        return False

def main():
    """Main setup function"""
    print("🔧 GOQII Health Data EDA Pipeline Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Verify directory structure
    if not verify_directory_structure():
        print("Please ensure all required files are present")
        return 1
    
    # Check if dependencies are already installed
    if not check_dependencies():
        # Try to install dependencies
        if not install_requirements():
            print("Please install dependencies manually: pip install -r requirements.txt")
            return 1
    
    # Run test with example data
    print("\n🚀 Setup complete! Running test pipeline...")
    if run_test_pipeline():
        print("\n" + "=" * 50)
        print("✅ SETUP SUCCESSFUL!")
        print("=" * 50)
        print("📖 Read README.md for detailed usage instructions")
        print("🔍 Example outputs are in the 'results' directory")
        print("\n📝 To run with your own data:")
        print("   1. Place your data files in data/input/")
        print("   2. Run: python src/eda_pipeline.py --input_dir data/input --output_dir results/")
        return 0
    else:
        print("\n❌ Setup completed but test failed. Check the error messages above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
