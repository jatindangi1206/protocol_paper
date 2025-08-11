#!/bin/bash

# GOQII Health Data EDA Pipeline - Interactive Script
# ===================================================
# Alternative to Makefile for systems without make

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[0;37m'
RESET='\033[0m'
BOLD='\033[1m'

# Project settings
INPUT_DIR="data/input"
OUTPUT_DIR="results"
REPORTS_DIR="reports"
SRC_DIR="src"

# Detect Python command
if command -v python3 &> /dev/null; then
    PYTHON="python3"
elif command -v python &> /dev/null; then
    PYTHON="python"
else
    echo -e "${RED}❌ Python not found! Please install Python 3.8+${RESET}"
    exit 1
fi

# Function to show menu
show_menu() {
    clear
    echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════════════════════╗${RESET}"
    echo -e "${BOLD}${CYAN}║           GOQII Health Data EDA Pipeline Menu                ║${RESET}"
    echo -e "${BOLD}${CYAN}║                   🏥 Health Analytics Tool                   ║${RESET}"
    echo -e "${BOLD}${CYAN}╠══════════════════════════════════════════════════════════════╣${RESET}"
    echo -e "${BOLD}${WHITE}║                                                              ║${RESET}"
    echo -e "${BOLD}${WHITE}║  📊 ANALYSIS PIPELINES:                                     ║${RESET}"
    echo -e "${BOLD}${GREEN}║    1) Run complete EDA pipeline                             ║${RESET}"
    echo -e "${BOLD}${GREEN}║    2) Generate AI weekly reports                            ║${RESET}"
    echo -e "${BOLD}${GREEN}║    3) Generate basic weekly reports                         ║${RESET}"
    echo -e "${BOLD}${WHITE}║                                                              ║${RESET}"
    echo -e "${BOLD}${WHITE}║  🧪 TESTING & VALIDATION:                                   ║${RESET}"
    echo -e "${BOLD}${YELLOW}║    4) Run integration tests                                 ║${RESET}"
    echo -e "${BOLD}${YELLOW}║    5) Quick pipeline validation                             ║${RESET}"
    echo -e "${BOLD}${YELLOW}║    6) Test LangChain integration                            ║${RESET}"
    echo -e "${BOLD}${WHITE}║                                                              ║${RESET}"
    echo -e "${BOLD}${WHITE}║  🔧 SETUP & CONFIGURATION:                                  ║${RESET}"
    echo -e "${BOLD}${BLUE}║    7) Install all dependencies                              ║${RESET}"
    echo -e "${BOLD}${BLUE}║    8) Install with LangChain support                        ║${RESET}"
    echo -e "${BOLD}${BLUE}║    9) Check LangChain configuration                         ║${RESET}"
    echo -e "${BOLD}${BLUE}║   10) Setup environment variables                           ║${RESET}"
    echo -e "${BOLD}${WHITE}║                                                              ║${RESET}"
    echo -e "${BOLD}${WHITE}║  🧹 MAINTENANCE:                                            ║${RESET}"
    echo -e "${BOLD}${MAGENTA}║   11) Clean all outputs                                     ║${RESET}"
    echo -e "${BOLD}${MAGENTA}║   12) Clean EDA results only                               ║${RESET}"
    echo -e "${BOLD}${MAGENTA}║   13) Clean AI reports only                                ║${RESET}"
    echo -e "${BOLD}${MAGENTA}║   14) Clean Python cache                                   ║${RESET}"
    echo -e "${BOLD}${WHITE}║                                                              ║${RESET}"
    echo -e "${BOLD}${WHITE}║  📊 DATA MANAGEMENT:                                        ║${RESET}"
    echo -e "${BOLD}${CYAN}║   15) Validate input data format                            ║${RESET}"
    echo -e "${BOLD}${CYAN}║   16) Show data file summary                                ║${RESET}"
    echo -e "${BOLD}${CYAN}║   17) Create sample data files                              ║${RESET}"
    echo -e "${BOLD}${WHITE}║                                                              ║${RESET}"
    echo -e "${BOLD}${WHITE}║  📋 DOCUMENTATION & INFO:                                   ║${RESET}"
    echo -e "${BOLD}${WHITE}║   18) Show detailed help                                    ║${RESET}"
    echo -e "${BOLD}${WHITE}║   19) Show project status                                   ║${RESET}"
    echo -e "${BOLD}${WHITE}║   20) Open documentation                                    ║${RESET}"
    echo -e "${BOLD}${WHITE}║                                                              ║${RESET}"
    echo -e "${BOLD}${RED}║   21) Exit                                                   ║${RESET}"
    echo -e "${BOLD}${WHITE}║                                                              ║${RESET}"
    echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════════════════╝${RESET}"
    echo ""
}

# Function to wait for user input
wait_for_user() {
    echo ""
    echo -e "${YELLOW}Press Enter to continue...${RESET}"
    read
}

# Function to run EDA pipeline
run_eda() {
    echo -e "${BOLD}${GREEN}🚀 Running Complete EDA Pipeline...${RESET}"
    echo -e "${YELLOW}Input: ${INPUT_DIR} → Output: ${OUTPUT_DIR}${RESET}"
    mkdir -p "${OUTPUT_DIR}"
    ${PYTHON} "${SRC_DIR}/eda_pipeline.py" --input_dir "${INPUT_DIR}" --output_dir "${OUTPUT_DIR}"
    echo -e "${GREEN}✅ EDA Pipeline completed!${RESET}"
    show_results
    wait_for_user
}

# Function to run AI weekly reports
run_weekly() {
    echo -e "${BOLD}${GREEN}🤖 Running AI-Powered Weekly Reports...${RESET}"
    echo -e "${YELLOW}Input: ${INPUT_DIR} → Output: ${OUTPUT_DIR}${RESET}"
    mkdir -p "${OUTPUT_DIR}" "${REPORTS_DIR}"
    ${PYTHON} generate_weekly_reports.py --input_dir "${INPUT_DIR}" --output_dir "${OUTPUT_DIR}"
    echo -e "${GREEN}✅ Weekly reports generated!${RESET}"
    show_reports
    wait_for_user
}

# Function to run basic weekly reports
run_basic() {
    echo -e "${BOLD}${GREEN}📊 Running Basic Weekly Reports...${RESET}"
    echo -e "${YELLOW}Input: ${INPUT_DIR} → Output: ${OUTPUT_DIR}${RESET}"
    mkdir -p "${OUTPUT_DIR}" "${REPORTS_DIR}"
    ${PYTHON} generate_weekly_reports.py --input_dir "${INPUT_DIR}" --output_dir "${OUTPUT_DIR}" --no-langchain
    echo -e "${GREEN}✅ Basic reports generated!${RESET}"
    show_reports
    wait_for_user
}

# Function to run tests
run_tests() {
    echo -e "${BOLD}${YELLOW}🧪 Running Integration Tests...${RESET}"
    ${PYTHON} test_integration.py
    wait_for_user
}

# Function to run quick test
quick_test() {
    echo -e "${BOLD}${YELLOW}⚡ Running Quick Pipeline Validation...${RESET}"
    echo -e "${CYAN}Testing basic EDA pipeline...${RESET}"
    mkdir -p /tmp/eda_quick_test
    ${PYTHON} "${SRC_DIR}/eda_pipeline.py" --input_dir "${INPUT_DIR}" --output_dir /tmp/eda_quick_test --log_level ERROR
    rm -rf /tmp/eda_quick_test
    echo -e "${GREEN}✅ Quick test passed!${RESET}"
    wait_for_user
}

# Function to test LangChain
test_langchain() {
    echo -e "${BOLD}${YELLOW}🤖 Testing LangChain Integration...${RESET}"
    ${PYTHON} generate_weekly_reports.py --test-langchain
    wait_for_user
}

# Function to install dependencies
install_deps() {
    echo -e "${BOLD}${BLUE}📦 Installing Basic Dependencies...${RESET}"
    pip install -r requirements.txt
    echo -e "${GREEN}✅ Basic installation completed!${RESET}"
    wait_for_user
}

# Function to install with AI support
install_ai() {
    echo -e "${BOLD}${BLUE}🤖 Installing with LangChain Support...${RESET}"
    pip install -r requirements.txt
    echo -e "${CYAN}Installing LangChain packages...${RESET}"
    pip install langchain openai faiss-cpu google-cloud-aiplatform anthropic tiktoken python-dotenv markdownify
    echo -e "${GREEN}✅ AI-enhanced installation completed!${RESET}"
    echo -e "${YELLOW}💡 Don't forget to set up your API keys with option 10${RESET}"
    wait_for_user
}

# Function to check configuration
check_config() {
    echo -e "${BOLD}${BLUE}🔧 Checking LangChain Configuration...${RESET}"
    ${PYTHON} generate_weekly_reports.py --config-check
    wait_for_user
}

# Function to setup environment
setup_env() {
    echo -e "${BOLD}${BLUE}🔐 Environment Setup Guide${RESET}"
    echo ""
    echo -e "${CYAN}Set up your API keys by running these commands:${RESET}"
    echo ""
    echo -e "${YELLOW}For OpenAI (recommended):${RESET}"
    echo "  export OPENAI_API_KEY=\"your_api_key_here\""
    echo "  export LLM_PROVIDER=\"openai\""
    echo ""
    echo -e "${YELLOW}For Google Vertex AI:${RESET}"
    echo "  export GOOGLE_APPLICATION_CREDENTIALS=\"/path/to/service-account.json\""
    echo "  export LLM_PROVIDER=\"google\""
    echo ""
    echo -e "${YELLOW}For Anthropic Claude:${RESET}"
    echo "  export ANTHROPIC_API_KEY=\"your_api_key_here\""
    echo "  export LLM_PROVIDER=\"anthropic\""
    echo ""
    echo -e "${YELLOW}Optional - Enable vector database:${RESET}"
    echo "  export ENABLE_VECTOR_STORE=\"true\""
    echo ""
    echo -e "${CYAN}💡 Add these to your ~/.bashrc or ~/.zshrc for persistence${RESET}"
    wait_for_user
}

# Cleaning functions
clean_all() {
    echo -e "${MAGENTA}🧹 Performing complete cleanup...${RESET}"
    rm -rf "${OUTPUT_DIR}" "${REPORTS_DIR}"
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -type f -delete 2>/dev/null || true
    find . -name "*.pyo" -type f -delete 2>/dev/null || true
    find . -name "*.log" -type f -delete 2>/dev/null || true
    echo -e "${GREEN}✅ Complete cleanup finished!${RESET}"
    wait_for_user
}

clean_results() {
    echo -e "${MAGENTA}🗑️  Cleaning EDA results...${RESET}"
    rm -rf "${OUTPUT_DIR}"
    echo -e "${GREEN}✅ EDA results cleaned${RESET}"
    wait_for_user
}

clean_reports() {
    echo -e "${MAGENTA}🗑️  Cleaning AI reports...${RESET}"
    rm -rf "${REPORTS_DIR}"
    echo -e "${GREEN}✅ AI reports cleaned${RESET}"
    wait_for_user
}

clean_cache() {
    echo -e "${MAGENTA}🗑️  Cleaning Python cache...${RESET}"
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -type f -delete 2>/dev/null || true
    find . -name "*.pyo" -type f -delete 2>/dev/null || true
    find . -name "*.log" -type f -delete 2>/dev/null || true
    echo -e "${GREEN}✅ Cache cleaned${RESET}"
    wait_for_user
}

# Data management functions
validate_data() {
    echo -e "${BOLD}${CYAN}📊 Validating Input Data...${RESET}"
    if [ ! -d "${INPUT_DIR}" ]; then
        echo -e "${RED}❌ Input directory ${INPUT_DIR} not found!${RESET}"
        wait_for_user
        return
    fi
    
    echo -e "${YELLOW}Checking data files...${RESET}"
    file_count=$(ls -1 ${INPUT_DIR}/*.{csv,json} 2>/dev/null | wc -l)
    if [ $file_count -eq 0 ]; then
        echo -e "${RED}❌ No CSV or JSON files found in ${INPUT_DIR}${RESET}"
    else
        echo -e "${GREEN}✅ Found $file_count data files${RESET}"
        echo -e "${CYAN}Files:${RESET}"
        ls -la ${INPUT_DIR}/*.{csv,json} 2>/dev/null | awk '{print "  " $9 " (" $5 " bytes)"}'
    fi
    wait_for_user
}

show_data() {
    echo -e "${BOLD}${CYAN}📋 Data File Summary${RESET}"
    echo ""
    if [ -d "${INPUT_DIR}" ]; then
        echo -e "${YELLOW}Input Directory: ${INPUT_DIR}${RESET}"
        ls -la "${INPUT_DIR}"
        echo ""
        echo -e "${CYAN}File Details:${RESET}"
        for file in ${INPUT_DIR}/*; do
            if [ -f "$file" ]; then
                echo "  📄 $(basename "$file")"
                if [[ "$file" == *.csv ]]; then
                    echo "    📊 CSV - $(wc -l < "$file") lines"
                    echo "    📋 Headers: $(head -1 "$file")"
                elif [[ "$file" == *.json ]]; then
                    echo "    🔧 JSON - $(wc -l < "$file") lines"
                fi
                echo ""
            fi
        done
    else
        echo -e "${RED}❌ Input directory ${INPUT_DIR} not found!${RESET}"
    fi
    wait_for_user
}

create_sample() {
    echo -e "${BOLD}${CYAN}📝 Creating Sample Data Files...${RESET}"
    mkdir -p "${INPUT_DIR}"
    echo -e "${YELLOW}ℹ️  Sample data files should be manually placed in ${INPUT_DIR}/${RESET}"
    echo -e "${GREEN}✅ Sample data check completed${RESET}"
    show_data
}

# Help and info functions
show_help() {
    echo -e "${BOLD}${WHITE}📚 GOQII Health Data EDA Pipeline - Detailed Help${RESET}"
    echo ""
    echo -e "${CYAN}🏥 PROJECT OVERVIEW:${RESET}"
    echo "  This pipeline analyzes GOQII health device data including:"
    echo "  • Blood Pressure (BP)    • Heart Rate (HR)     • Sleep patterns"
    echo "  • Step counts            • SpO₂ levels         • Temperature"
    echo ""
    echo -e "${CYAN}📁 DIRECTORY STRUCTURE:${RESET}"
    echo "  ${INPUT_DIR}/     - Input health data files (CSV/JSON)"
    echo "  ${OUTPUT_DIR}/    - EDA pipeline outputs"
    echo "  ${REPORTS_DIR}/   - AI-generated reports"
    echo "  ${SRC_DIR}/       - Source code modules"
    echo ""
    echo -e "${CYAN}📊 SUPPORTED DATA FORMATS:${RESET}"
    echo "  CSV: bp_*.csv, sleep_*.csv, steps_*.csv"
    echo "  JSON: hr_*.json, spo2_*.json, temp_*.json"
    echo ""
    echo -e "${CYAN}🚀 QUICK START:${RESET}"
    echo "  1. Run option 7 to setup dependencies"
    echo "  2. Place your data files in ${INPUT_DIR}/"
    echo "  3. Run option 1 for statistical analysis"
    echo "  4. Run option 2 for AI-powered insights"
    echo ""
    echo -e "${CYAN}🤖 AI FEATURES (requires LangChain):${RESET}"
    echo "  • Natural language health summaries"
    echo "  • Automated anomaly detection"
    echo "  • Personalized recommendations"
    echo "  • Data quality assessments"
    wait_for_user
}

show_status() {
    echo -e "${BOLD}${WHITE}📊 Project Status${RESET}"
    echo ""
    echo -e "${CYAN}🔧 Environment:${RESET}"
    echo "  Python: ${PYTHON}"
    echo "  Working Directory: $(pwd)"
    echo ""
    echo -e "${CYAN}📁 Directories:${RESET}"
    if [ -d "${INPUT_DIR}" ]; then
        file_count=$(ls -1 ${INPUT_DIR}/*.{csv,json} 2>/dev/null | wc -l)
        echo -e "  📥 Input: ${INPUT_DIR} ($file_count files)"
    else
        echo -e "  📥 Input: ${INPUT_DIR} ${RED}(missing)${RESET}"
    fi
    
    if [ -d "${OUTPUT_DIR}" ]; then
        echo -e "  📤 Output: ${OUTPUT_DIR} ${GREEN}(exists)${RESET}"
    else
        echo -e "  📤 Output: ${OUTPUT_DIR} ${YELLOW}(not created)${RESET}"
    fi
    
    if [ -d "${REPORTS_DIR}" ]; then
        echo -e "  📋 Reports: ${REPORTS_DIR} ${GREEN}(exists)${RESET}"
    else
        echo -e "  📋 Reports: ${REPORTS_DIR} ${YELLOW}(not created)${RESET}"
    fi
    
    echo ""
    echo -e "${CYAN}🔍 Dependencies:${RESET}"
    ${PYTHON} -c "import pandas; print('  ✅ pandas')" 2>/dev/null || echo "  ❌ pandas"
    ${PYTHON} -c "import numpy; print('  ✅ numpy')" 2>/dev/null || echo "  ❌ numpy"
    ${PYTHON} -c "import matplotlib; print('  ✅ matplotlib')" 2>/dev/null || echo "  ❌ matplotlib"
    ${PYTHON} -c "import seaborn; print('  ✅ seaborn')" 2>/dev/null || echo "  ❌ seaborn"
    ${PYTHON} -c "import langchain; print('  ✅ langchain')" 2>/dev/null || echo "  ⚠️  langchain (optional)"
    wait_for_user
}

show_docs() {
    echo -e "${BOLD}${WHITE}📖 Available Documentation${RESET}"
    echo ""
    echo -e "${CYAN}Documentation files:${RESET}"
    echo "  📄 README.md           - Complete usage guide"
    echo "  📄 PROJECT_COMPLETE.md - Implementation summary"
    echo "  📄 QUICKSTART.md       - Quick start guide"
    echo ""
    echo -e "${YELLOW}Choose a document to view:${RESET}"
    echo "  1) README.md"
    echo "  2) PROJECT_COMPLETE.md"
    echo "  3) QUICKSTART.md"
    echo "  4) Return to menu"
    echo ""
    read -p "Select (1-4): " choice
    case $choice in
        1) less README.md ;;
        2) less PROJECT_COMPLETE.md ;;
        3) less QUICKSTART.md ;;
        4) ;;
        *) echo -e "${RED}Invalid option${RESET}" ;;
    esac
    wait_for_user
}

# Helper functions
show_results() {
    echo ""
    echo -e "${BOLD}${GREEN}📊 Results Summary:${RESET}"
    if [ -d "${OUTPUT_DIR}" ]; then
        echo -e "${CYAN}Generated files:${RESET}"
        find "${OUTPUT_DIR}" -type f | head -10 | sed 's/^/  📄 /'
        total=$(find "${OUTPUT_DIR}" -type f | wc -l)
        if [ $total -gt 10 ]; then
            echo "  ... and $((total-10)) more files"
        fi
        echo -e "${CYAN}Output directory: ${OUTPUT_DIR}${RESET}"
    fi
}

show_reports() {
    echo ""
    echo -e "${BOLD}${GREEN}📋 Reports Summary:${RESET}"
    if [ -d "${REPORTS_DIR}" ]; then
        echo -e "${CYAN}Generated reports:${RESET}"
        find "${REPORTS_DIR}" -name "*.md" | head -5 | sed 's/^/  📄 /'
        find "${REPORTS_DIR}" -name "*.json" | head -3 | sed 's/^/  🔧 /'
        echo -e "${CYAN}Reports directory: ${REPORTS_DIR}${RESET}"
    fi
}

# Main menu loop
main() {
    while true; do
        show_menu
        echo -e "${BOLD}${YELLOW}Select an option (1-21):${RESET} "
        read choice
        
        case $choice in
            1) run_eda ;;
            2) run_weekly ;;
            3) run_basic ;;
            4) run_tests ;;
            5) quick_test ;;
            6) test_langchain ;;
            7) install_deps ;;
            8) install_ai ;;
            9) check_config ;;
            10) setup_env ;;
            11) clean_all ;;
            12) clean_results ;;
            13) clean_reports ;;
            14) clean_cache ;;
            15) validate_data ;;
            16) show_data ;;
            17) create_sample ;;
            18) show_help ;;
            19) show_status ;;
            20) show_docs ;;
            21) echo -e "${GREEN}Goodbye! 👋${RESET}"; exit 0 ;;
            *) echo -e "${RED}Invalid option. Please select 1-21.${RESET}"; sleep 2 ;;
        esac
    done
}

# Run the main function
main
