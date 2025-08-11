# GOQII Health Data EDA Pipeline - Interactive Makefile
# =====================================================

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
MAGENTA := \033[0;35m
CYAN := \033[0;36m
WHITE := \033[0;37m
RESET := \033[0m
BOLD := \033[1m

# Project directories
INPUT_DIR := data/input
OUTPUT_DIR := results
REPORTS_DIR := reports
SRC_DIR := src

# Python command (try python3 first, fallback to python)
PYTHON := $(shell command -v python3 2> /dev/null || echo python)

# Default target - show interactive menu
.PHONY: menu
menu:
	@echo "$(BOLD)$(CYAN)╔══════════════════════════════════════════════════════════════╗$(RESET)"
	@echo "$(BOLD)$(CYAN)║           GOQII Health Data EDA Pipeline Menu                ║$(RESET)"
	@echo "$(BOLD)$(CYAN)║                   🏥 Health Analytics Tool                   ║$(RESET)"
	@echo "$(BOLD)$(CYAN)╠══════════════════════════════════════════════════════════════╣$(RESET)"
	@echo "$(BOLD)$(WHITE)║                                                              ║$(RESET)"
	@echo "$(BOLD)$(WHITE)║  📊 ANALYSIS PIPELINES:                                     ║$(RESET)"
	@echo "$(BOLD)$(GREEN)║    1) run-eda          - Run complete EDA pipeline          ║$(RESET)"
	@echo "$(BOLD)$(GREEN)║    2) run-weekly       - Generate AI weekly reports         ║$(RESET)"
	@echo "$(BOLD)$(GREEN)║    3) run-basic        - Generate basic weekly reports      ║$(RESET)"
	@echo "$(BOLD)$(WHITE)║                                                              ║$(RESET)"
	@echo "$(BOLD)$(WHITE)║  🧪 TESTING & VALIDATION:                                   ║$(RESET)"
	@echo "$(BOLD)$(YELLOW)║    4) test             - Run integration tests              ║$(RESET)"
	@echo "$(BOLD)$(YELLOW)║    5) test-quick       - Quick pipeline validation          ║$(RESET)"
	@echo "$(BOLD)$(YELLOW)║    6) test-langchain   - Test LangChain integration         ║$(RESET)"
	@echo "$(BOLD)$(WHITE)║                                                              ║$(RESET)"
	@echo "$(BOLD)$(WHITE)║  🔧 SETUP & CONFIGURATION:                                  ║$(RESET)"
	@echo "$(BOLD)$(BLUE)║    7) install          - Install all dependencies           ║$(RESET)"
	@echo "$(BOLD)$(BLUE)║    8) install-ai       - Install with LangChain support     ║$(RESET)"
	@echo "$(BOLD)$(BLUE)║    9) check-config     - Check LangChain configuration      ║$(RESET)"
	@echo "$(BOLD)$(BLUE)║   10) setup-env        - Setup environment variables        ║$(RESET)"
	@echo "$(BOLD)$(WHITE)║                                                              ║$(RESET)"
	@echo "$(BOLD)$(WHITE)║  🧹 MAINTENANCE:                                            ║$(RESET)"
	@echo "$(BOLD)$(MAGENTA)║   11) clean            - Clean all outputs                  ║$(RESET)"
	@echo "$(BOLD)$(MAGENTA)║   12) clean-results    - Clean EDA results only            ║$(RESET)"
	@echo "$(BOLD)$(MAGENTA)║   13) clean-reports    - Clean AI reports only             ║$(RESET)"
	@echo "$(BOLD)$(MAGENTA)║   14) clean-cache      - Clean Python cache                ║$(RESET)"
	@echo "$(BOLD)$(WHITE)║                                                              ║$(RESET)"
	@echo "$(BOLD)$(WHITE)║  📊 DATA MANAGEMENT:                                        ║$(RESET)"
	@echo "$(BOLD)$(CYAN)║   15) validate-data    - Validate input data format         ║$(RESET)"
	@echo "$(BOLD)$(CYAN)║   16) show-data        - Show data file summary             ║$(RESET)"
	@echo "$(BOLD)$(CYAN)║   17) create-sample    - Create sample data files           ║$(RESET)"
	@echo "$(BOLD)$(WHITE)║                                                              ║$(RESET)"
	@echo "$(BOLD)$(WHITE)║  📋 DOCUMENTATION & INFO:                                   ║$(RESET)"
	@echo "$(BOLD)$(WHITE)║   18) help             - Show detailed help                 ║$(RESET)"
	@echo "$(BOLD)$(WHITE)║   19) status           - Show project status                ║$(RESET)"
	@echo "$(BOLD)$(WHITE)║   20) docs             - Open documentation                 ║$(RESET)"
	@echo "$(BOLD)$(WHITE)║                                                              ║$(RESET)"
	@echo "$(BOLD)$(RED)║   21) exit             - Exit menu                          ║$(RESET)"
	@echo "$(BOLD)$(WHITE)║                                                              ║$(RESET)"
	@echo "$(BOLD)$(CYAN)╚══════════════════════════════════════════════════════════════╝$(RESET)"
	@echo ""
	@echo "$(BOLD)$(YELLOW)Select an option (1-21):$(RESET) "
	@read choice; \
	case $$choice in \
		1) $(MAKE) run-eda ;; \
		2) $(MAKE) run-weekly ;; \
		3) $(MAKE) run-basic ;; \
		4) $(MAKE) test ;; \
		5) $(MAKE) test-quick ;; \
		6) $(MAKE) test-langchain ;; \
		7) $(MAKE) install ;; \
		8) $(MAKE) install-ai ;; \
		9) $(MAKE) check-config ;; \
		10) $(MAKE) setup-env ;; \
		11) $(MAKE) clean ;; \
		12) $(MAKE) clean-results ;; \
		13) $(MAKE) clean-reports ;; \
		14) $(MAKE) clean-cache ;; \
		15) $(MAKE) validate-data ;; \
		16) $(MAKE) show-data ;; \
		17) $(MAKE) create-sample ;; \
		18) $(MAKE) help ;; \
		19) $(MAKE) status ;; \
		20) $(MAKE) docs ;; \
		21) echo "$(GREEN)Goodbye! 👋$(RESET)"; exit 0 ;; \
		*) echo "$(RED)Invalid option. Please select 1-21.$(RESET)"; $(MAKE) menu ;; \
	esac

# ============================================================================
# ANALYSIS PIPELINES
# ============================================================================

.PHONY: run-eda
run-eda:
	@echo "$(BOLD)$(GREEN)🚀 Running Complete EDA Pipeline...$(RESET)"
	@echo "$(YELLOW)Input: $(INPUT_DIR) → Output: $(OUTPUT_DIR)$(RESET)"
	@mkdir -p $(OUTPUT_DIR)
	$(PYTHON) $(SRC_DIR)/eda_pipeline.py --input_dir $(INPUT_DIR) --output_dir $(OUTPUT_DIR)
	@echo "$(GREEN)✅ EDA Pipeline completed!$(RESET)"
	@$(MAKE) _show_results
	@$(MAKE) _return_to_menu

.PHONY: run-weekly
run-weekly:
	@echo "$(BOLD)$(GREEN)🤖 Running AI-Powered Weekly Reports...$(RESET)"
	@echo "$(YELLOW)Input: $(INPUT_DIR) → Output: $(OUTPUT_DIR)$(RESET)"
	@mkdir -p $(OUTPUT_DIR) $(REPORTS_DIR)
	$(PYTHON) generate_weekly_reports.py --input_dir $(INPUT_DIR) --output_dir $(OUTPUT_DIR)
	@echo "$(GREEN)✅ Weekly reports generated!$(RESET)"
	@$(MAKE) _show_reports
	@$(MAKE) _return_to_menu

.PHONY: run-basic
run-basic:
	@echo "$(BOLD)$(GREEN)📊 Running Basic Weekly Reports...$(RESET)"
	@echo "$(YELLOW)Input: $(INPUT_DIR) → Output: $(OUTPUT_DIR)$(RESET)"
	@mkdir -p $(OUTPUT_DIR) $(REPORTS_DIR)
	$(PYTHON) generate_weekly_reports.py --input_dir $(INPUT_DIR) --output_dir $(OUTPUT_DIR) --no-langchain
	@echo "$(GREEN)✅ Basic reports generated!$(RESET)"
	@$(MAKE) _show_reports
	@$(MAKE) _return_to_menu

# ============================================================================
# TESTING & VALIDATION
# ============================================================================

.PHONY: test
test:
	@echo "$(BOLD)$(YELLOW)🧪 Running Integration Tests...$(RESET)"
	$(PYTHON) test_integration.py
	@$(MAKE) _return_to_menu

.PHONY: test-quick
test-quick:
	@echo "$(BOLD)$(YELLOW)⚡ Running Quick Pipeline Validation...$(RESET)"
	@echo "$(CYAN)Testing basic EDA pipeline...$(RESET)"
	@mkdir -p /tmp/eda_quick_test
	$(PYTHON) $(SRC_DIR)/eda_pipeline.py --input_dir $(INPUT_DIR) --output_dir /tmp/eda_quick_test --log_level ERROR
	@rm -rf /tmp/eda_quick_test
	@echo "$(GREEN)✅ Quick test passed!$(RESET)"
	@$(MAKE) _return_to_menu

.PHONY: test-langchain
test-langchain:
	@echo "$(BOLD)$(YELLOW)🤖 Testing LangChain Integration...$(RESET)"
	$(PYTHON) generate_weekly_reports.py --test-langchain
	@$(MAKE) _return_to_menu

# ============================================================================
# SETUP & CONFIGURATION
# ============================================================================

.PHONY: install
install:
	@echo "$(BOLD)$(BLUE)📦 Installing Basic Dependencies...$(RESET)"
	pip install -r requirements.txt
	@echo "$(GREEN)✅ Basic installation completed!$(RESET)"
	@$(MAKE) _return_to_menu

.PHONY: install-ai
install-ai:
	@echo "$(BOLD)$(BLUE)🤖 Installing with LangChain Support...$(RESET)"
	pip install -r requirements.txt
	@echo "$(CYAN)Installing LangChain packages...$(RESET)"
	pip install langchain openai faiss-cpu google-cloud-aiplatform anthropic tiktoken python-dotenv markdownify
	@echo "$(GREEN)✅ AI-enhanced installation completed!$(RESET)"
	@echo "$(YELLOW)💡 Don't forget to set up your API keys with 'make setup-env'$(RESET)"
	@$(MAKE) _return_to_menu

.PHONY: check-config
check-config:
	@echo "$(BOLD)$(BLUE)🔧 Checking LangChain Configuration...$(RESET)"
	$(PYTHON) generate_weekly_reports.py --config-check
	@$(MAKE) _return_to_menu

.PHONY: setup-env
setup-env:
	@echo "$(BOLD)$(BLUE)🔐 Environment Setup Guide$(RESET)"
	@echo ""
	@echo "$(CYAN)Set up your API keys by running these commands:$(RESET)"
	@echo ""
	@echo "$(YELLOW)For OpenAI (recommended):$(RESET)"
	@echo "  export OPENAI_API_KEY=\"your_api_key_here\""
	@echo "  export LLM_PROVIDER=\"openai\""
	@echo ""
	@echo "$(YELLOW)For Google Vertex AI:$(RESET)"
	@echo "  export GOOGLE_APPLICATION_CREDENTIALS=\"/path/to/service-account.json\""
	@echo "  export LLM_PROVIDER=\"google\""
	@echo ""
	@echo "$(YELLOW)For Anthropic Claude:$(RESET)"
	@echo "  export ANTHROPIC_API_KEY=\"your_api_key_here\""
	@echo "  export LLM_PROVIDER=\"anthropic\""
	@echo ""
	@echo "$(YELLOW)Optional - Enable vector database:$(RESET)"
	@echo "  export ENABLE_VECTOR_STORE=\"true\""
	@echo ""
	@echo "$(CYAN)💡 Add these to your ~/.bashrc or ~/.zshrc for persistence$(RESET)"
	@$(MAKE) _return_to_menu

# ============================================================================
# MAINTENANCE
# ============================================================================

.PHONY: clean
clean: clean-results clean-reports clean-cache
	@echo "$(BOLD)$(MAGENTA)🧹 Complete cleanup finished!$(RESET)"
	@$(MAKE) _return_to_menu

.PHONY: clean-results
clean-results:
	@echo "$(MAGENTA)🗑️  Cleaning EDA results...$(RESET)"
	@rm -rf $(OUTPUT_DIR)
	@echo "$(GREEN)✅ EDA results cleaned$(RESET)"

.PHONY: clean-reports
clean-reports:
	@echo "$(MAGENTA)🗑️  Cleaning AI reports...$(RESET)"
	@rm -rf $(REPORTS_DIR)
	@echo "$(GREEN)✅ AI reports cleaned$(RESET)"

.PHONY: clean-cache
clean-cache:
	@echo "$(MAGENTA)🗑️  Cleaning Python cache...$(RESET)"
	@find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.pyc" -type f -delete 2>/dev/null || true
	@find . -name "*.pyo" -type f -delete 2>/dev/null || true
	@find . -name "*.log" -type f -delete 2>/dev/null || true
	@echo "$(GREEN)✅ Cache cleaned$(RESET)"

# ============================================================================
# DATA MANAGEMENT
# ============================================================================

.PHONY: validate-data
validate-data:
	@echo "$(BOLD)$(CYAN)📊 Validating Input Data...$(RESET)"
	@if [ ! -d "$(INPUT_DIR)" ]; then \
		echo "$(RED)❌ Input directory $(INPUT_DIR) not found!$(RESET)"; \
		$(MAKE) _return_to_menu; \
	fi
	@echo "$(YELLOW)Checking data files...$(RESET)"
	@file_count=$$(ls -1 $(INPUT_DIR)/*.{csv,json} 2>/dev/null | wc -l); \
	if [ $$file_count -eq 0 ]; then \
		echo "$(RED)❌ No CSV or JSON files found in $(INPUT_DIR)$(RESET)"; \
	else \
		echo "$(GREEN)✅ Found $$file_count data files$(RESET)"; \
		echo "$(CYAN)Files:$(RESET)"; \
		ls -la $(INPUT_DIR)/*.{csv,json} 2>/dev/null | awk '{print "  " $$9 " (" $$5 " bytes)"}'; \
	fi
	@$(MAKE) _return_to_menu

.PHONY: show-data
show-data:
	@echo "$(BOLD)$(CYAN)📋 Data File Summary$(RESET)"
	@echo ""
	@if [ -d "$(INPUT_DIR)" ]; then \
		echo "$(YELLOW)Input Directory: $(INPUT_DIR)$(RESET)"; \
		ls -la $(INPUT_DIR); \
		echo ""; \
		echo "$(CYAN)File Details:$(RESET)"; \
		for file in $(INPUT_DIR)/*; do \
			if [ -f "$$file" ]; then \
				echo "  📄 $$(basename $$file)"; \
				if [[ "$$file" == *.csv ]]; then \
					echo "    📊 CSV - $$(wc -l < "$$file") lines"; \
					echo "    📋 Headers: $$(head -1 "$$file")"; \
				elif [[ "$$file" == *.json ]]; then \
					echo "    🔧 JSON - $$(wc -l < "$$file") lines"; \
				fi; \
				echo ""; \
			fi; \
		done; \
	else \
		echo "$(RED)❌ Input directory $(INPUT_DIR) not found!$(RESET)"; \
	fi
	@$(MAKE) _return_to_menu

.PHONY: create-sample
create-sample:
	@echo "$(BOLD)$(CYAN)📝 Creating Sample Data Files...$(RESET)"
	@mkdir -p $(INPUT_DIR)
	$(PYTHON) -c "import sys; sys.path.append('src'); from data_preparation import DataPreparation; dp = DataPreparation('', ''); dp.create_sample_data('$(INPUT_DIR)')" 2>/dev/null || \
	echo "$(YELLOW)ℹ️  Sample data already exists or creation not implemented$(RESET)"
	@echo "$(GREEN)✅ Sample data check completed$(RESET)"
	@$(MAKE) show-data

# ============================================================================
# DOCUMENTATION & INFO
# ============================================================================

.PHONY: help
help:
	@echo "$(BOLD)$(WHITE)📚 GOQII Health Data EDA Pipeline - Detailed Help$(RESET)"
	@echo ""
	@echo "$(CYAN)🏥 PROJECT OVERVIEW:$(RESET)"
	@echo "  This pipeline analyzes GOQII health device data including:"
	@echo "  • Blood Pressure (BP)    • Heart Rate (HR)     • Sleep patterns"
	@echo "  • Step counts            • SpO₂ levels         • Temperature"
	@echo ""
	@echo "$(CYAN)📁 DIRECTORY STRUCTURE:$(RESET)"
	@echo "  $(INPUT_DIR)/     - Input health data files (CSV/JSON)"
	@echo "  $(OUTPUT_DIR)/    - EDA pipeline outputs"
	@echo "  $(REPORTS_DIR)/   - AI-generated reports"
	@echo "  $(SRC_DIR)/       - Source code modules"
	@echo ""
	@echo "$(CYAN)📊 SUPPORTED DATA FORMATS:$(RESET)"
	@echo "  CSV: bp_*.csv, sleep_*.csv, steps_*.csv"
	@echo "  JSON: hr_*.json, spo2_*.json, temp_*.json"
	@echo ""
	@echo "$(CYAN)🚀 QUICK START:$(RESET)"
	@echo "  1. Run 'make install' to setup dependencies"
	@echo "  2. Place your data files in $(INPUT_DIR)/"
	@echo "  3. Run 'make run-eda' for statistical analysis"
	@echo "  4. Run 'make run-weekly' for AI-powered insights"
	@echo ""
	@echo "$(CYAN)🤖 AI FEATURES (requires LangChain):$(RESET)"
	@echo "  • Natural language health summaries"
	@echo "  • Automated anomaly detection"
	@echo "  • Personalized recommendations"
	@echo "  • Data quality assessments"
	@echo ""
	@echo "$(CYAN)📖 DOCUMENTATION:$(RESET)"
	@echo "  README.md           - Complete usage guide"
	@echo "  PROJECT_COMPLETE.md - Implementation summary"
	@echo "  QUICKSTART.md       - Quick start guide"
	@echo ""
	@$(MAKE) _return_to_menu

.PHONY: status
status:
	@echo "$(BOLD)$(WHITE)📊 Project Status$(RESET)"
	@echo ""
	@echo "$(CYAN)🔧 Environment:$(RESET)"
	@echo "  Python: $(PYTHON)"
	@echo "  Working Directory: $$(pwd)"
	@echo ""
	@echo "$(CYAN)📁 Directories:$(RESET)"
	@if [ -d "$(INPUT_DIR)" ]; then \
		file_count=$$(ls -1 $(INPUT_DIR)/*.{csv,json} 2>/dev/null | wc -l); \
		echo "  📥 Input: $(INPUT_DIR) ($$file_count files)"; \
	else \
		echo "  📥 Input: $(INPUT_DIR) $(RED)(missing)$(RESET)"; \
	fi
	@if [ -d "$(OUTPUT_DIR)" ]; then \
		echo "  📤 Output: $(OUTPUT_DIR) $(GREEN)(exists)$(RESET)"; \
	else \
		echo "  📤 Output: $(OUTPUT_DIR) $(YELLOW)(not created)$(RESET)"; \
	fi
	@if [ -d "$(REPORTS_DIR)" ]; then \
		echo "  📋 Reports: $(REPORTS_DIR) $(GREEN)(exists)$(RESET)"; \
	else \
		echo "  📋 Reports: $(REPORTS_DIR) $(YELLOW)(not created)$(RESET)"; \
	fi
	@echo ""
	@echo "$(CYAN)🔍 Dependencies:$(RESET)"
	@$(PYTHON) -c "import pandas; print('  ✅ pandas')" 2>/dev/null || echo "  ❌ pandas"
	@$(PYTHON) -c "import numpy; print('  ✅ numpy')" 2>/dev/null || echo "  ❌ numpy"
	@$(PYTHON) -c "import matplotlib; print('  ✅ matplotlib')" 2>/dev/null || echo "  ❌ matplotlib"
	@$(PYTHON) -c "import seaborn; print('  ✅ seaborn')" 2>/dev/null || echo "  ❌ seaborn"
	@$(PYTHON) -c "import langchain; print('  ✅ langchain')" 2>/dev/null || echo "  ⚠️  langchain (optional)"
	@echo ""
	@echo "$(CYAN)🤖 LangChain Status:$(RESET)"
	@$(PYTHON) generate_weekly_reports.py --config-check 2>/dev/null || echo "  ⚠️  LangChain not configured"
	@$(MAKE) _return_to_menu

.PHONY: docs
docs:
	@echo "$(BOLD)$(WHITE)📖 Opening Documentation...$(RESET)"
	@echo ""
	@echo "$(CYAN)Available documentation files:$(RESET)"
	@echo "  📄 README.md           - Complete usage guide"
	@echo "  📄 PROJECT_COMPLETE.md - Implementation summary"
	@echo "  📄 QUICKSTART.md       - Quick start guide"
	@echo ""
	@echo "$(YELLOW)Choose a document to view:$(RESET)"
	@echo "  1) README.md"
	@echo "  2) PROJECT_COMPLETE.md"
	@echo "  3) QUICKSTART.md"
	@echo "  4) Return to menu"
	@echo ""
	@read -p "Select (1-4): " choice; \
	case $$choice in \
		1) less README.md ;; \
		2) less PROJECT_COMPLETE.md ;; \
		3) less QUICKSTART.md ;; \
		4) ;; \
		*) echo "$(RED)Invalid option$(RESET)" ;; \
	esac
	@$(MAKE) _return_to_menu

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

.PHONY: _show_results
_show_results:
	@echo ""
	@echo "$(BOLD)$(GREEN)📊 Results Summary:$(RESET)"
	@if [ -d "$(OUTPUT_DIR)" ]; then \
		echo "$(CYAN)Generated files:$(RESET)"; \
		find $(OUTPUT_DIR) -type f | head -10 | sed 's/^/  📄 /'; \
		total=$$(find $(OUTPUT_DIR) -type f | wc -l); \
		if [ $$total -gt 10 ]; then echo "  ... and $$((total-10)) more files"; fi; \
		echo "$(CYAN)Output directory: $(OUTPUT_DIR)$(RESET)"; \
	fi

.PHONY: _show_reports
_show_reports:
	@echo ""
	@echo "$(BOLD)$(GREEN)📋 Reports Summary:$(RESET)"
	@if [ -d "$(REPORTS_DIR)" ]; then \
		echo "$(CYAN)Generated reports:$(RESET)"; \
		find $(REPORTS_DIR) -name "*.md" | head -5 | sed 's/^/  📄 /'; \
		find $(REPORTS_DIR) -name "*.json" | head -3 | sed 's/^/  🔧 /'; \
		echo "$(CYAN)Reports directory: $(REPORTS_DIR)$(RESET)"; \
	fi

.PHONY: _return_to_menu
_return_to_menu:
	@echo ""
	@echo "$(YELLOW)Press Enter to return to menu...$(RESET)"
	@read dummy
	@$(MAKE) menu

# Make all targets phony by default
.DEFAULT_GOAL := menu
