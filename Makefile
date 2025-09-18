# LDM-EKI Integrated Makefile

# Directories
LDM_DIR = ldm
EKI_DIR = eki
INTEGRATION_DIR = integration
DATA_DIR = data
LOGS_DIR = logs

# Python environment
PYTHON = /opt/miniconda3/bin/python

.PHONY: all build clean test run-ldm run-eki run-coupled help setup

# Default target
all: build

# Build LDM
build-ldm:
	@echo "Building LDM..."
	cd $(LDM_DIR) && make

# Build everything
build: build-ldm
	@echo "Build complete"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	cd $(LDM_DIR) && make clean || true
	rm -rf $(DATA_DIR)/output/*
	rm -rf $(LOGS_DIR)/*
	@echo "Clean complete"

# Setup directories and permissions
setup:
	@echo "Setting up LDM-EKI environment..."
	mkdir -p $(DATA_DIR)/output/{ldm_results,eki_results,visualization}
	mkdir -p $(LOGS_DIR)/{ldm_logs,eki_logs,integration_logs}
	chmod +x $(LDM_DIR)/ldm || true
	chmod +x $(INTEGRATION_DIR)/workflows/coupled_simulation.py
	@echo "Setup complete"

# Run LDM server only
run-ldm: build-ldm
	@echo "Starting LDM server..."
	cd $(LDM_DIR) && CUDA_VISIBLE_DEVICES=0 ./ldm

# Run EKI client only
run-eki:
	@echo "Starting EKI client..."
	cd $(EKI_DIR) && $(PYTHON) src/RunEstimator.py config/input_config config/input_data

# Run coupled simulation
run-coupled: build setup
	@echo "Starting coupled LDM-EKI simulation..."
	$(PYTHON) $(INTEGRATION_DIR)/workflows/coupled_simulation.py

# Run function tracking analysis
track-functions:
	@echo "Analyzing function usage..."
	cd $(EKI_DIR) && $(PYTHON) -c "from tracking.function_tracker import print_usage_report; print_usage_report()"

# Run tests
test:
	@echo "Running tests..."
	@echo "Test framework not yet implemented"

# Show help
help:
	@echo "LDM-EKI Integrated Build System"
	@echo ""
	@echo "Available targets:"
	@echo "  all          - Build everything (default)"
	@echo "  build        - Build LDM and prepare system"
	@echo "  build-ldm    - Build LDM only"
	@echo "  clean        - Clean build artifacts and outputs"
	@echo "  setup        - Setup directories and permissions"
	@echo "  run-ldm      - Run LDM server only"
	@echo "  run-eki      - Run EKI client only"
	@echo "  run-coupled  - Run integrated LDM-EKI simulation"
	@echo "  track-functions - Show EKI function usage report"
	@echo "  test         - Run test suite"
	@echo "  help         - Show this help message"