# === LoRAForge Makefile ===
# Common developer shortcuts for reproducible workflows

PYTHON := python
CONFIG := configs/sft_default.yaml
VENV := .venv

.PHONY: setup install data train eval test clean format lint

# --- Environment setup ---
setup:
	@echo ">>> Creating virtual environment and installing dependencies..."
	$(PYTHON) -m venv $(VENV)
	. $(VENV)/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

install:
	@echo ">>> Installing LoRAForge as editable package..."
	. $(VENV)/bin/activate && pip install -e .

# --- Data preparation ---
data:
	@echo ">>> Preparing dataset..."
	. $(VENV)/bin/activate && $(PYTHON) scripts/prepare_data.py --config $(CONFIG)

# --- Training ---
train:
	@echo ">>> Starting LoRA fine-tuning..."
	. $(VENV)/bin/activate && $(PYTHON) src/training/train_lora.py --config $(CONFIG)

train-sft:
	@echo ">>> Starting supervised fine-tuning..."
	. $(VENV)/bin/activate && $(PYTHON) src/training/train_sft.py --config $(CONFIG)

# --- Evaluation ---
eval:
	@echo ">>> Running evaluation..."
	. $(VENV)/bin/activate && $(PYTHON) src/eval/evaluate.py --config $(CONFIG)

# --- Testing ---
test:
	@echo ">>> Running unit tests..."
	. $(VENV)/bin/activate && pytest -q

# --- Code quality ---
format:
	@echo ">>> Formatting code with black..."
	. $(VENV)/bin/activate && black src tests

lint:
	@echo ">>> Linting code with ruff..."
	. $(VENV)/bin/activate && ruff check src tests

# --- Cleanup ---
clean:
	@echo ">>> Cleaning outputs and caches..."
	rm -rf outputs/* data/processed/* .pytest_cache .ruff_cache
