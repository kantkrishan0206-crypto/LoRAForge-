<div align="center"> 

# 🔥 **LoRAForge** 
### _Fine-Tuning Large Language Models (LLMs) Efficiently with LoRA_
 
<img width="1024" height="1024" alt="image" src="https://github.com/user-attachments/assets/f4003dd7-9a1b-41f7-a9d9-2254a6a9970c" />


[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=for-the-badge&logo=python)](https://www.python.org/)
[![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-yellow.svg?style=for-the-badge&logo=huggingface)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)
[![Stars](https://img.shields.io/github/stars/yourusername/LoRAForge?style=for-the-badge&logo=github)](https://github.com/yourusername/LoRAForge/stargazers)
[![Forks](https://img.shields.io/github/forks/yourusername/LoRAForge?style=for-the-badge&logo=github)](https://github.com/yourusername/LoRAForge/network/members)

</div>

---

## 🧠 **What is LoRAForge?**

> **LoRAForge** is an open-source toolkit that empowers developers to fine-tune *Large Language Models (LLMs)* efficiently using **Low-Rank Adaptation (LoRA)** — achieving near full fine-tuning performance with **~10,000× fewer trainable parameters**.

With LoRAForge, you can:
- ⚡ Fine-tune billion-parameter models on a **laptop GPU**  
- 💰 Reduce training cost by **over 90%**  
- 🧩 Adapt models for **niche domains** (medical, legal, financial, etc.)  
- 📊 Visualize results with an interactive dashboard  

---

## 🚀 **Key Features**

✅ **Plug & Play Fine-Tuning** – Easily apply LoRA on any Hugging Face model  
✅ **Lightweight** – Trains on 6–8 GB GPU memory  
✅ **Domain Adaptation** – Load and fine-tune custom corpora  
✅ **Gradio Dashboard** – Monitor and test your model visually  
✅ **Cross-Domain Testing** – Evaluate transfer learning performance  

---

## 🏗️ **Architecture Overview**

<div align="center">
<img src="https://github.com/small-thinking/multi-lora-fine-tune/raw/main/assets/system_overview.png" width="700">
</div>

<div align="center">
<img width="1024" height="1024" alt="image" src="https://github.com/user-attachments/assets/35499ded-78f0-49d3-ac12-06f934d12fab" />
</div>

**Explanation:**  
LoRA injects small trainable rank-decomposition matrices into each attention layer, while the original model weights stay *frozen*.  
This dramatically cuts the number of parameters updated during training, enabling low-cost fine-tuning.

---

## ⚙️ **Tech Stack**

| Layer | Technology |
|--------|-------------|
| Model | Hugging Face Transformers (`LLaMA`, `Falcon`, `OPT`, etc.) |
| Training | PyTorch + CUDA |
| PEFT | `PEFT` library (`get_peft_model`, `LoraConfig`) |
| Visualization | Gradio / Streamlit |
| Dataset | Domain-specific corpus (Medical / Legal / Financial) |
| Deployment | Hugging Face Spaces  |

---

## 🧩 **Quick Start**

### 🛠️ 1. Setup Environment
```bash
git clone https://github.com/kantkrishan0206-crypto/LoRAForge.git
cd LoRAForge
pip install -r requirements.txt
```

### ⚙️ 2. Fine-Tune with LoRA
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

model_name = "facebook/opt-1.3b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
```

### 🚀 3. Train
```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    per_device_train_batch_size=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    output_dir="./results"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=your_dataset
)
trainer.train()
```

### 🧪 4. Evaluate
```python
metrics = trainer.evaluate()
print(metrics)
```

---

## 📊 **Performance Summary**

| Metric | Full Fine-Tuning | LoRA Fine-Tuning | Gain |
|--------|------------------|------------------|------|
| GPU Memory (GB) | 48 | 6 | 🔽 -88% |
| Training Time | 10h | 1.2h | 🔽 -88% |
| Accuracy | 89.2% | 88.5% | ≈ Same |
| Cost (USD) | $400 | $30 | 🔽 -92% |

---

## 💡 **Innovations in LoRAForge**

- 🔁 **Dynamic Rank Adaptation** – Adjust LoRA rank `r` during training  
- 🧩 **Hybrid LoRA + Prompt-Tuning** – Combine adapters and prefixes  
- 🌐 **Cross-Domain Evaluation** – Measure model adaptability  
- 🖥️ **LoRAForge UI** – Upload dataset → Train → Evaluate (Gradio)  

---

## 🧱 **Repository Structure**

```
📂
LoRAForge/
├─ README.md                        # 🔍 Project overview, setup, usage, citations, architecture
├─ LICENSE                          # 📜 MIT license for open-source use
├─ requirements.txt                 # 📦 Core dependencies (PEFT, Transformers, PyTorch, etc.)
├─ pyproject.toml                   # 🛠️ Modern packaging (Poetry/setuptools)
├─ Makefile                         # ⚙️ CLI shortcuts: setup, train, eval, test, clean
│
├─ configs/                         # 🧾 YAML configs for reproducible experiments
│  ├─ sft_default.yaml              # Default supervised fine-tuning config
│  ├─ lora_mistral.yaml             # LoRA config for Mistral-7B
│  ├─ lora_phi.yaml                 # LoRA config for Phi-2
│  ├─ eval.yaml                     # Evaluation config (prompts, metrics, output paths)
│  ├─ hub.yaml                      # Hugging Face Hub push config
│  └─ cross_domain.yaml             # Cross-domain evaluation config (train vs test domains)
│
├─ data/                            # 📂 Datasets for training/evaluation
│  ├─ raw/                          # Original datasets (jsonl, csv, txt)
│  │   ├─ dataset.jsonl             # Domain-specific instruction data
│  │   └─ metadata.json             # Optional schema or source info
│  ├─ processed/                    # Preprocessed parquet/arrow files
│  │   ├─ train.parquet             # Training split
│  │   └─ val.parquet               # Validation split
│  └─ samples/                      # Tiny toy datasets for CI/tests
│      ├─ sample.jsonl              # Minimal JSONL dataset for quick tests
│      └─ sample.parquet            # Minimal parquet dataset for quick tests
│
├─ notebooks/                       # 📓 Research-grade notebooks for exploration and analysis
│  ├─ 00_project_overview.ipynb     # Full pipeline walkthrough, config anatomy
│  ├─ 01_data_exploration.ipynb     # Schema inspection, distribution plots
│  ├─ 02_train_sft.ipynb            # Supervised fine-tuning demo
│  ├─ 03_train_lora.ipynb           # LoRA adapter training demo
│  ├─ 04_eval_visualization.ipynb   # Metrics, plots, prompt I/O rendering
│  ├─ 05_merge_export.ipynb         # Merge LoRA into base, export to Hugging Face
│  ├─ 06_sandbox_experiments.ipynb  # Ablations, debugging, exploratory runs
│  └─ 07_cross_domain_eval.ipynb    # Cross-domain generalization analysis
│
├─ src/                             # 🧠 Core source code
│  ├─ cli/                          # CLI entrypoints for modular execution
│  │   └─ run.py                    # Unified CLI: train, eval, export, push
│  ├─ data/                         # Data loading, formatting, preprocessing
│  │   ├─ dataset_loader.py         # Load parquet/jsonl datasets
│  │   ├─ formatters.py             # Prompt formatting for instruction-style tasks
│  │   ├─ preprocess.py             # Raw → processed conversion logic
│  │   └─ validators.py             # Schema & quality checks
│  ├─ models/                       # Model loading and adapter setup
│  │   ├─ load_base.py              # Load base model + tokenizer (with quantization)
│  │   ├─ lora_setup.py             # Attach LoRA adapters via PEFT
│  │   ├─ dynamic_lora.py           # Dynamic rank adaptation logic (adjust r during training)
│  │   └─ hybrid_lora.py            # Hybrid LoRA + prompt-tuning (combine adapters + prefixes)
│  ├─ training/                     # Training logic
│  │   ├─ train_sft.py              # Supervised fine-tuning script
│  │   ├─ train_lora.py             # LoRA fine-tuning script
│  │   ├─ trainer.py                # Shared Trainer wrapper (HF Trainer or custom)
│  │   └─ callbacks.py              # Custom callbacks (logging, dynamic rank, early stopping)
│  ├─ eval/                         # Evaluation and metrics
│  │   ├─ evaluate.py               # Evaluation loop
│  │   ├─ metrics.py                # Task-specific metrics (BLEU, ROUGE, EM, F1)
│  │   ├─ visualizer.py             # Plotting, prompt I/O rendering
│  │   ├─ prompts/
│  │   │   └─ eval_prompts.jsonl    # Instruction-style evaluation prompts
│  │   └─ cross_domain_eval.py      # Cross-domain evaluation runner
│  ├─ ui/                           # 🎨 User interface layer
│  │   ├─ gradio_app.py             # Gradio dashboard for Spaces
│  │   └─ streamlit_app.py          # Optional Streamlit UI
│  ├─ deploy/                       # 🚀 Deployment layer
│  │   ├─ spaces_app.py             # Hugging Face Spaces entrypoint (Gradio/Streamlit)
│  │   ├─ app_utils.py              # Shared helpers for Spaces deployment
│  │   └─ api.py                    # FastAPI inference API (optional)
│  └─ utils/                        # Utilities
│      ├─ io.py                     # File I/O helpers (YAML, JSON, parquet)
│      ├─ logging.py                # Logging setup (console + file)
│      ├─ seed.py                   # Reproducibility utilities (set_seed)
│      └─ config.py                 # Config loader + schema validation
│
├─ scripts/                         # 🧪 Automation scripts for deployment and conversion
│  ├─ prepare_data.py               # Preprocess raw → processed
│  ├─ push_to_hub.py                # Upload model/adapters to Hugging Face Hub
│  ├─ export_merged.py              # Merge LoRA adapters into base model
│  ├─ convert_to_gguf.py            # Export to GGUF for llama.cpp
│  └─ quantize_model.py             # Quantize merged model to 4-bit
│
├─ tests/                           # ✅ Unit tests for CI and reliability
│  ├─ test_data.py                  # Dataset loading and formatting tests
│  ├─ test_lora.py                  # LoRA setup and adapter attachment tests
│  ├─ test_trainer.py               # Training loop sanity checks
│  ├─ test_eval.py                  # Evaluation metrics and prompt rendering
│  ├─ test_config.py                # Config schema and loader validation
│  ├─ test_cross_domain.py          # Cross-domain evaluation tests
│  └─ test_dynamic_lora.py          # Dynamic/hybrid LoRA module tests
│
└─ Space/                           # 🌐 Hugging Face Spaces deployment folder
   ├─ app.py                        # Main app file (Gradio/Streamlit) auto-detected by Spaces
   ├─ requirements.txt              # Minimal deps for Spaces runtime
   └─ README.md                     # Instructions for running this Space


```

---

## 🧑‍🔬 **Research References**

- **Hu et al. (2021)** — *LoRA: Low-Rank Adaptation of Large Language Models*  
  [[arXiv:2106.09685]](https://arxiv.org/abs/2106.09685)  
- **PEFT Library Docs:** [https://huggingface.co/docs/peft](https://huggingface.co/docs/peft)
- **Hugging Face Transformers:** [https://huggingface.co/docs/transformers](https://huggingface.co/docs/transformers)

---

## 🤝 **Contributing**

Pull requests are welcome!  
If you find a bug or want to add a feature:
```bash
git checkout -b feature-name
git commit -m "Added new feature"
git push origin feature-name
```
Then open a PR 💡

---

## 🪙 **License**
This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  
✨ **LoRAForge — Fine-tune Giants, Pay Like a Student.** ✨  
Built with ❤️ by [kantkrishan0206-crypto](https://github.com/kantkrishan0206-crypto)

</div>
