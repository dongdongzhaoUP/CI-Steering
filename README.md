# CI-Steering

Code release for the paper _"Do LLMs Know What Is Private Internally? Probing and Steering Contextual Privacy Norms in Large Language Model Representations"_.

We investigate whether LLMs internally encode privacy norms as defined by Contextual Integrity (CI) framework. We provide tools for probing and steering privacy-related representations, and introduce **CI-Steering**, a compositional method that steers along per-CI-parameter axes (information type, recipient, transmission principle) for more effective and transferable privacy control.

## Setup

### Installation

```bash
conda create -n ci_steering python=3.11
conda activate ci_steering
pip install -r requirements.txt

# Or install as editable package
pip install -e .
```

### Environment Variables

```bash
# Required for GPT-as-judge evaluation
export OPENAI_API_KEY="your-openai-api-key"

# Optional: HuggingFace token for gated models (e.g., Llama)
export HF_TOKEN="your-hf-token"
```

### External Benchmarks

The following benchmarks must be cloned separately:

```bash
# CONFAIDE (Mireshghallah et al., ICLR 2024)
git clone https://github.com/skywalker023/confaide.git data/confaide

# PrivaCI-Bench (Li et al., ACL 2025)
git clone https://github.com/HKUST-KnowComp/PrivaCI-Bench.git data/privaci_bench
```

### Supported Models

All scripts accept a `--model` flag. The following models are tested:

| Model        | HuggingFace ID                       | Type     |
| ------------ | ------------------------------------ | -------- |
| Llama 3.1 8B | `meta-llama/Llama-3.1-8B-Instruct`   | Instruct |
| Qwen 2.5 7B  | `Qwen/Qwen2.5-7B-Instruct`           | Instruct |
| Mistral 7B   | `mistralai/Mistral-7B-Instruct-v0.3` | Instruct |
| Llama 2 7B   | `meta-llama/Llama-2-7b-hf`           | Base     |

## Pipeline

### Phase 1 — Generate Stimuli

```bash
python src/generate_stimuli.py \
    --num-pairs-per-type 50 \
    --num-function-pairs 200 \
    --num-ci-per-condition 100
```

### Phase 2 — Extract Activations

```bash
python src/extract_activations.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --batch-size 4
```

### Phase 3 — Probe Representations

```bash
python src/read_representations.py \
    --activations-dir outputs/activations/Llama-3.1-8B-Instruct
```

### Phase 4 — CI Decomposition

```bash
python src/ci_decomposition.py \
    --activations-dir outputs/activations/Llama-3.1-8B-Instruct \
    --output-dir outputs/ci_decomposition/Llama-3.1-8B-Instruct
```

### Phase 5 — Evaluation (CONFAIDE & PrivaCI-Bench)

```bash
# Monolithic steering on CONFAIDE
python src/confaide_evaluation.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --reader-dir outputs/reading/Llama-3.1-8B-Instruct/pca_reader

# CI-parametric steering on CONFAIDE
python src/confaide_ci_steering.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --ci-dir outputs/ci_decomposition/Llama-3.1-8B-Instruct

# PrivaCI-Bench
python src/privaci_evaluation.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --data-dir data/privaci_bench

# CI-parametric on PrivaCI-Bench
python src/privaci_ci_steering.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --ci-dir outputs/ci_decomposition/Llama-3.1-8B-Instruct
```

### Phase 6 — Tuning Baselines

```bash
# LoRRA
python src/lorra_finetune.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --reader-dir outputs/reading/Llama-3.1-8B-Instruct/pca_reader

# Representation Tuning
python src/rep_tuning.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --reader-dir outputs/reading/Llama-3.1-8B-Instruct/pca_reader
```

### Utility Evaluation

```bash
python src/utility_evaluation.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --reader-dir outputs/reading/Llama-3.1-8B-Instruct/probe_reader \
    --ci-dir outputs/ci_decomposition/Llama-3.1-8B-Instruct
```

## License

This project is released under the [MIT License](LICENSE).
