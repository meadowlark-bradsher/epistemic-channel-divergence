# Experiments Overview

This toolkit provides several experiments for probing belief-action divergence in LLMs.

## Available Experiments

| Experiment | Purpose | Key Output |
|------------|---------|------------|
| [Belief Probe Baseline](belief-probe.md) | Compare behavioral vs declarative beliefs across probe orderings | JS divergence by probe type |
| [Gravitational Pull](gravitational-pull.md) | Test whether constraints displace or eliminate hedging | Boundary mass trends |
| [Alignment Regularizer](alignment-regularizer.md) | Test whether incentives improve belief reporting | Effect of calibration prompts |

## Supported Providers

All experiments support multiple inference backends:

| Provider | Token Beliefs | Accuracy | Setup |
|----------|---------------|----------|-------|
| OpenAI | True logprobs | Best | `OPENAI_API_KEY` in `.env` |
| Gemini | True logprobs | Best | `GOOGLE_API_KEY` in `.env` |
| llama.cpp | True logprobs | Good | Run `llama-server -m model.gguf -c 4096` |
| Ollama | Sampling-based | Lower | `ollama serve` |

!!! warning "Logprobs vs Sampling"
    True logprobs are essential for confidence-related claims. Sampling-based estimation can invert correlations and mask effects.

## Quick Start

### Environment Setup

```bash
# Create .env file with API keys
cat > .env << EOF
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
EOF
```

### Local Model Setup (Optional)

```bash
# Install llama.cpp
brew install llama.cpp

# Download a model
huggingface-cli download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \
    --include "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" \
    --local-dir models/

# Start server
llama-server -m models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf -c 4096
```

### Run Experiments

```bash
# Belief probe with anti-uniform constraint
python experiments/belief_probe_baseline.py --experiment2 --anti-uniform

# Gravitational pull sweep
python experiments/gravitational_pull.py openai --multi-question

# Cross-provider comparison
python experiments/compare_providers.py --providers openai gemini
```

## Output Format

All experiments output JSON with consistent structure:

```json
{
  "provider": "openai",
  "model": "gpt-4o-mini",
  "timestamp": "2025-01-19T12:00:00",
  "results": [
    {
      "question_id": 1,
      "token_distribution": {"A": 0.7, "B": 0.2, "C": 0.05, "D": 0.05},
      "reported_distribution": {"A": 0.3, "B": 0.25, "C": 0.25, "D": 0.2},
      "js_divergence": 0.35,
      "token_entropy": 1.2,
      "report_entropy": 1.95
    }
  ]
}
```

## Key Metrics

- **JS Divergence**: Jensen-Shannon divergence between token and reported distributions
- **Token Entropy**: Shannon entropy of behavioral distribution
- **Report Entropy**: Shannon entropy of reported distribution
- **Boundary Mass**: Fraction of probability within 3% of constraint bounds
