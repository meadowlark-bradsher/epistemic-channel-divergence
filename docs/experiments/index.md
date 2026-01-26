# Experiments

How to run each experiment and what it tests. Each experiment has a clear epistemic job—a hypothesis it tests and claims it can or cannot support.

## Available Experiments

| Experiment | Purpose | Key Output |
|------------|---------|------------|
| [Belief Probe](belief-probe.md) | Establish decoupling + probe-order effects | JS divergence by probe type |
| [Gravitational Pull](gravitational-pull.md) | Show attractors persist under constraints | Boundary mass trends |
| [Alignment Regularizer](alignment-regularizer.md) | Falsify strategic hedging hypothesis | Effect of incentives |
| [ESI](esi.md) | Distinguish noise vs structural sensitivity | ESI_action / ESI_report ratio |
| [Ensemble Bootstrap](ensemble-bootstrap.md) | Turn diagnostics into measurement fix | JS reduction, optimal K |

## Supported Providers

| Provider | Token Beliefs | Setup |
|----------|---------------|-------|
| OpenAI | True logprobs | `OPENAI_API_KEY` in `.env` |
| Gemini | True logprobs | `GOOGLE_API_KEY` in `.env` |
| llama.cpp | True logprobs | `llama-server -m model.gguf` |
| Ollama | Sampling-based | `ollama serve` |

!!! warning "Logprobs vs Sampling"
    True logprobs are essential for confidence-related claims. Sampling-based estimation can invert correlations and mask effects. Use OpenAI, Gemini, or llama.cpp for final results.

## Quick Start

```bash
# Environment setup
cat > .env << EOF
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
EOF

# Core experiments
python experiments/belief_probe_baseline.py --experiment2 --anti-uniform
python experiments/gravitational_pull.py openai --multi-question
python experiments/alignment_regularizer.py openai

# ESI experiments
python experiments/esi_runner.py --providers openai gemini --full
python experiments/esi_ensemble.py --providers openai gemini --bootstrap
```

## Output Format

All experiments output JSON with consistent structure:

```json
{
  "provider": "openai",
  "model": "gpt-4o-mini",
  "results": [
    {
      "question_id": 1,
      "action_distribution": {"A": 0.7, "B": 0.2, "C": 0.05, "D": 0.05},
      "report_distribution": {"A": 0.3, "B": 0.25, "C": 0.25, "D": 0.2},
      "js_divergence": 0.35
    }
  ]
}
```

## Key Metrics

| Metric | Definition |
|--------|------------|
| JS Divergence | Jensen-Shannon divergence between action and report |
| ESI_action | Hellinger distance of action under intervention |
| ESI_report | Hellinger distance of report under intervention |
| Boundary Mass | Fraction of probability within 3% of constraint bounds |
| Argmax Flip Rate | Fraction of orderings where modal answer changes |
