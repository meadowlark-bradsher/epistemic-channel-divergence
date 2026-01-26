# ESI Experiment

Epistemic Sensitivity under Intervention—measuring how stable action and report channels are under semantic-preserving perturbations.

## Hypothesis

If action and report are separate systems with different properties, they should show different sensitivity to perturbations. Specifically:

- Action (token-level) should be sensitive to surface features
- Report (learned policy) should be stabilized by training pressure

## What This Experiment Falsifies

- **Claim**: "Self-reports are direct readouts of internal states"
- **Result**: Falsified. Reports are 3-6x more stable than actions under identical perturbations.

- **Claim**: "Position bias is a minor effect"
- **Result**: Falsified. ORDER intervention causes ~0.5 Hellinger distance—massive.

- **Claim**: "CoT improves reliability"
- **Result**: Falsified. CoT amplifies action sensitivity (highest ESI_action).

## What This Experiment Does NOT Claim

- That perturbation sensitivity is always bad (it reveals true variance)
- That all interventions have the same effect (SOC ≠ ORDER)
- That these results transfer to non-multiple-choice formats

## Interventions

### SOC (Skip-One-Char)

Remove characters from word tails:

```
"programming" → "programing"
"understanding" → "understading"
```

Semantic-preserving at the sentence level.

### ORDER (Option Position Shuffling)

Shuffle A/B/C/D positions while preserving content:

```
Original:  A=Paris, B=London, C=Berlin, D=Madrid
Shuffled:  A=Berlin, B=Paris, C=Madrid, D=London
```

Content preserved, only labels change.

## Usage

```bash
# Full experiment with both interventions
python experiments/esi_runner.py --providers openai gemini --full

# SOC only
python experiments/esi_runner.py openai --intervention soc

# ORDER only with more variants
python experiments/esi_runner.py openai --intervention order --n-variants 10

# Output to file
python experiments/esi_runner.py --providers openai gemini --full -o results/esi/run.jsonl
```

### Options

| Flag | Description |
|------|-------------|
| `--providers` | One or more providers |
| `--full` | Run all probes and interventions |
| `--intervention` | Specific intervention (soc, order) |
| `--n-variants` | Number of variants per item |
| `--max-items` | Limit number of questions |
| `-o FILE` | Output file (JSONL) |

## Metrics

| Metric | Definition |
|--------|------------|
| ESI_action | Mean Hellinger distance of action distributions under intervention |
| ESI_report | Mean Hellinger distance of report distributions under intervention |
| ΔJS | Change in JS(action\|\|report) under intervention |
| Ratio | ESI_action / ESI_report |

## Key Results

### SOC Intervention

| Provider | Probe | ESI_action | ESI_report | Ratio |
|----------|-------|------------|------------|-------|
| OpenAI | B | 0.161 | 0.048 | 3.4x |
| OpenAI | C | **0.297** | 0.048 | **6.2x** |
| Gemini | B | 0.314 | 0.079 | 4.0x |
| Gemini | C | **0.414** | 0.068 | **6.1x** |

### ORDER Intervention

| Provider | All Probes | ESI_action | ESI_report | Ratio |
|----------|------------|------------|------------|-------|
| OpenAI | Average | ~0.53 | ~0.09 | ~6x |
| Gemini | Average | ~0.50 | ~0.10 | ~5x |

## Delivers

- Action/report sensitivity asymmetry (3-6x ratio)
- Position bias magnitude (~0.5 Hellinger)
- CoT as variance amplifier (highest ESI_action)
- Probe B (Report→Act) as most stable

## Analysis

Run post-hoc analysis:

```bash
python experiments/esi_analysis.py results/esi/full_experiment.jsonl
```

Produces:
- Order-ensemble debiasing analysis
- Stratification by question category
- Gemini uniform attractor persistence

## Related

- [Findings: Structural Sensitivity](../findings/structural-sensitivity.md)
- [Ensemble Bootstrap](ensemble-bootstrap.md) - The measurement fix
