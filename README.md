# Epistemic Channel Divergence

Research toolkit for measuring the divergence between what LLMs **believe** (behavioral/token-level) and what they **report** (declarative self-reports).

## LLM Reliability: Bridging the Gap Between Action and Reported Belief

1.0 The Core Challenge: Default AI Reporting is Misleading

The reliability of any automated system powered by a Large Language Model (LLM) depends on a single factor: our ability to accurately interpret its confidence. Failure to do so exposes the organization to significant operational and reputational risk. Without a clear view into a model's certainty, we cannot safely delegate high-stakes decisions.

Our analysis dictates a fundamental challenge with the default behavior of LLMs. Absent any constraints, models will report a uniform probability distribution—for example, assigning exactly 25% confidence to each of four multiple-choice options—on 100% of queries. Critically, they do this regardless of how confident their underlying action is. A model may be internally prepared to select one answer with over 90% certainty but will still report a perfectly balanced 25/25/25/25 split when asked.

For leadership, the strategic imperative is to interpret this correctly: this behavior is not a sign of ignorance or incapacity. Instead, it is a learned "safe" policy developed during training to avoid being penalized for incorrect high-confidence answers. The key takeaway is that a model's default self-report on its confidence cannot be taken at face value. This misleading "safe" reporting is not random noise; it is the surface-level symptom of a deeper, more paradoxical behavior we term "belief-action decoupling."

2.0 The Decoupling Paradox: High Confidence in Action, High Uncertainty in Reporting

The mismatch between a model's internal conviction and its external communication can be described as belief-action decoupling. In simple terms, what the model does is disconnected from what it reports about its beliefs. Understanding and managing this decoupling is of paramount strategic importance for any team deploying LLMs in roles that require reliable decision-making.

Our research uncovered a startling paradox. Using precise measurements of the model's internal state—true logprobs (a direct and precise measurement of the model's internal probability calculations)—we found that the mismatch between action and reporting is highest precisely when the model is most committed to its chosen action. This isn't an occasional quirk; it is a predictable inverse correlation. As the model's certainty in its action rises, its reported uncertainty rises in lockstep. This means the model projects maximum uncertainty at the exact moment it has the highest internal conviction, creating a critical blind spot for risk management.

The model may appear most uncertain at the exact moment it is most decisive. This is a fundamental contradiction we must manage.

Recognizing this core paradox is the first step. The next is to deploy specific methods designed to diagnose and correct it, creating more aligned and reliable AI systems.

3.0 Strategic Interventions: A Toolkit for Improving AI Alignment

The behavioral issues we've identified are not immutable flaws; they are manageable dynamics. The following interventions represent a strategic toolkit for exerting deliberate control over model behavior, enabling our teams to move from passive operators to active managers of AI-driven outcomes.

3.1 The Foundational Fix: Eliciting Graded Beliefs

The misleading uniform reporting behavior can be effectively overcome. Our findings show that applying a simple and mild "anti-uniform constraint"—such as requiring that probabilities fall between 5% and 80%—compels the model to provide more informative and nuanced reports on its uncertainty. This intervention is remarkably effective, causing the model's use of misleading uniform reports to plummet from nearly 100% to less than 2%. This successfully breaks the model's habit of defaulting to a safe, uninformative state without forcing it to report extreme, unsupported confidence levels.

3.2 Choosing the Right Tool for the Task: A Comparative Analysis

Different prompting strategies yield dramatically different results depending on the nature of the task. The choice between these interventions is a strategic trade-off between performance on simple tasks and stability on complex ones. One optimizes for speed in known environments, while the other provides resilience in unpredictable ones. The following table compares two common approaches: Chain-of-Thought (CoT) and Introspective Framing.

Intervention Strategy	Strategic Application & Risk Profile
Chain-of-Thought (CoT)	Use Case: Best for easy, well-defined problems where it improves alignment.<br>Risk: Acts as a "variance amplifier" on ambiguous or adversarial problems, significantly worsening performance and leading to extreme failures. It collapses the model's reasoning into a single narrative, which is dangerously brittle when multiple interpretations are plausible.
Introspective Framing	Use Case: The most stable and reliable approach for ambiguous and complex problems.<br>Risk: It is not a pure "readout" of truth but a "stabilizer" that regularizes the model's behavior, making it a safer default for heterogeneous task environments where problem difficulty is unpredictable.

3.3 The Power of Sequencing: Anchoring Belief Before Action

A simple change in the sequence of operations can produce a powerful alignment effect. Our research consistently shows that requiring the model to first declare its beliefs before it takes an action serves as an effective anchoring mechanism. This "Report-then-Act" sequence consistently reduces the mismatch between the model's stated confidence and its subsequent choice, effectively encouraging the model to act in a way that is more consistent with its declared beliefs.

The interventions in our toolkit are powerful, but without the correct measurement infrastructure, their effects are invisible and their value cannot be proven. This brings us to the most foundational requirement for reliable AI systems.

4.0 The Foundational Requirement: Accurate Measurement is Non-Negotiable

The value of any intervention is lost without rigorous and accurate methods of measurement—a core principle of any data-driven decision-making process. In the context of LLMs, this means choosing the right tool to assess the model's internal state.

Our analysis revealed a critical difference between two methods for measuring a model's action confidence: "sampling-based" estimates and true logprobs. Relying on sampling, a common and computationally cheaper method, can produce highly misleading data. In our analysis, sampling-based methods not only failed to detect the belief-action decoupling but produced data suggesting the exact opposite—that belief and action were aligned. This creates a dangerously false sense of security.

For any serious analysis, monitoring, or reliable deployment of LLMs, using true logprobs is an essential, non-negotiable requirement. This is not merely a technical best practice; it is a prerequisite for governance. Any team deploying an LLM without this level of measurement is operating without the necessary controls to manage risk.

5.0 Strategic Summary and Forward Outlook

These findings provide a new, more nuanced framework for understanding and managing LLM behavior. Rather than viewing a model as a monolithic entity, we must approach its outputs with greater sophistication, recognizing the distinct and manageable components of its decision-making process.

The core conclusion of this research is that an LLM's belief, its action, and its reporting are "partially independent control surfaces." In strategic terms, this means we have distinct levers to pull to shape AI behavior. We can manage risk not by treating the model as an inscrutable black box, but by actively tuning its reporting, reasoning, and actions to align with our objectives.

This remains an active and vital area of research. Future work will focus on gaining finer-grained control over model reasoning and generalizing these alignment techniques across different models and tasks. Mastering these control surfaces is the next frontier in creating defensible, high-value AI applications, moving our organization from being a consumer of AI to a master of its behavior.


When asked to report their uncertainty, do language models give accurate probability estimates?

**Answer: No.** We find robust belief-action decoupling across providers, where models act confidently but report broad uncertainty—and this divergence is *largest* when the model is most confident.

## Key Findings

### F1. Uniform Reporting Attractor
Without constraints, models overwhelmingly report uniform distributions (25/25/25/25) regardless of their actual token probabilities. This is a learned "safe" policy, not ignorance.

### F2. Anti-Uniform Constraint Breaks It
A mild constraint (5-80% per option) forces informative reports without inducing gaming. The uniform attractor is optional, not fundamental.

### F3. Belief-Action Decoupling Increases with Confidence
With true logprobs, mismatch (JS divergence) is **highest when action is most peaked**. Models act decisively while reporting broad uncertainty.

### F4. Probe Order Matters
- **Report→Act (Probe B)**: Best alignment—declared belief anchors subsequent action
- **CoT→Act (Probe C)**: Variance amplifier—helps easy questions, hurts ambiguous ones
- **Act→Introspect (Probe D)**: Most stable across heterogeneous tasks

### F5. Gravitational Pull
When constrained away from uniform, models pile probability at constraint boundaries rather than expressing genuine interior beliefs. The uniform attractor is displaced, not eliminated.

## Experiments

### Belief Probe Baseline
Compare behavioral vs declarative beliefs across four probe orderings:

| Probe | Order | Purpose |
|-------|-------|---------|
| A | Act → Report | Post-hoc honesty |
| B | Report → Act | Belief anchoring |
| C | CoT → Act | Reasoning propagation |
| D | Act → Introspect | Retrospective report |

```bash
# Single probe run
python experiments/belief_probe_baseline.py llama3.1:latest --anti-uniform -n 50

# Multi-question experiment
python experiments/belief_probe_baseline.py --experiment2 --anti-uniform -o data/results.json
```

### Cross-Provider Comparison
Compare OpenAI, Gemini, and local models:

```bash
python experiments/compare_providers.py --providers openai gemini --quick
```

### Gravitational Pull
Test whether constraint tightening increases boundary piling:

```bash
# Single question sweep
python experiments/gravitational_pull.py openai --n-trials 10 --verbose

# Multi-question sweep across providers
python experiments/gravitational_pull.py openai --multi-question -o data/gp_results.json
```

### Alignment Regularizer
Test whether calibration incentives improve belief reporting:

```bash
python experiments/alignment_regularizer.py openai
```

## Providers

The toolkit supports multiple inference backends:

| Provider | Token Beliefs | Setup |
|----------|---------------|-------|
| OpenAI | True logprobs | `OPENAI_API_KEY` in `.env` |
| Gemini | True logprobs | `GOOGLE_API_KEY` in `.env` |
| llama.cpp | True logprobs | Run `llama-server -m model.gguf -c 4096` |
| Ollama | Sampling-based | `ollama serve` (less accurate) |

**Note:** True logprobs are essential for confidence-related claims. Sampling-based estimation can invert correlations.

## Project Structure

```
experiments/
├── belief_probe_baseline.py    # Core probe harness
├── compare_providers.py        # Cross-provider comparison
├── gravitational_pull.py       # Constraint sweep experiments
├── alignment_regularizer.py    # Calibration incentive tests
├── generate_mc_questions.py    # Question generation
├── providers.py                # Provider abstraction layer
└── test_gemini.py              # Provider smoke test

data/
├── mc_questions.json           # 40 test questions (easy/ambiguous/adversarial)
├── gp_mq_*.json               # Gravitational pull results by provider
├── provider_comparison.json    # Cross-provider experiment results
└── experiment2_results.json    # Multi-question probe results

docs/labnotes/
├── belief-probe-notes.md                  # Probe experiment notes
├── gravitational-pull-experiment.md       # Constraint sweep analysis
└── token-belief-vs-stated-belief.md       # Synthesis of findings
```

## Setup

1. Create `.env` with API keys:
   ```
   OPENAI_API_KEY=sk-...
   GOOGLE_API_KEY=...
   ```

2. For local models, install llama.cpp:
   ```bash
   brew install llama.cpp

   # Download a model (e.g., Llama 3.1 8B)
   huggingface-cli download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \
       --include "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" \
       --local-dir models/

   # Start server
   llama-server -m models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf -c 4096
   ```

3. No dependencies beyond Python stdlib (uses `urllib` for API calls).

## Metrics

- **JS Divergence**: Jensen-Shannon divergence between token distribution and reported probabilities
- **Boundary Mass**: Fraction of reported probability within 3% of constraint bounds
- **Report Entropy**: Shannon entropy of reported distribution
- **Token Entropy**: Shannon entropy of behavioral (action) distribution

## Results Summary

### Provider Comparison (Gravitational Pull)

| Provider | Boundary Mass (5,80) → (20,55) | Pull Strength |
|----------|-------------------------------|---------------|
| OpenAI gpt-4o-mini | 18% → 43% | **STRONG** |
| Qwen 2.5 7B | 18% → 38% | **STRONG** |
| Llama 3.1 8B | 10% → 16% | MODERATE |
| Gemini 2.0-flash | 28% → 31% | WEAK |

### Mean JS Divergence (Belief-Action Misalignment)

| Provider | Mean JS |
|----------|---------|
| OpenAI gpt-4o-mini | 0.368 |
| Gemini 2.0-flash | 0.373 |
| Llama 3.1 8B | 0.374 |
| Qwen 2.5 7B | 0.466 |

All models show substantial belief-action misalignment (~0.37-0.47 JS divergence).

## Takeaway

LLMs exhibit a robust, learned decoupling between action confidence and reported uncertainty. Belief expression, reasoning, and action are partially independent control surfaces—measurable, intervenable, and not interchangeable.
