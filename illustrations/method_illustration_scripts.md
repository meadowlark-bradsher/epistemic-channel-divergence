# Illustration Scripts: Experimental Method

Instructions for creating visual illustrations explaining **how** we measure LLM beliefs. Focus on methodology, not results.

---

## Illustration 1: Why Multiple Choice?

**Purpose:** Explain why we use multiple choice format — it constrains the output space to exactly 4 tokens we can measure.

**Scene:**

**Panel A: "Open-ended questions = unmeasurable"**
- Show a prompt: "What is the capital of France?"
- Show model outputting: "The capital of France is Paris, which has been..."
- Draw a cloud of tokens streaming out (messy, unbounded)
- Label: "Free-form output — no clear way to measure confidence"

**Panel B: "Multiple choice = measurable"**
- Show a prompt:
  ```
  What is the capital of France?

  A) London
  B) Paris
  C) Berlin
  D) Madrid

  Answer with A, B, C, or D:
  ```
- Show model outputting just: **"B"**
- Draw exactly 4 tokens (A, B, C, D) with the output constrained to one of them
- Label: "Constrained to 4 tokens — we can measure the probability of each"

**Key annotation:** "Multiple choice creates a closed, measurable decision space."

---

## Illustration 2: What Are Logprobs?

**Purpose:** Explain that when a model generates a token, it actually computes probabilities over its entire vocabulary, and we can access these.

**Scene:**

**Step 1: "The model sees a prompt"**
- Show the multiple choice prompt going into a model icon

**Step 2: "Before generating, it computes probabilities"**
- Show an "inside view" or "x-ray" of the model
- Display a simplified vocabulary with probabilities:
  ```
  Token        Probability
  ─────────────────────────
  "A"          0.05
  "B"          0.82
  "C"          0.08
  "D"          0.05
  "The"        0.00001
  "Paris"      0.00001
  "I"          0.00001
  ...          ...
  (50,000+ tokens)
  ```
- Label: "The model assigns a probability to every possible next token"

**Step 3: "It samples one token to output"**
- Show "B" being selected/sampled from the distribution
- Label: "Output: B (sampled from this distribution)"

**Key annotation:** "Logprobs let us see the full probability distribution, not just the sampled output."

---

## Illustration 3: Extracting Letter Token Probabilities

**Purpose:** Show how we search for specifically A, B, C, D in the logprobs and extract their probabilities.

**Scene:**

**Left side: "Raw logprobs from the model"**
- Show a list of the top tokens returned by the API:
  ```
  Top Logprobs (from API)
  ───────────────────────
  "B"      -0.20  (82%)
  "C"      -2.53  (8%)
  "A"      -3.00  (5%)
  "D"      -3.00  (5%)
  "The"    -8.50  (~0%)
  "I"      -9.20  (~0%)
  ...
  ```

**Middle: "Filter for answer tokens"**
- Draw a filter/funnel icon
- Show only A, B, C, D passing through
- Label: "Search for tokens: A, B, C, D"

**Right side: "Extracted belief distribution"**
- Show a clean bar chart:
  ```
  A  ████░░░░░░░░░░░░   5%
  B  █████████████████  82%
  C  ██████░░░░░░░░░░░   8%
  D  ████░░░░░░░░░░░░░   5%
  ```
- Label: "Token Belief Distribution"

**Key annotation:** "We extract exactly the 4 answer tokens to get a probability distribution over choices."

---

## Illustration 4: Token Belief vs. Stated Belief

**Purpose:** Show the two different ways we can measure what a model "believes" — what it does vs. what it says.

**Scene (two parallel paths):**

**Path 1: "Token Belief" (Behavioral)**
- Prompt → Model → Single letter output
- "X-ray view" showing logprobs bar chart
- Label: "What the model DOES"
- Subtitle: "Measured from token probabilities at decision point"

**Path 2: "Stated Belief" (Declarative)**
- Prompt asking: "Report your probability for each option as percentages"
- Model outputs: `{"A": 25, "B": 25, "C": 25, "D": 25}`
- Convert to bar chart (all equal)
- Label: "What the model SAYS"
- Subtitle: "Model's self-reported confidence"

**At bottom: Show both bar charts side by side**
- Token belief: peaked at B
- Stated belief: uniform
- Draw a "compare" or "vs" symbol between them
- Label: "We measure the divergence between these two distributions"

**No results annotation** — just show that we're comparing two measurements.

---

## Illustration 5: The Probe Variations (Method Overview)

**Purpose:** Introduce the different probe orderings as experimental conditions — variations in how we structure the interaction.

**Scene: Show 4 experimental conditions as flow diagrams**

**Probe A: Act → Report**
```
[Prompt] → [Get Action + Logprobs] → [Ask for Self-Report]
```
- Label: "Action first, then ask for beliefs"

**Probe B: Report → Act**
```
[Prompt] → [Ask for Self-Report] → [Get Action + Logprobs]
```
- Label: "Ask for beliefs first, then action"

**Probe C: CoT → Act**
```
[Prompt] → [Generate Reasoning] → [Get Action + Logprobs] → [Ask for Self-Report]
```
- Label: "Reasoning first, then action"

**Probe D: Act → Introspect**
```
[Prompt] → [Get Action + Logprobs] → [Ask for Retrospective Report]
```
- Label: "Action first, then introspective report"

**Key annotation:** "Different orderings let us test how the sequence of prompts affects belief-action alignment."

**Note:** Do NOT include findings or results. Just show these as experimental conditions.

---

## Illustration 6: The Complete Measurement Pipeline

**Purpose:** Show the full method from question to measured divergence.

**Scene: A horizontal pipeline diagram**

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Multiple  │    │    Model    │    │   Extract   │    │   Compare   │
│   Choice    │ →  │  Generates  │ →  │   A,B,C,D   │ →  │    Token    │
│   Question  │    │   Answer    │    │   Logprobs  │    │  vs Stated  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                          │
                          ↓
                   ┌─────────────┐
                   │  Ask Model  │
                   │  to Report  │
                   │   Beliefs   │
                   └─────────────┘
```

**Labels for each stage:**
1. "Constrain to 4 options"
2. "Generate single token"
3. "Filter logprobs for A, B, C, D"
4. "Measure divergence between distributions"

---

## Visual Style Notes

- Clean, technical, methodological tone
- Avoid value judgments or results (no "good"/"bad", no "finding:", no checkmarks/x-marks)
- Color coding:
  - Blue: Inputs/prompts
  - Green: Token beliefs (behavioral)
  - Orange: Stated beliefs (declarative)
  - Gray: Processing steps
- Use flowchart/pipeline visual language
- Emphasize the measurement apparatus, not what it measures
