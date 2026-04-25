# Methodology: Investigating Belief-Action-Report Decoupling in LLMs

## Introduction

The central objective of this research is to distinguish between a Large Language Model's (LLM) behavioral belief—what its actions imply—and its reported belief—what it explicitly states about its own certainty. This document outlines the controlled, multi-faceted methodology designed to probe, measure, and understand the decoupling between these two belief systems. The methods detailed here provide a systematic framework for analyzing this phenomenon across various models, framing interventions, and task conditions. Our findings suggest that belief expression, reasoning, and action are partially independent control surfaces—measurable, intervenable, and not interchangeable.

## 1. Research Scope and Core Objective

Precisely defining the research scope is of strategic importance for generating reproducible insights. This investigation focuses on how LLMs express, act upon, and report uncertainty within the controlled environment of multiple-choice questions. The primary goal is to establish a robust and replicable framework for studying the divergence between a model's implicit confidence and its explicit statements, thereby enabling a more nuanced understanding of its internal state.

The following core concepts are central to this investigation:

* **Behavioral Belief**: The model's implicit belief as revealed by its action distribution, derived from the softmax over token log-probabilities (logprobs) for the valid answer choices.
* **Reported Belief**: The model's explicit, stated probability distribution, elicited by directly prompting the model to report its confidence levels for each possible answer.
* **Belief-Action-Report Decoupling**: The core phenomenon under investigation, where a model's behavioral belief and its reported belief do not align. This manifests as a mismatch between the confidence implied by its actions and the uncertainty conveyed in its self-reporting.

The scope of the investigation encompassed several key dimensions: testing different probe orderings, applying reporting constraints, categorizing tasks by difficulty, utilizing distinct methods for measuring action distribution, and conducting cross-provider comparisons. The following sections detail the specific experimental design used to explore these concepts.

## 2. Experimental Design and Interventions

A series of structured probes were designed to systematically manipulate the sequence of model actions and reports. This experimental setup allows for a causal analysis of how different framings and orderings influence the alignment between a model's behavioral and reported beliefs. By controlling the context in which the model acts and reflects, we can isolate the effects of specific interventions.

The experimental probes and framings included:

* **Probe Orderings**: Different sequences of asking the model to act and report were tested to understand their causal impact on belief alignment. These sequences, referred to as A/B/C/D and their extended variants, systematically alter whether the model reports its beliefs before acting, after acting, or in conjunction with other reasoning steps.
* **Core Interventions**:
    * **Act vs. Report**: These represent the two fundamental prompts. An Act prompt requires the model to choose a final answer from the multiple-choice options. A Report prompt requires the model to provide a full probability distribution across all possible answers.
    * **Chain-of-Thought (CoT)**: This intervention prompts the model to generate step-by-step reasoning before committing to an action. It is designed to elicit the model's analytical process.
    * **Introspection**: This framing prompts the model to explicitly self-assess its own uncertainty and reasoning process, encouraging a more reflective mode of response.

### 2.1. Probe Inputs and Stratification

All experimental probes were based on multiple-choice (MC) questions, providing a structured format for measuring both actions and reported beliefs. To understand how task characteristics influence model behavior, these questions were stratified into three distinct categories:

* **Easy**: Questions with a clear, factually correct answer, designed to test the model's baseline performance and belief alignment under conditions of high certainty.
* **Ambiguous**: Questions where multiple answers could be plausible or where the correct answer is not definitively established, designed to probe behavior under inherent uncertainty.
* **Adversarial**: Questions designed to be tricky or misleading, testing the model's robustness and how it handles situations where its initial inclinations may be incorrect.

### 2.2. Models Tested

To ensure the generality of the findings, models from several major providers were evaluated. This cross-provider comparison helps distinguish between fundamental LLM behaviors and provider-specific artifacts. The study included models from:

* OpenAI
* Gemini
* Llama3.1
* Qwen2.5

This experimental setup, combining controlled probes with stratified inputs across multiple models, provides the foundation for the measurement and analysis techniques that follow.

## 3. Measurement and Analysis Techniques

Robust measurement is critical for quantifying an abstract concept like belief-action decoupling. Vague observations are insufficient; precise metrics are required to capture the nature and magnitude of the divergence between what a model does and what it says. This section details the specific techniques used to quantify model beliefs and measure the mismatch between them.

### 3.1. Quantifying Action and Reported Beliefs

Two distinct methods were used to estimate the model's action distribution, with one proving significantly more reliable for claims related to model confidence.

* **True Logprob-Based Distribution**: This method involves the direct measurement of the model's output logits (log probabilities) over the valid multiple-choice answer tokens. This technique provides the ground truth for the model's behavioral belief at the moment of action, reflecting its raw, unmediated output tendency.
* **Sampling-Based Distribution**: This is an estimation method where the action distribution is inferred by repeatedly sampling the model's output many times and calculating the frequency of each answer. This approach was found to introduce a "smoothing bias," which can obscure or even invert the true relationship between model confidence and belief-action decoupling.

### 3.2. Measuring Belief-Action Decoupling

To quantify the mismatch between the two belief systems, the following metrics were employed:

* The primary metric for measuring the divergence between the action distribution and the reported probability distribution was the **Jensen-Shannon (JS) divergence**. This provides a single, principled score for the degree of mismatch.
* **Correlation analysis** was used to understand the conditions under which decoupling is most severe. Specifically, the correlation between the JS divergence and the entropy of the action distribution (H_action) was calculated. A strong negative correlation (Corr(JS, H_action) ≈ −0.5 for OpenAI/Gemini models) indicates that the mismatch is greatest when the model's action is most confident (i.e., when action entropy is low).

### 3.3. Mitigating Reporting Biases

A significant methodological challenge was the discovery of a powerful bias in how models report uncertainty.

* **The "Uniform Reporting" Problem**: Historical local runs suggested that, when unconstrained, models often default to reporting uniform probability distributions (e.g., 25/25/25/25 for a four-option question). This behavior occurs regardless of the actual confidence of their chosen action, rendering naive self-reports uninformative. The paper-grade unconstrained raw artifact is not yet committed, so exact rates should be quoted from `data/unconstrained_baseline_results.json` after regeneration rather than from early notes.
* **The Solution**: To address this, an "anti-uniform constraint" was introduced into the reporting prompt (e.g., specifying that probabilities must be between a minimum of 5% and a maximum of 80%). This mild constraint is highly effective in the checked-in constrained baseline, where uniform reports fall to low single digits across probes. This compels the model to provide more informative, graded uncertainty reports without forcing it to "game" the boundaries.

These analytical techniques enabled a precise quantification of LLM behavior, leading to several robust and reproducible findings.

## 4. Summary of Methodological Findings

The application of the methodology described above yielded several robust and reproducible findings regarding LLM behavior. These discoveries highlight the complex and often counter-intuitive relationship between a model's internal state, its actions, and its self-reports. This section summarizes what these methods revealed, linking each discovery back to the techniques used to uncover it.

1. **F1 & F2: The Uniform Reporting Attractor and Its Mitigation** — Historical runs indicate that models exhibit a stable, learned policy to report uniform probabilities when unconstrained, while the checked-in constrained baseline shows that a mild anti-uniform constraint breaks this pattern and yields low-single-digit uniform rates. This supports the claim that models possess the ability to express graded uncertainty when properly prompted, while also making the repo's current evidentiary boundary explicit.

2. **F3: Decoupling is Greatest Under High Confidence** — Using true logprobs, analysis reveals a strong negative correlation (≈ −0.5) between belief-action mismatch and action entropy. This means decoupling is highest precisely when the model's action is most confident, a classic sign of a disconnect where models act decisively while simultaneously reporting broad uncertainty.

3. **F4: Probe Order and Framing Heavily Influence Alignment** — The sequence and framing of interventions have a significant, predictable impact. Prompting Report→Act acts as an anchoring device that reduces decoupling. Chain-of-Thought collapses narratives, not uncertainty; it improves alignment on easy tasks but severely worsens it on ambiguous ones, producing a near-zero correlation between JS divergence and action entropy that indicates a structural mismatch. An Introspective framing acts as a general stabilizer, but this should be understood as a useful regularization method, not as neutral 'truth access'.

4. **F5: True Logprobs are Essential for Accurate Measurement** — Methodology is paramount: sampling-based action estimation can mask or even invert true effects. For instance, sampling on Ollama models showed a positive correlation between decoupling and confidence, whereas true logprobs on OpenAI/Gemini models showed the opposite negative correlation. True logprobs are therefore essential for any claims related to model confidence.

These findings open up a new set of questions, setting a clear agenda for future work in this domain.

## 5. Open Questions and Future Directions

While the current methodology provided clear answers to its initial questions, it also surfaced a new set of unresolved challenges and opportunities. The findings confirm that belief, action, and reporting are partially independent and controllable facets of LLM behavior. This section outlines the next steps for research, defining a path for future investigations into these complex systems.

### What is the nature of the "reporting policy" itself?

Is this a distinct learned mechanism, or a conditional mode of generation triggered by specific language? Understanding its architecture and provider-specificity is a key next step.

**Next Steps**: Investigate with Report→Act→Report loops to test consistency; Act→Report→Act₂ sequences with flip analysis to measure influence; and cross-model prompt ablations to isolate triggers.

### Can belief revision be actively controlled, not just observed?

When a model is prompted to reconsider an answer, we need to distinguish between justified belief updates and mere confabulation.

**Next Steps**: Analyze behavior in Act→Report→Act₂ and Act→CoT→Act₂ sequences, using flip quality metrics like the change in entropy or total variation distance to measure the substance of belief changes.

### Can a generalized, adaptive probe-selection policy be developed?

Given that different probes (CoT, Introspection) excel on different task types, an automated policy could optimize performance by selecting the best probe for a given input.

**Next Steps**: Test model-internal signals, such as pre-action entropy, as a threshold for automatically routing queries to either a CoT or an Introspection probe.

### How does internal belief alignment relate to external environmental truth?

This study focused on the internal consistency between a model's beliefs and actions. The next phase must connect this internal alignment to external correctness.

**Next Steps**: Reintroduce environments with known ground truths (or posteriors) to measure whether probes that stabilize internal belief alignment also preserve or improve accuracy against the environment.

### What are the provider-specific effects and biases?

Subtle but consistent differences were observed across models, such as variations in logprob sharpness or residual rates of uniform reporting. These require deeper investigation.

**Next Steps**: Implement strict token canonicalization to rule out formatting artifacts and develop per-provider calibration curves to isolate and characterize unique model behaviors.
