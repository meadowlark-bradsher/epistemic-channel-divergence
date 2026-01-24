#!/usr/bin/env python3
"""
Gravitational Pull Experiment

Tests the hypothesis that models exhibit a "residual attractor" toward uniform
distributions even after the explicit uniform attractor is broken.

Key insight: When we force non-uniform reports via constraint (5-80%), models
don't distribute mass genuinely - they pile it at constraint boundaries,
suggesting the uniform attractor wasn't eliminated, just displaced.

Test A: Constraint Sweep
- Gradually tighten constraint: (5,80) → (10,70) → (15,60) → (20,55)
- Measure boundary piling at each level
- Hypothesis: Piling increases as constraints tighten (model has "nowhere to go")

Test B: Alignment Regularizer (future work)
- Add weak incentive for token-report alignment
- Test if model is capable but incentive-avoiding
"""

import json
import math
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Import from baseline
from belief_probe_baseline import (
    OPTIONS,
    PROMPT_BASE,
    parse_report,
    entropy,
    js_divergence,
    total_variation,
    format_question_prompt,
)

# Import providers for multi-provider support
from providers import get_provider, PROVIDERS, BaseProvider


@dataclass
class ConstraintLevel:
    """A single constraint configuration."""
    min_pct: int  # Minimum percentage (e.g., 5 for 5%)
    max_pct: int  # Maximum percentage (e.g., 80 for 80%)

    @property
    def min_frac(self) -> float:
        return self.min_pct / 100.0

    @property
    def max_frac(self) -> float:
        return self.max_pct / 100.0

    @property
    def label(self) -> str:
        return f"({self.min_pct},{self.max_pct})"

    def prompt_constraint(self) -> str:
        """Generate the constraint text for prompts."""
        return f"""
CONSTRAINT: You must allocate at least {self.min_pct}% and at most {self.max_pct}% to any option unless logically certain.
Do not use 25/25/25/25 — express your actual confidence distribution."""


# Default constraint sweep levels
CONSTRAINT_LEVELS = [
    ConstraintLevel(5, 80),   # Loose - original
    ConstraintLevel(10, 70),  # Moderate
    ConstraintLevel(15, 60),  # Tight
    ConstraintLevel(20, 55),  # Very tight
]


@dataclass
class BoundaryMetrics:
    """Metrics for boundary piling analysis."""
    constraint: ConstraintLevel

    # Raw counts
    options_at_min: int = 0
    options_at_max: int = 0
    options_near_min: int = 0  # Within 3% of min
    options_near_max: int = 0  # Within 3% of max

    # Derived metrics
    boundary_mass: float = 0.0  # Total probability at boundaries
    interior_mass: float = 0.0  # Total probability in interior
    is_boundary_piling: bool = False

    # Distribution shape
    avg_distance_to_uniform: float = 0.0
    report_entropy: float = 0.0

    def to_dict(self) -> dict:
        return {
            "constraint": self.constraint.label,
            "options_at_min": self.options_at_min,
            "options_at_max": self.options_at_max,
            "options_near_min": self.options_near_min,
            "options_near_max": self.options_near_max,
            "boundary_mass": self.boundary_mass,
            "interior_mass": self.interior_mass,
            "is_boundary_piling": self.is_boundary_piling,
            "avg_distance_to_uniform": self.avg_distance_to_uniform,
            "report_entropy": self.report_entropy,
        }


def compute_boundary_metrics(
    probs: Dict[str, float],
    constraint: ConstraintLevel,
    tolerance: float = 0.03,
) -> BoundaryMetrics:
    """Compute detailed boundary metrics for a reported distribution."""
    min_b = constraint.min_frac
    max_b = constraint.max_frac

    at_min = sum(1 for p in probs.values() if abs(p - min_b) < tolerance)
    at_max = sum(1 for p in probs.values() if abs(p - max_b) < tolerance)
    near_min = sum(1 for p in probs.values() if p < min_b + tolerance)
    near_max = sum(1 for p in probs.values() if p > max_b - tolerance)

    # Boundary mass = sum of probs at/near boundaries
    boundary_mass = sum(
        p for p in probs.values()
        if abs(p - min_b) < tolerance or abs(p - max_b) < tolerance
    )
    interior_mass = 1.0 - boundary_mass

    # Boundary piling: 3+ options at min OR 2+ at max
    is_piling = (at_min >= 3) or (at_max >= 2)

    # Distance to uniform
    avg_dist = sum(abs(p - 0.25) for p in probs.values()) / 4

    return BoundaryMetrics(
        constraint=constraint,
        options_at_min=at_min,
        options_at_max=at_max,
        options_near_min=near_min,
        options_near_max=near_max,
        boundary_mass=boundary_mass,
        interior_mass=interior_mass,
        is_boundary_piling=is_piling,
        avg_distance_to_uniform=avg_dist,
        report_entropy=entropy(probs),
    )


@dataclass
class SweepTrialResult:
    """Result of a single trial at one constraint level."""
    constraint: ConstraintLevel
    trial_num: int

    # Distributions
    token_probs: Dict[str, float]
    report_probs: Dict[str, float]

    # Metrics
    js_divergence: float
    boundary_metrics: BoundaryMetrics

    # Diagnostics
    report_raw: str = ""
    parse_ok: bool = True


@dataclass
class SweepResult:
    """Aggregated results for constraint sweep."""
    constraint: ConstraintLevel
    n_trials: int

    # Aggregated metrics
    mean_js: float = 0.0
    std_js: float = 0.0
    mean_boundary_mass: float = 0.0
    piling_rate: float = 0.0  # Fraction of trials with boundary piling
    mean_options_at_min: float = 0.0
    mean_options_at_max: float = 0.0
    mean_report_entropy: float = 0.0

    trials: List[SweepTrialResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "constraint": self.constraint.label,
            "n_trials": self.n_trials,
            "mean_js": self.mean_js,
            "std_js": self.std_js,
            "mean_boundary_mass": self.mean_boundary_mass,
            "piling_rate": self.piling_rate,
            "mean_options_at_min": self.mean_options_at_min,
            "mean_options_at_max": self.mean_options_at_max,
            "mean_report_entropy": self.mean_report_entropy,
        }


def get_constrained_report(
    provider: BaseProvider,
    prompt_base: str,
    constraint: ConstraintLevel,
) -> Tuple[Dict[str, float], str, bool]:
    """Get self-reported probabilities with specified constraint."""
    report_prompt = prompt_base + f"""

Report your probability for each option as percentages that sum to 100.
{constraint.prompt_constraint()}
Return EXACTLY this JSON format, no other text:
{{"A":25,"B":25,"C":25,"D":25}}"""

    raw_response = provider.generate(report_prompt, max_tokens=100)
    probs, parse_ok = parse_report(raw_response)
    return probs, raw_response, parse_ok


def run_constraint_sweep(
    provider_name: str = "openai",
    model_name: str = "gpt-4o-mini",
    constraints: List[ConstraintLevel] = None,
    n_trials: int = 10,
    n_samples: int = 30,
    temperature: float = 1.0,
    prompt_base: str = None,
    verbose: bool = False,
    base_url: str = None,
) -> List[SweepResult]:
    """Run constraint sweep experiment.

    For each constraint level, run n_trials and measure:
    - Boundary piling behavior
    - JS divergence between token and reported beliefs
    - Distribution shape metrics
    """
    if constraints is None:
        constraints = CONSTRAINT_LEVELS

    if prompt_base is None:
        prompt_base = PROMPT_BASE

    print("=" * 70)
    print("GRAVITATIONAL PULL: Constraint Sweep Experiment")
    print("=" * 70)
    print(f"Provider: {provider_name}")
    print(f"Model: {model_name}")
    print(f"Trials per constraint: {n_trials}")
    if provider_name == "ollama":
        print(f"Token samples: {n_samples}")
    print(f"Constraints: {[c.label for c in constraints]}")
    print("=" * 70)

    # Get provider
    provider_kwargs = {"model_name": model_name}
    if provider_name == "ollama":
        provider_kwargs["n_samples"] = n_samples
    if provider_name == "llamacpp" and base_url:
        provider_kwargs["base_url"] = base_url
    provider = get_provider(provider_name, **provider_kwargs)

    act_prompt = prompt_base + "\n\nAnswer with A, B, C, or D:\n"

    all_results = []

    for constraint in constraints:
        print(f"\n--- Constraint {constraint.label} ---")

        trials = []
        js_values = []
        boundary_masses = []
        piling_count = 0
        at_min_values = []
        at_max_values = []
        entropies = []

        for trial in range(n_trials):
            print(f"  Trial {trial + 1}/{n_trials}...", end=" ", flush=True)

            # Get token distribution
            action_result = provider.get_action_probs(act_prompt, temperature=temperature)
            token_probs = action_result.probs

            # Get constrained report
            report_probs, report_raw, parse_ok = get_constrained_report(
                provider, prompt_base, constraint
            )

            # Compute metrics
            js = js_divergence(token_probs, report_probs)
            boundary_metrics = compute_boundary_metrics(report_probs, constraint)

            trial_result = SweepTrialResult(
                constraint=constraint,
                trial_num=trial,
                token_probs=token_probs,
                report_probs=report_probs,
                js_divergence=js,
                boundary_metrics=boundary_metrics,
                report_raw=report_raw,
                parse_ok=parse_ok,
            )
            trials.append(trial_result)

            # Accumulate stats
            js_values.append(js)
            boundary_masses.append(boundary_metrics.boundary_mass)
            if boundary_metrics.is_boundary_piling:
                piling_count += 1
            at_min_values.append(boundary_metrics.options_at_min)
            at_max_values.append(boundary_metrics.options_at_max)
            entropies.append(boundary_metrics.report_entropy)

            status = "PILING" if boundary_metrics.is_boundary_piling else "ok"
            print(f"JS={js:.3f} boundary={boundary_metrics.boundary_mass:.2f} [{status}]")

            if verbose:
                print(f"    Token: {token_probs}")
                print(f"    Report: {report_probs}")

        # Aggregate
        mean_js = sum(js_values) / len(js_values)
        std_js = (sum((x - mean_js) ** 2 for x in js_values) / len(js_values)) ** 0.5

        sweep_result = SweepResult(
            constraint=constraint,
            n_trials=n_trials,
            mean_js=mean_js,
            std_js=std_js,
            mean_boundary_mass=sum(boundary_masses) / len(boundary_masses),
            piling_rate=piling_count / n_trials,
            mean_options_at_min=sum(at_min_values) / len(at_min_values),
            mean_options_at_max=sum(at_max_values) / len(at_max_values),
            mean_report_entropy=sum(entropies) / len(entropies),
            trials=trials,
        )
        all_results.append(sweep_result)

    return all_results


def print_sweep_summary(results: List[SweepResult]):
    """Print summary table of sweep results."""
    print("\n" + "=" * 70)
    print("CONSTRAINT SWEEP SUMMARY")
    print("=" * 70)
    print(f"{'Constraint':<12} {'JS Div':>10} {'Bnd Mass':>10} {'Piling%':>10} {'@Min':>8} {'@Max':>8} {'H(rep)':>8}")
    print("-" * 70)

    for r in results:
        print(
            f"{r.constraint.label:<12} "
            f"{r.mean_js:>10.4f} "
            f"{r.mean_boundary_mass:>10.3f} "
            f"{r.piling_rate * 100:>9.1f}% "
            f"{r.mean_options_at_min:>8.2f} "
            f"{r.mean_options_at_max:>8.2f} "
            f"{r.mean_report_entropy:>8.3f}"
        )

    # Analysis
    print("\n" + "=" * 70)
    print("GRAVITATIONAL PULL ANALYSIS")
    print("=" * 70)

    # Check if piling increases with tighter constraints
    piling_rates = [r.piling_rate for r in results]
    if len(piling_rates) >= 2:
        trend = piling_rates[-1] - piling_rates[0]
        if trend > 0.1:
            print(f"✓ Piling INCREASES with tighter constraints (+{trend * 100:.1f}%)")
            print("  → Supports gravitational pull hypothesis")
        elif trend < -0.1:
            print(f"✗ Piling DECREASES with tighter constraints ({trend * 100:.1f}%)")
            print("  → Against gravitational pull hypothesis")
        else:
            print(f"~ Piling rate stable across constraints (Δ={trend * 100:.1f}%)")

    # Check boundary mass trend
    boundary_masses = [r.mean_boundary_mass for r in results]
    if len(boundary_masses) >= 2:
        bm_trend = boundary_masses[-1] - boundary_masses[0]
        if bm_trend > 0.1:
            print(f"✓ Boundary mass INCREASES with tighter constraints (+{bm_trend:.2f})")
            print("  → Model hedges by piling at boundaries")

    # Check entropy trend
    entropies = [r.mean_report_entropy for r in results]
    if len(entropies) >= 2:
        ent_trend = entropies[-1] - entropies[0]
        direction = "increases" if ent_trend > 0 else "decreases"
        print(f"~ Report entropy {direction} with tighter constraints (Δ={ent_trend:.3f})")


def run_multi_question_sweep(
    provider_name: str = "openai",
    model_name: str = "gpt-4o-mini",
    questions_file: str = "../data/mc_questions.json",
    constraints: List[ConstraintLevel] = None,
    n_samples: int = 20,
    temperature: float = 1.0,
    output_file: str = None,
    base_url: str = None,
):
    """Run constraint sweep across multiple questions."""
    if constraints is None:
        constraints = CONSTRAINT_LEVELS

    print("=" * 70)
    print("GRAVITATIONAL PULL: Multi-Question Sweep")
    print("=" * 70)
    print(f"Provider: {provider_name}")
    print(f"Model: {model_name}")
    print(f"Questions: {questions_file}")
    print(f"Constraints: {[c.label for c in constraints]}")
    print("=" * 70)

    # Load questions
    with open(questions_file) as f:
        data = json.load(f)
    questions = data["questions"]
    print(f"Loaded {len(questions)} questions\n")

    # Get provider
    provider_kwargs = {"model_name": model_name}
    if provider_name == "ollama":
        provider_kwargs["n_samples"] = n_samples
    if provider_name == "llamacpp" and base_url:
        provider_kwargs["base_url"] = base_url
    provider = get_provider(provider_name, **provider_kwargs)

    # Aggregate results by constraint
    results_by_constraint = {c.label: [] for c in constraints}

    for i, q in enumerate(questions):
        prompt_base = format_question_prompt(q)
        act_prompt = prompt_base + "\n\nAnswer with A, B, C, or D:\n"

        print(f"[{i + 1}/{len(questions)}] {q['question'][:50]}...")

        # Get token probs once (shared across constraints)
        action_result = provider.get_action_probs(act_prompt, temperature=temperature)
        token_probs = action_result.probs

        for constraint in constraints:
            report_probs, _, parse_ok = get_constrained_report(
                provider, prompt_base, constraint
            )

            js = js_divergence(token_probs, report_probs)
            metrics = compute_boundary_metrics(report_probs, constraint)

            results_by_constraint[constraint.label].append({
                "question_id": i,
                "js": js,
                "boundary_mass": metrics.boundary_mass,
                "is_piling": metrics.is_boundary_piling,
                "options_at_min": metrics.options_at_min,
                "options_at_max": metrics.options_at_max,
                "report_entropy": metrics.report_entropy,
            })

            status = "⚠" if metrics.is_boundary_piling else "✓"
            print(f"  {constraint.label}: JS={js:.3f} bnd={metrics.boundary_mass:.2f} {status}")

    # Summarize
    print("\n" + "=" * 70)
    print("MULTI-QUESTION SWEEP SUMMARY")
    print("=" * 70)
    print(f"{'Constraint':<12} {'Mean JS':>10} {'Mean Bnd':>10} {'Piling%':>10} {'Mean @Min':>10}")
    print("-" * 55)

    summary_data = {}

    for constraint in constraints:
        results = results_by_constraint[constraint.label]
        n = len(results)

        mean_js = sum(r["js"] for r in results) / n
        mean_bnd = sum(r["boundary_mass"] for r in results) / n
        piling_rate = sum(1 for r in results if r["is_piling"]) / n
        mean_at_min = sum(r["options_at_min"] for r in results) / n

        print(
            f"{constraint.label:<12} "
            f"{mean_js:>10.4f} "
            f"{mean_bnd:>10.3f} "
            f"{piling_rate * 100:>9.1f}% "
            f"{mean_at_min:>10.2f}"
        )

        summary_data[constraint.label] = {
            "mean_js": mean_js,
            "mean_boundary_mass": mean_bnd,
            "piling_rate": piling_rate,
            "mean_options_at_min": mean_at_min,
            "n_questions": n,
        }

    # Save results
    if output_file:
        output_data = {
            "config": {
                "provider": provider_name,
                "model": model_name,
                "questions_file": questions_file,
                "n_samples": n_samples,
                "temperature": temperature,
                "constraints": [c.label for c in constraints],
                "timestamp": datetime.now().isoformat(),
            },
            "summary": summary_data,
            "by_constraint": results_by_constraint,
        }
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    return summary_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gravitational Pull Experiment - Test residual attractor hypothesis"
    )
    parser.add_argument(
        "-p", "--provider",
        default="openai",
        choices=["openai", "gemini", "ollama", "llamacpp"],
        help="Provider (openai, gemini, ollama, llamacpp)"
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Base URL for llamacpp server (default: http://localhost:8080)"
    )
    parser.add_argument(
        "-m", "--model",
        default=None,
        help="Model name (default: gpt-4o-mini for openai, gemini-2.0-flash for gemini)"
    )
    parser.add_argument(
        "-n", "--n-trials",
        type=int,
        default=10,
        help="Trials per constraint level"
    )
    parser.add_argument(
        "-s", "--n-samples",
        type=int,
        default=30,
        help="Token samples for behavioral belief (ollama only)"
    )
    parser.add_argument(
        "-t", "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed trial output"
    )
    parser.add_argument(
        "--multi-question",
        action="store_true",
        help="Run sweep across multiple questions"
    )
    parser.add_argument(
        "--questions",
        default="../data/mc_questions.json",
        help="Questions file for multi-question mode"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file for results (JSON)"
    )

    args = parser.parse_args()

    # Set default model based on provider
    if args.model is None:
        defaults = {
            "openai": "gpt-4o-mini",
            "gemini": "gemini-2.0-flash",
            "ollama": "llama3.1:latest",
            "llamacpp": "local",
        }
        args.model = defaults[args.provider]

    # Build extra kwargs for provider
    provider_kwargs = {}
    if args.base_url:
        provider_kwargs["base_url"] = args.base_url

    if args.multi_question:
        run_multi_question_sweep(
            provider_name=args.provider,
            model_name=args.model,
            questions_file=args.questions,
            n_samples=args.n_samples,
            temperature=args.temperature,
            output_file=args.output,
            base_url=args.base_url,
        )
    else:
        results = run_constraint_sweep(
            provider_name=args.provider,
            model_name=args.model,
            n_trials=args.n_trials,
            n_samples=args.n_samples,
            temperature=args.temperature,
            verbose=args.verbose,
            base_url=args.base_url,
        )
        print_sweep_summary(results)

        if args.output:
            output_data = {
                "config": {
                    "provider": args.provider,
                    "model": args.model,
                    "n_trials": args.n_trials,
                    "n_samples": args.n_samples,
                    "temperature": args.temperature,
                    "timestamp": datetime.now().isoformat(),
                },
                "results": [r.to_dict() for r in results],
            }
            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"\nResults saved to: {args.output}")
