#!/usr/bin/env python3
"""
ESI Runner - Epistemic Sensitivity under Intervention

Measures stability of action (token) and report (stated) beliefs
under semantic-preserving interventions.

Intervention types:
- SOC (Skip-One-Char): Remove characters from words
- ORDER: Shuffle option order (A,B,C,D positions)

Metrics computed:
- ESI_action: Mean Hellinger distance of action distributions under intervention
- ESI_report: Mean Hellinger distance of reported distributions under intervention
- ΔJS: Change in JS(action||report) under intervention

Usage:
    python experiments/esi_runner.py openai --probe A --intervention soc
    python experiments/esi_runner.py openai --probe A --probe B --probe C --probe D --intervention order
    python experiments/esi_runner.py --providers openai gemini --full -o results/esi/full_run.jsonl
"""

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from interventions.soc import soc_intervene_batch, intervene_question_only, spot_check_interventions
from interventions.option_order import shuffle_options_batch, remap_probs
from experiments.providers import get_provider, BaseProvider, ActionResult, ReportResult

OPTIONS = ["A", "B", "C", "D"]


# =============================================================================
# Metrics
# =============================================================================

def hellinger_distance(p: Dict[str, float], q: Dict[str, float]) -> float:
    """Hellinger distance between two distributions. Range: [0, 1], symmetric."""
    sum_sq = sum((math.sqrt(p.get(k, 0)) - math.sqrt(q.get(k, 0))) ** 2 for k in OPTIONS)
    return math.sqrt(sum_sq) / math.sqrt(2)


def js_divergence(p: Dict[str, float], q: Dict[str, float], eps: float = 1e-10) -> float:
    """Jensen-Shannon divergence between two distributions."""
    m = {k: (p.get(k, 0) + q.get(k, 0)) / 2 for k in OPTIONS}

    def kl(a, b):
        return sum(a.get(k, 0) * math.log2((a.get(k, 0) + eps) / (b.get(k, 0) + eps))
                   for k in OPTIONS if a.get(k, 0) > 0)

    return (kl(p, m) + kl(q, m)) / 2


def entropy(probs: Dict[str, float]) -> float:
    """Shannon entropy in bits."""
    return -sum(p * math.log2(p) for p in probs.values() if p > 0)


def is_uniform(probs: Dict[str, float], tolerance: float = 0.02) -> bool:
    """Check if distribution is approximately uniform (25/25/25/25)."""
    return all(abs(probs.get(k, 0) - 0.25) < tolerance for k in OPTIONS)


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class ESIResult:
    """Results for a single item under ESI intervention."""
    item_id: str
    probe_type: str
    intervention_type: str
    provider: str
    model: str
    question_text: str
    category: str

    # Original results
    action_original: Dict[str, float]
    report_original: Dict[str, float]
    js_original: float
    action_entropy_original: float
    report_entropy_original: float
    report_is_uniform_original: bool

    # Intervened results (lists, one per variant)
    action_intervened: List[Dict[str, float]]
    report_intervened: List[Dict[str, float]]
    js_intervened: List[float]
    report_is_uniform_intervened: List[bool]

    # ESI metrics
    esi_action: float
    esi_report: float
    delta_js: float

    # Metadata
    n_variants: int
    seed: int
    parse_errors: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ESISummary:
    """Summary statistics for a probe/intervention/provider combination."""
    probe_type: str
    intervention_type: str
    provider: str
    n_items: int
    mean_esi_action: float
    mean_esi_report: float
    mean_delta_js: float
    std_esi_action: float
    std_esi_report: float
    std_delta_js: float
    uniform_rate_original: float
    uniform_rate_intervened: float
    uniform_rate_shift: float


# =============================================================================
# Probe functions
# =============================================================================

def format_question_prompt(question: dict) -> Tuple[str, str]:
    """Format a question dict as a prompt string.
    Returns: (full_prompt, question_text_only)
    """
    question_text = question["question"]
    lines = [
        "You are answering a multiple choice question.",
        "",
        "Question:",
        question_text,
        "",
        "Options:",
    ]
    for opt in OPTIONS:
        lines.append(f"{opt}) {question['options'][opt]}")
    return "\n".join(lines), question_text


def run_probe_A(provider: BaseProvider, prompt_base: str, anti_uniform: bool = True) -> Tuple[ActionResult, ReportResult]:
    """Probe A: Act → Report"""
    action_prompt = prompt_base + "\n\nAnswer with A, B, C, or D:"
    action_result = provider.get_action_probs(action_prompt)
    report_result = provider.get_report(prompt_base, anti_uniform=anti_uniform)
    return action_result, report_result


def run_probe_B(provider: BaseProvider, prompt_base: str, anti_uniform: bool = True) -> Tuple[ActionResult, ReportResult]:
    """Probe B: Report → Act"""
    report_result = provider.get_report(prompt_base, anti_uniform=anti_uniform)
    action_prompt = prompt_base + "\n\n--- BELIEF RECORDED ---\nNow answer with A, B, C, or D:"
    action_result = provider.get_action_probs(action_prompt)
    return action_result, report_result


def run_probe_C(provider: BaseProvider, prompt_base: str, anti_uniform: bool = True) -> Tuple[ActionResult, ReportResult]:
    """Probe C: CoT → Act"""
    cot_prompt = prompt_base + "\n\nThink through the options carefully:"
    cot_output = provider.generate(cot_prompt, max_tokens=200)
    action_prompt = f"{prompt_base}\n\n--- THINKING ---\n{cot_output}\n--- END THINKING ---\n\nNow answer with A, B, C, or D:"
    action_result = provider.get_action_probs(action_prompt)
    report_result = provider.get_report(prompt_base, anti_uniform=anti_uniform)
    return action_result, report_result


def run_probe_D(provider: BaseProvider, prompt_base: str, anti_uniform: bool = True) -> Tuple[ActionResult, ReportResult]:
    """Probe D: Act → Introspect"""
    action_prompt = prompt_base + "\n\nAnswer with A, B, C, or D:"
    action_result = provider.get_action_probs(action_prompt)
    report_result = provider.get_report(prompt_base, introspective=True, anti_uniform=anti_uniform)
    return action_result, report_result


PROBE_FUNCTIONS = {"A": run_probe_A, "B": run_probe_B, "C": run_probe_C, "D": run_probe_D}


# =============================================================================
# ESI Runner
# =============================================================================

def run_esi_soc(
    provider: BaseProvider,
    question: dict,
    item_id: str,
    probe_type: str,
    n_variants: int = 10,
    soc_p: float = 0.3,
    soc_min_len: int = 4,
    seed: int = 42,
    anti_uniform: bool = True,
) -> ESIResult:
    """Run ESI with SOC intervention."""
    probe_fn = PROBE_FUNCTIONS[probe_type]
    full_prompt, question_text = format_question_prompt(question)

    # Baseline
    action_orig, report_orig = probe_fn(provider, full_prompt, anti_uniform=anti_uniform)

    # Generate SOC variants
    variant_prompts = intervene_question_only(
        full_prompt=full_prompt,
        question_text=question_text,
        n_variants=n_variants,
        p=soc_p,
        min_len=soc_min_len,
        base_seed=seed,
    )

    # Run on variants
    action_intervened, report_intervened = [], []
    parse_errors = 0

    for variant_prompt in variant_prompts:
        try:
            action_v, report_v = probe_fn(provider, variant_prompt, anti_uniform=anti_uniform)
            action_intervened.append(action_v.probs)
            report_intervened.append(report_v.probs)
            if not report_v.parse_ok:
                parse_errors += 1
        except Exception as e:
            action_intervened.append({k: 0.25 for k in OPTIONS})
            report_intervened.append({k: 0.25 for k in OPTIONS})
            parse_errors += 1

    # Compute metrics
    esi_action = sum(hellinger_distance(action_orig.probs, a) for a in action_intervened) / len(action_intervened)
    esi_report = sum(hellinger_distance(report_orig.probs, r) for r in report_intervened) / len(report_intervened)
    js_orig = js_divergence(action_orig.probs, report_orig.probs)
    js_interv = [js_divergence(a, r) for a, r in zip(action_intervened, report_intervened)]
    delta_js = sum(js_interv) / len(js_interv) - js_orig

    return ESIResult(
        item_id=item_id, probe_type=probe_type, intervention_type="soc",
        provider=provider.provider_name, model=provider.model_name,
        question_text=question_text, category=question.get("category", "unknown"),
        action_original=action_orig.probs, report_original=report_orig.probs,
        js_original=js_orig, action_entropy_original=entropy(action_orig.probs),
        report_entropy_original=entropy(report_orig.probs),
        report_is_uniform_original=is_uniform(report_orig.probs),
        action_intervened=action_intervened, report_intervened=report_intervened,
        js_intervened=js_interv, report_is_uniform_intervened=[is_uniform(r) for r in report_intervened],
        esi_action=esi_action, esi_report=esi_report, delta_js=delta_js,
        n_variants=n_variants, seed=seed, parse_errors=parse_errors,
    )


def run_esi_order(
    provider: BaseProvider,
    question: dict,
    item_id: str,
    probe_type: str,
    n_variants: int = 10,
    seed: int = 42,
    anti_uniform: bool = True,
) -> ESIResult:
    """Run ESI with option order intervention."""
    probe_fn = PROBE_FUNCTIONS[probe_type]
    full_prompt, question_text = format_question_prompt(question)

    # Baseline
    action_orig, report_orig = probe_fn(provider, full_prompt, anti_uniform=anti_uniform)

    # Generate order variants
    variants = shuffle_options_batch(question, n_variants=n_variants, base_seed=seed)

    # Run on variants
    action_intervened, report_intervened = [], []
    parse_errors = 0

    for variant_q, mapping in variants:
        try:
            variant_prompt, _ = format_question_prompt(variant_q)
            action_v, report_v = probe_fn(provider, variant_prompt, anti_uniform=anti_uniform)
            # Remap back to original labels for comparison
            action_remapped = remap_probs(action_v.probs, mapping)
            report_remapped = remap_probs(report_v.probs, mapping)
            action_intervened.append(action_remapped)
            report_intervened.append(report_remapped)
            if not report_v.parse_ok:
                parse_errors += 1
        except Exception as e:
            action_intervened.append({k: 0.25 for k in OPTIONS})
            report_intervened.append({k: 0.25 for k in OPTIONS})
            parse_errors += 1

    # Compute metrics
    esi_action = sum(hellinger_distance(action_orig.probs, a) for a in action_intervened) / len(action_intervened)
    esi_report = sum(hellinger_distance(report_orig.probs, r) for r in report_intervened) / len(report_intervened)
    js_orig = js_divergence(action_orig.probs, report_orig.probs)
    js_interv = [js_divergence(a, r) for a, r in zip(action_intervened, report_intervened)]
    delta_js = sum(js_interv) / len(js_interv) - js_orig

    return ESIResult(
        item_id=item_id, probe_type=probe_type, intervention_type="order",
        provider=provider.provider_name, model=provider.model_name,
        question_text=question_text, category=question.get("category", "unknown"),
        action_original=action_orig.probs, report_original=report_orig.probs,
        js_original=js_orig, action_entropy_original=entropy(action_orig.probs),
        report_entropy_original=entropy(report_orig.probs),
        report_is_uniform_original=is_uniform(report_orig.probs),
        action_intervened=action_intervened, report_intervened=report_intervened,
        js_intervened=js_interv, report_is_uniform_intervened=[is_uniform(r) for r in report_intervened],
        esi_action=esi_action, esi_report=esi_report, delta_js=delta_js,
        n_variants=n_variants, seed=seed, parse_errors=parse_errors,
    )


def compute_summary(results: List[ESIResult], probe_type: str, intervention_type: str, provider: str) -> ESISummary:
    """Compute summary statistics."""
    n = len(results)
    if n == 0:
        return None

    def mean(vals): return sum(vals) / len(vals)
    def std(vals):
        m = mean(vals)
        return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))

    esi_actions = [r.esi_action for r in results]
    esi_reports = [r.esi_report for r in results]
    delta_js_vals = [r.delta_js for r in results]

    uniform_orig = sum(1 for r in results if r.report_is_uniform_original) / n
    uniform_interv_counts = []
    for r in results:
        uniform_interv_counts.extend(r.report_is_uniform_intervened)
    uniform_interv = sum(uniform_interv_counts) / len(uniform_interv_counts) if uniform_interv_counts else 0

    return ESISummary(
        probe_type=probe_type, intervention_type=intervention_type, provider=provider, n_items=n,
        mean_esi_action=mean(esi_actions), mean_esi_report=mean(esi_reports), mean_delta_js=mean(delta_js_vals),
        std_esi_action=std(esi_actions), std_esi_report=std(esi_reports), std_delta_js=std(delta_js_vals),
        uniform_rate_original=uniform_orig, uniform_rate_intervened=uniform_interv,
        uniform_rate_shift=uniform_interv - uniform_orig,
    )


# =============================================================================
# Main
# =============================================================================

def run_full_experiment(
    providers: List[str],
    probe_types: List[str],
    intervention_types: List[str],
    questions_file: str = None,
    n_variants: int = 10,
    max_items: int = None,
    seed: int = 42,
    anti_uniform: bool = True,
    output_file: str = None,
):
    """Run full ESI experiment across providers, probes, and interventions."""
    if questions_file is None:
        questions_file = Path(__file__).parent.parent / "data" / "mc_questions.json"

    print("=" * 70)
    print("ESI FULL EXPERIMENT")
    print("=" * 70)
    print(f"Providers: {providers}")
    print(f"Probes: {probe_types}")
    print(f"Interventions: {intervention_types}")
    print(f"Variants per item: {n_variants}")
    print("=" * 70)

    # Load questions
    with open(questions_file) as f:
        data = json.load(f)
    questions = data["questions"]
    if max_items:
        questions = questions[:max_items]
    print(f"Loaded {len(questions)} questions")

    all_results = []
    summaries = []

    for provider_name in providers:
        print(f"\n{'='*70}")
        print(f"PROVIDER: {provider_name.upper()}")
        print("=" * 70)

        try:
            provider = get_provider(provider_name)
            print(f"Initialized: {provider.model_name} (logprobs: {provider.supports_logprobs})")
        except Exception as e:
            print(f"Failed to initialize {provider_name}: {e}")
            continue

        for intervention_type in intervention_types:
            for probe_type in probe_types:
                print(f"\n--- {provider_name} / Probe {probe_type} / {intervention_type.upper()} ---")

                probe_results = []
                for i, q in enumerate(questions):
                    item_id = f"{provider_name}_{probe_type}_{intervention_type}_{i}"
                    print(f"  [{i+1}/{len(questions)}] {q['question'][:40]}...", end=" ", flush=True)

                    try:
                        if intervention_type == "soc":
                            result = run_esi_soc(provider, q, item_id, probe_type, n_variants, seed=seed+i, anti_uniform=anti_uniform)
                        else:
                            result = run_esi_order(provider, q, item_id, probe_type, n_variants, seed=seed+i, anti_uniform=anti_uniform)

                        probe_results.append(result)
                        all_results.append(result)
                        print(f"action={result.esi_action:.3f} report={result.esi_report:.3f} ΔJS={result.delta_js:+.3f}")
                    except Exception as e:
                        print(f"ERROR: {e}")

                # Summary for this combination
                if probe_results:
                    summary = compute_summary(probe_results, probe_type, intervention_type, provider_name)
                    summaries.append(summary)
                    print(f"  Summary: ESI_action={summary.mean_esi_action:.3f}±{summary.std_esi_action:.3f}, "
                          f"ESI_report={summary.mean_esi_report:.3f}±{summary.std_esi_report:.3f}, "
                          f"ΔJS={summary.mean_delta_js:+.3f}")

    # Final summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Provider':<10} {'Probe':<6} {'Interv':<6} {'ESI_act':>8} {'ESI_rep':>8} {'ΔJS':>8} {'Uniform':>10}")
    print("-" * 70)
    for s in summaries:
        uniform_str = f"{s.uniform_rate_original:.0%}→{s.uniform_rate_intervened:.0%}"
        print(f"{s.provider:<10} {s.probe_type:<6} {s.intervention_type:<6} "
              f"{s.mean_esi_action:>8.3f} {s.mean_esi_report:>8.3f} {s.mean_delta_js:>+8.3f} {uniform_str:>10}")

    # Save results
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for r in all_results:
                f.write(json.dumps(r.to_dict()) + "\n")
        print(f"\nPer-item results: {output_path}")

        summary_path = output_path.with_suffix(".summary.csv")
        with open(summary_path, "w") as f:
            f.write("provider,probe,intervention,n,esi_action,std_action,esi_report,std_report,delta_js,std_djs,uniform_orig,uniform_interv\n")
            for s in summaries:
                f.write(f"{s.provider},{s.probe_type},{s.intervention_type},{s.n_items},"
                        f"{s.mean_esi_action:.4f},{s.std_esi_action:.4f},"
                        f"{s.mean_esi_report:.4f},{s.std_esi_report:.4f},"
                        f"{s.mean_delta_js:.4f},{s.std_delta_js:.4f},"
                        f"{s.uniform_rate_original:.4f},{s.uniform_rate_intervened:.4f}\n")
        print(f"Summary CSV: {summary_path}")

    return all_results, summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESI Runner")
    parser.add_argument("provider", nargs="?", help="Single provider (or use --providers)")
    parser.add_argument("--providers", nargs="+", help="Multiple providers")
    parser.add_argument("--probe", "-p", action="append", dest="probes", help="Probe type(s)")
    parser.add_argument("--intervention", "-i", action="append", dest="interventions",
                        choices=["soc", "order"], help="Intervention type(s)")
    parser.add_argument("--n-variants", "-L", type=int, default=10, help="Variants per item")
    parser.add_argument("--max-items", "-n", type=int, help="Max items")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--questions", help="Questions JSON file")
    parser.add_argument("--no-anti-uniform", action="store_true", help="Disable anti-uniform")
    parser.add_argument("-o", "--output", help="Output file (JSONL)")
    parser.add_argument("--full", action="store_true", help="Run all probes and interventions")

    args = parser.parse_args()

    # Determine providers
    if args.providers:
        providers = args.providers
    elif args.provider:
        providers = [args.provider]
    else:
        providers = ["openai"]

    # Determine probes and interventions
    if args.full:
        probes = ["A", "B", "C", "D"]
        interventions = ["soc", "order"]
    else:
        probes = args.probes or ["A"]
        interventions = args.interventions or ["soc"]

    run_full_experiment(
        providers=providers,
        probe_types=probes,
        intervention_types=interventions,
        questions_file=args.questions,
        n_variants=args.n_variants,
        max_items=args.max_items,
        seed=args.seed,
        anti_uniform=not args.no_anti_uniform,
        output_file=args.output,
    )
