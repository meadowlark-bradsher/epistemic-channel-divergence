#!/usr/bin/env python3
"""
ESI Analysis - Post-hoc analysis of ESI experiment results.

Analyses:
1. Order-averaged (ensemble) action beliefs - does debiasing reduce divergence?
2. Stratification by question category (easy vs ambiguous)
3. Gemini uniform attractor persistence analysis

Usage:
    python experiments/esi_analysis.py results/esi/full_experiment.jsonl
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

OPTIONS = ["A", "B", "C", "D"]


def hellinger_distance(p: Dict[str, float], q: Dict[str, float]) -> float:
    sum_sq = sum((math.sqrt(p.get(k, 0)) - math.sqrt(q.get(k, 0))) ** 2 for k in OPTIONS)
    return math.sqrt(sum_sq) / math.sqrt(2)


def js_divergence(p: Dict[str, float], q: Dict[str, float], eps: float = 1e-10) -> float:
    m = {k: (p.get(k, 0) + q.get(k, 0)) / 2 for k in OPTIONS}
    def kl(a, b):
        return sum(a.get(k, 0) * math.log2((a.get(k, 0) + eps) / (b.get(k, 0) + eps))
                   for k in OPTIONS if a.get(k, 0) > 0)
    return (kl(p, m) + kl(q, m)) / 2


def entropy(probs: Dict[str, float]) -> float:
    return -sum(p * math.log2(p) for p in probs.values() if p > 0)


def is_uniform(probs: Dict[str, float], tolerance: float = 0.02) -> bool:
    return all(abs(probs.get(k, 0) - 0.25) < tolerance for k in OPTIONS)


def average_distributions(dists: List[Dict[str, float]]) -> Dict[str, float]:
    """Compute element-wise average of distributions."""
    if not dists:
        return {k: 0.25 for k in OPTIONS}
    avg = {k: 0.0 for k in OPTIONS}
    for d in dists:
        for k in OPTIONS:
            avg[k] += d.get(k, 0)
    n = len(dists)
    return {k: v / n for k, v in avg.items()}


def load_results(filepath: str) -> List[dict]:
    """Load JSONL results file."""
    results = []
    with open(filepath) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def analyze_ensemble_debiasing(results: List[dict]):
    """
    Analysis A: Order-averaged action beliefs.

    For ORDER intervention results, compute:
    - p̄_action = mean of unpermuted action distributions
    - Compare JS(p̄_action || p_report) vs JS(p_action_original || p_report)
    """
    print("\n" + "=" * 70)
    print("ANALYSIS A: Order-Averaged (Ensemble) Action Beliefs")
    print("=" * 70)
    print("Does averaging over option permutations reduce action-report divergence?")
    print()

    order_results = [r for r in results if r["intervention_type"] == "order"]

    if not order_results:
        print("No ORDER intervention results found.")
        return

    # Group by provider and probe
    grouped = defaultdict(list)
    for r in order_results:
        key = (r["provider"], r["probe_type"])
        grouped[key].append(r)

    print(f"{'Provider':<10} {'Probe':<6} {'JS(orig)':<10} {'JS(ensemble)':<12} {'Δ':>8} {'Improvement':>12}")
    print("-" * 70)

    improvements = []

    for (provider, probe), items in sorted(grouped.items()):
        js_orig_list = []
        js_ensemble_list = []

        for item in items:
            p_action_orig = item["action_original"]
            p_report_orig = item["report_original"]
            action_variants = item["action_intervened"]  # Already unpermuted

            # Original JS
            js_orig = js_divergence(p_action_orig, p_report_orig)

            # Ensemble: average the unpermuted action distributions
            p_action_ensemble = average_distributions(action_variants)

            # Ensemble JS
            js_ensemble = js_divergence(p_action_ensemble, p_report_orig)

            js_orig_list.append(js_orig)
            js_ensemble_list.append(js_ensemble)

        mean_js_orig = sum(js_orig_list) / len(js_orig_list)
        mean_js_ensemble = sum(js_ensemble_list) / len(js_ensemble_list)
        delta = mean_js_ensemble - mean_js_orig
        pct_improvement = -delta / mean_js_orig * 100 if mean_js_orig > 0 else 0

        improvements.append(pct_improvement)

        print(f"{provider:<10} {probe:<6} {mean_js_orig:<10.4f} {mean_js_ensemble:<12.4f} "
              f"{delta:>+8.4f} {pct_improvement:>+11.1f}%")

    print("-" * 70)
    print(f"{'Average improvement:':<40} {sum(improvements)/len(improvements):>+11.1f}%")
    print()
    print("Interpretation:")
    print("  Positive % = ensemble reduces divergence (debiasing helps)")
    print("  Negative % = ensemble increases divergence (debiasing hurts)")


def analyze_by_category(results: List[dict]):
    """
    Analysis C: Stratify results by question category (easy vs ambiguous).
    """
    print("\n" + "=" * 70)
    print("ANALYSIS C: Results Stratified by Question Category")
    print("=" * 70)

    # Group by provider, probe, intervention, category
    grouped = defaultdict(list)
    for r in results:
        key = (r["provider"], r["probe_type"], r["intervention_type"], r["category"])
        grouped[key].append(r)

    # Compute means
    summary = []
    for (provider, probe, interv, category), items in grouped.items():
        esi_action = sum(r["esi_action"] for r in items) / len(items)
        esi_report = sum(r["esi_report"] for r in items) / len(items)
        delta_js = sum(r["delta_js"] for r in items) / len(items)
        uniform_orig = sum(1 for r in items if r["report_is_uniform_original"]) / len(items)
        uniform_interv = sum(
            sum(r["report_is_uniform_intervened"]) / len(r["report_is_uniform_intervened"])
            for r in items
        ) / len(items)

        summary.append({
            "provider": provider,
            "probe": probe,
            "intervention": interv,
            "category": category,
            "n": len(items),
            "esi_action": esi_action,
            "esi_report": esi_report,
            "delta_js": delta_js,
            "uniform_orig": uniform_orig,
            "uniform_interv": uniform_interv,
        })

    # Print by category
    for category in ["easy", "ambiguous"]:
        print(f"\n--- {category.upper()} questions ---")
        print(f"{'Provider':<10} {'Probe':<6} {'Interv':<6} {'n':>4} {'ESI_act':>8} {'ESI_rep':>8} {'ΔJS':>8} {'Unif%':>8}")
        print("-" * 70)

        cat_rows = [s for s in summary if s["category"] == category]
        for s in sorted(cat_rows, key=lambda x: (x["provider"], x["intervention"], x["probe"])):
            print(f"{s['provider']:<10} {s['probe']:<6} {s['intervention']:<6} {s['n']:>4} "
                  f"{s['esi_action']:>8.3f} {s['esi_report']:>8.3f} {s['delta_js']:>+8.3f} "
                  f"{s['uniform_orig']*100:>7.0f}%")

    # Comparison summary
    print("\n--- Category Comparison (means across all conditions) ---")
    print(f"{'Category':<12} {'ESI_action':>12} {'ESI_report':>12} {'ΔJS':>12} {'Uniform%':>12}")
    print("-" * 60)

    for category in ["easy", "ambiguous"]:
        cat_rows = [s for s in summary if s["category"] == category]
        if cat_rows:
            mean_action = sum(s["esi_action"] for s in cat_rows) / len(cat_rows)
            mean_report = sum(s["esi_report"] for s in cat_rows) / len(cat_rows)
            mean_djs = sum(s["delta_js"] for s in cat_rows) / len(cat_rows)
            mean_uniform = sum(s["uniform_orig"] for s in cat_rows) / len(cat_rows)
            print(f"{category:<12} {mean_action:>12.3f} {mean_report:>12.3f} {mean_djs:>+12.3f} {mean_uniform*100:>11.1f}%")


def analyze_gemini_uniform(results: List[dict]):
    """
    Analysis D: Gemini uniform attractor persistence.

    Table: provider × probe × intervention showing uniform rates.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS D: Uniform Attractor Persistence")
    print("=" * 70)
    print("Uniform report rate (% of reports that are ~25/25/25/25)")
    print()

    # Group by provider, probe, intervention
    grouped = defaultdict(list)
    for r in results:
        key = (r["provider"], r["probe_type"], r["intervention_type"])
        grouped[key].append(r)

    # Compute uniform rates
    print(f"{'Provider':<10} {'Probe':<6} {'Interv':<8} {'Orig%':>8} {'Interv%':>8} {'Shift':>8}")
    print("-" * 55)

    for provider in ["openai", "gemini"]:
        for interv in ["soc", "order"]:
            for probe in ["A", "B", "C", "D"]:
                key = (provider, probe, interv)
                items = grouped.get(key, [])
                if not items:
                    continue

                # Original uniform rate
                uniform_orig = sum(1 for r in items if r["report_is_uniform_original"]) / len(items)

                # Intervened uniform rate (average across variants)
                uniform_interv_rates = []
                for r in items:
                    if r["report_is_uniform_intervened"]:
                        rate = sum(r["report_is_uniform_intervened"]) / len(r["report_is_uniform_intervened"])
                        uniform_interv_rates.append(rate)
                uniform_interv = sum(uniform_interv_rates) / len(uniform_interv_rates) if uniform_interv_rates else 0

                shift = uniform_interv - uniform_orig

                print(f"{provider:<10} {probe:<6} {interv:<8} {uniform_orig*100:>7.1f}% {uniform_interv*100:>7.1f}% {shift*100:>+7.1f}%")

        print()

    # Summary by provider
    print("--- Summary by Provider ---")
    for provider in ["openai", "gemini"]:
        provider_items = [r for r in results if r["provider"] == provider]
        if not provider_items:
            continue

        uniform_orig = sum(1 for r in provider_items if r["report_is_uniform_original"]) / len(provider_items)

        uniform_interv_rates = []
        for r in provider_items:
            if r["report_is_uniform_intervened"]:
                rate = sum(r["report_is_uniform_intervened"]) / len(r["report_is_uniform_intervened"])
                uniform_interv_rates.append(rate)
        uniform_interv = sum(uniform_interv_rates) / len(uniform_interv_rates) if uniform_interv_rates else 0

        print(f"{provider}: Original {uniform_orig*100:.1f}%, Under intervention {uniform_interv*100:.1f}%")


def analyze_action_entropy_correlation(results: List[dict]):
    """
    Bonus: Check if high-entropy (uncertain) actions are more stable.
    """
    print("\n" + "=" * 70)
    print("BONUS: Action Entropy vs ESI Sensitivity")
    print("=" * 70)

    # Bin by action entropy
    low_entropy = []  # < 1 bit
    mid_entropy = []  # 1-1.5 bits
    high_entropy = [] # > 1.5 bits

    for r in results:
        h = r["action_entropy_original"]
        esi = r["esi_action"]
        if h < 1.0:
            low_entropy.append(esi)
        elif h < 1.5:
            mid_entropy.append(esi)
        else:
            high_entropy.append(esi)

    def safe_mean(lst):
        return sum(lst) / len(lst) if lst else 0

    print(f"{'Entropy bin':<20} {'N':>6} {'Mean ESI_action':>15}")
    print("-" * 45)
    print(f"{'Low (<1 bit)':<20} {len(low_entropy):>6} {safe_mean(low_entropy):>15.3f}")
    print(f"{'Mid (1-1.5 bits)':<20} {len(mid_entropy):>6} {safe_mean(mid_entropy):>15.3f}")
    print(f"{'High (>1.5 bits)':<20} {len(high_entropy):>6} {safe_mean(high_entropy):>15.3f}")
    print()
    print("Interpretation: Lower entropy (more confident) actions may be more/less stable")


def main(filepath: str):
    print("=" * 70)
    print("ESI EXPERIMENT ANALYSIS")
    print("=" * 70)
    print(f"Input: {filepath}")

    results = load_results(filepath)
    print(f"Loaded {len(results)} results")

    # Run analyses
    analyze_ensemble_debiasing(results)
    analyze_by_category(results)
    analyze_gemini_uniform(results)
    analyze_action_entropy_correlation(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESI Analysis")
    parser.add_argument("results_file", help="Path to results JSONL file")
    args = parser.parse_args()
    main(args.results_file)
