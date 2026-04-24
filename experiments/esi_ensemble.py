#!/usr/bin/env python3
"""
ESI Ensemble Runner - Order-ensemble as first-class measurement protocol.

Features:
1. Always computes both single-shot and order-ensemble action beliefs
2. Content-identity check: verifies argmax preservation across permutations
3. Bootstrap analysis: JS vs K (number of orderings)

Usage:
    python experiments/esi_ensemble.py openai --max-items 25 -o results/esi/ensemble_run.jsonl
    python experiments/esi_ensemble.py --providers openai gemini --bootstrap -o results/esi/bootstrap.jsonl
"""

import argparse
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from interventions.option_order import shuffle_options_batch, remap_probs
from experiments.providers import get_provider, BaseProvider

OPTIONS = ["A", "B", "C", "D"]


# =============================================================================
# Metrics
# =============================================================================

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


def argmax(probs: Dict[str, float]) -> str:
    return max(probs, key=probs.get)


def average_distributions(dists: List[Dict[str, float]]) -> Dict[str, float]:
    if not dists:
        return {k: 0.25 for k in OPTIONS}
    avg = {k: 0.0 for k in OPTIONS}
    for d in dists:
        for k in OPTIONS:
            avg[k] += d.get(k, 0)
    return {k: v / len(dists) for k, v in avg.items()}


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class EnsembleResult:
    """Results for a single item with ensemble analysis."""
    item_id: str
    provider: str
    model: str
    question_text: str
    category: str

    # Single-shot results
    action_single: Dict[str, float]
    report_single: Dict[str, float]
    js_single: float
    argmax_single: str

    # Order-ensemble results
    action_ensemble: Dict[str, float]
    js_ensemble: float
    argmax_ensemble: str
    improvement_pct: float  # (js_single - js_ensemble) / js_single * 100

    # Content-identity analysis
    n_orderings: int
    argmax_flip_count: int  # How many orderings had different argmax
    argmax_flip_rate: float
    ensemble_restores_argmax: bool  # True if ensemble argmax == single argmax

    # Per-ordering details (for bootstrap)
    per_ordering_action: List[Dict[str, float]]
    per_ordering_argmax: List[str]

    # Bootstrap results (optional)
    bootstrap_js_by_k: Dict[int, float] = None  # K -> mean JS at that K

    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# Probe function (simplified - just Act→Report)
# =============================================================================

def format_question_prompt(question: dict) -> str:
    lines = [
        "You are answering a multiple choice question.",
        "",
        "Question:",
        question["question"],
        "",
        "Options:",
    ]
    for opt in OPTIONS:
        lines.append(f"{opt}) {question['options'][opt]}")
    return "\n".join(lines)


def run_probe(provider: BaseProvider, prompt: str, anti_uniform: bool = True):
    """Run Act→Report probe, return (action_probs, report_probs)."""
    action_prompt = prompt + "\n\nAnswer with A, B, C, or D:"
    action_result = provider.get_action_probs(action_prompt)
    report_result = provider.get_report(prompt, anti_uniform=anti_uniform)
    return action_result.probs, report_result.probs


# =============================================================================
# Ensemble runner
# =============================================================================

def run_ensemble_item(
    provider: BaseProvider,
    question: dict,
    item_id: str,
    n_orderings: int = 10,
    seed: int = 42,
    anti_uniform: bool = True,
    do_bootstrap: bool = False,
) -> EnsembleResult:
    """Run order-ensemble analysis on a single item."""

    # Single-shot (canonical ordering)
    prompt_single = format_question_prompt(question)
    action_single, report_single = run_probe(provider, prompt_single, anti_uniform)
    argmax_single = argmax(action_single)
    js_single = js_divergence(action_single, report_single)

    # Generate orderings
    variants = shuffle_options_batch(question, n_variants=n_orderings, base_seed=seed)

    # Run on each ordering
    per_ordering_action = []
    per_ordering_argmax = []
    argmax_flips = 0

    for variant_q, mapping in variants:
        variant_prompt = format_question_prompt(variant_q)
        try:
            action_v, _ = run_probe(provider, variant_prompt, anti_uniform)
            # Unpermute back to canonical labels
            action_remapped = remap_probs(action_v, mapping)
            per_ordering_action.append(action_remapped)

            am = argmax(action_remapped)
            per_ordering_argmax.append(am)
            if am != argmax_single:
                argmax_flips += 1
        except Exception as e:
            # Fallback to uniform
            per_ordering_action.append({k: 0.25 for k in OPTIONS})
            per_ordering_argmax.append("?")

    # Compute ensemble
    action_ensemble = average_distributions(per_ordering_action)
    argmax_ensemble = argmax(action_ensemble)
    js_ensemble = js_divergence(action_ensemble, report_single)

    improvement_pct = (js_single - js_ensemble) / js_single * 100 if js_single > 0 else 0

    # Bootstrap analysis
    bootstrap_js_by_k = None
    if do_bootstrap and len(per_ordering_action) >= 4:
        bootstrap_js_by_k = {}
        for k in [2, 3, 4, 5, min(8, len(per_ordering_action)), len(per_ordering_action)]:
            if k > len(per_ordering_action):
                continue
            # Sample 20 bootstrap replicates
            js_samples = []
            for _ in range(20):
                subset = random.sample(per_ordering_action, k)
                ensemble_k = average_distributions(subset)
                js_k = js_divergence(ensemble_k, report_single)
                js_samples.append(js_k)
            bootstrap_js_by_k[k] = sum(js_samples) / len(js_samples)

    return EnsembleResult(
        item_id=item_id,
        provider=provider.provider_name,
        model=provider.model_name,
        question_text=question["question"],
        category=question.get("category", "unknown"),
        # Single
        action_single=action_single,
        report_single=report_single,
        js_single=js_single,
        argmax_single=argmax_single,
        # Ensemble
        action_ensemble=action_ensemble,
        js_ensemble=js_ensemble,
        argmax_ensemble=argmax_ensemble,
        improvement_pct=improvement_pct,
        # Identity
        n_orderings=n_orderings,
        argmax_flip_count=argmax_flips,
        argmax_flip_rate=argmax_flips / n_orderings if n_orderings > 0 else 0,
        ensemble_restores_argmax=(argmax_ensemble == argmax_single),
        # Details
        per_ordering_action=per_ordering_action,
        per_ordering_argmax=per_ordering_argmax,
        bootstrap_js_by_k=bootstrap_js_by_k,
    )


def run_ensemble_experiment(
    providers: List[str],
    questions_file: str = None,
    n_orderings: int = 10,
    max_items: int = None,
    seed: int = 42,
    anti_uniform: bool = True,
    do_bootstrap: bool = False,
    output_file: str = None,
):
    """Run ensemble experiment."""
    if questions_file is None:
        questions_file = Path(__file__).parent.parent / "data" / "mc_questions.json"

    print("=" * 70)
    print("ESI ENSEMBLE EXPERIMENT")
    print("=" * 70)
    print(f"Providers: {providers}")
    print(f"Orderings per item: {n_orderings}")
    print(f"Bootstrap: {do_bootstrap}")
    print("=" * 70)

    with open(questions_file) as f:
        data = json.load(f)
    questions = data["questions"]
    if max_items:
        questions = questions[:max_items]
    print(f"Loaded {len(questions)} questions")

    all_results = []
    provider_models = {}

    for provider_name in providers:
        print(f"\n{'='*70}")
        print(f"PROVIDER: {provider_name.upper()}")
        print("=" * 70)

        try:
            provider = get_provider(provider_name)
            print(f"Initialized: {provider.model_name}")
            provider_models[provider_name] = provider.model_name
        except Exception as e:
            print(f"Failed: {e}")
            continue

        for i, q in enumerate(questions):
            item_id = f"{provider_name}_{i}"
            print(f"[{i+1}/{len(questions)}] {q['question'][:40]}...", end=" ", flush=True)

            try:
                result = run_ensemble_item(
                    provider, q, item_id,
                    n_orderings=n_orderings,
                    seed=seed + i,
                    anti_uniform=anti_uniform,
                    do_bootstrap=do_bootstrap,
                )
                all_results.append(result)
                print(f"JS: {result.js_single:.3f}→{result.js_ensemble:.3f} "
                      f"({result.improvement_pct:+.0f}%) "
                      f"flips:{result.argmax_flip_count}/{n_orderings}")
            except Exception as e:
                print(f"ERROR: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for provider_name in providers:
        results = [r for r in all_results if r.provider == provider_name]
        if not results:
            continue

        print(f"\n--- {provider_name.upper()} ---")

        # JS improvement
        mean_js_single = sum(r.js_single for r in results) / len(results)
        mean_js_ensemble = sum(r.js_ensemble for r in results) / len(results)
        mean_improvement = sum(r.improvement_pct for r in results) / len(results)

        print(f"JS(single):   {mean_js_single:.4f}")
        print(f"JS(ensemble): {mean_js_ensemble:.4f}")
        print(f"Improvement:  {mean_improvement:+.1f}%")

        # Argmax analysis
        total_flips = sum(r.argmax_flip_count for r in results)
        total_orderings = sum(r.n_orderings for r in results)
        flip_rate = total_flips / total_orderings if total_orderings > 0 else 0
        restore_rate = sum(1 for r in results if r.ensemble_restores_argmax) / len(results)

        print(f"Argmax flip rate:     {flip_rate:.1%}")
        print(f"Ensemble restores:    {restore_rate:.1%}")

        # By category
        for cat in ["easy", "ambiguous"]:
            cat_results = [r for r in results if r.category == cat]
            if cat_results:
                mean_imp = sum(r.improvement_pct for r in cat_results) / len(cat_results)
                mean_flip = sum(r.argmax_flip_rate for r in cat_results) / len(cat_results)
                print(f"  {cat}: improvement={mean_imp:+.1f}%, flip_rate={mean_flip:.1%}")

    # Bootstrap results
    if do_bootstrap:
        print("\n" + "=" * 70)
        print("BOOTSTRAP: JS vs K (number of orderings)")
        print("=" * 70)

        # Aggregate by K
        k_values = set()
        for r in all_results:
            if r.bootstrap_js_by_k:
                k_values.update(r.bootstrap_js_by_k.keys())

        if k_values:
            print(f"{'K':>4} {'Mean JS':>10} {'Δ from K=1':>12}")
            print("-" * 30)

            # K=1 baseline (single-shot)
            mean_js_1 = sum(r.js_single for r in all_results) / len(all_results)
            print(f"{'1':>4} {mean_js_1:>10.4f} {'—':>12}")

            for k in sorted(k_values):
                js_at_k = [r.bootstrap_js_by_k[k] for r in all_results
                          if r.bootstrap_js_by_k and k in r.bootstrap_js_by_k]
                if js_at_k:
                    mean_js_k = sum(js_at_k) / len(js_at_k)
                    delta = mean_js_k - mean_js_1
                    print(f"{k:>4} {mean_js_k:>10.4f} {delta:>+12.4f}")

    # Save results
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for r in all_results:
                f.write(json.dumps(r.to_dict()) + "\n")
        print(f"\nResults saved to: {output_path}")

        metadata_path = output_path.with_suffix(".metadata.json")
        metadata = {
            "generated_at": datetime.now().isoformat(),
            "script": str(Path(__file__).resolve()),
            "questions_file": str(questions_file),
            "providers": providers,
            "provider_models": provider_models,
            "n_orderings": n_orderings,
            "max_items": max_items,
            "seed": seed,
            "anti_uniform": anti_uniform,
            "bootstrap": do_bootstrap,
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata JSON: {metadata_path}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESI Ensemble Runner")
    parser.add_argument("provider", nargs="?", help="Single provider")
    parser.add_argument("--providers", nargs="+", help="Multiple providers")
    parser.add_argument("--n-orderings", "-K", type=int, default=10, help="Number of orderings")
    parser.add_argument("--max-items", "-n", type=int, help="Max items")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--bootstrap", action="store_true", help="Run bootstrap analysis")
    parser.add_argument("--no-anti-uniform", action="store_true")
    parser.add_argument("-o", "--output", help="Output file (JSONL)")
    parser.add_argument("--questions", help="Questions file")

    args = parser.parse_args()

    if args.providers:
        providers = args.providers
    elif args.provider:
        providers = [args.provider]
    else:
        providers = ["openai"]

    run_ensemble_experiment(
        providers=providers,
        questions_file=args.questions,
        n_orderings=args.n_orderings,
        max_items=args.max_items,
        seed=args.seed,
        anti_uniform=not args.no_anti_uniform,
        do_bootstrap=args.bootstrap,
        output_file=args.output,
    )
