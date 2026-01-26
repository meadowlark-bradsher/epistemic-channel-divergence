"""
Option Order Intervention

Semantic-preserving intervention that shuffles the order of multiple choice options.
The correct answer remains the same, but its letter label changes.

This tests whether the model's beliefs are about the *content* of options
or anchored to *position* (A, B, C, D labels).
"""

import random
from typing import Dict, List, Tuple


def shuffle_options(
    question: dict,
    seed: int = None,
) -> Tuple[dict, Dict[str, str]]:
    """Shuffle the order of options in a question.

    Args:
        question: Question dict with 'question', 'options' keys
        seed: Random seed for reproducibility

    Returns:
        (new_question, mapping) where mapping is {new_label: old_label}
        e.g., {'A': 'C', 'B': 'A', 'C': 'D', 'D': 'B'} means
        new option A contains what was originally option C
    """
    if seed is not None:
        random.seed(seed)

    labels = ["A", "B", "C", "D"]
    original_options = question["options"]

    # Create shuffled order
    shuffled_labels = labels.copy()
    random.shuffle(shuffled_labels)

    # Build new options and mapping
    new_options = {}
    mapping = {}  # new_label -> old_label

    for new_label, old_label in zip(labels, shuffled_labels):
        new_options[new_label] = original_options[old_label]
        mapping[new_label] = old_label

    new_question = {
        **question,
        "options": new_options,
    }

    return new_question, mapping


def shuffle_options_batch(
    question: dict,
    n_variants: int = 10,
    base_seed: int = None,
) -> List[Tuple[dict, Dict[str, str]]]:
    """Generate multiple shuffled variants of a question.

    Args:
        question: Original question dict
        n_variants: Number of variants to generate
        base_seed: Base seed (each variant uses base_seed + i)

    Returns:
        List of (shuffled_question, mapping) tuples
    """
    variants = []
    for i in range(n_variants):
        seed = (base_seed + i) if base_seed is not None else None
        variant, mapping = shuffle_options(question, seed=seed)
        variants.append((variant, mapping))
    return variants


def remap_probs(
    probs: Dict[str, float],
    mapping: Dict[str, str],
) -> Dict[str, float]:
    """Remap probabilities from shuffled labels back to original labels.

    Args:
        probs: Probabilities with shuffled labels {'A': 0.5, 'B': 0.3, ...}
        mapping: {new_label: old_label} from shuffle_options

    Returns:
        Probabilities with original labels
    """
    # mapping is new_label -> old_label
    # We want to convert probs[new_label] to probs[old_label]
    remapped = {}
    for new_label, old_label in mapping.items():
        remapped[old_label] = probs.get(new_label, 0.25)
    return remapped


def inverse_mapping(mapping: Dict[str, str]) -> Dict[str, str]:
    """Get inverse mapping (old_label -> new_label).

    Args:
        mapping: {new_label: old_label}

    Returns:
        {old_label: new_label}
    """
    return {v: k for k, v in mapping.items()}


# Quick test
if __name__ == "__main__":
    test_question = {
        "question": "What is the capital of France?",
        "options": {
            "A": "London",
            "B": "Paris",
            "C": "Berlin",
            "D": "Madrid"
        },
        "category": "easy"
    }

    print("Original question:")
    print(f"  {test_question['question']}")
    for opt, text in test_question["options"].items():
        print(f"  {opt}) {text}")

    print("\nShuffled variants:")
    variants = shuffle_options_batch(test_question, n_variants=3, base_seed=42)

    for i, (variant, mapping) in enumerate(variants):
        print(f"\n  Variant {i+1} (mapping: {mapping}):")
        for opt, text in variant["options"].items():
            print(f"    {opt}) {text}")

        # Test remapping
        fake_probs = {"A": 0.1, "B": 0.6, "C": 0.2, "D": 0.1}
        remapped = remap_probs(fake_probs, mapping)
        print(f"    Original probs: {fake_probs}")
        print(f"    Remapped probs: {remapped}")
