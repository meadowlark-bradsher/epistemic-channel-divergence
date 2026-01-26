"""
SOC (Skip-One-Char) Intervention

Semantic-preserving intervention that randomly removes characters from words.
Based on ESI paper methodology.

Key properties:
- Only touches words with length >= min_len
- Only deletes from tail portion (after first min_len-1 chars) to preserve word roots
- Leaves punctuation and formatting intact
- Designed to preserve semantic meaning while creating measurable distribution shifts
"""

import random
import re
from typing import List, Tuple


def soc_intervene(
    text: str,
    p: float = 0.3,
    min_len: int = 4,
    tail_only: bool = True,
    seed: int = None,
) -> str:
    """Apply Skip-One-Char intervention to text.

    Args:
        text: Input text to intervene on
        p: Probability/fraction of eligible words to modify (0.0-1.0)
        min_len: Minimum word length to consider for intervention
        tail_only: If True, only delete chars from position [min_len-1, end)
        seed: Random seed for reproducibility

    Returns:
        Intervened text with some characters removed
    """
    if seed is not None:
        random.seed(seed)

    # Split into tokens preserving whitespace and punctuation
    # Pattern: word characters vs everything else
    tokens = re.findall(r'\w+|\W+', text)

    # Find indices of words eligible for intervention
    eligible_indices = []
    for i, token in enumerate(tokens):
        if token.isalnum() and len(token) >= min_len:
            eligible_indices.append(i)

    if not eligible_indices:
        return text

    # Select which words to modify
    n_to_modify = max(1, int(len(eligible_indices) * p))
    indices_to_modify = random.sample(eligible_indices, min(n_to_modify, len(eligible_indices)))

    # Apply intervention
    result_tokens = tokens.copy()
    for idx in indices_to_modify:
        word = result_tokens[idx]
        result_tokens[idx] = _skip_one_char(word, min_len, tail_only)

    return ''.join(result_tokens)


def _skip_one_char(word: str, min_len: int, tail_only: bool) -> str:
    """Remove one character from a word.

    Args:
        word: The word to modify
        min_len: Minimum length threshold
        tail_only: If True, only remove from tail portion

    Returns:
        Word with one character removed
    """
    if len(word) < min_len:
        return word

    if tail_only:
        # Only delete from positions [min_len-1, end)
        start_pos = min_len - 1
        if start_pos >= len(word):
            return word
        delete_pos = random.randint(start_pos, len(word) - 1)
    else:
        delete_pos = random.randint(0, len(word) - 1)

    return word[:delete_pos] + word[delete_pos + 1:]


def soc_intervene_batch(
    text: str,
    n_variants: int = 10,
    p: float = 0.3,
    min_len: int = 4,
    tail_only: bool = True,
    base_seed: int = None,
) -> List[str]:
    """Generate multiple SOC variants of a text.

    Args:
        text: Input text
        n_variants: Number of variants to generate
        p: Fraction of eligible words to modify
        min_len: Minimum word length for intervention
        tail_only: Only delete from tail portion of words
        base_seed: Base seed (each variant uses base_seed + i)

    Returns:
        List of n_variants intervened texts
    """
    variants = []
    for i in range(n_variants):
        seed = (base_seed + i) if base_seed is not None else None
        variant = soc_intervene(text, p=p, min_len=min_len, tail_only=tail_only, seed=seed)
        variants.append(variant)
    return variants


def intervene_question_only(
    full_prompt: str,
    question_text: str,
    n_variants: int = 10,
    p: float = 0.3,
    min_len: int = 4,
    base_seed: int = None,
) -> List[str]:
    """Apply SOC intervention only to the question portion of a probe prompt.

    This preserves formatting instructions, system prompts, and answer options
    while only intervening on the actual question content.

    Args:
        full_prompt: Complete probe prompt
        question_text: The question text to intervene on (must be substring of full_prompt)
        n_variants: Number of variants to generate
        p: Fraction of words to modify
        min_len: Minimum word length
        base_seed: Random seed base

    Returns:
        List of full prompts with only the question portion intervened
    """
    if question_text not in full_prompt:
        raise ValueError("question_text must be a substring of full_prompt")

    variants = []
    for i in range(n_variants):
        seed = (base_seed + i) if base_seed is not None else None
        intervened_question = soc_intervene(
            question_text, p=p, min_len=min_len, tail_only=True, seed=seed
        )
        variant_prompt = full_prompt.replace(question_text, intervened_question, 1)
        variants.append(variant_prompt)

    return variants


def spot_check_interventions(
    texts: List[str],
    n_samples: int = 20,
    p: float = 0.3,
    min_len: int = 4,
    seed: int = 42,
) -> List[Tuple[str, str]]:
    """Generate sample (original, intervened) pairs for manual review.

    Args:
        texts: List of original texts
        n_samples: Number of samples to generate
        p: SOC intervention probability
        min_len: Minimum word length
        seed: Random seed

    Returns:
        List of (original, intervened) tuples
    """
    random.seed(seed)
    samples = random.sample(texts, min(n_samples, len(texts)))

    pairs = []
    for i, text in enumerate(samples):
        intervened = soc_intervene(text, p=p, min_len=min_len, seed=seed + i)
        pairs.append((text, intervened))

    return pairs


# Quick test
if __name__ == "__main__":
    test_text = "Which programming paradigm emphasizes immutable data structures?"

    print("Original:")
    print(f"  {test_text}")
    print("\nSOC Variants (p=0.3, min_len=4):")

    variants = soc_intervene_batch(test_text, n_variants=5, p=0.3, min_len=4, base_seed=42)
    for i, v in enumerate(variants):
        print(f"  {i+1}. {v}")

    # Show what changed
    print("\nSpot check:")
    for orig, interv in spot_check_interventions([test_text], n_samples=1, seed=42):
        print(f"  Original:   {orig}")
        print(f"  Intervened: {interv}")
