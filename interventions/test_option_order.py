#!/usr/bin/env python3
"""
Unit tests for option order intervention.

Verifies:
1. Unpermute recovers original vector exactly
2. Content strings match after shuffling (no duplication/truncation)
3. Mapping is bijective
"""

import unittest
from option_order import shuffle_options, shuffle_options_batch, remap_probs, inverse_mapping


class TestOptionOrder(unittest.TestCase):

    def setUp(self):
        self.question = {
            "question": "What is the capital of France?",
            "options": {
                "A": "London",
                "B": "Paris",
                "C": "Berlin",
                "D": "Madrid"
            },
            "category": "easy"
        }

    def test_shuffle_preserves_content(self):
        """Content strings must match exactly one original option."""
        for seed in range(100):
            shuffled, mapping = shuffle_options(self.question, seed=seed)

            # All original content should appear exactly once
            original_contents = set(self.question["options"].values())
            shuffled_contents = set(shuffled["options"].values())

            self.assertEqual(original_contents, shuffled_contents,
                           f"Content mismatch at seed {seed}")

    def test_mapping_is_bijective(self):
        """Mapping should be a valid permutation (bijective)."""
        for seed in range(100):
            _, mapping = shuffle_options(self.question, seed=seed)

            # Keys and values should both be {A, B, C, D}
            self.assertEqual(set(mapping.keys()), {"A", "B", "C", "D"})
            self.assertEqual(set(mapping.values()), {"A", "B", "C", "D"})

    def test_unpermute_recovers_original(self):
        """Remap should recover the original distribution exactly."""
        original_probs = {"A": 0.1, "B": 0.6, "C": 0.2, "D": 0.1}

        for seed in range(100):
            _, mapping = shuffle_options(self.question, seed=seed)

            # Permute the probs (simulate what model sees)
            # mapping is new_label -> old_label
            # So if mapping = {'A': 'C', ...}, then new option A has content of old C
            # Model outputs probs for new labels
            # We need to remap: new_label's prob goes to old_label

            # Simulate: model sees shuffled options and outputs probs for new labels
            # based on content (which came from old labels)
            inv = inverse_mapping(mapping)  # old_label -> new_label

            # If model responds to content, P(new_A) = P_original(content of new_A) = P_original(mapping['A'])
            permuted_probs = {}
            for new_label in ["A", "B", "C", "D"]:
                old_label = mapping[new_label]  # content came from old_label
                permuted_probs[new_label] = original_probs[old_label]

            # Now remap back
            recovered = remap_probs(permuted_probs, mapping)

            for label in ["A", "B", "C", "D"]:
                self.assertAlmostEqual(
                    recovered[label], original_probs[label], places=10,
                    msg=f"Mismatch at label {label}, seed {seed}, mapping {mapping}"
                )

    def test_content_mapping_consistency(self):
        """Verify mapping correctly describes content movement."""
        for seed in range(50):
            shuffled, mapping = shuffle_options(self.question, seed=seed)

            for new_label, old_label in mapping.items():
                # New option should have content from old option
                self.assertEqual(
                    shuffled["options"][new_label],
                    self.question["options"][old_label],
                    f"Content mismatch: new {new_label} should have old {old_label}'s content"
                )

    def test_batch_produces_different_shuffles(self):
        """Batch should produce different permutations (with high probability)."""
        variants = shuffle_options_batch(self.question, n_variants=10, base_seed=42)

        # Extract mappings
        mappings = [tuple(sorted(m.items())) for _, m in variants]

        # Should have some variety (not all identical)
        unique_mappings = set(mappings)
        self.assertGreater(len(unique_mappings), 1,
                          "Batch should produce different permutations")

    def test_identity_mapping_possible(self):
        """At least one seed should produce identity mapping."""
        found_identity = False
        for seed in range(1000):
            _, mapping = shuffle_options(self.question, seed=seed)
            if mapping == {"A": "A", "B": "B", "C": "C", "D": "D"}:
                found_identity = True
                break

        # Identity has 1/24 probability, so should appear within 1000 tries
        self.assertTrue(found_identity, "Should find identity mapping within 1000 seeds")


class TestRemapProbs(unittest.TestCase):

    def test_remap_simple(self):
        """Test basic remapping."""
        # mapping: new_label -> old_label
        # If A->C, it means new A has content of old C
        # So prob that model assigned to new A should go to old C
        mapping = {"A": "C", "B": "A", "C": "D", "D": "B"}
        probs = {"A": 0.5, "B": 0.2, "C": 0.1, "D": 0.2}

        remapped = remap_probs(probs, mapping)

        # new A (0.5) -> old C
        # new B (0.2) -> old A
        # new C (0.1) -> old D
        # new D (0.2) -> old B
        self.assertAlmostEqual(remapped["C"], 0.5)
        self.assertAlmostEqual(remapped["A"], 0.2)
        self.assertAlmostEqual(remapped["D"], 0.1)
        self.assertAlmostEqual(remapped["B"], 0.2)

    def test_remap_identity(self):
        """Identity mapping should return same probs."""
        mapping = {"A": "A", "B": "B", "C": "C", "D": "D"}
        probs = {"A": 0.1, "B": 0.6, "C": 0.2, "D": 0.1}

        remapped = remap_probs(probs, mapping)

        for label in ["A", "B", "C", "D"]:
            self.assertAlmostEqual(remapped[label], probs[label])


if __name__ == "__main__":
    unittest.main()
