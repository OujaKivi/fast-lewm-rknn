#!/usr/bin/env python3
"""Guard descriptive paired timing analysis against invalid or mixed samples."""

import math
import unittest

from analyze_smolvla_vision_service_matrix import paired_metrics, summarize


class PairedMetricsTests(unittest.TestCase):
    def test_pairing_is_not_difference_of_medians(self):
        result = paired_metrics([1, 10, 2], [2, 3, 11])
        self.assertEqual(result["paired_median_saved_ms"], 1)
        self.assertEqual(result["baseline_median_ms"] - result["candidate_median_ms"], 1)
        self.assertEqual(result["wins"], 2)
        self.assertEqual(result["trials"], 3)

    def test_median_difference_can_disagree_with_paired_median(self):
        result = paired_metrics([1, 2, 100], [2, 100, 101])
        self.assertEqual(result["paired_median_saved_ms"], 1)
        self.assertEqual(result["baseline_median_ms"] - result["candidate_median_ms"], 98)

    def test_rejects_empty_and_unmatched_samples(self):
        for candidate, baseline in (([], []), ([1], [1, 2])):
            with self.assertRaises(ValueError):
                paired_metrics(candidate, baseline)

    def test_rejects_invalid_latency(self):
        for value in (0, -1, math.nan, math.inf):
            with self.assertRaises(ValueError):
                paired_metrics([value], [1])

    def test_incomplete_session_is_not_a_result(self):
        for status in ("running", "failed"):
            with self.assertRaises(ValueError):
                summarize({"status": status})


if __name__ == "__main__":
    unittest.main()
