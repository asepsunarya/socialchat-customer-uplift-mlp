"""
Unit test ringan untuk rekomendasi.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from socialchat_uplift.recommender import get_recommendation


class TestRecommender(unittest.TestCase):
    def test_high_free(self):
        rec = get_recommendation("High", "Free")
        self.assertEqual(rec["priority"], 1)
        self.assertIn("Premium", rec["recommended_action"])

    def test_low_basic(self):
        rec = get_recommendation("Low", "Basic")
        self.assertEqual(rec["priority"], 3)
        self.assertIn("retensi", rec["recommended_action"].lower())


if __name__ == "__main__":
    unittest.main()
