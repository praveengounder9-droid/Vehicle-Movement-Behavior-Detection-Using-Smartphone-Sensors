"""
Test suite to verify that core_analysis.py reproduces the exact results
from Project.ipynb on Accelerometer.csv and Gyroscope.csv.
"""

import sys
import unittest
import numpy as np
from core_analysis import (
    process_sensor_datasets,
    explain_single_observation,
    answer_user_query,
    generate_markdown_report,
)


class TestSensorPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.acc_path = "Accelerometer.csv"
        cls.gyro_path = "Gyroscope.csv"
        cls.result = process_sensor_datasets(cls.acc_path, cls.gyro_path)
        cls.df = cls.result["dataframe"]
        cls.stats = cls.result["stats"]

    def test_row_counts(self):
        """Verify row counts match Project.ipynb outputs."""
        self.assertEqual(self.stats["raw_acc_rows"], 102051)
        self.assertEqual(self.stats["raw_gyro_rows"], 105541)
        self.assertEqual(self.stats["acc_clean_rows"], 8445)
        self.assertEqual(self.stats["gyro_clean_rows"], 11848)
        self.assertEqual(self.stats["merged_rows"], 1733)
        self.assertEqual(len(self.df), 1733)

    def test_thresholds(self):
        """Verify thresholds match exact values calculated in notebook."""
        # Expected from notebook:
        # acc_threshold: ~2.7958166861869818
        # gyro_threshold: ~1.273287406360153
        self.assertAlmostEqual(self.stats["acc_threshold"], 2.7958166861869818, places=4)
        self.assertAlmostEqual(self.stats["gyro_threshold"], 1.273287406360153, places=4)

    def test_classifications(self):
        """Verify classification distribution matches notebook output."""
        # Expected:
        # Normal Movement: 1639
        # Sudden Acceleration/Braking: 53
        # Turning: 41
        counts = self.stats["event_counts"]
        self.assertEqual(counts["Normal Movement"], 1639)
        self.assertEqual(counts["Sudden Acceleration/Braking"], 53)
        self.assertEqual(counts["Turning"], 41)

    def test_observation_explanation(self):
        """Verify observation explanation generates detailed reasoning."""
        # Test row 18 (classified as Turning in Project.ipynb)
        row_18 = self.df.iloc[18]
        self.assertEqual(row_18["Event"], "Turning")
        exp = explain_single_observation(
            row_18, self.stats["acc_threshold"], self.stats["gyro_threshold"]
        )
        self.assertEqual(exp["event"], "Turning")
        self.assertTrue(exp["is_gyro_high"])
        self.assertIn("Turning", exp["reasoning_summary"])

        # Test a Normal Movement row (row 0)
        row_0 = self.df.iloc[0]
        self.assertEqual(row_0["Event"], "Normal Movement")
        exp_0 = explain_single_observation(
            row_0, self.stats["acc_threshold"], self.stats["gyro_threshold"]
        )
        self.assertEqual(exp_0["event"], "Normal Movement")

        # Test a Sudden Acceleration/Braking row
        brake_rows = self.df[self.df["Event"] == "Sudden Acceleration/Braking"]
        self.assertGreater(len(brake_rows), 0)
        brake_row = brake_rows.iloc[0]
        exp_b = explain_single_observation(
            brake_row, self.stats["acc_threshold"], self.stats["gyro_threshold"]
        )
        self.assertEqual(exp_b["event"], "Sudden Acceleration/Braking")
        self.assertTrue(exp_b["is_acc_change_high"])

    def test_query_answering(self):
        """Test standard queries produce factual and deterministic answers."""
        sample_questions = [
            "What does my dataset contain?",
            "What is the most common detected event?",
            "How many observations were classified as Normal Movement?",
            "What percentage of observations were classified as Turning?",
            "What are the calculated thresholds?",
            "At which timestamp is the gyroscope magnitude highest?",
            "Why was this observation classified as Sudden Acceleration/Braking?",
            "Summarize the main findings from my uploaded data.",
            "What are the limitations of these results?",
        ]
        for q in sample_questions:
            res = answer_user_query(q, self.result, selected_row_idx=18)
            self.assertTrue(res["matched"], f"Question failed to match: {q}")
            self.assertIn("answer", res)
            self.assertGreater(len(res["answer"]), 20)

    def test_report_generation(self):
        """Test markdown report generation."""
        report = generate_markdown_report(self.result)
        self.assertIn("1,733", report)
        self.assertIn("2.7958", report)
        self.assertIn("1.2733", report)
        self.assertIn("L.D. College of Engineering", report)


if __name__ == "__main__":
    unittest.main()
