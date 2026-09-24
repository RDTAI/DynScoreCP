import unittest

import numpy as np

from dynscore.calibration import SplitConformalClassifier, conformal_quantile


class CalibrationTests(unittest.TestCase):
    def test_finite_sample_quantile(self) -> None:
        scores = np.arange(1, 11, dtype=float)
        self.assertEqual(conformal_quantile(scores, alpha=0.2), 9.0)

    def test_prediction_set_uses_less_than_or_equal_threshold(self) -> None:
        classifier = SplitConformalClassifier(np.array([0.1, 0.2, 0.3, 0.4]))
        predicted = classifier.predict(np.array([[0.1, 0.3, 0.5]]), alpha=0.4)
        np.testing.assert_array_equal(predicted, [[True, True, False]])


if __name__ == "__main__":
    unittest.main()
