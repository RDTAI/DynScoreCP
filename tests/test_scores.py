import unittest

import numpy as np

from dynscore.scores import aps_score, lac_score, margin_score, raps_score


class ScoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.probabilities = np.array([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]])

    def test_lac_score(self) -> None:
        np.testing.assert_allclose(lac_score(self.probabilities), 1 - self.probabilities)

    def test_margin_score(self) -> None:
        expected = np.array([[-0.3, 0.3, 0.5], [0.3, -0.2, 0.2]])
        np.testing.assert_allclose(margin_score(self.probabilities), expected)

    def test_deterministic_aps_is_cumulative_probability(self) -> None:
        expected = np.array([[0.6, 0.9, 1.0], [1.0, 0.5, 0.8]])
        np.testing.assert_allclose(aps_score(self.probabilities), expected)

    def test_raps_penalizes_low_rank_labels(self) -> None:
        aps = aps_score(self.probabilities)
        raps = raps_score(self.probabilities, penalty=0.1, regularize_after_rank=1)
        self.assertAlmostEqual(raps[0, 0], aps[0, 0])
        self.assertAlmostEqual(raps[0, 2], aps[0, 2] + 0.2)


if __name__ == "__main__":
    unittest.main()
