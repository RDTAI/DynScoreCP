import unittest
import numpy as np
from dynscore.checkpoints import checkpoint_features, dynamic_score, wilson_interval
from dynscore.calibration import SplitConformalClassifier


class CheckpointTests(unittest.TestCase):
    def test_equal_final_confidence_different_trajectories(self):
        history = np.array([[[.9, .1], [.3, .7]], [[.9, .1], [.9, .1]]])
        features = checkpoint_features(history)
        np.testing.assert_allclose(features["final"][0], features["final"][1])
        self.assertGreater(dynamic_score(history)[1, 0], dynamic_score(history)[0, 0])
        self.assertEqual(features["rank_change"][1, 0], 1)

    def test_zero_penalty_is_ensemble_lac(self):
        p = np.array([[[.8, .2]], [[.6, .4]]])
        np.testing.assert_allclose(dynamic_score(p, 0), [[.3, .7]])

    def test_class_permutation_equivariance(self):
        p = np.array([[[.8, .2]], [[.6, .4]]])
        np.testing.assert_allclose(dynamic_score(p[:, :, ::-1]), dynamic_score(p)[:, ::-1])

    def test_invalid_history(self):
        for p in [np.ones((1, 2, 2)), np.full((2, 2, 2), np.nan), np.ones((2, 2, 2))]:
            with self.assertRaises(ValueError):
                checkpoint_features(p)

    def test_small_alpha_returns_all_labels(self):
        calibrator = SplitConformalClassifier(np.array([.1, .2]))
        self.assertEqual(calibrator.threshold(alpha=.01), float("inf"))
        self.assertTrue(calibrator.predict([[.9, 100]], alpha=.01).all())

    def test_empty_group_is_not_zero_coverage(self):
        self.assertIsNone(wilson_interval(0, 0))
