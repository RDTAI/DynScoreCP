import unittest

import numpy as np

from dynscore.difficulty import TrainingDynamicsDifficulty, forgetting_events


class DifficultyTests(unittest.TestCase):
    def test_forgetting_events_count_correct_to_incorrect_transitions(self) -> None:
        history = np.array([[0, 1], [1, 0], [0, 1], [1, 0]])
        np.testing.assert_array_equal(forgetting_events(history), [1.0, 2.0])

    def test_knn_transfers_local_difficulty(self) -> None:
        model = TrainingDynamicsDifficulty(
            np.array([[0.0], [1.0], [10.0], [11.0]]),
            np.array([0.0, 0.0, 4.0, 4.0]),
            neighbours=2,
        )
        np.testing.assert_allclose(model.predict(np.array([[0.2], [10.2]])), [0.0, 4.0])


if __name__ == "__main__":
    unittest.main()
