"""Synthetic benchmark for training-dynamics-aware conformal scores."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dynscore import (
    ConditionalCDFRectifier,
    LocationScaleRectifier,
    SplitConformalClassifier,
    TrainingDynamicsDifficulty,
    aps_score,
    evaluate_prediction_sets,
    forgetting_events,
    lac_score,
    raps_score,
    softmax,
)
from dynscore.scores import true_label_scores


CENTERS = np.array([[-2.2, -1.1], [2.2, -1.0], [0.0, 2.2]], dtype=np.float32)


def sample_data(
    rng: np.random.Generator, *, samples_per_class: int, scale: float = 1.25
) -> tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    for label, center in enumerate(CENTERS):
        x.append(rng.normal(center, scale, size=(samples_per_class, 2)))
        y.append(np.full(samples_per_class, label))
    features = np.concatenate(x).astype(np.float32)
    labels = np.concatenate(y).astype(np.int64)
    order = rng.permutation(labels.size)
    return features[order], labels[order]


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def probabilities(model: nn.Module, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return softmax(model(torch.from_numpy(x)))


def plot_difficulty_results(
    path: Path,
    train_x: np.ndarray,
    train_difficulty_signal: np.ndarray,
    test_x: np.ndarray,
    test_difficulty: np.ndarray,
    results: dict[str, dict[str, object]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(17, 5))
    first = axes[0].scatter(
        train_x[:, 0],
        train_x[:, 1],
        c=train_difficulty_signal,
        cmap="magma",
        s=14,
        alpha=0.7,
    )
    axes[0].set_title("Training-dynamics difficulty")
    figure.colorbar(
        first,
        ax=axes[0],
        label="forgetting + error frequency + confidence variability",
    )
    second = axes[1].scatter(
        test_x[:, 0], test_x[:, 1], c=test_difficulty, cmap="magma", s=14, alpha=0.7
    )
    axes[1].set_title("Predicted local difficulty")
    figure.colorbar(second, ax=axes[1], label="difficulty")

    methods = list(results)
    width = 0.8 / len(methods)
    for method_index, method in enumerate(methods):
        groups = results[method]["difficulty_groups"]
        positions = np.arange(len(groups)) + (method_index - (len(methods) - 1) / 2) * width
        axes[2].bar(
            positions,
            [group["coverage"] for group in groups],
            width,
            label=method,
        )
    axes[2].axhline(0.9, color="black", linestyle="--", linewidth=1, label="target")
    axes[2].set_xticks(np.arange(5), ["easiest", "2", "3", "4", "hardest"])
    axes[2].set_ylim(0.55, 1.02)
    axes[2].set_ylabel("Coverage")
    axes[2].set_title("Coverage by difficulty quintile")
    axes[2].legend(fontsize=7)
    for axis in axes[:2]:
        axis.set_xlabel("x1")
        axis.set_ylabel("x2")
        axis.grid(alpha=0.15)
    figure.suptitle("DynScoreCP: training dynamics as a nonconformity-score context")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(seed: int, epochs: int, alpha: float, output_dir: Path) -> dict[str, object]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    train_x, train_y = sample_data(rng, samples_per_class=300)
    score_fit_x, score_fit_y = sample_data(rng, samples_per_class=200)
    calibration_x, calibration_y = sample_data(rng, samples_per_class=200)
    test_x, test_y = sample_data(rng, samples_per_class=200)

    train_tensor = torch.from_numpy(train_x)
    label_tensor = torch.from_numpy(train_y)
    loader = DataLoader(
        TensorDataset(train_tensor, label_tensor),
        batch_size=64,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    model = TinyMLP()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()
    correctness = []
    true_label_confidence = []
    for _ in range(epochs):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            audit_logits = model(train_tensor)
            correctness.append((audit_logits.argmax(dim=1) == label_tensor).numpy())
            audit_probabilities = torch.softmax(audit_logits, dim=1)
            true_label_confidence.append(
                audit_probabilities[torch.arange(train_y.size), label_tensor].numpy()
            )

    train_forgetting = forgetting_events(np.stack(correctness))
    train_error_frequency = 1.0 - np.stack(correctness).mean(axis=0)
    train_confidence_variability = np.stack(true_label_confidence).std(axis=0)
    train_difficulty_signal = (
        train_forgetting + train_error_frequency + train_confidence_variability
    )
    difficulty_model = TrainingDynamicsDifficulty(
        train_x, train_difficulty_signal, neighbours=30
    )
    score_fit_difficulty = difficulty_model.predict(score_fit_x)
    calibration_difficulty = difficulty_model.predict(calibration_x)
    test_difficulty = difficulty_model.predict(test_x)

    score_fit_probabilities = probabilities(model, score_fit_x)
    calibration_probabilities = probabilities(model, calibration_x)
    test_probabilities = probabilities(model, test_x)
    score_fit_uniforms = rng.random(score_fit_y.size)
    calibration_uniforms = rng.random(calibration_y.size)
    test_uniforms = rng.random(test_y.size)

    score_matrices = {
        "LAC": (
            lac_score(score_fit_probabilities),
            lac_score(calibration_probabilities),
            lac_score(test_probabilities),
        ),
        "APS": (
            aps_score(score_fit_probabilities, randomization=score_fit_uniforms),
            aps_score(calibration_probabilities, randomization=calibration_uniforms),
            aps_score(test_probabilities, randomization=test_uniforms),
        ),
        "RAPS": (
            raps_score(
                score_fit_probabilities,
                penalty=0.02,
                regularize_after_rank=1,
                randomization=score_fit_uniforms,
            ),
            raps_score(
                calibration_probabilities,
                penalty=0.02,
                regularize_after_rank=1,
                randomization=calibration_uniforms,
            ),
            raps_score(
                test_probabilities,
                penalty=0.02,
                regularize_after_rank=1,
                randomization=test_uniforms,
            ),
        ),
    }

    lac_fit, lac_calibration, lac_test = score_matrices["LAC"]
    lac_fit_true = true_label_scores(lac_fit, score_fit_y)
    cdf = ConditionalCDFRectifier.fit(lac_fit_true, score_fit_difficulty, bins=5)
    normalized = LocationScaleRectifier.fit(lac_fit_true, score_fit_difficulty, bins=5)
    score_matrices["DynCDF-LAC"] = (
        cdf.transform(lac_fit, score_fit_difficulty),
        cdf.transform(lac_calibration, calibration_difficulty),
        cdf.transform(lac_test, test_difficulty),
    )
    score_matrices["DynNorm-LAC"] = (
        normalized.transform(lac_fit, score_fit_difficulty),
        normalized.transform(lac_calibration, calibration_difficulty),
        normalized.transform(lac_test, test_difficulty),
    )

    results: dict[str, dict[str, object]] = {}
    for name, (_, calibration_scores, test_scores) in score_matrices.items():
        calibrator = SplitConformalClassifier(
            true_label_scores(calibration_scores, calibration_y)
        )
        prediction_sets = calibrator.predict(test_scores, alpha=alpha)
        results[name] = evaluate_prediction_sets(
            prediction_sets,
            test_y,
            difficulty=test_difficulty,
            difficulty_bins=5,
        )
        results[name]["threshold"] = calibrator.threshold(alpha=alpha)

    report: dict[str, object] = {
        "seed": seed,
        "epochs": epochs,
        "alpha": alpha,
        "splits": {
            "model_train": int(train_y.size),
            "score_fit": int(score_fit_y.size),
            "calibration": int(calibration_y.size),
            "test": int(test_y.size),
        },
        "training_samples_with_forgetting": int((train_forgetting > 0).sum()),
        "mean_training_error_frequency": float(train_error_frequency.mean()),
        "mean_training_confidence_variability": float(train_confidence_variability.mean()),
        "methods": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    plot_difficulty_results(
        output_dir / "difficulty_coverage.png",
        train_x,
        train_difficulty_signal,
        test_x,
        test_difficulty,
        results,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/synthetic"))
    args = parser.parse_args()
    report = run(args.seed, args.epochs, args.alpha, args.output_dir)
    compact = {
        name: {
            "coverage": values["coverage"],
            "mean_set_size": values["mean_set_size"],
            "worst_difficulty_coverage": values["worst_difficulty_coverage"],
        }
        for name, values in report["methods"].items()
    }
    print(json.dumps(compact, indent=2))
    print(f"Artifacts written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
