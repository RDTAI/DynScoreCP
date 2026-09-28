"""Frozen-checkpoint score benchmark with development-only selection and diagnostics.

Run from the repository root: PYTHONPATH=src python examples/demo_checkpoints.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from demo_synthetic import CENTERS, TinyMLP, probabilities
from dynscore.calibration import SplitConformalClassifier
from dynscore.checkpoints import checkpoint_features, wilson_interval
from dynscore.metrics import evaluate_prediction_sets
from dynscore.rectification import ConditionalCDFRectifier
from dynscore.scores import aps_score, lac_score, raps_score, true_label_scores


def describe(sets, labels, mask):
    n = int(mask.sum())
    if not n:
        return {"n": 0, "coverage": None, "mean_set_size": None, "coverage_ci95": None}
    covered = sets[np.arange(len(labels)), labels][mask]
    return {"n": n, "coverage": float(covered.mean()),
            "mean_set_size": float(sets[mask].sum(axis=1).mean()),
            "coverage_ci95": wilson_interval(int(covered.sum()), n)}


def run(seed, epochs, checkpoints, alphas, output):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    # Three development roles keep CDF fitting and lambda selection separate.
    names = ["train", "score_fit", "tune_cal", "tune_eval", "calibration", "test"]
    data = {}
    for name in names:
        n = 900 if name == "train" else 600
        labels = rng.integers(0, 3, size=n, dtype=np.int64)
        x = (CENTERS[labels] + rng.normal(0, 1.25, size=(n, 2))).astype(np.float32)
        data[name] = (x, labels)
    train_x, train_y = data["train"]
    loader = DataLoader(TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
                        batch_size=64, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    model = TinyMLP()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.001)
    history = {name: [] for name in names[1:]}
    saved_epochs = np.linspace(epochs // 2, epochs - 1, checkpoints, dtype=int)
    for epoch in range(epochs):
        model.train()
        for x, y in loader:
            optimizer.zero_grad()
            nn.functional.cross_entropy(model(x), y).backward()
            optimizer.step()
        if epoch in saved_epochs:
            for name in history:
                history[name].append(probabilities(model, data[name][0]))
    features = {name: checkpoint_features(np.stack(h)) for name, h in history.items()}
    scores = {}
    for name, f in features.items():
        u = rng.random(len(data[name][1]))
        scores[name] = {"Final-LAC": lac_score(f["final"]),
                        "Final-APS": aps_score(f["final"], randomization=u),
                        "Final-RAPS": raps_score(f["final"], randomization=u, penalty=.02),
                        "Mean-LAC": lac_score(f["mean"])}
    # Identical rectifier; only the context changes. Final-label std is label-free.
    def context(f, kind):
        if kind == "confidence":
            return 1 - f["final"].max(axis=1)
        return f["std"][np.arange(len(f["final"])), f["final"].argmax(axis=1)]
    for kind in ["confidence", "dynamics", "shuffled"]:
        fit_context = context(features["score_fit"], "dynamics" if kind == "shuffled" else kind)
        if kind == "shuffled":
            fit_context = rng.permutation(fit_context)
        rectifier = ConditionalCDFRectifier.fit(
            true_label_scores(scores["score_fit"]["Final-LAC"], data["score_fit"][1]),
            fit_context, bins=5)
        for name, f in features.items():
            scores[name]["CDF-" + kind] = rectifier.transform(
                scores[name]["Final-LAC"],
                context(f, "dynamics" if kind == "shuffled" else kind))

    # Freeze diagnostic boundaries using score-fit inputs only.
    instability_cutoff = float(np.quantile(context(features["score_fit"], "dynamics"), .8))
    f = features["test"]
    confidence = f["final"].max(axis=1)
    instability = context(f, "dynamics")
    masks = {"all": np.ones(len(confidence), dtype=bool),
             "high_confidence": confidence >= .9,
             "high_confidence_unstable": (confidence >= .9) & (instability >= instability_cutoff),
             "high_confidence_stable": (confidence >= .9) & (instability < instability_cutoff)}
    for lo, hi in [(0, .6), (.6, .8), (.8, .9), (.9, 1.01)]:
        for unstable in [False, True]:
            masks[f"confidence_{lo}_{hi}_unstable_{unstable}"] = (
                (confidence >= lo) & (confidence < hi) & ((instability >= instability_cutoff) == unstable))
    for label in range(3):
        masks[f"class_{label}"] = data["test"][1] == label

    report = {"seed": seed, "checkpoint_epochs": (saved_epochs + 1).tolist(),
              "split_sizes": {n: len(y) for n, (_, y) in data.items()},
              "instability_cutoff": instability_cutoff, "alphas": {}}
    grid = [0., .25, .5, 1., 2.]
    for alpha in alphas:
        selection = []
        for penalty in grid:
            def candidate(name):
                return 1 - features[name]["mean"] + penalty * features[name]["std"]
            calibrator = SplitConformalClassifier(true_label_scores(candidate("tune_cal"), data["tune_cal"][1]))
            metrics = evaluate_prediction_sets(calibrator.predict(candidate("tune_eval"), alpha=alpha), data["tune_eval"][1])
            selection.append({"penalty": penalty, "coverage": metrics["coverage"],
                              "mean_set_size": metrics["mean_set_size"]})
        feasible = [r for r in selection if r["coverage"] >= 1 - alpha]
        best = min(feasible, key=lambda r: (r["mean_set_size"], r["penalty"])) if feasible else max(
            selection, key=lambda r: (r["coverage"], -r["mean_set_size"], -r["penalty"]))
        penalty = best["penalty"]
        for name in features:
            scores[name]["Dynamic-tuned"] = 1 - features[name]["mean"] + penalty * features[name]["std"]
            scores[name]["Dynamic-fixed-1"] = 1 - features[name]["mean"] + features[name]["std"]
        results = {}
        for method, test_scores in scores["test"].items():
            calibrator = SplitConformalClassifier(true_label_scores(scores["calibration"][method], data["calibration"][1]))
            sets = calibrator.predict(test_scores, alpha=alpha)
            metrics = evaluate_prediction_sets(sets, data["test"][1])
            metrics["groups"] = {name: describe(sets, data["test"][1], mask) for name, mask in masks.items()}
            results[method] = metrics
        report["alphas"][str(alpha)] = {"selected_penalty": penalty,
            "selection_feasible": bool(feasible), "development_grid": selection, "methods": results}
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / f"seed_{seed}_histories.npz",
                        **{name: np.stack(h) for name, h in history.items()},
                        test_labels=data["test"][1], test_features=data["test"][0])
    (output / f"seed_{seed}.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 23, 42])
    parser.add_argument("--alphas", nargs="+", type=float, default=[.05, .1, .2])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--checkpoints", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/checkpoints"))
    args = parser.parse_args()
    if args.epochs < 4 or not 2 <= args.checkpoints <= args.epochs - args.epochs // 2:
        parser.error("need epochs>=4 and 2<=checkpoints<=number of second-half epochs")
    if any(not 0 < a < 1 for a in args.alphas):
        parser.error("alphas must lie in (0, 1)")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique")
    torch.set_num_threads(1)
    reports = [run(s, args.epochs, args.checkpoints, args.alphas, args.output_dir) for s in args.seeds]
    summary = {}
    for alpha in args.alphas:
        key = str(alpha)
        summary[key] = {}
        for method in reports[0]["alphas"][key]["methods"]:
            summary[key][method] = {}
            for metric in ["coverage", "mean_set_size", "empty_rate", "singleton_rate"]:
                values = [r["alphas"][key]["methods"][method][metric] for r in reports]
                summary[key][method][metric] = {"mean": float(np.mean(values)),
                    "seed_std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                    "per_seed": values}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    for alpha, methods in summary.items():
        print("alpha", alpha)
        for method, metrics in methods.items():
            print(f"{method:20} coverage={metrics['coverage']['mean']:.4f} size={metrics['mean_set_size']['mean']:.4f}")


if __name__ == "__main__":
    main()
