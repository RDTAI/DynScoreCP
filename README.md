# DynScoreCP

**Training-dynamics-aware nonconformity scores for conformal classification.**

DynScoreCP is a research-oriented implementation for studying how the choice of
nonconformity score affects marginal coverage, difficulty-conditional coverage,
and prediction-set efficiency. Its central question is:

> Can training dynamics provide a better notion of sample difficulty for
> constructing adaptive conformal classification scores?

The project is experimental. It does not claim distribution-free conditional
coverage or validity under arbitrary distribution shift.

## Included in v0.1

- LAC, probability-margin, randomized APS, and RAPS candidate-label scores;
- finite-sample split-conformal calibration;
- training forgetting, error frequency, and confidence-variability signals;
- inverse-distance-weighted k-NN difficulty transfer;
- difficulty-bin location/scale normalization (`DynNorm-LAC`);
- empirical conditional-CDF rectification (`DynCDF-LAC`);
- marginal, class-conditional, and difficulty-conditional evaluation;
- a leakage-aware synthetic benchmark with four disjoint data roles.

Every score uses the same convention: **larger means less conforming**.

## Quick start

```bash
git clone https://github.com/RDTAI/DynScoreCP.git
cd DynScoreCP
python -m pip install -e '.[demo]'
python examples/demo_synthetic.py
```

The demo writes:

```text
outputs/synthetic/
├── metrics.json
└── difficulty_coverage.png
```

## Method

Let the base LAC score be

```text
A0(x, y) = 1 - p(y | x).
```

The model-training split supplies a transparent difficulty target combining
forgetting events, error frequency across epochs, and true-label confidence
variability. A weighted k-NN model transfers this signal to unseen inputs.

On a separate score-fitting split, `DynCDF-LAC` estimates the conditional score
distribution within difficulty bins and transforms a candidate score into its
local empirical rank:

```text
ADynCDF(x, y) = F_hat[A0 | difficulty](A0(x, y) | d_hat(x)).
```

The transformation is frozen before conformal calibration. Calibration and test
data are never used to fit the difficulty model or rectifier.

## Reference synthetic result

One run with seed 17, 40 epochs, and target coverage 0.90 produced:

| Method | Marginal coverage | Mean set size | Worst difficulty-quintile coverage |
| --- | ---: | ---: | ---: |
| LAC | 0.917 | 1.003 | 0.725 |
| APS | 0.915 | 1.120 | 0.867 |
| RAPS | 0.915 | 1.103 | 0.858 |
| DynCDF-LAC | 0.905 | 1.087 | 0.867 |
| DynNorm-LAC | 0.917 | 1.057 | 0.842 |

In this run, ordinary LAC had acceptable marginal coverage while severely
undercovering the hardest region. Conditional-CDF rectification substantially
reduced that gap, with a modest increase in set size. It matched APS on the
worst difficulty group; this is a proof-of-concept result, not evidence of
universal superiority.

## Validity contract

DynScoreCP separates four roles:

1. model training and dynamics collection;
2. score/difficulty fitting;
3. conformal calibration;
4. final testing.

With exchangeable calibration and test examples, and a score function frozen
before calibration, split conformal prediction retains marginal finite-sample
coverage. See [`docs/validity_protocol.md`](docs/validity_protocol.md).

## Test

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Research roadmap

- repeated-seed uncertainty intervals and hyperparameter validation;
- learned difficulty predictors on frozen embeddings;
- DynCDF transforms for APS, RAPS, and SAPS;
- class-conditional and group-conditional calibration;
- label-noise and class-imbalance stress tests;
- MedMNIST and multi-site medical-imaging experiments.

## Research foundations

- Romano, Sesia, and Candès, [Classification with Valid and Adaptive Coverage](https://arxiv.org/abs/2006.02544).
- Angelopoulos et al., [Uncertainty Sets for Image Classifiers using Conformal Prediction](https://arxiv.org/abs/2009.14193).
- Huang et al., [Conformal Prediction for Deep Classifier via Label Ranking](https://proceedings.mlr.press/v235/huang24aa.html).
- Plassier et al., [Rectifying Conformity Scores for Better Conditional Coverage](https://proceedings.mlr.press/v267/plassier25a.html).

DynScoreCP is an independent educational and research implementation.

## License

MIT
