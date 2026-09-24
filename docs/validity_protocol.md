# Validity and Data-Splitting Protocol

DynScoreCP uses four disjoint roles:

1. **Model training:** fit the classifier and record training dynamics.
2. **Score fitting:** fit the difficulty-to-score transformation.
3. **Calibration:** estimate the finite-sample conformal threshold.
4. **Testing:** evaluate coverage and efficiency once.

The score-fitting stage must finish before calibration begins. Selecting a
rectifier, bin count, penalty, or model after observing calibration/test
coverage invalidates a clean held-out evaluation unless an additional validation
split or nested procedure is used.

For calibration scores `s_1, ..., s_n`, DynScoreCP uses the order statistic at

```text
ceil((n + 1) * (1 - alpha))
```

clipped to the largest observed score. Candidate labels with scores at most this
threshold are included in the prediction set. Under exchangeability and a score
function fixed before calibration, this provides marginal finite-sample
coverage. It does not provide distribution-free conditional coverage and does
not protect against arbitrary deployment shift.

The synthetic demo uses input coordinates for neighbourhood transfer so that
the geometry is inspectable. Image experiments should use a frozen embedding
and patient-level splits. Hyperparameters must not be selected on the final
hospital/site test set.
