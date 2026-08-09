# Phase 3 C-MAPSS Recommendation

This note freezes the first Phase 3 C-MAPSS uncertainty and monotonicity
decision after the FD001 audit and tail-fallback sweep. It summarizes the
policy posture; the supporting run evidence is in the Evidence Snapshot below.

**Every interval in this note is validation-fitted calibration and carries no
coverage guarantee.** The coverage figures below are what was observed on the
official test set at a requested confidence; nothing in the method bounds the
probability that a band contains the truth on a new engine, and "reaches the
requested coverage" should be read as an observation, not a property. Intervals
with a distribution-free guarantee were added later — unit-grouped split
conformal, measured in [writeup.md](writeup.md) §6 — and they do not supersede
this recommendation, which still governs the deployed policy.

## Recommendation

Keep the global validation-residual interval as the deployable baseline for the
current C-MAPSS deep candidate. It is simple, inference-safe, validation-fitted,
and reaches the requested official-test coverage in the current audit.

Track two tail-fallback variants as research candidates, not deployed policy:

- Balanced experiment: predicted-tail fallback at threshold `91`, confidence
  `0.95`. It improves coverage and covers unit `16`, but unit `67` remains
  uncovered.
- Aggressive safety candidate: predicted-tail fallback at threshold `76`,
  confidence `0.99`. It covers both late-overestimate misses, but with a large
  interval-width cost.

Do not adopt raw predicted-bin intervals or global-floor predicted-bin
intervals as the default policy. The raw predicted-bin strategy loses coverage;
the global-floor variant restores coverage but does not reduce late-risk
failures.

Do not add constrained training losses yet. The available official-test final
rows show zero raw and calibrated monotonicity violations, so the evidence points
to tail calibration and high-RUL compression before model-loss changes.

## Evidence Snapshot

| Policy | Status | Coverage | Mean width | Uncovered late predictions | Notes |
|---|---|---:|---:|---:|---|
| Global validation residual interval | Deployable baseline | `0.900000` | `48.665198` | `2` | Meets nominal coverage with the narrowest accepted baseline. |
| Predicted-bin interval | Diagnostic only | `0.880000` | `43.826050` | `2` | Narrower, but below nominal coverage. |
| Predicted-bin interval with global floor | Guardrail candidate only | `0.900000` | `50.679584` | `2` | Restores coverage, but does not reduce late failures. |
| Tail fallback `91/0.95` | Balanced experiment | `0.940000` | `51.534539` | `1` | Covers unit `16`; unit `67` remains uncovered. |
| Tail fallback `76/0.99` | Aggressive safety candidate | `0.970000` | `64.717455` | `0` | Covers units `16` and `67`; width rises by `16.052257` cycles over global. |

## Decision Boundary

This is operational research evidence, not certified PHM uncertainty. Official
test labels are used only for audit reporting. Inference-safe candidate policies
may use predicted RUL, validation residuals, and validation-fitted thresholds;
they must not use actual RUL bins at inference time.

The global interval remains the product baseline until a future policy improves
late-risk coverage with an acceptable width and operational trade-off. The
`91/0.95` and `76/0.99` tail-fallback candidates are useful evidence for that
future decision, but neither should silently replace the baseline.

## Next Milestone

Start ESA-ADB as a separate protocol-intake milestone before model
implementation. Use this C-MAPSS evidence loop as the pattern: protocol first,
then calibration and diagnostics, then policy recommendation.
