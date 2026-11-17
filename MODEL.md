# Model notes — gta6-twitch-ccu-forecast-01

## What the program does
For each hour h = 2..24 it fits, on the case file's own comparable launches:
- **Training set ("regime"):** tier-comparable launches whose release-day
  peak is at most 20x their hour-1 print. This drops
  early-access-contaminated and weak launches, and launches whose audience
  explodes later — neither resembles a midnight console launch. 19 of the
  37 comparables in a practice case qualify (38 in the final, which also
  includes the four practice launches).
- **Model (a) log-log OLS:** log y_h = a + b * log p1.
- **Model (b) two-feature ridge:** log y_h ~ log p1 + log channels_1,
  ridge penalty 1.0 on the centred slopes.
- **Blend:** median = mean of the two medians; spread = max of the two
  residual SDs, scaled by 0.75. Quantiles = median * exp(z_tau * spread).
- **Fallback:** fewer than 5 comparables with data for an hour → the house
  persistence baseline for that hour. Missing hour-1 channels → the ridge
  is the OLS fit, so the blend degrades to OLS alone.

## Why this config (validation, 2026-11-17)
Pre-flight tried ~12 variants over 6 rounds. The honest floor is
~0.17–0.18 practice loss against a top-20 bar of 0.145 — the gap is
intrinsic launch-to-launch shape variability (residual log-SD ~0.3–0.5),
not functional form. This build then tested principled variants, each
judged by leave-one-out (LOO) on comparables, never by practice loss
alone (a grid search that reached 0.1622 on practice had LOO 0.4600 —
pure overfit, rejected). Results:

| config | practice | LOO (19 regime) | LOO (8 largest) |
|---|---|---|---|
| OLS, regime, spread 0.75 (pre-flight best) | 0.1744 | 0.2671 | 0.1949 |
| OLS, regime, spread 0.9 | 0.1793 | 0.2648 | 0.1963 |
| ridge2, spread 0.75 | 0.1720 | 0.2644 | 0.1986 |
| **blend (OLS + ridge2), spread 0.75** | **0.1728** | **0.2602** | **0.1916** |

The blend is nominally best on both LOO measures and near-best on
practice, and it hedges a real risk: the practice channels are SullyGnome
hourly averages while the final's are Twitch GraphQL counts, so a pure
channels model could misbehave on the final; the blend is half OLS, which
never touches channels. Per-case practice loss of the blend:
p00 0.234, p01 0.154, p02 0.116, p03 0.187. It is best on the biggest
launch (p02, hour-1 print 269,517) — the final is one huge launch, which
is the bet this entry makes.

## Honest odds
The paid top 20 sits at a practice loss of 0.145; this config's honest
practice loss is ~0.173, so a paid place is a long shot. The entry is a
lottery ticket on the final being a huge launch like p02, where the model
scores ~0.12. It beats the house baseline (0.387) with high probability.
