# Causal Inference — IHDP

This folder implements the non-mandatory Causal Inference task from the Intelligence SIG Recs 2026 repository.

## Objectives
- Estimate the naive treatment effect.
- Correct confounding using propensity-score methods.
- Estimate heterogeneous treatment effects with a causal forest.
- Check overlap/positivity.
- Compare estimated ITEs with IHDP ground truth.

## Dataset
Use one IHDP replication, commonly distributed in the CEVAE CSV layout:
`treatment, factual outcome, counterfactual outcome, mu0, mu1, x1...x25`.

Example:
`ihdp_npci_1.csv`.

Dataset files are intentionally not committed.
