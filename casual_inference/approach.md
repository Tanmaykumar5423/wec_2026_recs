# Approach — Causal Inference

## 1. Problem formulation

We want the causal effect of treatment T on outcome Y. In the potential-outcomes framework:

ATE = E[Y(1) - Y(0)].

The fundamental problem is that for one individual we only observe one potential outcome.

## 2. IHDP benchmark

Use an IHDP semi-synthetic replication. The standard CEVAE distribution stores treatment, factual outcome, counterfactual outcome, conditional means mu0/mu1, and 25 covariates. Because the benchmark supplies ground-truth potential outcomes/conditional means, we can evaluate both ATE and heterogeneous treatment effect quality.

## 3. Baseline — naive difference in means

Compute:
naive = E[Y | T=1] - E[Y | T=0].

This is generally biased when treatment assignment depends on covariates that also affect outcome.

## 4. Propensity-score methods

Estimate e(X) = P(T=1 | X) with a probabilistic classifier.

Use two corrections:
- Inverse Probability Weighting (IPW).
- Propensity-score nearest-neighbor matching.

For IPW, stabilized weights can be used:
w_i = T_i * p_bar/e_i + (1-T_i) * (1-p_bar)/(1-e_i).

For ATT, a treated unit receives weight 1 and a control receives e/(1-e). For this task we primarily report the ATE using stabilized ATE weights.

## 5. Positivity / overlap

Inspect propensity-score distributions for treated and control groups. Values close to 0 or 1 indicate weak overlap and unstable weighting. Clip propensity estimates only as a numerical safeguard and report that clipping can introduce bias.

## 6. Causal forest

Use EconML CausalForestDML with flexible outcome/treatment nuisance models. Estimate ITE/CATE for the held-out evaluation sample.

The forest should be trained only from observed outcomes, treatments and covariates. Ground-truth mu0/mu1 are used only for evaluation, never as features.

## 7. Evaluation

Report:
- Naive ATE
- IPW ATE
- Matching ATE
- Causal-forest ATE
- Absolute ATE error
- PEHE = sqrt(mean((tau_hat - tau_true)^2))
- ITE correlation
- ITE scatter plot against ground truth
- propensity overlap plot

## 8. Controlled interpretation

The central experiment is not simply “which number is largest.” Explain:
- how much naive confounding bias exists;
- whether propensity correction moves the ATE toward the truth;
- whether forest estimates recover heterogeneity;
- where poor overlap makes estimates unreliable.

## 9. Caveats

IHDP is semi-synthetic. Strong benchmark performance does not prove that the same identification assumptions hold in a real observational dataset. A useful follow-up is Lalonde, where individual ground truth is unavailable.
