# AlphaTrim-Exp

Simulation scripts for the paper "Alpha-Trimming: Locally Adaptive Tree Pruning for Random Forests," submitted to *Computational Statistics & Data Analysis*.

## Requirements

You will need the modified `ranger` package with alpha-trimming support installed.
See the [RFtrim.R](https://github.com/nikola-sur/RFtrim.R-CSDA) repository.

## Simulation files

- `full_sim_study.Rmd`: Full simulation study on 46 data sets (Section 4.1).
- `varying_SNR_sim.R`: Varying SNR simulations (Section 4.2, single replicate).
- `varying_SNR_sim_20rep.R`: Varying SNR simulations with CC-Trim (Section 4.2, 20 replicates).
- `high_dim_elbow_sim.R`: High-dimensional multi-scale simulation (single replicate).
- `high_dim_multiscale_20rep.R`: High-dimensional multi-scale simulation (Section 4.3, 20 replicates).
- `mtry_sensitivity_elbow.R`: Sensitivity to `mtry` on the Elbow DGP (Appendix).

## Notice about data

Some of the benchmark data sets from Chipman et al. (2010) were obtained from a collaborator and cannot be redistributed. These have been replaced with placeholder files in `data/`. The synthetic data sets (Constant, Elbow, Logistic, Sine) are included.
