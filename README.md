# Transfer learning for treatment-effect estimation in small samples

Code, simulated data and results for the paper of the same title, which evaluates the transfer-learning framework for TARNet of Aloui et al. (2023), TL-TARNet and the CITA score, on small and possibly non-randomized targets.

The paper is under double-blind review. Author and citation information will be added after the review.

## Contents

| Folder | Content |
|---|---|
| `Functions/` | `tltarnet.py`: the network, the estimators, CITA and the error measures. `generate_data.py` and `generate_nn.py`: the data generators of Simulations 1–2 and of Simulation 3 |
| `Data/` | Source and target data sets, one folder per simulation |
| `Results/` | Errors, biases and CITA for every target data set, one folder per simulation and one for Appendix C |

Requires Python 3 with NumPy, pandas and PyTorch.

## Data

CSV files with the covariates `X1`, `X2`, ..., the treatment `Tr`, the outcome `Y` and the true effect `ite`. A target file stacks the replications of one condition (column `rep`), and `biased` in its name is the non-randomized assignment. `manifest.csv` lists the design of every file.

The seeds are fixed in the scripts; the stored files are the ones analysed in the paper.

```bash
python Functions/generate_data.py sim1 --outdir regenerated/sim1
python Functions/generate_data.py sim2 --outdir regenerated/sim2
python Functions/generate_nn.py --dims 5 --rescale-target --outdir regenerated/sim3
```

## Results

Each row holds the results for one target data set. In the column names, `target` is target-only CFR, `target0` target-only TARNet, `source` the source-only model, `pool` pooled CFR, `tl` TL-TARNet and `warm` the warm start. The normalized error of the tables is `pehe_<estimator> / true_ite_sd**2`.

| Paper | Files |
|---|---|
| Table 1, Figures 2, 3 and 6 | `Results/sim1/main_with_warm_tlearner.csv` |
| Table 2, Figure 4 | `Results/sim2/main_with_warm_tlearner.csv` |
| Table 3 | `Results/sim2/chunk_ablation_*.csv` |
| Tables 5 and 6, Figure 5 | `Results/sim3/chunk_main_*.csv` |
| Table 8 | `Results/appendix_c/appendix_c/lr1e-3_wsrc0/blocks_*.csv` |
| Table 9, Figure 7 | `Results/appendix_c/appendix_c/lambda_*.csv` |

