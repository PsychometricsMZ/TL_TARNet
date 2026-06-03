#!/usr/bin/env python3
"""
generate_data.py  --  Python port of generate_data.R (no R required).

Produces byte-compatible CSVs + manifest for the TL-TARNet mismatch sweep:
identical DGP, identical column schema, identical target filenames. Drop-in
for run.py / aggregate.py. The data is statistically equivalent to the R
generator (same distributions); it is NOT bit-identical, since numpy and R use
different RNGs -- but it is internally reproducible via the `seed` below.

Nonlinear DGP (lambda=0 recovers the paper's linear model):
  x^S ~ N(0, I_k);  x^T ~ N(delta*1, (1-rho)I + rho 11')      [covariate shift]
  g(x)=sin x1 + x2 x3 + x4^2 ;  q(x)=x1 x2 + cos x3            [shared nonlinearity]
  b(x)=gamma'x + lambda g(x) ;  tau(x)=beta + omega'x + lambda q(x)   [ITE = tau]
  Y(0)=b+e0,  Y(1)=b+tau+e1,  e~N(0,sigma^2)
  mechanism change:  gamma_T=gamma_S + m_gamma v,  omega_T=omega_S + m_omega v,
                     v=(1,-1,1,-1,1)/sqrt(k)   (gamma cancels in tau)
  assignment: source always Bern(1/2); target Bern(1/2) or Bern(expit(std(Y0))).

Usage:
  python generate_data.py            # full grid -> ./data
  python generate_data.py pilot      # tiny smoke-test grid
"""

import os
import sys
import numpy as np
import pandas as pd

# ---- fixed DGP constants (match generate_data.R) ---------------------------
k       = 5
gammaS  = np.full(k, 0.5)
omegaS  = np.full(k, 0.5)
beta    = 1.0
lam     = 1.0          # nonlinearity strength (0 -> linear DGP)
sigma   = 1.0          # residual sd (held fixed across all cells)
vdir    = np.array([1, -1, 1, -1, 1], dtype=float) / np.sqrt(k)


def gfun(X):  # prognostic nonlinearity
    return np.sin(X[:, 0]) + X[:, 1] * X[:, 2] + X[:, 3] ** 2


def qfun(X):  # effect-modification nonlinearity
    return X[:, 0] * X[:, 1] + np.cos(X[:, 2])


def _expit(z):
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


def draw(rng, n, mu=0.0, rho=0.0, m_gamma=0.0, m_omega=0.0, biased=False):
    """One dataset. m_gamma perturbs prognostic coeffs (NOT in ITE);
    m_omega perturbs effect-modification coeffs (IS the ITE)."""
    Sig = (1 - rho) * np.eye(k) + rho * np.ones((k, k))
    X = rng.multivariate_normal(np.full(k, mu), Sig, size=n)
    gT = gammaS + m_gamma * vdir
    wT = omegaS + m_omega * vdir
    base = X @ gT + lam * gfun(X)
    ite = beta + X @ wT + lam * qfun(X)          # true ITE
    Y0 = base + rng.normal(0, sigma, n)
    Y1 = base + ite + rng.normal(0, sigma, n)
    if biased:
        z = (Y0 - Y0.mean()) / Y0.std(ddof=1)    # standardized (R's scale())
        Tr = rng.binomial(1, _expit(z)).astype(float)
    else:
        Tr = rng.binomial(1, 0.5, n).astype(float)
    Y = np.where(Tr == 1, Y1, Y0)
    df = pd.DataFrame(X, columns=[f"X{i+1}" for i in range(k)])
    df["Tr"] = Tr
    df["Y"] = Y
    df["ite"] = ite                               # Y0/Y1 omitted (ite is scored)
    return df


def kl_covshift(delta, rho):
    """KL(target || source) for N(delta*1, Sigma_rho) || N(0, I)."""
    return 0.5 * (k * delta ** 2 - (k - 1) * np.log(1 - rho) - np.log(1 + (k - 1) * rho))


def generate(outdir="data",
             source_sizes=(1000, 5000),
             target_sizes=(50, 100, 500),
             reps=100,
             cov_levels=(0, 0.25, 0.50, 0.75, 1.00),
             mech_levels=(0, 0.25, 0.50, 0.75, 1.00),
             delta_max=1.0, rho_max=0.6,
             seed=20240601):
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(seed)

    # ONE source dataset per size (always randomized), reused everywhere.
    for sn in source_sizes:
        draw(rng, sn).to_csv(os.path.join(outdir, f"source_n{sn}.csv"), index=False)

    assignments = {"randomized": False, "biased": True}

    # Three isolation slices (each varies exactly one bound term):
    #   covshift   : P(X) only        -> covariate-IPM term, gamma* = 0
    #   effect     : omega only       -> changes the ITE, inflates gamma*
    #   prognostic : gamma only        -> ITE UNCHANGED, inflates gamma*
    def cfg(level, delta, rho, mg, mw):
        return dict(level=level, delta=delta, rho=rho, m_gamma=mg, m_omega=mw)

    slices = {
        "covshift":   [cfg(c, c * delta_max, c * rho_max, 0.0, 0.0) for c in cov_levels],
        "effect":     [cfg(c, 0.0, 0.0, 0.0, c) for c in mech_levels],
        "prognostic": [cfg(c, 0.0, 0.0, c, 0.0) for c in mech_levels],
    }

    rows = []
    for tn in target_sizes:
        for asg, biased in assignments.items():
            for sl, cfgs in slices.items():
                for c in cfgs:
                    lvl, d, rho, mg, mw = c["level"], c["delta"], c["rho"], c["m_gamma"], c["m_omega"]
                    # stack all reps into one file (rep ids 0..reps-1)
                    parts = []
                    for r in range(reps):
                        dd = draw(rng, tn, mu=d, rho=rho, m_gamma=mg, m_omega=mw, biased=biased)
                        dd.insert(0, "rep", r)
                        parts.append(dd)
                    tfile = f"target_n{tn}_{sl}_L{lvl:.2f}_{asg}.csv"
                    pd.concat(parts, ignore_index=True).to_csv(os.path.join(outdir, tfile), index=False)

                    KL = kl_covshift(d, rho) if sl == "covshift" else 0.0
                    for sn in source_sizes:
                        for r in range(reps):
                            rows.append(dict(
                                target_file=tfile, rep=r,
                                source_tag=f"source_n{sn}", source_file=f"source_n{sn}.csv",
                                source_n=sn, target_n=tn, slice=sl, level=lvl, assignment=asg,
                                delta=d, rho=rho, m_gamma=mg, m_omega=mw,
                                dgamma_norm=mg, domega_norm=mw, KL=KL))  # ||change||=m (||v||=1)

    man = pd.DataFrame(rows).sort_values(
        ["source_tag", "target_file", "rep"]).reset_index(drop=True)  # localise array shards
    man.to_csv(os.path.join(outdir, "manifest.csv"), index=False)
    print(f"Wrote {len(source_sizes)} source files, "
          f"{man['target_file'].nunique()} target files, "
          f"manifest with {len(man)} cells -> {outdir}/")
    return man


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "pilot":
        generate(source_sizes=(1000,), target_sizes=(50, 100), reps=5,
                 cov_levels=(0, 0.5, 1.0), mech_levels=(0, 0.5, 1.0))
    else:
        generate()
