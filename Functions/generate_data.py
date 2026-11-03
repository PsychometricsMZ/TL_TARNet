
import argparse
import os

import numpy as np
import pandas as pd

# ---- DGP constants ---------------------------------------------------------
beta = 1.0
lam = 1.0          # nonlinearity strength (0 -> linear DGP)
sigma = 1.0        # residual sd
k = gammaS = omegaS = vdir = None   # set by configure()


def configure(p=5):
    """Set the number of covariates.  gamma = omega = 0.5 for every covariate; the
    unit perturbation direction alternates in sign; the non-linear terms g and q are
    repeated over consecutive blocks of four covariates (X1..X4, X5..X8, ...), any
    remaining covariates enter linearly only.  p = 5 reproduces the original DGP."""
    global k, gammaS, omegaS, vdir
    k = int(p)
    gammaS = np.full(k, 0.5)
    omegaS = np.full(k, 0.5)
    vdir = np.array([(-1.0) ** i for i in range(k)]) / np.sqrt(k)


configure(5)


def gfun(X):  # prognostic nonlinearity, one block per four covariates
    out = np.zeros(len(X))
    for b in range(k // 4):
        x1, x2, x3, x4 = X[:, 4 * b], X[:, 4 * b + 1], X[:, 4 * b + 2], X[:, 4 * b + 3]
        out += np.sin(x1) + x2 * x3 + x4 ** 2
    return out


def qfun(X):  # effect-modification nonlinearity, one block per four covariates
    out = np.zeros(len(X))
    for b in range(k // 4):
        x1, x2, x3 = X[:, 4 * b], X[:, 4 * b + 1], X[:, 4 * b + 2]
        out += x1 * x2 + np.cos(x3)
    return out


def _expit(z):
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


def covariance(rho):
    return (1 - rho) * np.eye(k) + rho * np.ones((k, k))


def base_fun(X, m_gamma=0.0):
    return X @ (gammaS + m_gamma * vdir) + lam * gfun(X)


def tau_fun(X, m_omega=0.0):
    return beta + X @ (omegaS + m_omega * vdir) + lam * qfun(X)


# ---------------------------------------------------------------------------
def draw(rng, n, mu=0.0, rho=0.0, m_gamma=0.0, m_omega=0.0, assignment="randomized",
         gamma_msm=1.0, u_delta=0.0):
    """One dataset.  m_gamma perturbs prognostic coefficients (NOT in the ITE);
    m_omega perturbs effect-modification coefficients (IS the ITE)."""
    X = rng.multivariate_normal(np.full(k, mu), covariance(rho), size=n)
    base = base_fun(X, m_gamma)
    ite = tau_fun(X, m_omega)
    U = rng.normal(0, 1, n)
    Y0 = base + u_delta * U + rng.normal(0, sigma, n)
    Y1 = base + ite + u_delta * U + rng.normal(0, sigma, n)
    if assignment == "randomized":
        Tr = rng.binomial(1, 0.5, n)
    elif assignment == "biased":
        z = (Y0 - Y0.mean()) / Y0.std(ddof=1)        # standardised (R's scale())
        Tr = rng.binomial(1, _expit(z))
    elif assignment == "msm":
        p1 = gamma_msm / (1.0 + gamma_msm)           # odds = Gamma  (U > 0)
        p0 = 1.0 / (1.0 + gamma_msm)                 # odds = 1/Gamma (U < 0)
        Tr = rng.binomial(1, np.where(U > 0, p1, p0))
    else:
        raise ValueError(assignment)
    Tr = Tr.astype(float)
    Y = np.where(Tr == 1, Y1, Y0)
    df = pd.DataFrame(X, columns=[f"X{i + 1}" for i in range(k)])
    df["Tr"] = Tr
    df["Y"] = Y
    df["ite"] = ite                                  # true CATE of the TARGET
    if u_delta > 0:
        df["U"] = U                                  # unmeasured; not a feature
    return df


def kl_covshift(delta, rho):
    """KL(target || source) for N(delta*1, Sigma_rho) || N(0, I)."""
    return 0.5 * (k * delta ** 2 - (k - 1) * np.log(1 - rho) - np.log(1 + (k - 1) * rho))


def cell_distances(mu, rho, m_gamma, m_omega, n_mc=20000, seed=12345):
    """Monte-Carlo D_tau, D_mu under the target covariate distribution and the
    variance of tau_S under the source distribution (for normalisation)."""
    rng = np.random.default_rng(seed)
    XT = rng.multivariate_normal(np.full(k, mu), covariance(rho), size=n_mc)
    XS = rng.multivariate_normal(np.zeros(k), np.eye(k), size=n_mc)
    d_tau = float(np.mean((tau_fun(XT, 0.0) - tau_fun(XT, m_omega)) ** 2))
    d_mu = float(np.mean((base_fun(XT, 0.0) - base_fun(XT, m_gamma)) ** 2))
    var_tau_s = float(np.var(tau_fun(XS, 0.0)))
    return dict(D_tau=d_tau, D_mu=d_mu, D_tau_norm=d_tau / var_tau_s,
                D_mu_norm=d_mu / var_tau_s, var_tau_source=var_tau_s)


# ---------------------------------------------------------------------------
def _write_sources(outdir, rng, source_sizes, u_delta=0.0):
    """ONE source dataset per size (always randomized), reused by every cell."""
    for sn in source_sizes:
        draw(rng, sn, u_delta=u_delta).to_csv(os.path.join(outdir, f"source_n{sn}.csv"), index=False)


def _write_cells(outdir, rng, experiment, cells, source_sizes, reps):
    """cells: list of dicts with keys target_n, assignment, gamma_msm, u_delta,
    mu, rho, m_gamma, m_omega, slice, level, tag.  Writes one CSV per cell
    (all reps stacked) and returns the manifest rows."""
    rows = []
    for c in cells:
        parts = []
        for r in range(reps):
            dd = draw(rng, c["target_n"], mu=c["mu"], rho=c["rho"], m_gamma=c["m_gamma"],
                      m_omega=c["m_omega"], assignment=c["assignment"],
                      gamma_msm=c["gamma_msm"], u_delta=c["u_delta"])
            dd.insert(0, "rep", r)
            parts.append(dd)
        tfile = f"target_{c['tag']}.csv"
        pd.concat(parts, ignore_index=True).to_csv(os.path.join(outdir, tfile), index=False)
        dist = cell_distances(c["mu"], c["rho"], c["m_gamma"], c["m_omega"])
        KL = kl_covshift(c["mu"], c["rho"]) if (c["mu"] != 0 or c["rho"] != 0) else 0.0
        for sn in source_sizes:
            for r in range(reps):
                rows.append(dict(
                    experiment=experiment, target_file=tfile, rep=r,
                    source_tag=f"source_n{sn}", source_file=f"source_n{sn}.csv",
                    source_n=sn, target_n=c["target_n"], assignment=c["assignment"], p=k,
                    gamma_msm=c["gamma_msm"], u_delta=c["u_delta"],
                    slice=c["slice"], level=c["level"], delta=c["mu"], rho=c["rho"],
                    m_gamma=c["m_gamma"], m_omega=c["m_omega"], KL=KL, **dist))
    return rows


def _finish(outdir, rows):
    man = pd.DataFrame(rows).sort_values(["source_tag", "target_file", "rep"]).reset_index(drop=True)
    man.to_csv(os.path.join(outdir, "manifest.csv"), index=False)
    print(f"{outdir}: {man['source_tag'].nunique()} source files, "
          f"{man['target_file'].nunique()} target files, manifest with {len(man)} cells")
    return man


# ---------------------------------------------------------------------------
def generate_sim1(outdir, source_sizes=(1000, 5000, 10000), target_sizes=(50, 100, 250, 500),
                  reps=100, seed=20240601):
    """Simulation 1: shared population (level 0), target sample size x assignment."""
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(seed)
    _write_sources(outdir, rng, source_sizes)
    cells = [dict(target_n=tn, assignment=asg, gamma_msm=1.0, u_delta=0.0, mu=0.0, rho=0.0,
                  m_gamma=0.0, m_omega=0.0, slice="none", level=0.0, tag=f"n{tn}_{asg}")
             for tn in target_sizes for asg in ("randomized", "biased")]
    return _finish(outdir, _write_cells(outdir, rng, "sim1", cells, source_sizes, reps))


def generate_sim2(outdir, source_sizes=(1000, 5000), target_sizes=(50, 100, 500), reps=100,
                  cov_levels=(0, 0.25, 0.50, 0.75, 1.00), mech_levels=(0, 0.25, 0.50, 0.75, 1.00),
                  delta_max=1.0, rho_max=0.6, seed=20240601):
    """Simulation 2: source-target mismatch, one bound term at a time."""
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(seed)
    _write_sources(outdir, rng, source_sizes)
    slices = {
        "covshift":   [(c, c * delta_max, c * rho_max, 0.0, 0.0) for c in cov_levels],   # P(X) only
        "effect":     [(c, 0.0, 0.0, 0.0, c) for c in mech_levels],                     # omega -> ITE changes
        "prognostic": [(c, 0.0, 0.0, c, 0.0) for c in mech_levels],                     # gamma -> ITE unchanged
    }
    cells = []
    for tn in target_sizes:
        for asg in ("randomized", "biased"):
            for sl, cfgs in slices.items():
                for lvl, d, rho, mg, mw in cfgs:
                    cells.append(dict(target_n=tn, assignment=asg, gamma_msm=1.0, u_delta=0.0,
                                      mu=d, rho=rho, m_gamma=mg, m_omega=mw, slice=sl, level=lvl,
                                      tag=f"n{tn}_{sl}_L{lvl:.2f}_{asg}"))
    return _finish(outdir, _write_cells(outdir, rng, "sim2", cells, source_sizes, reps))


def generate_sim3(outdir, source_sizes=(5000,), target_sizes=(100, 500), reps=100,
                  gammas=(1.0, 1.5, 2.0, 3.0, 5.0), u_delta=1.0, seed=20240601):
    """Simulation 3: hidden confounding of known strength (MSM parameter Gamma),
    plus the paper's expit(z) mechanism as a reference cell."""
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(seed)
    _write_sources(outdir, rng, source_sizes, u_delta=u_delta)
    cells = []
    for tn in target_sizes:
        for g in gammas:
            cells.append(dict(target_n=tn, assignment="msm", gamma_msm=g, u_delta=u_delta, mu=0.0,
                              rho=0.0, m_gamma=0.0, m_omega=0.0, slice="none", level=g,
                              tag=f"n{tn}_msm_G{g:.2f}"))
        cells.append(dict(target_n=tn, assignment="biased", gamma_msm=float("nan"), u_delta=u_delta,
                          mu=0.0, rho=0.0, m_gamma=0.0, m_omega=0.0, slice="none", level=float("nan"),
                          tag=f"n{tn}_biased_ref"))
    return _finish(outdir, _write_cells(outdir, rng, "sim3", cells, source_sizes, reps))


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", choices=["sim1", "sim2", "sim3"])
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20240601)
    ap.add_argument("--pilot", action="store_true", help="tiny version for smoke tests")
    ap.add_argument("--k", type=int, default=5, help="number of covariates (default 5 = original DGP)")
    ap.add_argument("--source-sizes", type=int, nargs="+", default=None)
    ap.add_argument("--target-sizes", type=int, nargs="+", default=None)
    ap.add_argument("--levels", type=float, nargs="+", default=None, help="mismatch levels (sim2 only)")
    a = ap.parse_args()
    configure(a.k)
    outdir = a.outdir or os.path.join("data", a.experiment)
    extra = {}
    if a.source_sizes:
        extra["source_sizes"] = tuple(a.source_sizes)
    if a.target_sizes:
        extra["target_sizes"] = tuple(a.target_sizes)
    if a.experiment == "sim1":
        if a.pilot:
            generate_sim1(outdir, source_sizes=(1000,), target_sizes=(50, 100), reps=3, seed=a.seed)
        else:
            generate_sim1(outdir, reps=a.reps, seed=a.seed, **extra)
    elif a.experiment == "sim2":
        if a.pilot:
            generate_sim2(outdir, source_sizes=(1000,), target_sizes=(50, 100), reps=3,
                          cov_levels=(0, 1.0), mech_levels=(0, 1.0), seed=a.seed)
        else:
            if a.levels:
                extra["cov_levels"] = extra["mech_levels"] = tuple(a.levels)
            generate_sim2(outdir, reps=a.reps, seed=a.seed, **extra)
    else:
        if a.pilot:
            generate_sim3(outdir, source_sizes=(1000,), target_sizes=(100,), reps=3,
                          gammas=(1.0, 3.0), seed=a.seed)
        else:
            generate_sim3(outdir, reps=a.reps, seed=a.seed)

