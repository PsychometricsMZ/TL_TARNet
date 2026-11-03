
import argparse
import os

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
class RandomMLP:
    """tanh MLP with random weights; only the first ``n_informative`` inputs are used."""

    def __init__(self, rng, p, n_informative, hidden=(64, 64), gain=1.5):
        self.p, self.n_informative = p, n_informative
        sizes = [p] + list(hidden) + [1]
        self.W, self.b = [], []
        for d_in, d_out in zip(sizes[:-1], sizes[1:]):
            self.W.append(rng.normal(0.0, gain / np.sqrt(d_in), (d_in, d_out)))
            self.b.append(rng.normal(0.0, 0.1, d_out))
        # disconnect the noise covariates; the effective fan-in of the first
        # layer is then n_informative, not p, so rescale to gain / sqrt(n_informative)
        self.W[0][n_informative:, :] = 0.0
        self.W[0] *= np.sqrt(p / n_informative)

    def __call__(self, X):
        h = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W + b
            if i < len(self.W) - 1:
                h = np.tanh(h)
        return h[:, 0]

    def perturbed(self, rng, eps):
        """Layer-wise relative Gaussian perturbation of all weights and biases."""
        new = RandomMLP.__new__(RandomMLP)
        new.p, new.n_informative = self.p, self.n_informative
        new.W, new.b = [], []
        for W, b in zip(self.W, self.b):
            sW = W[W != 0].std() if np.any(W != 0) else 0.0
            sb = b.std() if b.size > 1 else 0.0
            Wn = W + eps * sW * rng.normal(size=W.shape) * (W != 0)   # keep noise inputs disconnected
            bn = b + eps * sb * rng.normal(size=b.shape)
            new.W.append(Wn)
            new.b.append(bn)
        return new


class Generator:
    """Source generator (mu0 net, tau net) with fixed output scaling."""

    def __init__(self, rng, p, n_informative=5, hidden=(64, 64), gain=2.0, tau_mean=1.0,
                 tau_sd=1.5, n_ref=100_000):
        self.p = p
        self.mu_net = RandomMLP(rng, p, n_informative, hidden, gain)
        self.tau_net = RandomMLP(rng, p, n_informative, hidden, gain)
        Xr = rng.normal(size=(n_ref, p))
        m_raw, t_raw = self.mu_net(Xr), self.tau_net(Xr)
        self.mu_loc, self.mu_scale = m_raw.mean(), m_raw.std() + 1e-12
        self.tau_loc, self.tau_scale = t_raw.mean(), t_raw.std() + 1e-12
        self.tau_mean, self.tau_sd = tau_mean, tau_sd
        # how non-linear are the source functions? R^2 of the best linear fit
        self.lin_r2_mu = _linear_r2(Xr, self.mu(Xr))
        self.lin_r2_tau = _linear_r2(Xr, self.tau(Xr))
        self.var_tau = float(np.var(self.tau(Xr)))

    # the SAME affine constants are applied to perturbed networks
    def mu(self, X, net=None):
        net = net or self.mu_net
        return (net(X) - self.mu_loc) / self.mu_scale

    def tau(self, X, net=None):
        net = net or self.tau_net
        return self.tau_mean + self.tau_sd * (net(X) - self.tau_loc) / self.tau_scale


def _linear_r2(X, y):
    A = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    return float(1.0 - resid.var() / y.var())


def _expit(z):
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


class Rescaled:
    """Wraps a perturbed network so that, under the target covariate distribution, its
    rescaled output has the same mean and sd as the source function (shape-only change)."""

    def __init__(self, net, loc, scale):
        self.net, self.loc, self.scale = net, loc, scale

    def __call__(self, X):
        return (self.net(X) - self.loc) / self.scale


def rescale_to_source(rng, gen, net, which, n_ref=100_000):
    """Return a Rescaled wrapper of ``net`` whose output, passed through gen.mu / gen.tau,
    has the source function's mean and sd on X ~ N(0, I_p)."""
    Xr = rng.normal(size=(n_ref, gen.p))
    raw = net(Xr)
    ref = gen.mu_net(Xr) if which == "mu" else gen.tau_net(Xr)
    # match raw moments to the source network's raw moments, so the source's affine map gives equal mean/sd
    return Rescaled(net, raw.mean() - ref.mean() * raw.std() / ref.std(), raw.std() / ref.std())


def draw(rng, gen, n, mu_net=None, tau_net=None, assignment="randomized", sigma=1.0):
    """One dataset from (possibly perturbed) networks; ``ite`` is the TARGET's CATE."""
    X = rng.normal(size=(n, gen.p))
    base = gen.mu(X, mu_net)
    ite = gen.tau(X, tau_net)
    Y0 = base + rng.normal(0, sigma, n)
    Y1 = base + ite + rng.normal(0, sigma, n)
    if assignment == "randomized":
        Tr = rng.binomial(1, 0.5, n)
    elif assignment == "biased":
        z = (Y0 - Y0.mean()) / Y0.std(ddof=1)
        Tr = rng.binomial(1, _expit(z))
    else:
        raise ValueError(assignment)
    Tr = Tr.astype(float)
    df = pd.DataFrame(X, columns=[f"X{i + 1}" for i in range(gen.p)])
    df["Tr"] = Tr
    df["Y"] = np.where(Tr == 1, Y1, Y0)
    df["ite"] = ite
    return df


def distances(rng, gen, mu_net, tau_net, n_mc=20000):
    X = rng.normal(size=(n_mc, gen.p))
    d_tau = float(np.mean((gen.tau(X) - gen.tau(X, tau_net)) ** 2))
    d_mu = float(np.mean((gen.mu(X) - gen.mu(X, mu_net)) ** 2))
    return dict(D_tau=d_tau, D_mu=d_mu, D_tau_norm=d_tau / gen.var_tau,
                D_mu_norm=d_mu / gen.var_tau, var_tau_source=gen.var_tau)


# ---------------------------------------------------------------------------
def generate(outdir="data/sim2b", dims=(5, 10, 20), n_informative=5, n_seeds=10,
             source_n=5000, target_sizes=(100, 500), draws=30, eps_min=0.01, eps_max=1.0,
             zero_share=0.1, hidden=(64, 64), gain=2.0, seed=20240602, rescale_target=False,
             all_informative=False):
    """One source (and one source model) per (p, generator seed); per cell
    (p, seed, target_n, assignment, perturb_type) ``draws`` targets, each with
    its own eps and its own perturbation."""
    os.makedirs(outdir, exist_ok=True)
    rows = []
    for p in dims:
        for g in range(n_seeds):
            rng = np.random.default_rng([seed, p, g])
            gen = Generator(rng, p, p if all_informative else n_informative, hidden, gain)
            src_tag = f"source_p{p}_g{g}"
            draw(rng, gen, source_n).to_csv(os.path.join(outdir, src_tag + ".csv"), index=False)
            for tn in target_sizes:
                for asg in ("randomized", "biased"):
                    for ptype in ("effect", "prognostic"):
                        parts = []
                        tfile = f"target_p{p}_g{g}_n{tn}_{ptype}_{asg}.csv"
                        for r in range(draws):
                            if r < int(round(zero_share * draws)):
                                eps = 0.0
                            else:
                                eps = float(np.exp(rng.uniform(np.log(eps_min), np.log(eps_max))))
                            mu_net = gen.mu_net.perturbed(rng, eps) if ptype == "prognostic" else gen.mu_net
                            tau_net = gen.tau_net.perturbed(rng, eps) if ptype == "effect" else gen.tau_net
                            if rescale_target and eps > 0:
                                # shape-only mismatch: perturbed function keeps the source's mean and sd,
                                # so D_tau / Var(tau_S) = 2 (1 - corr(tau_S, tau_T))
                                if ptype == "effect":
                                    tau_net = rescale_to_source(rng, gen, tau_net, "tau")
                                else:
                                    mu_net = rescale_to_source(rng, gen, mu_net, "mu")
                            dd = draw(rng, gen, tn, mu_net, tau_net, asg)
                            dd.insert(0, "rep", r)
                            parts.append(dd)
                            dist = distances(rng, gen, mu_net, tau_net)
                            rows.append(dict(
                                experiment="sim2b", target_file=tfile, rep=r,
                                source_tag=src_tag, source_file=src_tag + ".csv",
                                source_n=source_n, target_n=tn, assignment=asg,
                                p=p, n_informative=n_informative, gen_seed=g,
                                perturb_type=ptype, eps=eps, rescaled=bool(rescale_target),
                                lin_r2_mu=gen.lin_r2_mu, lin_r2_tau=gen.lin_r2_tau, **dist))
                        pd.concat(parts, ignore_index=True).to_csv(os.path.join(outdir, tfile), index=False)
    man = pd.DataFrame(rows).sort_values(["source_tag", "target_file", "rep"]).reset_index(drop=True)
    man.to_csv(os.path.join(outdir, "manifest.csv"), index=False)
    print(f"{outdir}: {man['source_tag'].nunique()} source files, "
          f"{man['target_file'].nunique()} target files, manifest with {len(man)} cells")
    return man


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="data/sim2b")
    ap.add_argument("--dims", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--draws", type=int, default=30)
    ap.add_argument("--eps-max", type=float, default=1.0)
    ap.add_argument("--gain", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=20240602)
    ap.add_argument("--rescale-target", action="store_true",
                    help="re-standardise the perturbed function to the source mean/sd (shape-only distance)")
    ap.add_argument("--all-informative", action="store_true",
                    help="connect all p covariates to the networks instead of the first 5")
    ap.add_argument("--pilot", action="store_true")
    a = ap.parse_args()
    if a.pilot:
        generate(a.outdir, dims=(5,), n_seeds=2, source_n=1000, target_sizes=(100,), draws=6,
                 eps_max=a.eps_max, gain=a.gain, seed=a.seed, rescale_target=a.rescale_target,
                 all_informative=a.all_informative)
    else:
        generate(a.outdir, dims=tuple(a.dims), n_seeds=a.n_seeds, draws=a.draws,
                 eps_max=a.eps_max, gain=a.gain, seed=a.seed, rescale_target=a.rescale_target,
                 all_informative=a.all_informative)

