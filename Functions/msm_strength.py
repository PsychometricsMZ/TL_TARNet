"""Strength of the non-randomized selection in Simulation 1, computed inside the paper's own DGP.

DGP (Eq. 16):  X ~ N(0, I_5),  mu0(x) = 0.5 * sum(x) + sin(x1) + x2*x3 + x4^2,
               Y0 = mu0(X) + eps0,  eps0 ~ N(0, 1),  tau(x) = 1 + 0.5 * sum(x) + x1*x2 + cos(x3).
Selection:     pi(x, eps) = P(T = 1 | X = x, eps0 = eps) = expit(a * (mu0(x) + eps - m) / s),  m = E[Y0] = 1, s = SD(Y0) = 2.51
               (a times the standardized control outcome, the main text's rule: treated share 1/2 at every a).
               CENTER = 0 gives the uncentered form expit(a * Y0 / s) (treated share 0.58 at a = 1).
               a = 0 is the randomized target; a = 1 reproduces the hidden contrast 0.33 quoted in the paper.

For each a the script reports
  Lambda_q         q-quantile of max(OR, 1/OR), OR(x, eps) = odds(pi(x, eps)) / odds(e(x)),  e(x) = E[pi(x, eps0) | x]:
                   the marginal-sensitivity parameter that covers a fraction q of the population
  AUC(T | X)       AUC of the nominal propensity e(x) for T: strength of the observed (covariate) component
  contrast         E[eps0 | T = 1] - E[eps0 | T = 0]  (marginal, as quoted in the paper)
  b_X              -E_X[ E[eps0 | T = 0, X] ]: bias of the mean effect for an estimator that adjusts for X perfectly,
                   when the treated residual is independent of the selection (as in the paper's note)
  b_X (shared)     E_X[ E[eps | T = 1, X] - E[eps | T = 0, X] ]: the same if Y1 and Y0 share one residual (Eq. 16 as printed)
  naive            E[Y | T = 1] - E[Y | T = 0] - E[tau(X)]: bias with no adjustment at all (independent residuals)

All expectations over eps0 are Gauss-Hermite quadrature (60 nodes); averages over X are Monte Carlo over 200,000 draws.
"""
import numpy as np
from numpy.polynomial.hermite_e import hermegauss

rng = np.random.default_rng(1)
N = 200_000
X = rng.standard_normal((N, 5))
mu0 = 0.5 * X.sum(1) + np.sin(X[:, 0]) + X[:, 1] * X[:, 2] + X[:, 3] ** 2
tau = 1 + 0.5 * X.sum(1) + X[:, 0] * X[:, 1] + np.cos(X[:, 2])
eps0 = rng.standard_normal(N)
eps1 = rng.standard_normal(N)
Y0 = mu0 + eps0
Y1 = mu0 + tau + eps1
S = Y0.std()
CENTER = 1.0                             # E[Y0]; set to 0.0 for the uncentered selection

nodes, weights = hermegauss(60)          # probabilists' Hermite: integrates against N(0, 1)
weights = weights / weights.sum()


def expit(z):
    return 1.0 / (1.0 + np.exp(-z))


def auc(score, t):
    order = np.argsort(score)
    ranks = np.empty(N)
    ranks[order] = np.arange(1, N + 1)
    n1 = t.sum()
    return (ranks[t].sum() - n1 * (n1 + 1) / 2) / (n1 * (N - n1))


def analyse(a):
    P = expit(a * (mu0[:, None] + nodes[None, :] - CENTER) / S)   # pi(x, eps) on the quadrature grid
    e_x = P @ weights                                          # nominal propensity e(x)
    E_eps_c = ((1 - P) * nodes[None, :]) @ weights / (1 - e_x)  # E[eps0 | T = 0, x]
    E_eps_t = (P * nodes[None, :]) @ weights / e_x              # E[eps0 | T = 1, x]
    b_X = -E_eps_c.mean()
    b_X_shared = (E_eps_t - E_eps_c).mean()
    p_full = expit(a * (Y0 - CENTER) / S)                       # pi at the realised eps0
    with np.errstate(divide="ignore", invalid="ignore"):
        OR = (p_full / (1 - p_full)) / (e_x / (1 - e_x))
        L = np.maximum(OR, 1 / OR)
    lam = {q: np.quantile(L, q) for q in (0.5, 0.9, 0.95)}
    T = rng.random(N) < p_full
    contrast = eps0[T].mean() - eps0[~T].mean()
    Y = np.where(T, Y1, Y0)
    naive = Y[T].mean() - Y[~T].mean() - tau.mean()
    return dict(a=a, p1=T.mean(), lam=lam, auc_x=auc(e_x, T), contrast=contrast,
                b_X=b_X, b_X_shared=b_X_shared, naive=naive)


if __name__ == "__main__":
    print(f"SD(Y0) = {S:.3f};  Var(mu0) = {mu0.var():.3f}  ->  share of the selection on X = {mu0.var() / (mu0.var() + 1):.3f};  "
          f"selection expit(a * (Y0 - {CENTER:g}) / s)")
    print(f"{'a':>5} {'P(T=1)':>7} {'L50':>6} {'L90':>6} {'L95':>6} {'AUC(T|X)':>8} {'contrast':>8} "
          f"{'b_X':>6} {'b_X(sh)':>7} {'naive':>6}")
    for a in (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0):
        r = analyse(a)
        print(f"{r['a']:>5.2f} {r['p1']:>7.3f} {r['lam'][0.5]:>6.2f} {r['lam'][0.9]:>6.2f} {r['lam'][0.95]:>6.2f} "
              f"{r['auc_x']:>8.3f} {r['contrast']:>8.3f} {r['b_X']:>6.3f} {r['b_X_shared']:>7.3f} {r['naive']:>6.3f}")
