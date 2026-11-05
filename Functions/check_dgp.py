
import sys

import numpy as np
import pandas as pd

import tltarnet as tl
import appendix_c as ac

rng = np.random.default_rng(7)
U = ac.draw_units(400_000, rng)
E_Y0 = U["Y0"].mean()


def implied(link, label):
    p = link(U["Y0"])
    t = U["U"] < p
    eps0 = U["Y0"] - ac.mu0_fn(U["X"])
    return f"{label:>34s}  P(T=1) {t.mean():.3f}  E[eps0|T=0] {eps0[~t].mean():+.3f}  E[eps0|T=1] {eps0[t].mean():+.3f}"


print(f"s = SD(Y0) = {ac.SD_Y0:.4f}  (Monte Carlo {U['Y0'].std():.4f}),  E[Y0] = {E_Y0:.3f}")
for path in sys.argv[1:]:
    df = pd.read_csv(path)
    X = df[tl.x_columns(df)].to_numpy(float)
    t = df["Tr"].to_numpy(float) > 0.5
    y = df["Y"].to_numpy(float)
    m0, tau = ac.mu0_fn(X), ac.tau_fn(X)
    eps0_c = y[~t] - m0[~t]
    eps1_t = y[t] - m0[t] - tau[t]
    print(f"\n{path}: {len(df)} rows" + (f", {df.rep.nunique()} reps" if "rep" in df.columns else ""))
    print(f"  max |ite - tau(x)| = {np.abs(df['ite'].to_numpy(float) - tau).max():.2e}")
    print(f"  P(T=1) = {t.mean():.3f}   E[eps0|T=0] = {eps0_c.mean():+.3f} (SD {eps0_c.std():.3f})   "
          f"E[eps1|T=1] = {eps1_t.mean():+.3f} (SD {eps1_t.std():.3f})")
    print(f"  mean(X) = {X.mean(0).round(2)}, SD(X) = {X.std(0).round(2)}")

print("\nimplied by the DGP (Eq. 16) under each selection link (the row matching the file is its selection):")
for a in (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0):
    print(implied(lambda y0, a=a: ac.expit(a * (y0 - E_Y0) / ac.SD_Y0), f"centered, a = {a:g}"))
for a in (0.5, 1.0, 2.0):
    print(implied(lambda y0, a=a: ac.expit(a * y0 / ac.SD_Y0), f"uncentered, a = {a:g}"))
print(implied(lambda y0: ac.expit(y0), "expit(Y0), no standardization"))
