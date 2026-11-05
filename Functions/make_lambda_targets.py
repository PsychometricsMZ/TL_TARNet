
import argparse
import os

import numpy as np
import pandas as pd

import tltarnet as tl
import appendix_c as ac

p = argparse.ArgumentParser()
p.add_argument("--outdir", required=True)
p.add_argument("--manifest", default=None)
p.add_argument("--a-grid", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0, 2.0, 3.0])
p.add_argument("--n-t", type=int, nargs="+", default=[50, 500])
p.add_argument("--n-reps", type=int, default=50)
p.add_argument("--seed", type=int, default=20261)
p.add_argument("--sd-y0", type=float, default=ac.SD_Y0, help="the s of the selection (default: exact SD(Y0) of Eq. 16)")
p.add_argument("--center", type=float, default=ac.E_Y0, help="the m of the selection (default: exact E[Y0] = 1; 0 = uncentered)")
p.add_argument("--subsample-source", default=None, help="source csv: draw the covariates from it instead of N(0, I)")
args = p.parse_args()

src_X = None
if args.subsample_source:
    src = pd.read_csv(args.subsample_source)
    src_X = src[tl.x_columns(src)].to_numpy(float)

os.makedirs(args.outdir, exist_ok=True)
frames = {(a, n_t): [] for a in args.a_grid for n_t in args.n_t}
for n_t in args.n_t:
    for r in range(args.n_reps):
        rng = np.random.default_rng([args.seed, n_t, r])          # one stream per (N_T, rep), shared by all a
        units = ac.draw_units(n_t, rng)
        if src_X is not None:                                       # same population, units taken from the source
            X = src_X[rng.choice(len(src_X), n_t, replace=False)]
            m0 = ac.mu0_fn(X)
            units = dict(X=X, Y0=m0 + rng.standard_normal(n_t), Y1=m0 + ac.tau_fn(X) + rng.standard_normal(n_t),
                         ite=ac.tau_fn(X), U=rng.random(n_t))
        for a in args.a_grid:
            df = ac.assign_treatment(units, a, args.sd_y0, args.center)
            df.insert(0, "rep", r)
            frames[(a, n_t)].append(df)

rows = []
for (a, n_t), parts in frames.items():
    path = os.path.join(args.outdir, f"target_n{n_t}_a{a:g}.csv")
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(path, index=False)
    p1 = df.groupby("rep").Tr.mean()
    print(f"{path}: {df.rep.nunique()} reps x {n_t} units, P(T=1) = {df.Tr.mean():.3f} "
          f"(per rep {p1.min():.2f}-{p1.max():.2f})")
    rows += [dict(path=path, N_T=n_t, rep=r, a=a) for r in range(args.n_reps)]

print(f"selection expit(a * (Y0 - {args.center:g}) / {args.sd_y0:.4f}); a-grid {args.a_grid}; {len(rows)} targets")
if args.manifest:
    os.makedirs(os.path.dirname(args.manifest) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(args.manifest, index=False)
    print("manifest:", args.manifest)
