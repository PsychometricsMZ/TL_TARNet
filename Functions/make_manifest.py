
import argparse
import glob

import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--n-reps", type=int, default=50)
p.add_argument("--spec", action="append", required=True, help="PATTERN:N_T:a (repeatable)")
args = p.parse_args()

rows = []
for spec in args.spec:
    pattern, n_t, a = spec.rsplit(":", 2)
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no files match {pattern}")
    stacked = [f for f in files if "rep" in pd.read_csv(f, nrows=0).columns]
    if stacked:
        for f in stacked:
            reps = sorted(pd.read_csv(f, usecols=["rep"])["rep"].unique())[: args.n_reps]
            rows += [dict(path=f, N_T=int(n_t), rep=int(r), a=float(a)) for r in reps]
    else:
        rows += [dict(path=f, N_T=int(n_t), rep=r, a=float(a)) for r, f in enumerate(files[: args.n_reps])]
pd.DataFrame(rows).to_csv(args.out, index=False)
print(f"wrote {args.out}: {len(rows)} targets")
