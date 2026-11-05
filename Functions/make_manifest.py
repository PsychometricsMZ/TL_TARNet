"""Build a manifest for run_appendix_c.py from file patterns.

    # C.2 (blocks): the main text's non-randomized targets, stacked files with a ``rep`` column
    python make_manifest.py --out manifests/appc_blocks.csv --n-reps 50 \
        --spec "data/sim1/target_n50_biased.csv:50:1" --spec "data/sim1/target_n500_biased.csv:500:1"

    # C.3 (lambda) from the main text's files plus the generated grid (make_lambda_targets.py also
    # writes a manifest of its own files, so this is only needed to mix the two)
    python make_manifest.py --out manifests/appc_lambda.csv --n-reps 50 \
        --spec "data/sim1/target_n50_random.csv:50:0"  --spec "data/sim1/target_n500_random.csv:500:0" \
        --spec "data/sim1/target_n50_biased.csv:50:1"  --spec "data/sim1/target_n500_biased.csv:500:1" \
        --spec "data/appendix_c/target_n50_a0.25.csv:50:0.25" ...

Each --spec is PATTERN:N_T:a.  A matching file with a ``rep`` column contributes one manifest row per
replication (the first --n-reps values of rep, sorted); files without it are sorted and numbered
rep = 0, 1, ... (the first --n-reps are kept).  Output columns: path, N_T, rep, a.
"""
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
