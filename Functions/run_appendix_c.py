"""Runner for Appendix C (manifest-driven, array-job friendly).

    python run_appendix_c.py --exp blocks --source data/sim1/source_n5000.csv --manifest manifests/appc_blocks.csv \
        --out results/appendix_c/blocks_3.csv --chunk 3 --n-chunks 20 [--source-model models/appc_source_source_n5000.pt]

    python run_appendix_c.py --exp lambda --source data/sim1/source_n5000.csv --manifest manifests/appc_lambda.csv \
        --out results/appendix_c/lambda_0.csv --chunk 0 --n-chunks 40

Manifest: CSV with columns  path, N_T, rep  and, for --exp lambda,  a  (selection strength; a = 1 is the main text).
``path`` is either one target file (X1..Xp, Tr, Y, ite) or one of the pipeline's stacked files with a ``rep``
column (e.g. data/sim1/target_n50_biased.csv: 100 replications x 50 rows), from which the rows with
rep == ``rep`` are taken.  Rows of the manifest are split into --n-chunks contiguous chunks.
The source model is trained once per source file and cached at --source-model (trained if missing).
Fine-tuning settings default to the main text (Sim 1: lr 1e-3, no source factual term, kappa = 1); the
target-only CFR and the source model default to the pipeline's settings (Table 9).
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

import tltarnet as tl
import appendix_c as ac


def _load_checkpoint(path_model, in_dim, n_source, device):
    """Accepts a {'cfg', 'state'} checkpoint written here, a full pickled TARNet, or a bare state_dict
    (architecture then taken from the size rule, as in the pipeline)."""
    try:
        ck = torch.load(path_model, map_location=device, weights_only=False)
    except TypeError:                      # torch < 1.13 has no weights_only
        ck = torch.load(path_model, map_location=device)
    if isinstance(ck, tl.TARNet):
        model = ck
    elif isinstance(ck, dict) and "state" in ck:
        model = tl.TARNet(**ck["cfg"])
        model.load_state_dict(ck["state"])
    elif isinstance(ck, dict) and any(k.startswith("encoder.") for k in ck):
        model = tl.TARNet(in_dim, **tl.arch_for_n(n_source))
        model.load_state_dict(ck)
    else:
        raise ValueError(f"unrecognised checkpoint format in {path_model}")
    return model.to(device).eval()


def load_or_train_source(path_csv, path_model, seed, device, src_kwargs):
    df = pd.read_csv(path_csv)
    xs, ts, ys, _ = tl.to_tensors(df, device)
    if path_model and os.path.exists(path_model):
        model = _load_checkpoint(path_model, xs.shape[1], len(xs), device)
    else:
        model, info = tl.train_tarnet(xs, ts, ys, seed=seed, device=device, **src_kwargs)
        print("trained source model:", {k: v for k, v in info.items() if k not in ("train_idx", "val_idx")})
        if path_model:
            os.makedirs(os.path.dirname(path_model) or ".", exist_ok=True)
            torch.save({"cfg": model.cfg, "state": model.state_dict()}, path_model)
    return model, (xs, ts, ys)


class TargetLoader:
    """Reads a manifest row's target; stacked files (with a ``rep`` column) are read once per process."""

    def __init__(self):
        self.cache = {}

    def __call__(self, path, rep):
        if path not in self.cache:
            self.cache[path] = pd.read_csv(path)
        df = self.cache[path]
        if "rep" in df.columns:
            df = df[df["rep"] == rep]
            if len(df) == 0:
                raise ValueError(f"{path}: no rows with rep == {rep}")
        return df.reset_index(drop=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", choices=["blocks", "lambda"], required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--source-model", default=None)
    p.add_argument("--manifest", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--chunk", type=int, default=0)
    p.add_argument("--n-chunks", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--no-probe", action="store_true", help="skip the probe AUC of the representation")
    p.add_argument("--train-source-only", action="store_true",
                   help="train / cache the source model and exit (run once before an array job)")
    # fine-tuning (tl.transfer_finetune), defaults = main text, Simulation 1
    p.add_argument("--n-unfrozen", type=int, default=1)
    p.add_argument("--w-src", type=float, default=0.0, help="weight of the source factual term (main text: 0)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--min-epochs", type=int, default=20)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--main-kappa", type=float, default=1.0, help="kappa of the main-text objective (hybrids are taken from it)")
    # target-only CFR (tl.train_tarnet)
    p.add_argument("--to-alpha", type=float, default=1.0)
    p.add_argument("--to-lr", type=float, default=5e-3)
    p.add_argument("--to-batch", type=int, default=32)
    p.add_argument("--to-val-frac", type=float, default=0.2)
    p.add_argument("--to-patience", type=int, default=20)
    p.add_argument("--to-min-epochs", type=int, default=50)
    p.add_argument("--to-epochs", type=int, default=400)
    # source model, if it has to be trained here (tl.train_tarnet)
    p.add_argument("--src-alpha", type=float, default=1.0)
    p.add_argument("--src-lr", type=float, default=5e-3)
    p.add_argument("--src-batch", type=int, default=64)
    p.add_argument("--src-val-frac", type=float, default=0.2)
    p.add_argument("--src-patience", type=int, default=20)
    p.add_argument("--src-min-epochs", type=int, default=50)
    p.add_argument("--src-epochs", type=int, default=400)
    args = p.parse_args()

    torch.set_num_threads(1)
    ft_kwargs = dict(w_src=args.w_src, lr=args.lr, batch=args.batch, val_frac=args.val_frac,
                     patience=args.patience, min_epochs=args.min_epochs, epochs=args.epochs)
    to_kwargs = dict(alpha=args.to_alpha, lr=args.to_lr, batch=args.to_batch, val_frac=args.to_val_frac,
                     patience=args.to_patience, min_epochs=args.to_min_epochs, epochs=args.to_epochs)
    src_kwargs = dict(alpha=args.src_alpha, lr=args.src_lr, batch=args.src_batch, val_frac=args.src_val_frac,
                      patience=args.src_patience, min_epochs=args.src_min_epochs, epochs=args.src_epochs)

    source_model, (xs, ts, ys) = load_or_train_source(args.source, args.source_model, args.seed, args.device, src_kwargs)
    if args.train_source_only:
        print("source model ready:", args.source_model)
        return

    if not args.manifest or not args.out:
        p.error("--manifest and --out are required unless --train-source-only is given")
    manifest = pd.read_csv(args.manifest)
    if "a" not in manifest.columns:
        manifest["a"] = 1.0
    chunks = np.array_split(np.arange(len(manifest)), args.n_chunks)
    load_target = TargetLoader()
    rows_out = []
    for i in chunks[args.chunk]:
        m = manifest.iloc[i]
        df = load_target(m["path"], int(m["rep"]))
        xt, tt, yt, ite = tl.to_tensors(df, args.device)
        if len(xt) != int(m["N_T"]):
            print(f"warning: {m['path']} rep {m['rep']} has {len(xt)} rows, manifest says N_T={m['N_T']}", flush=True)
        meta = dict(path=m["path"], N_T=int(m["N_T"]), rep=int(m["rep"]), a=float(m["a"]),
                    w_src=args.w_src, lr=args.lr, main_kappa=args.main_kappa)
        seed = args.seed + 1000 * int(m["rep"]) + int(m["N_T"])     # same seed for every a of one (rep, N_T)
        rows_out += ac.run_target(source_model, xt, tt, yt, ite, xs, ts, ys, exp=args.exp, meta=meta,
                                  ft_kwargs=ft_kwargs, to_kwargs=to_kwargs, n_unfrozen=args.n_unfrozen,
                                  main_kappa=args.main_kappa, probe=not args.no_probe, seed=seed,
                                  device=args.device)
        print(f"done {m['path']} (N_T={m['N_T']}, a={m['a']}, rep={m['rep']})", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pd.DataFrame(rows_out).to_csv(args.out, index=False)
    print("wrote", args.out, len(rows_out), "rows")


if __name__ == "__main__":
    main()
