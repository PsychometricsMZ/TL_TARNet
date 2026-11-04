"""Patch tltarnet.py and run.py in place: add a T-learner base model (two
independent networks, no shared encoder) so that Kunzel et al. (2018), Alg. 8
(warm start on a T-learner) can be run with the existing pipeline.

Run from the project directory:   python patch_tlearner.py
Exits with a message (and changes nothing) if a file does not match the expected
text.  Afterwards:

    python run.py --experiment sim2 --config configs/tlearner.json sources      # T-learner sources
    python run.py --experiment sim2 --config configs/tlearner.json chunk ... \\
        --variants configs/variants_kunzel.json --estimators tl --tag warm_tlearner

What changes
* tltarnet.py: class TLearner (each arm = the TARNet encoder layers followed by
  the TARNet head layers, nothing shared), build_model(), and a ``kind``
  argument of train_tarnet.
* run.py: config key ``model`` ("tarnet" | "tlearner"), checkpoint names
  <source_tag>__tlearner__seed<s>.pt and sources_tlearner.csv for the new kind
  (TARNet checkpoints and sources.csv are untouched), load_model dispatch, a
  ``model_kind`` output column, and ``kind`` passed to every from-scratch fit.
"""
import sys

edits = {"tltarnet.py": [], "run.py": []}

# --------------------------------------------------------------------------- tltarnet.py
edits["tltarnet.py"].append(('''# ---------------------------------------------------------------------------
# IPM: RBF-MMD^2 with the median-distance bandwidth heuristic''',
'''class TLearner(TARNet):
    """T-learner: two independent networks, nothing shared (Kunzel et al., 2018,
    Alg. 8).  Each arm is the full TARNet path -- the encoder layers followed by
    the head layers -- so per-arm depth and width equal one TARNet path; the
    parameter count is about 1.6x TARNet because the encoder is duplicated.
    ``encoder`` is an empty Sequential (identity), so repr(x) = x."""

    def __init__(self, in_dim, rep_dim=16, rep_layers=3, head_dim=8, head_layers=2):
        nn.Module.__init__(self)
        self.cfg = dict(in_dim=in_dim, rep_dim=rep_dim, rep_layers=rep_layers,
                        head_dim=head_dim, head_layers=head_layers, kind="tlearner")
        self.encoder = nn.Sequential()

        def make_arm():
            layers, d = [], in_dim
            for _ in range(rep_layers):
                layers += [nn.Linear(d, rep_dim), nn.ReLU()]
                d = rep_dim
            for _ in range(head_layers):
                layers += [nn.Linear(d, head_dim), nn.ReLU()]
                d = head_dim
            layers += [nn.Linear(d, 1)]
            return nn.Sequential(*layers)

        self.h0, self.h1 = make_arm(), make_arm()


MODEL_KINDS = {"tarnet": TARNet, "tlearner": TLearner}


def build_model(in_dim, kind="tarnet", **arch):
    """TARNet (shared encoder + two heads) or TLearner (two separate networks)."""
    arch = {k: v for k, v in arch.items() if k != "kind"}
    return MODEL_KINDS[kind](in_dim, **arch)


# ---------------------------------------------------------------------------
# IPM: RBF-MMD^2 with the median-distance bandwidth heuristic'''))

edits["tltarnet.py"].append(('''def train_tarnet(x, t, y, arch=None, alpha=1.0, epochs=400, lr=5e-3, batch=64,
                 weight_decay=1e-4, val_frac=0.0, patience=20, min_epochs=20,
                 refit_full=False, split=None, seed=0, device="cpu"):''',
'''def train_tarnet(x, t, y, arch=None, alpha=1.0, epochs=400, lr=5e-3, batch=64,
                 weight_decay=1e-4, val_frac=0.0, patience=20, min_epochs=20,
                 refit_full=False, split=None, seed=0, device="cpu", kind="tarnet"):'''))

edits["tltarnet.py"].append(('''        model = TARNet(x.shape[1], **arch).to(device)''',
'''        model = build_model(x.shape[1], kind=kind, **arch).to(device)'''))

# --------------------------------------------------------------------------- run.py
edits["run.py"].append(('''    cita_nsample=256,''',
'''    model="tarnet",        # base learner of every fit: "tarnet" | "tlearner" (Kunzel Alg. 8)
    cita_nsample=256,'''))

edits["run.py"].append(('''def load_model(path):
    blob = torch.load(path, map_location=DEVICE)
    m = M.TARNet(**blob["cfg"]).to(DEVICE)
    m.load_state_dict(blob["state"])
    m.eval()
    return m''',
'''def load_model(path):
    blob = torch.load(path, map_location=DEVICE)
    cfg = dict(blob["cfg"])
    kind = cfg.pop("kind", "tarnet")
    m = M.build_model(kind=kind, **cfg).to(DEVICE)
    m.load_state_dict(blob["state"])
    m.eval()
    return m


def ckpt_path(ckpt_dir, src_tag, seed, kind="tarnet"):
    """TARNet sources keep their old names; other kinds get a ``__<kind>`` infix."""
    infix = "" if kind == "tarnet" else f"__{kind}"
    return os.path.join(ckpt_dir, f"{src_tag}{infix}__seed{seed}.pt")'''))

edits["run.py"].append(('''            ckpt = os.path.join(ckpt_dir, f"{src_tag}__seed{seed}.pt")''',
'''            ckpt = ckpt_path(ckpt_dir, src_tag, seed, cfg["model"])'''))

edits["run.py"].append(('''                model, info = M.train_tarnet(xs, ts, ys, epochs=c["epochs"], lr=c["lr"], batch=c["batch"],
                                             alpha=c["alpha"], val_frac=c["val_frac"], patience=c["patience"],
                                             min_epochs=c["min_epochs"], seed=eff_seed, device=DEVICE)''',
'''                model, info = M.train_tarnet(xs, ts, ys, epochs=c["epochs"], lr=c["lr"], batch=c["batch"],
                                             alpha=c["alpha"], val_frac=c["val_frac"], patience=c["patience"],
                                             min_epochs=c["min_epochs"], seed=eff_seed, device=DEVICE,
                                             kind=cfg["model"])'''))

edits["run.py"].append(('''                       n_params=model.n_params(), alpha=c["alpha"], failed=bool(failed))''',
'''                       n_params=model.n_params(), alpha=c["alpha"], failed=bool(failed),
                       kind=cfg["model"])'''))

edits["run.py"].append(('''        out = os.path.join(ckpt_dir, "sources.csv")''',
'''        out = os.path.join(ckpt_dir, "sources.csv" if cfg["model"] == "tarnet" else f"sources_{cfg['model']}.csv")'''))

edits["run.py"].append(('''    base = dict(true_ite_sd=float(it.std()), true_ite_mean=float(it.mean()),
                n_train=int(len(train_idx)), n_val=int(len(val_idx)))''',
'''    base = dict(true_ite_sd=float(it.std()), true_ite_mean=float(it.mean()),
                n_train=int(len(train_idx)), n_val=int(len(val_idx)), model_kind=cfg["model"])'''))

edits["run.py"].append(('''        tg, info = M.train_tarnet(xt, tt, yt, epochs=c["epochs"], lr=c["lr"], batch=c["batch"],
                                  alpha=c["alpha"], patience=c["patience"], min_epochs=c["min_epochs"],
                                  refit_full=split_cfg["refit_full"], split=split, seed=seed, device=DEVICE)''',
'''        tg, info = M.train_tarnet(xt, tt, yt, epochs=c["epochs"], lr=c["lr"], batch=c["batch"],
                                  alpha=c["alpha"], patience=c["patience"], min_epochs=c["min_epochs"],
                                  refit_full=split_cfg["refit_full"], split=split, seed=seed, device=DEVICE,
                                  kind=cfg["model"])'''))

edits["run.py"].append(('''        tg0, info0 = M.train_tarnet(xt, tt, yt, epochs=c["epochs"], lr=c["lr"], batch=c["batch"],
                                    alpha=0.0, patience=c["patience"], min_epochs=c["min_epochs"],
                                    refit_full=split_cfg["refit_full"], split=split, seed=seed, device=DEVICE)''',
'''        tg0, info0 = M.train_tarnet(xt, tt, yt, epochs=c["epochs"], lr=c["lr"], batch=c["batch"],
                                    alpha=0.0, patience=c["patience"], min_epochs=c["min_epochs"],
                                    refit_full=split_cfg["refit_full"], split=split, seed=seed, device=DEVICE,
                                    kind=cfg["model"])'''))

edits["run.py"].append(('''            m_k, info_k = M.train_tarnet(xp, tp, yp, epochs=c["epochs"], lr=c["lr"], batch=c["batch"],
                                         alpha=c["alpha"], val_frac=c["val_frac"], patience=c["patience"],
                                         min_epochs=c["min_epochs"], split=split_p, seed=seed + 1000 * k,
                                         device=DEVICE)''',
'''            m_k, info_k = M.train_tarnet(xp, tp, yp, epochs=c["epochs"], lr=c["lr"], batch=c["batch"],
                                         alpha=c["alpha"], val_frac=c["val_frac"], patience=c["patience"],
                                         min_epochs=c["min_epochs"], split=split_p, seed=seed + 1000 * k,
                                         device=DEVICE, kind=cfg["model"])'''))

# the load_model(...) calls in cmd_chunk (12 spaces) and cmd_pilot (8 spaces);
# the leading newline keeps the two patterns from matching each other
edits["run.py"].append(('''\n            sm = load_model(os.path.join(ckpt_dir, f"{r['source_tag']}__seed{args.source_seed}.pt"))''',
'''\n            sm = load_model(ckpt_path(ckpt_dir, r["source_tag"], args.source_seed, cfg["model"]))'''))
edits["run.py"].append(('''\n        sm = load_model(os.path.join(ckpt_dir, f"{r['source_tag']}__seed{args.source_seed}.pt"))''',
'''\n        sm = load_model(ckpt_path(ckpt_dir, r["source_tag"], args.source_seed, cfg["model"]))'''))

# --------------------------------------------------------------------------- apply
for path, reps in edits.items():
    src = open(path).read()
    if "tlearner" in src:
        sys.exit(f"{path} already patched -- nothing to do")
    for old, new in reps:
        n = src.count(old)
        if n != 1:
            sys.exit(f"{path}: expected exactly one match for\n{old[:80]!r}...\nfound {n}; file left unchanged")
        src = src.replace(old, new)
    open(path, "w").write(src)
    print(f"patched {path}: {len(reps)} edits")
