
import sys

path = "run.py"
src = open(path).read()
if "n_inits" in src:
    sys.exit("run.py already patched -- nothing to do")

old_cfg = '''    # pooled baseline: CFR on source + target
    pool=dict(epochs=300, lr=5e-3, batch=64, alpha=1.0, val_frac=0.2, patience=20, min_epochs=50),'''
new_cfg = '''    # pooled baseline: CFR on source + target.  n_inits > 1: train from several
    # random initialisations (same held-out split) and keep the fit with the
    # lowest held-out factual loss -- from-scratch fits on the random outcome
    # functions of Simulation 3 collapse to a near-constant effect prediction in
    # about a quarter of the attempts (about 1% with the parametric functions).
    pool=dict(epochs=300, lr=5e-3, batch=64, alpha=1.0, val_frac=0.2, patience=20, min_epochs=50,
              n_inits=3),'''

old_fit = '''        pm, infop = M.train_tarnet(xp, tp, yp, epochs=c["epochs"], lr=c["lr"], batch=c["batch"],
                                   alpha=c["alpha"], val_frac=c["val_frac"], patience=c["patience"],
                                   min_epochs=c["min_epochs"], seed=seed, device=DEVICE)
        base.update(_model_metrics("pool", pm, xt, it, val_idx))
        base.update(stop_epoch_pool=infop["stop_epoch"], pool_epochs_max=c["epochs"],
                    pool_min_epochs=c["min_epochs"])'''
new_fit = '''        # one held-out split shared by all initialisations, so that their
        # validation losses are comparable
        split_p = M.split_indices(tp, c["val_frac"], seed)
        n_inits = max(int(c.get("n_inits", 1)), 1)
        pm, infop, pick = None, None, 0
        for k in range(n_inits):
            m_k, info_k = M.train_tarnet(xp, tp, yp, epochs=c["epochs"], lr=c["lr"], batch=c["batch"],
                                         alpha=c["alpha"], val_frac=c["val_frac"], patience=c["patience"],
                                         min_epochs=c["min_epochs"], split=split_p, seed=seed + 1000 * k,
                                         device=DEVICE)
            if infop is None or info_k["val_loss_best"] < infop["val_loss_best"]:
                pm, infop, pick = m_k, info_k, k
        base.update(_model_metrics("pool", pm, xt, it, val_idx))
        base.update(stop_epoch_pool=infop["stop_epoch"], val_loss_pool=infop["val_loss_best"],
                    pool_init=pick, pool_n_inits=n_inits, pool_epochs_max=c["epochs"],
                    pool_min_epochs=c["min_epochs"])'''

for old, what in [(old_cfg, "pool config"), (old_fit, "pooled fit block")]:
    if src.count(old) != 1:
        sys.exit(f"could not find the {what} in run.py (found {src.count(old)} matches) -- nothing changed")

src = src.replace(old_cfg, new_cfg).replace(old_fit, new_fit)
open(path, "w").write(src)
print("run.py patched: pooled baseline now uses n_inits=3 (best held-out loss kept)")
