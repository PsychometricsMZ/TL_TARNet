"""Appendix C: where the target's selection enters the network, and how it scales with the selection.

Imports the unchanged tltarnet.py; the two additions it needs (``set_trainable`` and
``transfer_finetune_block``, a copy of ``tl.transfer_finetune`` whose only change is
which blocks are trainable) live here, so nothing in the pipeline is modified.
Two experiments, both on the design of Simulation 1 with N_S = 5000 and N_T in {50, 500}:

  C.2  blocks : on the non-randomized targets of the main text (a = 1)
               source-only | heads | encoder x kappa in {0, 1, 10} | both x kappa in {0, 1, 10}
               + the two recombined models (Phi_T, h_S) and (Phi_S, h_T) from "both, kappa = 1"
               + target-only CFR
  C.3  lambda : on targets whose selection strength a varies (manifest column ``a``;
               Eq. 24: P(T = 1 | x, eps) = expit(a * (mu0(x) + eps) / SD(Y0)))
               source-only | heads only | encoder only (kappa = 1) | TL-TARNet (both, kappa = 1)
               + the two recombined models from TL-TARNet | target-only CFR
               -> bias, error and the block attribution of the re-imported bias as a function of a
                  (Lambda_q, b_X(a) and the naive bias for each a come from msm_strength.py)

``kappa`` sets both IPM weights of the fine-tuning objective, alpha = alpha_cross = kappa.
Every row of the output carries bias and PEHE on ALL target units against the target's
own ITE, the within-target MMD^2 of the representation, the probe AUC of the
representation for T, the relative movement of the two trainable blocks and the
stopping epoch.  ``summarize_appendix_c.py`` turns the rows into Table C.2 / Table C.3 / Figure C.1.

The module also holds the data-generating process of Simulation 1 (Eq. 16) with the
selection of Eq. 24, used by ``make_lambda_targets.py`` to draw the targets of C.3 and
by ``check_dgp.py`` to verify that the existing non-randomized targets are Eq. 24 at a = 1.
"""
import copy
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import tltarnet as tl


# --------------------------------------------------------------------------- trainable blocks

# Which blocks are trainable during fine-tuning (last n_unfrozen linear layers of each):
#   'both'    = tl.freeze_but_last (the main text): encoder AND each head
#   'encoder' = encoder only (heads frozen)
#   'heads'   = each head only (encoder frozen -> the IPM terms get no gradient)
#   'none'    = nothing (the frozen source model)
BLOCKS = {"both": ("encoder", "h0", "h1"), "encoder": ("encoder",), "heads": ("h0", "h1"), "none": ()}


def set_trainable(model, block="both", n_unfrozen=1):
    if block not in BLOCKS:
        raise ValueError(f"block must be one of {list(BLOCKS)}, got {block!r}")
    for p in model.parameters():
        p.requires_grad_(False)
    for name in BLOCKS[block]:
        for lin in tl._last_linears(getattr(model, name), n_unfrozen):
            for p in lin.parameters():
                p.requires_grad_(True)
    return model


def transfer_finetune_block(source_model, xt, tt, yt, xs, ts, ys, n_unfrozen=1,
                            alpha=1.0, alpha_cross=1.0, w_src=1.0, epochs=300, lr=1e-4,
                            batch=32, val_frac=0.0, patience=20, min_epochs=20,
                            refit_full=False, split=None, seed=0, device="cpu", block="both"):
    """Copy of tl.transfer_finetune with one difference: ``block`` selects the trainable blocks
    ('both' reproduces tl.transfer_finetune exactly; 'none' returns an untrained copy of the source).
    Same objective, optimiser, early stopping and info dict as the original."""
    xt, tt, yt = xt.to(device), tt.to(device), yt.to(device)
    xs, ts, ys = xs.to(device), ts.to(device), ys.to(device)
    train_idx, val_idx = split if split is not None else tl.split_indices(tt, val_frac, seed)
    use_val = len(val_idx) > 0

    def _fit(n_epochs, tr_idx, early):
        torch.manual_seed(seed)
        gen = torch.Generator().manual_seed(seed)
        model = copy.deepcopy(source_model).to(device)
        set_trainable(model, block, n_unfrozen)                                   # <- the only change
        params = [p for p in model.parameters() if p.requires_grad]
        if not params:                                                            # block == 'none'
            return model, 0, float("nan")
        opt = torch.optim.Adam(params, lr=lr)
        stopper = tl.EarlyStopper(patience, min_epochs) if early else None
        last = n_epochs
        n_src_b = min(batch, len(xs))
        for epoch in range(1, n_epochs + 1):
            model.train()
            perm = tr_idx[torch.randperm(len(tr_idx), generator=gen).to(tr_idx.device)]
            for i in range(0, len(perm), batch):
                idx = perm[i:i + batch]
                xb, tb, yb = xt[idx], tt[idx], yt[idx]
                sidx = torch.randint(0, len(xs), (n_src_b,), generator=gen).to(xs.device)
                xsb, tsb, ysb = xs[sidx], ts[sidx], ys[sidx]

                loss = tl.factual_loss_weighted(model, xb, tb, yb)                          # target factual
                if w_src > 0:
                    loss = loss + w_src * tl.factual_loss_weighted(model, xsb, tsb, ysb)   # source factual
                if alpha > 0 or alpha_cross > 0:
                    phi_t, phi_s = model.repr(xb), model.repr(xsb)
                    if alpha > 0 and tb.gt(0.5).any() and tb.le(0.5).any():                 # within-target t1 vs t0
                        loss = loss + alpha * tl.mmd_rbf(phi_t[tb > 0.5], phi_t[tb <= 0.5])
                    if alpha_cross > 0:                                                     # cross-domain, per arm
                        for tv in (0, 1):
                            tm = (tb > 0.5) if tv else (tb <= 0.5)
                            sm = (tsb > 0.5) if tv else (tsb <= 0.5)
                            if tm.any() and sm.any():
                                loss = loss + alpha_cross * tl.mmd_rbf(phi_t[tm], phi_s[sm])
                opt.zero_grad()
                loss.backward()
                opt.step()
            if stopper is not None:
                model.eval()
                with torch.no_grad():
                    vl = tl.factual_mse(model, xt[val_idx], tt[val_idx], yt[val_idx]).item()
                if stopper.step(epoch, vl, model):
                    last = epoch
                    break
        if stopper is not None:
            stopper.restore(model)
            return model, stopper.best_epoch, stopper.best
        return model, last, float("nan")

    src_before = tl.source_factual_loss(source_model, xs, ts, ys, seed=seed)
    mmd_before = tl.within_target_mmd(source_model, xt, tt)
    model, stop_epoch, val_best = _fit(epochs, train_idx, use_val)
    if use_val and refit_full:
        model, stop_epoch, _ = _fit(max(stop_epoch, 1), torch.arange(len(xt), device=device), False)
    model.eval()
    info = dict(stop_epoch=int(stop_epoch), val_loss_best=float(val_best),
                src_floss_before=src_before,
                src_floss_after=tl.source_factual_loss(model, xs, ts, ys, seed=seed),
                tgt_mmd_before=mmd_before, tgt_mmd_after=tl.within_target_mmd(model, xt, tt),
                n_train=int(len(train_idx)), n_val=int(len(val_idx)),
                train_idx=train_idx.cpu().numpy(), val_idx=val_idx.cpu().numpy())
    return model, info


# --------------------------------------------------------------------------- helpers

def hybrid_model(enc_from, heads_from):
    """Encoder of ``enc_from`` with the two heads of ``heads_from`` (no training)."""
    m = copy.deepcopy(enc_from)
    m.h0.load_state_dict(heads_from.h0.state_dict())
    m.h1.load_state_dict(heads_from.h1.state_dict())
    m.eval()
    return m


def block_movement(model, source_model, block, n_unfrozen=1):
    """Relative movement ||W_T - W_S|| / ||W_S|| over the linear layers that ``block`` trains."""
    num = den = 0.0
    for name in BLOCKS[block]:
        lins_new = tl._last_linears(getattr(model, name), n_unfrozen)
        lins_old = tl._last_linears(getattr(source_model, name), n_unfrozen)
        for ln, lo in zip(lins_new, lins_old):
            for pn, po in zip(ln.parameters(), lo.parameters()):
                num += float(((pn.detach() - po.detach()) ** 2).sum())
                den += float((po.detach() ** 2).sum())
    return math.sqrt(num / den) if den > 0 else float("nan")


def covariate_mmd(x, t):
    """Within-target MMD^2 on the raw covariates (the imbalance present in the data)."""
    with torch.no_grad():
        return tl.mmd_rbf(x[t > 0.5], x[t <= 0.5]).item()


def shapley_attribution(b_SS, b_TS, b_ST, b_TT):
    """Shapley values of the re-imported bias b(Phi_T,h_T) - b(Phi_S,h_S) for the two blocks.

    b_SS = source-only, b_TS = (Phi_T, h_S), b_ST = (Phi_S, h_T), b_TT = fine-tuned model.
    Returns (phi_representation, phi_heads); they sum to b_TT - b_SS.
    """
    phi_rep = 0.5 * ((b_TS - b_SS) + (b_TT - b_ST))
    phi_heads = 0.5 * ((b_ST - b_SS) + (b_TT - b_TS))
    return phi_rep, phi_heads


def _metrics(model, x, t, ite, source_model=None, block=None, n_unfrozen=1, probe=True, device="cpu"):
    pehe = tl.pehe(model, x, ite, device=device)
    row = dict(bias=tl.mean_bias(model, x, ite, device=device), pehe=pehe,
               pehe_norm=pehe / float(np.var(ite)),
               mmd_phi=tl.within_target_mmd(model, x, t),
               probe_auc_phi=tl.probe_auc(model, x, (t > 0.5).cpu().numpy().astype(int), device=device) if probe else float("nan"))
    if source_model is not None and block not in (None, "none"):
        trained = BLOCKS.get(block, ())          # () for the target-only model (block = 'all'): no movement defined
        row["move_enc"] = block_movement(model, source_model, "encoder", n_unfrozen) if "encoder" in trained else float("nan")
        row["move_heads"] = block_movement(model, source_model, "heads", n_unfrozen) if "h0" in trained else float("nan")
    return row


# --------------------------------------------------------------------------- experiments

# (block, kappa) fine-tuning variants of each experiment
BLOCK_CONFIGS = [("heads", 0.0), ("encoder", 0.0), ("encoder", 1.0), ("encoder", 10.0),
                 ("both", 0.0), ("both", 1.0), ("both", 10.0)]
# C.3: the main-text model (both blocks, kappa = 1), and each block alone so that the robustness of
# the representation (IPM terms active) and of the heads (factual loss only) can be compared along a
LAMBDA_CONFIGS = [("both", 1.0), ("heads", 0.0), ("encoder", 1.0)]

VARIANT_NAME = {"both": "TL-TARNet", "encoder": "encoder only", "heads": "heads only"}


def run_target(source_model, xt, tt, yt, ite, xs, ts, ys, *, exp, meta, ft_kwargs, to_kwargs,
               n_unfrozen=1, main_kappa=1.0, probe=True, seed=0, device="cpu"):
    """All estimators of one experiment on one target.  Returns a list of result rows.

    ft_kwargs: keyword arguments of transfer_finetune_block except alpha / alpha_cross / block
               (w_src, lr, batch, val_frac, patience, min_epochs, epochs, refit_full)
    to_kwargs: keyword arguments of tl.train_tarnet for the target-only CFR
               (alpha, lr, batch, val_frac, patience, min_epochs, epochs, weight_decay)
    In both experiments the two recombined models (Phi_T, h_S) and (Phi_S, h_T) are taken from the
    'both' variant with kappa = main_kappa, so the Shapley attribution of Eqs. 27-28 is available per target.
    """
    rows = []
    base = dict(meta, n_unfrozen=n_unfrozen, p_treated=float(tt.mean()), mmd_X=covariate_mmd(xt, tt))

    def add(variant, block, kappa, model, info=None, extra=None):
        r = dict(base, variant=variant, block=block, kappa=kappa,
                 stop_epoch=(info or {}).get("stop_epoch", 0),
                 src_floss_after=(info or {}).get("src_floss_after", float("nan")))
        r.update(_metrics(model, xt, tt, ite, source_model, block, n_unfrozen, probe, device))
        if extra:
            r.update(extra)
        rows.append(r)

    # frozen source model
    add("source-only", "none", float("nan"), source_model)

    configs = BLOCK_CONFIGS if exp == "blocks" else LAMBDA_CONFIGS
    for block, kappa in configs:
        model, info = transfer_finetune_block(source_model, xt, tt, yt, xs, ts, ys, n_unfrozen=n_unfrozen,
                                           alpha=kappa, alpha_cross=kappa, block=block, seed=seed,
                                           device=device, **ft_kwargs)
        add(VARIANT_NAME[block], block, kappa, model, info)
        if block == "both" and kappa == main_kappa:
            add("hybrid PhiT hS", "both", kappa, hybrid_model(model, source_model))
            add("hybrid PhiS hT", "both", kappa, hybrid_model(source_model, model))

    # target-only CFR, trained from scratch with the size rule
    model, info = tl.train_tarnet(xt, tt, yt, seed=seed, device=device, **to_kwargs)
    add("target-only CFR", "all", to_kwargs.get("alpha", 1.0), model, info)
    return rows


# --------------------------------------------------------------------------- the DGP of Simulation 1 (C.3 targets)

# Eq. 16 with alpha = 0, beta = 1, gamma = omega = (0.5, ..., 0.5), lambda = 1, sigma^2 = 1:
#   mu0(x) = 0.5 * sum(x) + g(x),       g(x) = sin(x1) + x2 * x3 + x4^2        (prognostic part)
#   tau(x) = 1 + 0.5 * sum(x) + q(x),   q(x) = x1 * x2 + cos(x3)               (the ITE; the file column ``ite``)
#   Y0 = mu0(X) + eps0,  Y1 = mu0(X) + tau(X) + eps1,  eps0, eps1 ~ N(0, 1) independent
# Selection:  P(T = 1 | x, eps0) = expit(a * (Y0 - m) / s),  m = E[Y0] = 1,  s = SD(Y0) = 2.508,
# i.e. a times the standardized control outcome; a = 0 randomized, a = 1 the main text (P(T = 1) = 1/2 at every a).
# center = 0 gives the uncentered form expit(a * Y0 / s) (P(T = 1) = 0.58 at a = 1).
P_COV = 5
VAR_MU0 = 1.25 + (1.0 - math.exp(-2.0)) / 2.0 + 1.0 + 2.0 + math.exp(-0.5)   # = 5.289 (the "5.28" of C.1)
SD_Y0 = math.sqrt(VAR_MU0 + 1.0)                                              # = 2.508 (the "s" of Eq. 24)
E_Y0 = 1.0                                                                    # E[mu0(X)] = E[X4^2] = 1


def mu0_fn(X):
    X = np.asarray(X, float)
    return 0.5 * X.sum(1) + np.sin(X[:, 0]) + X[:, 1] * X[:, 2] + X[:, 3] ** 2


def tau_fn(X):
    X = np.asarray(X, float)
    return 1.0 + 0.5 * X.sum(1) + X[:, 0] * X[:, 1] + np.cos(X[:, 2])


def expit(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, float)))


def draw_units(n, rng):
    """Covariates, potential outcomes and a selection uniform for n units of Simulation 1.
    Returns a dict with X (n x 5), Y0, Y1, ite and U (the uniform used to assign treatment)."""
    X = rng.standard_normal((n, P_COV))
    m0, tau = mu0_fn(X), tau_fn(X)
    Y0 = m0 + rng.standard_normal(n)
    Y1 = m0 + tau + rng.standard_normal(n)
    return dict(X=X, Y0=Y0, Y1=Y1, ite=tau, U=rng.random(n))


def assign_treatment(units, a, sd_y0=SD_Y0, center=E_Y0):
    """Selection with slope ``a`` on the standardized control outcome, expit(a * (Y0 - center) / sd_y0);
    the same uniform U for every a, so that for one unit the treatment is monotone in a (only the
    selection differs between the targets of a grid).  center = E_Y0 is the main text's rule,
    center = 0 the uncentered form.  Returns a data frame with the columns X1..X5, Tr, Y, ite."""
    p = expit(a * (units["Y0"] - center) / sd_y0)
    t = (units["U"] < p).astype(float)
    y = np.where(t > 0.5, units["Y1"], units["Y0"])
    df = pd.DataFrame(units["X"], columns=[f"X{j + 1}" for j in range(P_COV)])
    df["Tr"], df["Y"], df["ite"] = t, y, units["ite"]
    return df
