
import copy
import math
import re

import numpy as np
import torch
import torch.nn as nn

X_COL = re.compile(r"^X\d+$")


# ---------------------------------------------------------------------------
# architecture, scaled to dataset size (paper: ~N/10 trainable parameters)
# NOTE: the architecture is held fixed for n >= 5000.
# ---------------------------------------------------------------------------
def arch_for_n(n):
    if n >= 5000:
        return dict(rep_dim=16, rep_layers=3, head_dim=8, head_layers=2)
    if n >= 1000:
        return dict(rep_dim=8, rep_layers=3, head_dim=8, head_layers=2)
    if n >= 250:
        return dict(rep_dim=8, rep_layers=2, head_dim=4, head_layers=1)
    if n >= 100:
        return dict(rep_dim=4, rep_layers=2, head_dim=4, head_layers=1)
    return dict(rep_dim=4, rep_layers=1, head_dim=2, head_layers=1)  # n = 50


class TARNet(nn.Module):
    """Shared encoder Phi + two outcome heads h0, h1 (Shalit et al., 2017)."""

    def __init__(self, in_dim, rep_dim=16, rep_layers=3, head_dim=8, head_layers=2):
        super().__init__()
        self.cfg = dict(in_dim=in_dim, rep_dim=rep_dim, rep_layers=rep_layers,
                        head_dim=head_dim, head_layers=head_layers)
        enc, d = [], in_dim
        for _ in range(rep_layers):
            enc += [nn.Linear(d, rep_dim), nn.ReLU()]
            d = rep_dim
        self.encoder = nn.Sequential(*enc)

        def make_head():
            layers, d = [], rep_dim
            for _ in range(head_layers):
                layers += [nn.Linear(d, head_dim), nn.ReLU()]
                d = head_dim
            layers += [nn.Linear(d, 1)]
            return nn.Sequential(*layers)

        self.h0, self.h1 = make_head(), make_head()

    def repr(self, x):
        return self.encoder(x)

    def forward(self, x):
        """Returns (y0_hat, y1_hat), each of shape (n,)."""
        phi = self.encoder(x)
        return self.h0(phi).squeeze(-1), self.h1(phi).squeeze(-1)

    def predict_ite(self, x):
        y0, y1 = self.forward(x)
        return y1 - y0

    def n_params(self, trainable_only=False):
        return sum(p.numel() for p in self.parameters() if (p.requires_grad or not trainable_only))


class TLearner(TARNet):
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
# IPM: RBF-MMD^2 with the median-distance bandwidth heuristic
# ---------------------------------------------------------------------------
def _pdist2(a, b):
    a2 = (a * a).sum(1, keepdim=True)
    b2 = (b * b).sum(1, keepdim=True)
    return (a2 + b2.T - 2.0 * a @ b.T).clamp_min(0.0)


def mmd_rbf(X, Y):
    """Biased RBF-MMD^2 estimate; 0 if either sample has fewer than 2 points."""
    if len(X) < 2 or len(Y) < 2:
        return X.new_tensor(0.0)
    with torch.no_grad():
        med = _pdist2(torch.cat([X, Y]), torch.cat([X, Y])).median().clamp_min(1e-6)
    sig2 = med
    Kxx = torch.exp(-_pdist2(X, X) / (2 * sig2))
    Kyy = torch.exp(-_pdist2(Y, Y) / (2 * sig2))
    Kxy = torch.exp(-_pdist2(X, Y) / (2 * sig2))
    return Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean()


# ---------------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------------
def factual_loss_weighted(model, x, t, y):
    """Sample-weighted factual MSE (Eq. 6 of the paper); used for TRAINING."""
    y0, y1 = model(x)
    u = t.mean().clamp(0.05, 0.95)
    w = t / (2 * u) + (1 - t) / (2 * (1 - u))
    pred = torch.where(t > 0.5, y1, y0)
    return (w * (pred - y) ** 2).mean()


def factual_mse(model, x, t, y):
    """Plain factual MSE; used for VALIDATION, the Fisher and the source-loss checks."""
    y0, y1 = model(x)
    pred = torch.where(t > 0.5, y1, y0)
    return ((pred - y) ** 2).mean()


# ---------------------------------------------------------------------------
# held-out split and early stopping
# ---------------------------------------------------------------------------
def split_indices(t, val_frac, seed=0):
    """Stratified (by treatment) train / validation split.

    Returns (train_idx, val_idx) as LongTensors on t's device.  ``val_frac = 0``
    gives all indices in train and an empty validation set.
    """
    n = len(t)
    if val_frac <= 0.0:
        return torch.arange(n, device=t.device), torch.empty(0, dtype=torch.long, device=t.device)
    rng = np.random.default_rng(seed)
    t_np = t.detach().cpu().numpy()
    val = []
    for arm in (0, 1):
        idx = np.where((t_np > 0.5) == bool(arm))[0]
        rng.shuffle(idx)
        n_val = int(round(val_frac * len(idx)))
        val.extend(idx[:n_val].tolist())
    val = np.array(sorted(val), dtype=np.int64)
    train = np.setdiff1d(np.arange(n), val)
    return (torch.as_tensor(train, device=t.device),
            torch.as_tensor(val, device=t.device))


class EarlyStopper:
    """Keeps the best weights (lowest validation loss); stops after ``patience``
    epochs without improvement, but never before ``min_epochs``."""

    def __init__(self, patience=20, min_epochs=20, tol=1e-6):
        self.patience, self.min_epochs, self.tol = patience, min_epochs, tol
        self.best, self.best_epoch, self.best_state, self.bad = float("inf"), 0, None, 0

    def step(self, epoch, val_loss, model):
        """``epoch`` is 1-based.  Returns True when training should stop."""
        if val_loss < self.best - self.tol:
            self.best, self.best_epoch, self.bad = val_loss, epoch, 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.bad += 1
        return self.bad >= self.patience and epoch >= self.min_epochs

    def restore(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def _tarnet_epoch(model, opt, x, t, y, train_idx, batch, alpha, gen):
    """One epoch of TARNet / CFR training on the rows in ``train_idx``."""
    perm = train_idx[torch.randperm(len(train_idx), generator=gen, device="cpu").to(train_idx.device)]
    for i in range(0, len(perm), batch):
        idx = perm[i:i + batch]
        xb, tb, yb = x[idx], t[idx], y[idx]
        loss = factual_loss_weighted(model, xb, tb, yb)
        if alpha > 0 and tb.gt(0.5).any() and tb.le(0.5).any():
            phi = model.repr(xb)
            loss = loss + alpha * mmd_rbf(phi[tb > 0.5], phi[tb <= 0.5])
        opt.zero_grad()
        loss.backward()
        opt.step()


# ---------------------------------------------------------------------------
# training: TARNet (alpha = 0) / CFR (alpha > 0); source-only, target-only and
# pooled models all use this
# ---------------------------------------------------------------------------
def train_tarnet(x, t, y, arch=None, alpha=1.0, epochs=400, lr=5e-3, batch=64,
                 weight_decay=1e-4, val_frac=0.0, patience=20, min_epochs=20,
                 refit_full=False, split=None, seed=0, device="cpu", kind="tarnet"):
    """Train from scratch.  Returns (model, info).

    info: stop_epoch, val_loss_best (nan without validation), n_train, n_val,
          train_idx, val_idx (numpy).
    """
    x, t, y = x.to(device), t.to(device), y.to(device)
    arch = arch or arch_for_n(len(x))
    train_idx, val_idx = split if split is not None else split_indices(t, val_frac, seed)
    use_val = len(val_idx) > 0

    def _fit(n_epochs, tr_idx, early):
        torch.manual_seed(seed)
        gen = torch.Generator().manual_seed(seed)
        model = build_model(x.shape[1], kind=kind, **arch).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        stopper = EarlyStopper(patience, min_epochs) if early else None
        last = n_epochs
        for epoch in range(1, n_epochs + 1):
            model.train()
            _tarnet_epoch(model, opt, x, t, y, tr_idx, batch, alpha, gen)
            if stopper is not None:
                model.eval()
                with torch.no_grad():
                    vl = factual_mse(model, x[val_idx], t[val_idx], y[val_idx]).item()
                if stopper.step(epoch, vl, model):
                    last = epoch
                    break
        if stopper is not None:
            stopper.restore(model)
            return model, stopper.best_epoch, stopper.best
        return model, last, float("nan")

    model, stop_epoch, val_best = _fit(epochs, train_idx, use_val)
    if use_val and refit_full:
        # retrain on ALL rows for the number of epochs found by early stopping
        model, stop_epoch, _ = _fit(max(stop_epoch, 1), torch.arange(len(x), device=device), False)
    model.eval()
    info = dict(stop_epoch=int(stop_epoch), val_loss_best=float(val_best),
                n_train=int(len(train_idx)), n_val=int(len(val_idx)),
                train_idx=train_idx.cpu().numpy(), val_idx=val_idx.cpu().numpy())
    return model, info


# ---------------------------------------------------------------------------
# TL-TARNet: warm start from the source model, freeze all but the last
# ``n_unfrozen`` linear layers of the encoder and of each head, fine-tune on
# the target with the joint objective in the module docstring
# ---------------------------------------------------------------------------
def _last_linears(seq, k):
    lins = [m for m in seq if isinstance(m, nn.Linear)]
    return lins[-k:] if k > 0 else []


def freeze_but_last(model, n_unfrozen=1):
    for p in model.parameters():
        p.requires_grad_(False)
    for blk in (model.encoder, model.h0, model.h1):
        for lin in _last_linears(blk, n_unfrozen):
            for p in lin.parameters():
                p.requires_grad_(True)
    return model


def source_factual_loss(model, xs, ts, ys, n_max=2000, seed=0):
    """Plain factual MSE of ``model`` on (a fixed random subset of) the source."""
    model.eval()
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(xs), generator=g)[:min(n_max, len(xs))].to(xs.device)
    with torch.no_grad():
        return factual_mse(model, xs[idx], ts[idx], ys[idx]).item()


def within_target_mmd(model, x, t):
    """MMD^2 between treated and control representations of ALL target units."""
    model.eval()
    with torch.no_grad():
        phi = model.repr(x)
        return mmd_rbf(phi[t > 0.5], phi[t <= 0.5]).item()


def transfer_finetune(source_model, xt, tt, yt, xs, ts, ys, n_unfrozen=1,
                      alpha=1.0, alpha_cross=1.0, w_src=1.0, epochs=300, lr=1e-4,
                      batch=32, val_frac=0.0, patience=20, min_epochs=20,
                      refit_full=False, split=None, seed=0, device="cpu"):
    """Fine-tune a copy of ``source_model`` on the target.  Returns (model, info).

    info: stop_epoch, val_loss_best, src_floss_before/after (source factual MSE of
          the pre-trained / fine-tuned model), tgt_mmd_before/after (within-target
          MMD^2 of the representation), n_train, n_val, train_idx, val_idx.
    """
    xt, tt, yt = xt.to(device), tt.to(device), yt.to(device)
    xs, ts, ys = xs.to(device), ts.to(device), ys.to(device)
    train_idx, val_idx = split if split is not None else split_indices(tt, val_frac, seed)
    use_val = len(val_idx) > 0

    def _fit(n_epochs, tr_idx, early):
        torch.manual_seed(seed)
        gen = torch.Generator().manual_seed(seed)
        model = copy.deepcopy(source_model).to(device)
        freeze_but_last(model, n_unfrozen)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.Adam(params, lr=lr)
        stopper = EarlyStopper(patience, min_epochs) if early else None
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

                loss = factual_loss_weighted(model, xb, tb, yb)              # target factual
                if w_src > 0:
                    loss = loss + w_src * factual_loss_weighted(model, xsb, tsb, ysb)  # source factual
                if alpha > 0 or alpha_cross > 0:
                    phi_t, phi_s = model.repr(xb), model.repr(xsb)
                    if alpha > 0 and tb.gt(0.5).any() and tb.le(0.5).any():   # within-target t1 vs t0
                        loss = loss + alpha * mmd_rbf(phi_t[tb > 0.5], phi_t[tb <= 0.5])
                    if alpha_cross > 0:                                       # cross-domain, per arm
                        for tv in (0, 1):
                            tm = (tb > 0.5) if tv else (tb <= 0.5)
                            sm = (tsb > 0.5) if tv else (tsb <= 0.5)
                            if tm.any() and sm.any():
                                loss = loss + alpha_cross * mmd_rbf(phi_t[tm], phi_s[sm])
                opt.zero_grad()
                loss.backward()
                opt.step()
            if stopper is not None:
                model.eval()
                with torch.no_grad():
                    vl = factual_mse(model, xt[val_idx], tt[val_idx], yt[val_idx]).item()
                if stopper.step(epoch, vl, model):
                    last = epoch
                    break
        if stopper is not None:
            stopper.restore(model)
            return model, stopper.best_epoch, stopper.best
        return model, last, float("nan")

    src_before = source_factual_loss(source_model, xs, ts, ys, seed=seed)
    mmd_before = within_target_mmd(source_model, xt, tt)
    model, stop_epoch, val_best = _fit(epochs, train_idx, use_val)
    if use_val and refit_full:
        model, stop_epoch, _ = _fit(max(stop_epoch, 1), torch.arange(len(xt), device=device), False)
    model.eval()
    info = dict(stop_epoch=int(stop_epoch), val_loss_best=float(val_best),
                src_floss_before=src_before,
                src_floss_after=source_factual_loss(model, xs, ts, ys, seed=seed),
                tgt_mmd_before=mmd_before, tgt_mmd_after=within_target_mmd(model, xt, tt),
                n_train=int(len(train_idx)), n_val=int(len(val_idx)),
                train_idx=train_idx.cpu().numpy(), val_idx=val_idx.cpu().numpy())
    return model, info


# ---------------------------------------------------------------------------
# CITA: diagonal Fisher, symmetrised over the binary label permutation
# ---------------------------------------------------------------------------
def diag_fisher(model, x, t, y, n_sample=512, seed=0, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(x), generator=g)[:min(n_sample, len(x))]
    params = list(model.parameters())
    old_req = [p.requires_grad for p in params]
    for p in params:
        p.requires_grad_(True)
    fisher = [torch.zeros_like(p) for p in params]
    for i in idx:
        model.zero_grad()
        factual_mse(model, x[i:i + 1].to(device), t[i:i + 1].to(device),
                    y[i:i + 1].to(device)).backward()
        for f, p in zip(fisher, params):
            if p.grad is not None:
                f += p.grad.detach() ** 2
    for p, r in zip(params, old_req):
        p.requires_grad_(r)
    model.zero_grad()
    return [f / len(idx) for f in fisher]


def _unit_sqrt(F):
    """Flatten the diagonal Fisher, take the elementwise sqrt and normalise to unit
    L2 norm, so that (1/sqrt2)||u_ss - u_st|| lies in [0, 1]."""
    v = torch.cat([f.clamp_min(0).sqrt().flatten() for f in F])
    return v / v.norm().clamp_min(1e-12)


def cita(source_model, xs, ts, ys, xt, tt, yt, n_sample=512, seed=0, device="cpu"):
    u_ss = _unit_sqrt(diag_fisher(source_model, xs, ts, ys, n_sample, seed, device))
    u_id = _unit_sqrt(diag_fisher(source_model, xt, tt, yt, n_sample, seed, device))
    u_sw = _unit_sqrt(diag_fisher(source_model, xt, 1.0 - tt, yt, n_sample, seed, device))
    d_id = float((u_ss - u_id).norm().item() / math.sqrt(2))
    d_sw = float((u_ss - u_sw).norm().item() / math.sqrt(2))
    return min(d_id, d_sw)


# ---------------------------------------------------------------------------
# metrics (true ITE known in simulation -> evaluated against the TARGET's ITE)
# ---------------------------------------------------------------------------
def _predict(model, x, device="cpu"):
    model.eval()
    with torch.no_grad():
        return model.predict_ite(x.to(device)).cpu().numpy().astype(np.float64)


def pehe(model, x, ite_true, idx=None, device="cpu"):
    """Mean squared error of the CATE prediction; ``idx`` restricts to a subset
    (e.g. the held-out rows).  nan if the subset is empty."""
    pred = _predict(model, x, device)
    if idx is not None:
        if len(idx) == 0:
            return float("nan")
        pred, ite_true = pred[idx], ite_true[idx]
    return float(np.mean((pred - ite_true) ** 2))


def mean_bias(model, x, ite_true, idx=None, device="cpu"):
    pred = _predict(model, x, device)
    if idx is not None:
        if len(idx) == 0:
            return float("nan")
        pred, ite_true = pred[idx], ite_true[idx]
    return float(pred.mean() - ite_true.mean())


def ite_summary(model, x, ite_true, device="cpu"):
    """Distribution of the predicted CATEs over the target units: mean (the ATE
    estimate), within-sample SD (dispersion) and correlation with the true CATE
    (nan if either is constant)."""
    pred = _predict(model, x, device)
    ite = np.asarray(ite_true, dtype=np.float64)
    if pred.std() < 1e-12 or ite.std() < 1e-12:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(pred, ite)[0, 1])
    return dict(pred_mean=float(pred.mean()), pred_sd=float(pred.std()), corr=corr)


def constant_ate_metrics(t, y, ite_true):
    """Baselines that predict one number for everybody.

    oracle: the true mean ITE of the target sample (PEHE = Var(tau)).
    dim:    the difference in means of the observed outcomes (what a practitioner
            could compute).
    """
    t, y, ite = np.asarray(t, float), np.asarray(y, float), np.asarray(ite_true, float)
    if (t > 0.5).sum() == 0 or (t <= 0.5).sum() == 0:
        dim = float("nan")
    else:
        dim = float(y[t > 0.5].mean() - y[t <= 0.5].mean())
    return dict(pehe_const_oracle=float(ite.var()),
                pehe_const_dim=float(np.mean((dim - ite) ** 2)),
                bias_const_dim=float(dim - ite.mean()),
                ate_dim=dim)


def _average_ranks(s):
    """1-based ranks with ties averaged (for the Mann-Whitney AUC)."""
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ss = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return ranks


def auc_score(labels, scores):
    """Area under the ROC curve (Mann-Whitney statistic); no sklearn dependency."""
    y = np.asarray(labels).astype(int)
    s = np.asarray(scores, dtype=float)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = _average_ranks(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _logreg_proba(Xtr, ytr, Xte, l2=1e-2, max_iter=200):
    """L2-regularised logistic regression (L-BFGS) on standardised features;
    returns P(y = 1) for Xte."""
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-8
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    w = torch.zeros(Xtr.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(Xtr @ w + b, ytr) + l2 * (w * w).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        return torch.sigmoid(Xte @ w + b)


def probe_auc(model, x, labels, n_splits=5, seed=0, device="cpu"):
    """How much of ``labels`` (binary) is linearly decodable from Phi(x).

    5-fold stratified cross-validated AUC of a logistic regression on the
    standardised representation.  nan if a class is missing or too small.
    """
    lab = np.asarray(labels).astype(int)
    if len(np.unique(lab)) < 2 or np.bincount(lab).min() < n_splits:
        return float("nan")
    model.eval()
    with torch.no_grad():
        phi = model.repr(x.to(device)).detach().cpu().float()
    y = torch.as_tensor(lab, dtype=torch.float32)
    # stratified folds: shuffle within each class, assign round-robin
    rng = np.random.default_rng(seed)
    fold = np.empty(len(lab), dtype=int)
    for c in (0, 1):
        idx = np.where(lab == c)[0]
        rng.shuffle(idx)
        fold[idx] = np.arange(len(idx)) % n_splits
    scores = np.empty(len(lab), dtype=float)
    for f in range(n_splits):
        te, tr = np.where(fold == f)[0], np.where(fold != f)[0]
        scores[te] = _logreg_proba(phi[tr], y[tr], phi[te]).numpy()
    return auc_score(lab, scores)


# ---------------------------------------------------------------------------
# convenience loaders
# ---------------------------------------------------------------------------
def x_columns(df):
    return [c for c in df.columns if X_COL.match(c)]


def to_tensors(df, device="cpu"):
    """(x, t, y, ite_true) from a data frame with columns X1..Xp, Tr, Y, ite."""
    x = torch.tensor(df[x_columns(df)].values, dtype=torch.float32, device=device)
    t = torch.tensor(df["Tr"].values, dtype=torch.float32, device=device)
    y = torch.tensor(df["Y"].values, dtype=torch.float32, device=device)
    ite = df["ite"].values.astype(np.float64)
    return x, t, y, ite

