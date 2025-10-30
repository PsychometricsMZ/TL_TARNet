import math
import torch
from copy import deepcopy

wass = Wassertein_Loss()
def IPM(phi_control, phi_treated):
    wasserstein_distance = wass(phi_control, phi_treated)
    return wasserstein_distance

def IPM_distance(phi_source_treatment, phi_source_control, phi_target_treatment, phi_target_control, IPM, device):
    phi_source_treatment_fixed = phi_source_treatment.detach()
    phi_source_control_fixed = phi_source_control.detach()
    ipm_loss = (
        IPM(phi_target_treatment, phi_source_treatment_fixed) +
        IPM(phi_target_control, phi_source_control_fixed) +
        IPM(phi_target_control, phi_target_treatment)
    )
    return ipm_loss

def _make_inverted_loader(tensor_dataset, batch_size=32, shuffle=False):
    """
    Takes a TensorDataset(X, a, y) and returns a DataLoader with a' = 1 - a.
    """
    X, a, y = tensor_dataset.tensors
    a_inv = 1.0 - a
    ds_inv = torch.utils.data.TensorDataset(X, a_inv, y)
    return torch.utils.data.DataLoader(ds_inv, batch_size=batch_size, shuffle=shuffle)

def _unit_trace(diag_flat: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Normalize a diagonal Fisher (flattened) to have trace 1.
    trace(F) = sum of diagonal entries.
    """
    tr = torch.clamp(diag_flat.sum(), min=eps)
    return diag_flat / tr

def compute_fisher_diag_noalpha(
    model,
    loader,
    device,
    loss_fn=None,          
    use_running_weights=True
):
    """
    Diagonal Fisher w.r.t. TARNet factual loss only (NO alpha/IPM).
    Empirical Fisher ≈ average over samples of (∇θ L_i)(∇θ L_i)^T, taking only the diagonal.
    Returns a single flattened vector containing the diagonal Fisher entries (UN-normalized).
    """
    assert loss_fn is not None, "Provide an elementwise loss, e.g., MSELoss(reduction='none')."

    was_training = model.training
    model.eval()

    # Temporarily enable grads for all params to measure Fisher
    old_req = {n: p.requires_grad for n, p in model.named_parameters()}
    for p in model.parameters():
        p.requires_grad = True

    fisher_accums = [torch.zeros_like(p, device=device) for p in model.parameters()]
    total_N = 0

    for batch in loader:
        x, a, y = [t.to(device) for t in batch]
        B = x.size(0)
        total_N += B

        model.zero_grad(set_to_none=True)
        phi, y0_pred, y1_pred = model(x)

        # TARNet weights
        if use_running_weights:
            v = float(a.mean().item())
            v = min(max(v, 1e-6), 1 - 1e-6)  # clamp to avoid div-by-zero
            weights = (a / (2 * v) + (1 - a) / (2 * (1 - v))).to(device)
        else:
            weights = torch.ones_like(a, device=device)

        # Flatten preds/targets
        y0_pred = y0_pred.view(-1)
        y1_pred = y1_pred.view(-1)
        y_true  = y.view(-1)

        mask_c = (a == 0)
        mask_t = (a == 1)

        # Build per-batch average loss (weighted), typical TARNet implementation
        loss = torch.tensor(0.0, device=device)
        if mask_c.any():
            l_c = loss_fn(y0_pred[mask_c], y_true[mask_c])  # elementwise
            loss = loss + (l_c * weights[mask_c]).sum() / B
        if mask_t.any():
            l_t = loss_fn(y1_pred[mask_t], y_true[mask_t])  # elementwise
            loss = loss + (l_t * weights[mask_t]).sum() / B

        loss.backward()

        # Accumulate squared grads *by batch size* so we can divide by total_N later
        for i, p in enumerate(model.parameters()):
            if p.grad is not None:
                fisher_accums[i] += (p.grad.detach() ** 2) * B

    # Average over samples
    total_N = max(int(total_N), 1)
    fisher_accums = [fa / float(total_N) for fa in fisher_accums]

    # Restore flags / mode
    for (n, p) in model.named_parameters():
        p.requires_grad = old_req[n]
    if was_training:
        model.train()

    # Flatten to a single vector
    fisher_flat = torch.cat([fa.reshape(-1) for fa in fisher_accums])
    return fisher_flat

def fisher_task_distance(f_ss_flat, f_st_flat, eps=1e-12):
    """
    TAS: d[s,t] = (1/sqrt(2)) * || sqrt(F_ss) - sqrt(F_st) ||_F
    (Here F are diagonal, provided as flattened vectors.)
    Assumes f_ss_flat and f_st_flat are *unit-trace normalized* diagonals.
    """
    f_ss_sqrt = torch.sqrt(torch.clamp(f_ss_flat, min=eps))
    f_st_sqrt = torch.sqrt(torch.clamp(f_st_flat, min=eps))
    diff = f_ss_sqrt - f_st_sqrt
    return (1.0 / math.sqrt(2.0)) * torch.linalg.vector_norm(diff, ord=2)

def compute_CITA(
    model,
    source_loader,          # DataLoader over source (X_s, a_s, y_s)
    target_dataset,         # TensorDataset(X_t, a_t, y_t)
    device,
    batch_size=32,
    shuffle=False
):
    """
    Returns (tas_plus, tas_minus, cita) with NO alpha (no IPM term).
    Each Fisher is unit-trace normalized so TAS, CITA ∈ [0,1].
    """
    loss_elem = torch.nn.MSELoss(reduction='none')

    # Build loaders for target: original and label-flipped
    target_loader     = torch.utils.data.DataLoader(target_dataset, batch_size=batch_size, shuffle=shuffle)
    target_loader_inv = _make_inverted_loader(target_dataset, batch_size=batch_size, shuffle=shuffle)

    # Fisher on source (source data)
    f_ss = compute_fisher_diag_noalpha(model, source_loader, device, loss_fn=loss_elem)

    # Fisher on target (original labels)
    f_st = compute_fisher_diag_noalpha(model, target_loader, device, loss_fn=loss_elem)

    # Fisher on target (inverted labels)
    f_st_inv = compute_fisher_diag_noalpha(model, target_loader_inv, device, loss_fn=loss_elem)

    # --- Unit-trace normalization (crucial for [0,1] range) ---
    f_ss = _unit_trace(f_ss)
    f_st = _unit_trace(f_st)
    f_st_inv = _unit_trace(f_st_inv)

    # TAS and CITA
    tas_plus  = fisher_task_distance(f_ss, f_st).item()
    tas_minus = fisher_task_distance(f_ss, f_st_inv).item()
    cita      = min(tas_plus, tas_minus)

    return tas_plus, tas_minus, cita

# Example call:
tas_p, tas_m, cita_val = compute_CITA(
     model=source_model,
     source_loader=train_loader,
     target_dataset=all_target,
     device=device,
     batch_size=32,
     shuffle=False
 )
print(f"TAS (original target labels): {tas_p:.6f}")
print(f"TAS (inverted target labels): {tas_m:.6f}")
print(f"CITA (min of the two):        {cita_val:.6f}")
