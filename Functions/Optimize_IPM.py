def optimize_ipm(model_representation, optimizer_ipm, train_loader_target, IPM_distance, device, max_epochs_phase_1, phi_source_treatment, phi_source_control):
    model_representation.train()
    phi_target_dict = {}

    for epoch in range(max_epochs_phase_1):
        total_ipm_loss = 0.0

        for idx, batch in enumerate(train_loader_target):
            x_, a, y = [item.to(device) for item in batch]
            optimizer_ipm.zero_grad()

            # Compute target embeddings
            phi_target = model_representation(x_)

            # Store embeddings for factual loss optimization
            phi_target_dict[idx] = (phi_target.clone().detach(), a.clone().detach(), y.clone().detach())

            # Compute IPM loss
            phi_target_control_tensor = phi_target[a == 0]
            phi_target_treatment_tensor = phi_target[a == 1]
            ipm_loss = IPM_distance(
                phi_source_treatment.detach(), phi_source_control.detach(),
                phi_target_treatment_tensor, phi_target_control_tensor,IPM,
                device
            )
            
            ipm_loss.backward()
            torch.nn.utils.clip_grad_norm_(model_representation.parameters(), max_norm=1.0) 
            optimizer_ipm.step()

            total_ipm_loss += ipm_loss.item() * x_.size(0)

        avg_ipm_loss = total_ipm_loss / len(train_loader_target.dataset)
        print(f"Phase 1 - Epoch [{epoch + 1}/{max_epochs_phase_1}], Average IPM Loss: {avg_ipm_loss:.4f}")

    return model_representation, phi_target_dict
