def loss_function(model_prediction, optimizer_factual, train_loader_target, device, max_epochs_phase_2, phi_target_dict): #phi_target_dict comes from optimize_ipm
    model_prediction.train()

    for epoch in range(max_epochs_phase_2):
        total_factual_loss = 0.0

        for idx, batch in enumerate(train_loader_target):
            x_, a, y = [item.to(device) for item in batch]
            optimizer_factual.zero_grad()

            # Retrieve stored embeddings from Phase 1
            phi_target_batch, a_batch, y_batch = phi_target_dict[idx]
            phi_target_batch = phi_target_batch.to(device)
            a_batch = a_batch.to(device)
            y_batch = y_batch.to(device)

            # Predict outcomes
            y0_pred = model_prediction['control'](phi_target_batch).squeeze()
            y1_pred = model_prediction['treatment'](phi_target_batch).squeeze()

            # Compute factual loss
            factual_loss = 0.0
            control_indices = (a == 0).nonzero(as_tuple=True)[0]
            treatment_indices = (a == 1).nonzero(as_tuple=True)[0]
            # Written with if statement because some minibatches might contain only treated or only control units
            if len(control_indices) > 0:
                factual_loss += mse_loss(y0_pred[control_indices], y[control_indices])
            if len(treatment_indices) > 0:
                factual_loss += mse_loss(y1_pred[treatment_indices], y[treatment_indices])

            factual_loss.backward()
            optimizer_factual.step()

            total_factual_loss += factual_loss.item() * x_.size(0)

        avg_factual_loss = total_factual_loss / len(train_loader_target.dataset)
        print(f"Phase 2 - Epoch [{epoch + 1}/{max_epochs_phase_2}], Average Factual Loss: {avg_factual_loss:.4f}")

    return model_prediction, float(avg_factual_loss)

