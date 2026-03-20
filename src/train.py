"""Training function for Quadratic RNN with sequential inputs."""

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader


def train(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    epochs: int = 2000,
    verbose_interval: int = 100,
    grad_clip: float | None = None,
    eval_dataloader: DataLoader | None = None,
    save_param_interval: int | None = None,
    reduction_threshold: float | None = None,
    dense_save_until: int = 0,
) -> tuple[list[float], list[float], list[dict[str, torch.Tensor]], list[int], int]:
    """
    Train a model with sequential inputs (offline/epoch-based).

    Args:
        model: The model to train
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        epochs: Maximum number of training epochs
        verbose_interval: Print progress every N epochs
        grad_clip: Optional gradient clipping value
        eval_dataloader: Optional separate validation loader
        save_param_interval: If provided, save params every N epochs.
                            If None, only save initial and final params (memory efficient!)
        reduction_threshold: If provided, stop training when loss reduction reaches
                            this threshold (e.g., 0.99 = 99% reduction). If None,
                            train for full epochs.
        dense_save_until: Save params every epoch for the first N epochs (default 0).

    Returns:
        tuple: (train_loss_history, val_loss_history, param_history,
                param_save_epochs, final_epoch)
    """
    train_loss_history, val_loss_history, param_history = [], [], []
    param_save_epochs = []

    # --- BEFORE TRAINING (epoch 0) ---
    model.eval()
    with torch.no_grad():
        if eval_dataloader is not None:
            X_eval, Y_eval = next(iter(eval_dataloader))
            out = model(X_eval)
            val_loss0 = criterion(out, Y_eval).item()
        else:
            X_eval, Y_eval = next(iter(dataloader))
            out = model(X_eval)
            val_loss0 = criterion(out, Y_eval).item()

        snap0 = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}

    train_loss_history.append(val_loss0)
    val_loss_history.append(val_loss0)
    param_history.append(snap0)
    param_save_epochs.append(0)
    initial_loss = val_loss0

    if reduction_threshold is not None:
        print(f"  Initial loss: {initial_loss:.6f}")
        print(f"  Early stopping at {reduction_threshold * 100:.1f}% reduction")

    final_epoch = epochs

    # --- TRAINING LOOP (epochs 1..epochs) ---
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for X_batch, Y_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            running += loss.item()

        avg_loss = running / len(dataloader)
        train_loss_history.append(avg_loss)

        model.eval()
        with torch.no_grad():
            if eval_dataloader is not None:
                X_eval, Y_eval = next(iter(eval_dataloader))
                out = model(X_eval)
                val_loss = criterion(out, Y_eval).item()
            else:
                X_eval, Y_eval = next(iter(dataloader))
                out = model(X_eval)
                val_loss = criterion(out, Y_eval).item()

        val_loss_history.append(val_loss)

        should_save = (
            epoch <= dense_save_until
            or (
                save_param_interval is not None
                and (epoch % save_param_interval == 0 or epoch == epochs)
            )
            or (save_param_interval is None and epoch == epochs)
        )

        if should_save:
            with torch.no_grad():
                snap = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
            param_history.append(snap)
            param_save_epochs.append(epoch)

        # Compute reduction for logging and early stopping
        reduction = 1 - avg_loss / initial_loss if initial_loss > 0 else 0

        # Check early stopping
        if reduction_threshold is not None and reduction >= reduction_threshold:
            final_epoch = epoch
            # Save final params if not already saved
            if param_save_epochs[-1] != epoch:
                with torch.no_grad():
                    snap = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
                param_history.append(snap)
                param_save_epochs.append(epoch)
            print(
                f"\n[CONVERGED] Epoch {epoch}: {reduction * 100:.1f}% reduction >= {reduction_threshold * 100:.1f}% threshold"
            )
            break

        if epoch % verbose_interval == 0:
            print(
                f"[Epoch {epoch:>5}/{epochs}] loss: {avg_loss:.6f} | reduction: {reduction * 100:>6.1f}%"
            )

    return train_loss_history, val_loss_history, param_history, param_save_epochs, final_epoch


def train_online(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    num_steps: int = 10000,
    verbose_interval: int = 100,
    grad_clip: float | None = None,
    eval_dataloader: DataLoader | None = None,
    save_param_interval: int | None = None,
    reduction_threshold: float | None = None,
) -> tuple[list[float], list[float], list[dict[str, torch.Tensor]], list[int], int]:
    """
    Train with online data generation (step-based instead of epoch-based).

    Args:
        model: The model to train
        dataloader: Training data loader (online/infinite)
        criterion: Loss function
        optimizer: Optimizer
        num_steps: Maximum number of training steps
        verbose_interval: Print progress every N steps
        grad_clip: Optional gradient clipping value
        eval_dataloader: Optional separate validation loader
        save_param_interval: If provided, save params every N steps.
                            If None, only save initial and final params (memory efficient!)
        reduction_threshold: If provided, stop training when loss reduction reaches
                            this threshold (e.g., 0.99 = 99% reduction). If None,
                            train for full num_steps.

    Returns:
        tuple: (train_loss_history, val_loss_history, param_history,
                param_save_steps, final_step)
    """
    train_loss_history, val_loss_history, param_history = [], [], []
    param_save_steps = []

    # Initial evaluation (step 0)
    model.eval()
    with torch.no_grad():
        if eval_dataloader is not None:
            X_eval, Y_eval = next(iter(eval_dataloader))
            out = model(X_eval)
            val_loss0 = criterion(out, Y_eval).item()
        else:
            X_batch, Y_batch = next(iter(dataloader))
            out = model(X_batch)
            val_loss0 = criterion(out, Y_batch).item()

        snap0 = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}

    train_loss_history.append(val_loss0)
    val_loss_history.append(val_loss0)
    param_history.append(snap0)
    param_save_steps.append(0)
    initial_loss = val_loss0

    if reduction_threshold is not None:
        print(f"  Initial loss: {initial_loss:.6f}")
        print(f"  Early stopping at {reduction_threshold * 100:.1f}% reduction")

    # Training loop
    model.train()
    data_iter = iter(dataloader)
    final_step = num_steps

    for step in range(1, num_steps + 1):
        # Get fresh batch
        X_batch, Y_batch = next(data_iter)

        # Training step
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, Y_batch)
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        # Record training loss
        current_loss = loss.item()
        train_loss_history.append(current_loss)

        # Evaluation on validation set
        model.eval()
        with torch.no_grad():
            if eval_dataloader is not None:
                X_eval, Y_eval = next(iter(eval_dataloader))
                out = model(X_eval)
                val_loss = criterion(out, Y_eval).item()
            else:
                X_eval, Y_eval = next(data_iter)
                out = model(X_eval)
                val_loss = criterion(out, Y_eval).item()

            val_loss_history.append(val_loss)

            # Only save parameters at specified intervals or at the end
            # If save_param_interval is None, only save at the very end
            should_save = (
                save_param_interval is not None
                and (step % save_param_interval == 0 or step == num_steps)
            ) or (save_param_interval is None and step == num_steps)

            if should_save:
                snap = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
                param_history.append(snap)
                param_save_steps.append(step)

        model.train()

        # Compute reduction for logging and early stopping
        reduction = 1 - current_loss / initial_loss if initial_loss > 0 else 0

        # Check early stopping
        if reduction_threshold is not None and reduction >= reduction_threshold:
            final_step = step
            # Save final params if not already saved
            if param_save_steps[-1] != step:
                with torch.no_grad():
                    snap = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
                param_history.append(snap)
                param_save_steps.append(step)
            print(
                f"\n[CONVERGED] Step {step}: {reduction * 100:.1f}% reduction >= {reduction_threshold * 100:.1f}% threshold"
            )
            break

        if step % verbose_interval == 0:
            print(
                f"[Step {step:>6}/{num_steps}] loss: {current_loss:.6f} | reduction: {reduction * 100:>6.1f}%"
            )

    return train_loss_history, val_loss_history, param_history, param_save_steps, final_step
