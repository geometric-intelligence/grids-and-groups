"""Training function for TwoLayerMLP or QuadraticRNN with sequential inputs."""

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader


def _clone_param_snapshots(
    snapshots: list[dict[str, torch.Tensor]],
) -> list[dict[str, torch.Tensor]]:
    return [{k: v.detach().cpu().clone() for k, v in sd.items()} for sd in snapshots]


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
    save_param_snapshots: bool = True,
    reduction_threshold: float | None = None,
    dense_save_until: int = 0,
    resume_state: dict | None = None,
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
        save_param_snapshots: If False, never clone weights during training (no trajectory).
        reduction_threshold: If provided, stop training when loss reduction reaches
                            this threshold (e.g., 0.99 = 99% reduction). If None,
                            train for full epochs.
        dense_save_until: Save params every epoch for the first N epochs (default 0).
        resume_state: If set, continue from prior ``train_loss_history`` / ``param_history``
            (see ``main._try_load_resume_training_state``). Skips the initial snapshot;
            new epochs are indexed globally for ``param_save_epochs``.

    Returns:
        tuple: (train_loss_history, val_loss_history, param_history,
                param_save_epochs, final_epoch)
    """
    if resume_state is not None:
        train_loss_history = list(resume_state["train_loss_history"])
        val_loss_history = list(resume_state["val_loss_history"])
        param_history = _clone_param_snapshots(resume_state["param_history"])
        param_save_epochs = list(resume_state["param_save_indices"])
        initial_loss = float(resume_state["initial_loss"])
        epoch_offset = len(train_loss_history) - 1
        if len(train_loss_history) != len(val_loss_history):
            raise ValueError(
                "resume_state: train_loss_history and val_loss_history length mismatch"
            )
        if len(param_history) != len(param_save_epochs):
            raise ValueError("resume_state: param_history and param_save_indices length mismatch")
        if reduction_threshold is not None:
            print(f"  Resuming: initial loss (run start) {initial_loss:.6f}")
            print(f"  Early stopping at {reduction_threshold * 100:.1f}% reduction")
    else:
        train_loss_history, val_loss_history, param_history = [], [], []
        param_save_epochs = []

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

            if save_param_snapshots:
                snap0 = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}

        train_loss_history.append(val_loss0)
        val_loss_history.append(val_loss0)
        if save_param_snapshots:
            param_history.append(snap0)
            param_save_epochs.append(0)
        initial_loss = val_loss0
        epoch_offset = 0

        if reduction_threshold is not None:
            print(f"  Initial loss: {initial_loss:.6f}")
            print(f"  Early stopping at {reduction_threshold * 100:.1f}% reduction")

    final_epoch = epochs

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

        global_epoch = epoch_offset + epoch
        should_save = save_param_snapshots and (
            epoch <= dense_save_until
            or (
                save_param_interval is not None
                and (global_epoch % save_param_interval == 0 or epoch == epochs)
            )
            or (save_param_interval is None and epoch == epochs)
        )

        if should_save:
            with torch.no_grad():
                snap = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
            param_history.append(snap)
            param_save_epochs.append(global_epoch)

        # Compute reduction for logging and early stopping
        reduction = 1 - avg_loss / initial_loss if initial_loss > 0 else 0

        # Check early stopping
        if reduction_threshold is not None and reduction >= reduction_threshold:
            final_epoch = epoch
            # Save final params if not already saved
            if save_param_snapshots and (
                not param_save_epochs or param_save_epochs[-1] != global_epoch
            ):
                with torch.no_grad():
                    snap = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
                param_history.append(snap)
                param_save_epochs.append(global_epoch)
            print(
                f"\n[CONVERGED] Epoch {global_epoch} (segment {epoch}/{epochs}): "
                f"{reduction * 100:.1f}% reduction >= {reduction_threshold * 100:.1f}% threshold"
            )
            break

        if epoch % verbose_interval == 0:
            print(
                f"[Epoch {global_epoch:>5} (seg {epoch:>5}/{epochs})] "
                f"loss: {avg_loss:.6f} | reduction: {reduction * 100:>6.1f}%"
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
    save_param_snapshots: bool = True,
    reduction_threshold: float | None = None,
    resume_state: dict | None = None,
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
        save_param_snapshots: If False, never clone weights during training (no trajectory).
        reduction_threshold: If provided, stop training when loss reduction reaches
                            this threshold (e.g., 0.99 = 99% reduction). If None,
                            train for full num_steps.

    Returns:
        tuple: (train_loss_history, val_loss_history, param_history,
                param_save_steps, final_step)
    """
    if resume_state is not None:
        train_loss_history = list(resume_state["train_loss_history"])
        val_loss_history = list(resume_state["val_loss_history"])
        param_history = _clone_param_snapshots(resume_state["param_history"])
        param_save_steps = list(resume_state["param_save_indices"])
        initial_loss = float(resume_state["initial_loss"])
        step_offset = len(train_loss_history) - 1
        if len(train_loss_history) != len(val_loss_history):
            raise ValueError(
                "resume_state: train_loss_history and val_loss_history length mismatch"
            )
        if len(param_history) != len(param_save_steps):
            raise ValueError("resume_state: param_history and param_save_indices length mismatch")
        if reduction_threshold is not None:
            print(f"  Resuming: initial loss (run start) {initial_loss:.6f}")
            print(f"  Early stopping at {reduction_threshold * 100:.1f}% reduction")
    else:
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

            if save_param_snapshots:
                snap0 = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}

        train_loss_history.append(val_loss0)
        val_loss_history.append(val_loss0)
        if save_param_snapshots:
            param_history.append(snap0)
            param_save_steps.append(0)
        initial_loss = val_loss0
        step_offset = 0

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

            global_step = step_offset + step
            # Only save parameters at specified intervals or at the end
            # If save_param_interval is None, only save at the very end
            should_save = save_param_snapshots and (
                (
                    save_param_interval is not None
                    and (global_step % save_param_interval == 0 or step == num_steps)
                )
                or (save_param_interval is None and step == num_steps)
            )

            if should_save:
                snap = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
                param_history.append(snap)
                param_save_steps.append(global_step)

        model.train()

        # Compute reduction for logging and early stopping
        reduction = 1 - current_loss / initial_loss if initial_loss > 0 else 0

        # Check early stopping
        if reduction_threshold is not None and reduction >= reduction_threshold:
            final_step = step
            global_step = step_offset + step
            # Save final params if not already saved
            if save_param_snapshots and (
                not param_save_steps or param_save_steps[-1] != global_step
            ):
                with torch.no_grad():
                    snap = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
                param_history.append(snap)
                param_save_steps.append(global_step)
            print(
                f"\n[CONVERGED] Step {global_step} (segment {step}/{num_steps}): "
                f"{reduction * 100:.1f}% reduction >= {reduction_threshold * 100:.1f}% threshold"
            )
            break

        if step % verbose_interval == 0:
            global_step = step_offset + step
            print(
                f"[Step {global_step:>6} (seg {step:>6}/{num_steps})] "
                f"loss: {current_loss:.6f} | reduction: {reduction * 100:>6.1f}%"
            )

    return train_loss_history, val_loss_history, param_history, param_save_steps, final_step
