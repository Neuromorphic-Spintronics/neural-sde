from __future__ import annotations

from typing import List, Tuple, Callable, Optional, Mapping

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_with_validation(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    early_stopping_patience: int,
    device: torch.device | str,
    batch_preparation_fn: Optional[Callable] = None,
) -> Tuple[nn.Module, List[float], List[float]]:
    """
    Generic training loop with validation and early stopping.

    Args:
        model: The neural network model to train.
        train_loader: DataLoader for the training set.
        val_loader: DataLoader for the validation set.
        num_epochs: Maximum number of epochs to train.
        learning_rate: Learning rate for the Adam optimizer.
        early_stopping_patience: Number of epochs to wait for improvement before stopping.
        device: The device to train on ('cpu' or 'cuda').
        batch_preparation_fn: Optional function to convert a raw batch into
            (inputs, targets). If None, expects each batch to be either a
            tuple/list of (inputs, targets) or a mapping with keys
            {'inputs','targets'}. Inputs/targets are moved to 'device'.

    Returns:
        A tuple containing:
        - The best performing model based on validation loss.
        - A list of training losses for each epoch.
        - A list of validation losses for each epoch.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.SmoothL1Loss()
    model.to(device)

    training_losses = []
    validation_losses = []
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    with tqdm(range(num_epochs), desc="Training") as pbar:
        for epoch in pbar:
            # --- Training Phase ---
            model.train()
            epoch_train_loss = 0.0
            for raw_batch in train_loader:
                if batch_preparation_fn is not None:
                    net_input, true_derivatives = batch_preparation_fn(raw_batch, device)
                else:
                    if isinstance(raw_batch, (tuple, list)) and len(raw_batch) == 2:
                        net_input, true_derivatives = raw_batch
                    elif isinstance(raw_batch, Mapping):
                        net_input = raw_batch.get("inputs")
                        true_derivatives = raw_batch.get("targets")
                        if net_input is None or true_derivatives is None:
                            raise ValueError("Mapping batch must have 'inputs' and 'targets' keys when batch_preparation_fn is None.")
                    else:
                        raise TypeError("Batch must be (inputs, targets) or mapping when batch_preparation_fn is None.")

                    if hasattr(net_input, "device") and net_input.device != device:
                        net_input = net_input.to(device, non_blocking=True)
                    if hasattr(true_derivatives, "device") and true_derivatives.device != device:
                        true_derivatives = true_derivatives.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                predicted_derivatives = model(net_input)
                loss = criterion(predicted_derivatives, true_derivatives)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_train_loss += loss.item()

            avg_epoch_train_loss = epoch_train_loss / max(len(train_loader), 1)
            training_losses.append(avg_epoch_train_loss)

            # --- Validation Phase ---
            model.eval()
            epoch_val_loss = 0.0
            with torch.no_grad():
                for raw_batch in val_loader:
                    if batch_preparation_fn is not None:
                        net_input, true_derivatives = batch_preparation_fn(raw_batch, device)
                    else:
                        if isinstance(raw_batch, (tuple, list)) and len(raw_batch) == 2:
                            net_input, true_derivatives = raw_batch
                        elif isinstance(raw_batch, Mapping):
                            net_input = raw_batch.get("inputs")
                            true_derivatives = raw_batch.get("targets")
                            if net_input is None or true_derivatives is None:
                                raise ValueError("Mapping batch must have 'inputs' and 'targets' keys when batch_preparation_fn is None.")
                        else:
                            raise TypeError("Batch must be (inputs, targets) or mapping when batch_preparation_fn is None.")

                        if hasattr(net_input, "device") and net_input.device != device:
                            net_input = net_input.to(device, non_blocking=True)
                        if hasattr(true_derivatives, "device") and true_derivatives.device != device:
                            true_derivatives = true_derivatives.to(device, non_blocking=True)

                    predicted_derivatives = model(net_input)
                    val_loss = criterion(predicted_derivatives, true_derivatives)
                    epoch_val_loss += val_loss.item()

            avg_epoch_val_loss = epoch_val_loss / max(len(val_loader), 1)
            validation_losses.append(avg_epoch_val_loss)

            pbar.set_postfix(
                {
                    "Train Loss": f"{avg_epoch_train_loss:.6f}",
                    "Val Loss": f"{avg_epoch_val_loss:.6f}",
                }
            )

            # --- Early Stopping Check ---
            if avg_epoch_val_loss < best_val_loss - 1e-9:
                best_val_loss = avg_epoch_val_loss
                patience_counter = 0
                best_model_state = {
                    k: v.detach().cpu() for k, v in model.state_dict().items()
                }
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}.")
                break

    # Load the best model state before finishing
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(device)
        print("Loaded best model weights from early stopping.")

    return model, training_losses, validation_losses
