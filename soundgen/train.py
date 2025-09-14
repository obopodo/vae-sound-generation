from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn, save
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from soundgen.vae import VAE

tb_writer = SummaryWriter()


def save_model(model: nn.Module, epoch=None):
    model_folder = Path(__file__).parent.parent / "models"
    model_folder.exists() or model_folder.mkdir(parents=True, exist_ok=True)
    model_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_checkpoint.pth"
    if epoch is not None:
        model_name = model_name.replace("checkpoint", f"checkpoint_e{epoch}")
    save(model.state_dict(), model_folder / model_name)


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    *,
    loss_fn: nn.Module,
    optimizer: Optimizer,
    device: str,
    epoch: int,
    tb_writer: SummaryWriter,
):
    model.train()
    train_loss = 0.0
    for batch, (X, y) in enumerate(data_loader):
        X, _ = X.to(device), y.to(device)

        # Forward pass
        preds = model(X)
        if isinstance(model, VAE):
            loss = loss_fn(preds, X, model.mu, model.log_var)
        else:
            loss = loss_fn(preds, X)

        # Backward pass
        optimizer.zero_grad()  # Clear gradients from previous step
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        if batch == 0 or batch % 100 == 99:
            last_loss = train_loss / 100
            tb_writer.add_scalar("Loss/train", last_loss, batch + (epoch - 1) * len(data_loader))
            train_loss = 0.0

    return last_loss


def evaluate(model: nn.Module, data_loader: DataLoader, loss_fn: nn.Module, device: str):
    model.eval()
    valid_loss = 0.0
    with torch.no_grad():
        for batch, (X, y) in enumerate(data_loader):
            X, _ = X.to(device), y.to(device)
            preds = model(X)
            if isinstance(model, VAE):
                loss = loss_fn(preds, X, model.mu, model.log_var)
            else:
                loss = loss_fn(preds, X)
            valid_loss += loss.item()
    return valid_loss / len(data_loader)


def train(
    model: nn.Module,
    train_data_loader: DataLoader,
    valid_data_loader: DataLoader | None,
    *,
    loss_fn: nn.Module,
    optimizer: Optimizer,
    device: str,
    epochs: int,
    save_checkpoint: bool,
):
    min_valid_loss = np.inf
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        train_loss = train_one_epoch(
            model,
            train_data_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            tb_writer=tb_writer,
        )

        if valid_data_loader is not None:
            valid_loss = evaluate(model, valid_data_loader, loss_fn, device)
        else:
            valid_loss = np.nan

        tb_writer.add_scalars(
            "Training vs. Validation Loss",
            {"Training": train_loss, "Validation": valid_loss},
            epoch,
        )
        tb_writer.flush()

        print(f"LOSS train {train_loss} | valid {valid_loss}")
        print("-" * 30)

        if save_checkpoint and valid_loss < min_valid_loss:
            save_model(model, epoch=epoch)
            min_valid_loss = valid_loss
    print("Training complete.")
