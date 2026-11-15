import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from protgen.data import ProteinDataset


@hydra.main(config_path="config", config_name="default", version_base=None)
def main(cfg):
    print(OmegaConf.to_yaml(cfg))

    torch.manual_seed(cfg.seed)

    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)

    train_dataset = ProteinDataset("data/train.csv")
    validation_dataset = ProteinDataset("data/val.csv")

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    validation_loader = DataLoader(validation_dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    model = hydra.utils.instantiate(
        cfg.model.init,
        vocab_size=train_dataset.vocab_size,
        max_length=train_dataset.max_length,
    ).to(device)

    optimizer = hydra.utils.instantiate(cfg.optimizer)(params=model.parameters())

    step = 0
    model.train()
    
    train_losses = []
    validation_losses = []
    best_validation_loss = float("inf")
    checks_since_best = 0

    for epoch in range(cfg.num_epochs):

        train_losses_batches = []
        for x, x_shifted, _, _ in train_loader:
            x, x_shifted = x.to(device), x_shifted.to(device)

            loss = model.loss(x, x_shifted)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            train_losses_batches.append(loss.item())

            if step % cfg.validation_every_steps == 0:

                train_losses.append(np.mean(train_losses_batches))
                train_losses_batches = []

                validation_losses_batches = []
                with torch.no_grad():

                    model.eval()
                    for x, x_shifted, _, _ in validation_loader:
                        x, x_shifted = x.to(device), x_shifted.to(device)
                        loss = model.loss(x, x_shifted)
                        validation_losses_batches.append(loss.item() * len(x))

                    model.train()

                validation_losses.append(sum(validation_losses_batches) / len(validation_dataset))

                print(f"Step {step}")
                print(f"  training loss: {train_losses[-1]}")
                print(f"  validation loss: {validation_losses[-1]}")

                if validation_losses[-1] < best_validation_loss:
                    best_validation_loss = validation_losses[-1]
                    checks_since_best = 0
                    torch.save(model.state_dict(), cfg.checkpoint_path)
                else:
                    checks_since_best += 1

                if checks_since_best >= cfg.patience:
                    break

        if checks_since_best >= cfg.patience: #early stopping check
            break

    print("Finished training.")


if __name__ == "__main__":
    main()
