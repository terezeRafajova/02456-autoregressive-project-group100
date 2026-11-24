import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from protgen.data import ProteinDataset
from protgen.models.classifier import FamilyClassifier

# The prefix lengths the trained head is reported on
FRACTIONS = (0.1, 0.25, 0.5, 0.75, 1.0)


@hydra.main(config_path="config", config_name="classifier", version_base=None)
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

    backbone = hydra.utils.instantiate(
        cfg.model.init,
        vocab_size=train_dataset.vocab_size,
        max_length=train_dataset.max_length,
    ).to(device)
    backbone.load_state_dict(torch.load(cfg.checkpoint_path, map_location=device))

    model = FamilyClassifier(backbone, num_classes=len(train_dataset.fam_to_idx)).to(device)

    # Only the head is trained
    optimizer = hydra.utils.instantiate(cfg.optimizer)(params=model.head.parameters())

    step = 0
    model.head.train()

    train_losses = []
    validation_losses = []
    best_validation_loss = float("inf")
    checks_since_best = 0

    for epoch in range(cfg.num_epochs):

        train_losses_batches = []
        for x, _, y, lengths in train_loader:
            x, y, lengths = x.to(device), y.to(device), lengths.to(device)

            loss = model.loss(x, y, lengths)
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

                    model.head.eval()
                    for x, _, y, lengths in validation_loader:
                        x, y, lengths = x.to(device), y.to(device), lengths.to(device)
                        loss = model.loss(x, y, lengths)
                        validation_losses_batches.append(loss.item() * len(x))

                    model.head.train()

                validation_losses.append(sum(validation_losses_batches) / len(validation_dataset))

                print(f"Step {step}")
                print(f"  training loss: {train_losses[-1]}")
                print(f"  validation loss: {validation_losses[-1]}")

                if validation_losses[-1] < best_validation_loss:
                    best_validation_loss = validation_losses[-1]
                    checks_since_best = 0
                    torch.save(model.head.state_dict(), cfg.classifier_path)
                else:
                    checks_since_best += 1

                if checks_since_best >= cfg.patience:
                    break

        if checks_since_best >= cfg.patience: #early stopping check
            break

    print("Finished training.")

    # How much of a sequence the head needs before it knows the family
    model.head.load_state_dict(torch.load(cfg.classifier_path, map_location=device))
    model.head.eval()

    accuracies = prefix_accuracy(model, validation_loader, device)
    print("Validation accuracy by observed fraction of the sequence:")
    for fraction, accuracy in zip(FRACTIONS, accuracies):
        print(f"  {fraction:>5}: {accuracy:.3f}")


@torch.no_grad()
def prefix_accuracy(model, loader, device):
    correct = torch.zeros(len(FRACTIONS), device=device)
    total = 0

    for x, _, y, lengths in loader:
        x, y, lengths = x.to(device), y.to(device), lengths.to(device)

        predictions = model(x).argmax(dim=-1)

        for index, fraction in enumerate(FRACTIONS):
            residues = (lengths * fraction).round().long().clamp(min=1)
            correct[index] += (predictions.gather(1, residues[:, None]).squeeze(1) == y).sum()

        total += len(x)

    return (correct / total).tolist()


if __name__ == "__main__":
    main()
