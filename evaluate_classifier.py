import hydra
import pandas as pd
import torch
from torch.utils.data import DataLoader

from generate_guided import predicted_membership
from protgen.data import ProteinDataset
from protgen.models.classifier import FamilyClassifier
from train_classifier import FRACTIONS, prefix_accuracy

# The guidance strengths the existing guided samples were generated with
LAMBDAS = ("0.0", "0.5", "1.0", "2.0", "5.0")


@hydra.main(config_path="config", config_name="classifier", version_base=None)
def main(cfg):
    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)

    dataset = ProteinDataset("data/train.csv")

    backbone = hydra.utils.instantiate(
        cfg.model.init,
        vocab_size=dataset.vocab_size,
        max_length=dataset.max_length,
    ).to(device)
    backbone.load_state_dict(torch.load(cfg.checkpoint_path, map_location=device))

    model = FamilyClassifier(backbone, num_classes=len(dataset.fam_to_idx)).to(device)
    model.head.load_state_dict(torch.load(cfg.classifier_path, map_location=device))
    model.eval()

    # Accuracy of the noisy predictor on partial and finished sequences
    rows = []
    for split in ("val", "test"):
        loader = DataLoader(ProteinDataset(f"data/{split}.csv"), batch_size=cfg.batch_size, shuffle=False)
        for fraction, accuracy in zip(FRACTIONS, prefix_accuracy(model, loader, device)):
            rows.append({"split": split, "fraction": fraction, "accuracy": accuracy})

    accuracy = pd.DataFrame(rows)
    accuracy.to_csv(f"results/{cfg.model.name}_classifier_accuracy.csv", index=False)
    print(accuracy.to_string(index=False))

    # Predicted target-family membership of the unguided and guided samples
    family = cfg.guidance.family
    target = dataset.fam_to_idx[family]
    sets = {"unguided": f"results/sequences/{cfg.model.name}.csv"}
    sets |= {lam: f"results/sequences/{cfg.model.name}_guided_{family}_lam{lam}.csv" for lam in LAMBDAS}

    rows = []
    for name, path in sets.items():
        sequences = pd.read_csv(path)["sequence"].fillna("").tolist()
        membership = predicted_membership(model, dataset, sequences, target, device)
        rows.append({"set": name, "family": family, "membership": membership})

    membership = pd.DataFrame(rows)
    membership.to_csv(f"results/{cfg.model.name}_guided_{family}_membership.csv", index=False)
    print(membership.to_string(index=False))


if __name__ == "__main__":
    main()