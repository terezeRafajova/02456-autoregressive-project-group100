import hydra
import pandas as pd
import torch

from protgen.data import ProteinDataset


@hydra.main(config_path="config", config_name="default", version_base=None)
def main(cfg):
    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)

    dataset = ProteinDataset("data/train.csv")
    model = hydra.utils.instantiate(
        cfg.model.init,
        vocab_size=dataset.vocab_size,
        max_length=dataset.max_length,
    ).to(device)

    model.load_state_dict(torch.load(cfg.checkpoint_path, map_location=device))
    model.eval()

    tokens = model.generate(cfg.num_samples, temperature=cfg.temperature)
    sequences = [dataset.decode(row) for row in tokens]

    pd.DataFrame({"sequence": sequences}).to_csv(cfg.samples_path, index=False)

if __name__ == "__main__":
    main()
