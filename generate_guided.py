import math

import hydra
import pandas as pd
import torch
import torch.nn.functional as F

from protgen.data import AA, ProteinDataset
from protgen.models.classifier import FamilyClassifier


@hydra.main(config_path="config", config_name="classifier", version_base=None)
def main(cfg):
    torch.manual_seed(cfg.seed)

    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)

    dataset = ProteinDataset("data/train.csv")
    target = dataset.fam_to_idx[cfg.guidance.family]

    backbone = hydra.utils.instantiate(
        cfg.model.init,
        vocab_size=dataset.vocab_size,
        max_length=dataset.max_length,
    ).to(device)
    backbone.load_state_dict(torch.load(cfg.checkpoint_path, map_location=device))

    model = FamilyClassifier(backbone, num_classes=len(dataset.fam_to_idx)).to(device)
    model.head.load_state_dict(torch.load(cfg.classifier_path, map_location=device))
    model.eval()

    candidates = torch.tensor(
        [dataset.vocab["<EOS>"]] + [dataset.vocab[residue] for residue in AA],
        device=device,
    )

    sequences = []
    remaining = cfg.guidance.num_samples
    while remaining > 0:
        n = min(cfg.guidance.chunk_size, remaining)
        tokens = generate_guided(
            model,
            dataset,
            candidates,
            n=n,
            target=target,
            lam=cfg.guidance.lam,
            temperature=cfg.guidance.temperature,
            device=device,
        )
        sequences += [dataset.decode(row) for row in tokens]
        remaining -= n
        print(f"{len(sequences)} / {cfg.guidance.num_samples} sequences")

    pd.DataFrame({"sequence": sequences}).to_csv(cfg.samples_path, index=False)

    membership = predicted_membership(model, dataset, sequences, target, device)
    print(f"Wrote {cfg.samples_path}")
    print(f"Predicted {cfg.guidance.family} membership: {membership:.3f}")


@torch.no_grad()
def generate_guided(model, dataset, candidates, *, n, target, lam, temperature, device):
    bos_id, eos_id, pad_id = dataset.vocab["<BOS>"], dataset.vocab["<EOS>"], dataset.vocab["<PAD>"]

    x = torch.full((n, 1), bos_id, dtype=torch.long, device=device)
    done = torch.zeros(n, dtype=torch.bool, device=device)

    for _ in range(dataset.max_length - 1):
        log_probs = F.log_softmax(model.backbone(x)[:, -1, :] / temperature, dim=-1)

        prefixes = torch.cat(
            [
                x.repeat_interleave(len(candidates), dim=0),
                candidates.repeat(n)[:, None],
            ],
            dim=1,
        )
        guidance = model.log_prob(prefixes, target).view(n, len(candidates))

        # Everything that is not a continuation of the protein, <BOS> and <PAD>, stays impossible
        logits = torch.full_like(log_probs, -math.inf)
        logits[:, candidates] = log_probs[:, candidates] + lam * guidance

        nxt = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1).squeeze(1)
        nxt = torch.where(done, torch.full_like(nxt, pad_id), nxt)
        x = torch.cat([x, nxt[:, None]], dim=1)

        done = done | (nxt == eos_id)
        if done.all():
            break

    return x


@torch.no_grad()
def predicted_membership(model, dataset, sequences, target, device):
    # A sample that never wrote <EOS> can be one residue longer than the longest protein
    sequences = [sequence[: dataset.max_length - 2] for sequence in sequences]
    tokens = torch.stack([dataset.encode(sequence) for sequence in sequences]).to(device)
    lengths = torch.tensor([len(sequence) for sequence in sequences], device=device)

    # The finished sequence is the prefix that ends at <EOS>
    predictions = model(tokens).argmax(dim=-1).gather(1, (lengths + 1)[:, None]).squeeze(1)

    return (predictions == target).float().mean().item()


if __name__ == "__main__":
    main()
