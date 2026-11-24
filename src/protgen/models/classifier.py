import torch
import torch.nn as nn
import torch.nn.functional as F


class FamilyClassifier(nn.Module):
    def __init__(self, backbone, num_classes: int = 3):
        super().__init__()

        self.backbone = backbone
        self.backbone.requires_grad_(False)

        self.head = nn.Linear(backbone.d_model, num_classes)

    def forward(self, x):
        self.backbone.eval()
        with torch.no_grad():
            h = self.backbone.features(x)

        return self.head(h)

    def loss(self, x, y, lengths):
        logits = self(x)
        mask = prefix_mask(lengths, x.shape[1])

        return F.cross_entropy(
            logits[mask],
            y[:, None].expand_as(mask)[mask],
        )

    @torch.no_grad()
    def log_prob(self, x, y):
        log_probs = F.log_softmax(self(x)[:, -1, :], dim=-1)

        return log_probs[:, y]


def prefix_mask(lengths, max_length):
    positions = torch.arange(max_length, device=lengths.device)

    return positions[None, :] <= (lengths[:, None] + 1)
