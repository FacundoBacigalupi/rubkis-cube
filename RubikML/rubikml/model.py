import torch
import torch.nn as nn


class RubikPolicyNet(nn.Module):
    def __init__(self, input_size: int = 324, num_moves: int = 12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_moves),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RubikPolicyValueNet(nn.Module):
    def __init__(self, input_size: int = 324, num_moves: int = 12):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_size, 1024), nn.ReLU(),
            nn.Linear(1024, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, num_moves),
        )
        self.value_head = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor):
        z = self.backbone(x)
        policy_logits = self.policy_head(z)
        value = self.value_head(z).squeeze(-1)
        return policy_logits, value


class _ResBlock(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(size, size),
            nn.LayerNorm(size),
            nn.ReLU(),
            nn.Linear(size, size),
            nn.LayerNorm(size),
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.net(x))


class RubikPolicyValueNetV2(nn.Module):
    """Residual network with LayerNorm — better gradient flow for deep scrambles.

    Default (hidden=1024, num_blocks=4) has ~9.5 M params vs ~2.2 M for V1.
    Trains from scratch; value target must use a fixed normalisation constant
    (value_norm in TrainConfig) so the scale is stable across curriculum phases.
    """

    def __init__(
        self,
        input_size: int = 324,
        num_moves: int = 12,
        hidden: int = 1024,
        num_blocks: int = 4,
    ):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )
        self.blocks = nn.ModuleList([_ResBlock(hidden) for _ in range(num_blocks)])
        self.policy_head = nn.Sequential(
            nn.Linear(hidden, 512),
            nn.ReLU(),
            nn.Linear(512, num_moves),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor):
        z = self.embed(x)
        for block in self.blocks:
            z = block(z)
        policy_logits = self.policy_head(z)
        value = self.value_head(z).squeeze(-1)
        return policy_logits, value


def build_model(model_type: str = "policy") -> nn.Module:
    if model_type == "policy":
        return RubikPolicyNet()
    if model_type == "policy_value":
        return RubikPolicyValueNet()
    if model_type == "policy_value_v2":
        return RubikPolicyValueNetV2()
    raise ValueError(f"Unknown model_type: {model_type!r}")
