import torch
from torch import nn


class UnitNorm(nn.Module):
    def __init__(
        self,
        num_features: int = 512,
        dim: int = -1,
        affine: bool = False,
        k: float | None = None,
        learnable_k: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.D = num_features
        # self.linear = nn.Linear(num_features, num_features) if affine else nn.Identity()
        if affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))
        else:
            self.gamma = None
            self.beta = None
        if k is not None:
            if learnable_k:
                self.k = nn.Parameter(torch.ones(num_features) * k)
            else:
                self.k = k
        else:
            self.k = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_hat = x / torch.linalg.norm(x, dim=self.dim, keepdim=True)
        if self.gamma is not None and self.beta is not None:
            x_hat = x_hat * self.gamma
            x_hat = x_hat + self.beta
        if self.k is not None:
            x_hat = x_hat * (self.D ** (self.k / 2))
        return x_hat


class BatchNormWrapper(nn.Module):
    def __init__(self, batch_norm) -> None:
        super().__init__()
        self.batch_norm = batch_norm

    def forward(self, x: torch.Tensor):
        match len(x.shape):
            case 4:
                return self.batch_norm(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            case 3:
                return self.batch_norm(x.permute(0, 2, 1)).permute(0, 2, 1)
            case 2:
                return self.batch_norm(x)
            case _:
                raise ValueError("Only support 2d, 3d and 4d input")


class SeasonalLayernorm(nn.Module):
    """
    Special designed layernorm for the seasonal part
    """

    def __init__(self, channels):
        super().__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias
