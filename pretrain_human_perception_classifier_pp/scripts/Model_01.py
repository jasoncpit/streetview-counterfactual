import torch
from torch import nn
from torchvision.models import ViT_B_16_Weights, vit_b_16


class Net(nn.Module):
    def __init__(self, num_class: int):
        super().__init__()
        self.model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
        num_fc = self.model.heads.head.in_features
        self.model.heads.head = nn.Sequential(
            nn.Linear(num_fc, 512, bias=True),
            nn.ReLU(True),
            nn.Linear(512, 256, bias=True),
            nn.ReLU(True),
            nn.Linear(256, num_class, bias=True),
        )
        nn.init.xavier_uniform_(self.model.heads.head[0].weight)
        nn.init.xavier_uniform_(self.model.heads.head[2].weight)
        nn.init.xavier_uniform_(self.model.heads.head[4].weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
