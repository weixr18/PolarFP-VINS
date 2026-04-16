"""Convert SuperPoint .pth weights to TorchScript .pt for LibTorch C++ inference."""

import torch
import torch.nn as nn
import torch.nn.functional as F

WEIGHTS = "superpoint_v1.pth"
OUTPUT  = "superpoint_v1.pt"


class SuperPoint(nn.Module):
    """Standard SuperPoint architecture (encoder + detector + descriptor heads)."""

    def __init__(self):
        super().__init__()
        # Shared encoder
        self.conv1a = nn.Conv2d(1, 64, 3, 1, 1)
        self.conv1b = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv2a = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv2b = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv3a = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv3b = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv4a = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv4b = nn.Conv2d(128, 128, 3, 1, 1)

        # Detector head: 128 -> 256 (3x3) -> 65 (1x1), output at 1/8 resolution
        self.convPa = nn.Conv2d(128, 256, 3, 1, 1)
        self.convPb = nn.Conv2d(256, 65, 1, 1, 0)

        # Descriptor head: 128 -> 256 (3x3) -> 256 (1x1), output at 1/8 resolution, L2-normalized
        self.convDa = nn.Conv2d(128, 256, 3, 1, 1)
        self.convDb = nn.Conv2d(256, 256, 1, 1, 0)

    def forward(self, x):
        # Encoder
        x = F.relu(self.conv1a(x))
        x = F.relu(self.conv1b(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2a(x))
        x = F.relu(self.conv2b(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3a(x))
        x = F.relu(self.conv3b(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv4a(x))
        x = F.relu(self.conv4b(x))

        # Detector head
        c1 = F.relu(self.convPa(x))
        semi = self.convPb(c1)

        # Descriptor head
        c2 = F.relu(self.convDa(x))
        desc = self.convDb(c2)
        desc = F.normalize(desc, p=2, dim=1)

        return semi, desc


if __name__ == "__main__":
    model = SuperPoint()

    # Load weights
    state_dict = torch.load(WEIGHTS, map_location="cpu")
    model.load_state_dict(state_dict)
    print("Weights loaded OK")

    # Trace with dummy input (480x640 grayscale)
    dummy = torch.randn(1, 1, 480, 640)
    traced = torch.jit.trace(model, dummy)
    torch.jit.save(traced, OUTPUT)
    print(f"Saved TorchScript model to {OUTPUT}")

    # Verify
    loaded = torch.jit.load(OUTPUT)
    out = loaded(dummy)
    print(f"Detector output shape: {out[0].shape}")
    print(f"Descriptor output shape: {out[1].shape}")
