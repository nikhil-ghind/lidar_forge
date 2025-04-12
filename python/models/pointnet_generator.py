import torch
import torch.nn as nn
import torch.nn.functional as F


class TNet(nn.Module):
    def __init__(self, k: int):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        B = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.max(dim=2)[0]
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        eye = torch.eye(self.k, device=x.device).flatten().unsqueeze(0).expand(B, -1)
        x = x + eye
        return x.view(B, self.k, self.k)


class PointNetEncoder(nn.Module):
    def __init__(self, in_channels: int = 4, feature_dim: int = 1024):
        super().__init__()
        self.tnet = TNet(in_channels)
        self.conv1 = nn.Conv1d(in_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, feature_dim, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(feature_dim)

    def forward(self, x):
        T = self.tnet(x)
        x = torch.bmm(T, x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x.max(dim=2)[0]


class PointNetGenerator(nn.Module):
    """Conditional point cloud generator: condition vector -> N x (x,y,z,intensity)."""

    def __init__(self, condition_dim: int = 64, num_points: int = 16384,
                 latent_dim: int = 256, out_channels: int = 4):
        super().__init__()
        self.num_points = num_points
        self.latent_dim = latent_dim

        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_dim, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
        )
        self.fc_latent = nn.Linear(256, latent_dim * num_points // 256)

        self.decoder = nn.Sequential(
            nn.Conv1d(latent_dim, 512, 1), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Conv1d(512, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, out_channels, 1),
        )

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        B = condition.size(0)
        h = self.condition_encoder(condition)
        feat = self.fc_latent(h)
        feat = feat.view(B, self.latent_dim, -1)
        feat = F.interpolate(feat, size=self.num_points, mode='nearest')
        out = self.decoder(feat)
        return out.permute(0, 2, 1)


class PointNetDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 4):
        super().__init__()
        self.encoder = PointNetEncoder(in_channels, feature_dim=1024)
        self.head = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        feat = self.encoder(x)
        return self.head(feat)
