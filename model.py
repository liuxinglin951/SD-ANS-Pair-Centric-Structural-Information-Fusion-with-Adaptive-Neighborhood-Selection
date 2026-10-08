import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv


def mlp(input_dim, hidden_dim):
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                         nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                         nn.Linear(hidden_dim, 1))


class SDANS(nn.Module):
    def __init__(self, input_dim, hidden_dim, selector_dim, layers, dropout, tau):
        super().__init__()
        if min(input_dim, hidden_dim, selector_dim, layers) < 1 or tau <= 0:
            raise ValueError("Dimensions, depth and tau must be positive.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must lie in [0, 1).")
        self.encoder = nn.ModuleList([
            GCNConv(input_dim if index == 0 else hidden_dim, hidden_dim)
            for index in range(layers)
        ])
        self.selector = mlp(hidden_dim, selector_dim)
        self.predictor = mlp(hidden_dim, hidden_dim)
        self.dropout = dropout
        self.tau = tau

    def encode(self, x, edge_index, distance, community):
        if self.training:
            signs = 2 * torch.randint(2, (community.size(1),), device=x.device) - 1
            community = community * signs
        h = torch.cat((x, distance, community), dim=-1)
        outputs = []
        for index, layer in enumerate(self.encoder):
            h = layer(h, edge_index)
            outputs.append(h)
            if index + 1 < len(self.encoder):
                h = F.dropout(F.relu(h), p=self.dropout, training=self.training)
        return torch.stack(outputs, dim=1)

    def decode(self, h, pairs, hard=False):
        left, right = h[pairs[:, 0]], h[pairs[:, 1]]
        candidates = (left[:, :, None, :] * right[:, None, :, :]).flatten(1, 2)
        scores = self.selector(candidates).squeeze(-1)
        if hard:
            selected = scores.argmax(dim=1)
            z = candidates[torch.arange(len(pairs), device=h.device), selected]
        else:
            weights = (scores / self.tau).softmax(dim=1)
            z = (weights.unsqueeze(-1) * candidates).sum(dim=1)
        return self.predictor(z).squeeze(-1)

    def forward(self, x, edge_index, distance, community, pairs, hard=False):
        h = self.encode(x, edge_index, distance, community)
        return self.decode(h, pairs, hard=hard)


@torch.no_grad()
def predict(model, graph, pairs, batch_size=1024):
    model.eval()
    h = model.encode(*graph)
    return torch.cat([model.decode(h, batch, hard=True).sigmoid()
                      for batch in pairs.split(batch_size)])
