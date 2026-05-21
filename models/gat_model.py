"""

================
  1. 2 GAT layers instead of 4  (prevents over-smoothing on 200-node graphs)
  2. Residual connections        (stabilises training, preserves node identity)
  3. Dropout reduced to 0.2     (less aggressive regularisation)
  4. Focal loss option           (down-weights easy examples, focuses on hard/minority)
  5. No edge_dim by default      (edge attrs optional — pass use_edge_attr=True to enable)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class FixedFaultDetectionGAT(nn.Module):
    """
    2-layer GAT with residual connections for network intrusion detection.

    Architecture rationale:
      - 2 layers: each node sees up to 2-hop neighbourhood.  On a k=5 kNN
        graph of 200 nodes, 4 layers makes every node see ~200 neighbours
        (over-smoothing: all embeddings converge to the same vector).
      - Residual skip: x = x_proj + conv(x) keeps the original features
        alive through the layers.
      - heads=4 (not 8): halves the parameter count, reduces overfitting
        on small graphs.
      - hidden_dim=64: leaner than 128; the kNN graphs are small and the
        features (38 dims) are already informative.
    """

    def __init__(
        self,
        num_node_features: int = 38,
        num_edge_features: int = 2,
        hidden_dim: int = 64,
        num_classes: int = 5,
        heads: int = 4,
        dropout: float = 0.2,
        use_edge_attr: bool = True,
    ):
        super().__init__()
        self.use_edge_attr = use_edge_attr
        edge_dim = num_edge_features if use_edge_attr else None

        # --- GAT layer 1 ---
        self.conv1 = GATConv(
            in_channels=num_node_features,
            out_channels=hidden_dim,
            heads=heads,
            dropout=dropout,
            edge_dim=edge_dim,
            add_self_loops=True,
        )
        self.bn1 = nn.BatchNorm1d(hidden_dim * heads)

        # --- GAT layer 2 (single head, concat=False averages the heads) ---
        self.conv2 = GATConv(
            in_channels=hidden_dim * heads,
            out_channels=hidden_dim,
            heads=heads,
            concat=False,           # output shape: [N, hidden_dim]  (averaged)
            dropout=dropout,
            edge_dim=edge_dim,
            add_self_loops=True,
        )
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        # --- Residual projection: map input features → hidden_dim so we can add ---
        self.res_proj = nn.Linear(num_node_features, hidden_dim)

        # --- Classifier head ---
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, num_classes)

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for m in [self.fc1, self.fc2, self.res_proj]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        edge_kw = {'edge_attr': edge_attr} if (self.use_edge_attr and edge_attr is not None) else {}

        # Residual identity
        identity = self.res_proj(x)             # [N, hidden_dim]

        # Layer 1
        h = self.conv1(x, edge_index, **edge_kw)
        h = self.bn1(h)
        h = F.elu(h)
        h = self.dropout(h)

        # Layer 2  +  residual
        h = self.conv2(h, edge_index, **edge_kw)
        h = self.bn2(h)
        h = F.elu(h + identity)                 # residual add before activation

        # Classifier
        h = self.dropout(h)
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        out = self.fc2(h)
        return out

    def get_embeddings(self, x, edge_index, edge_attr=None):
        """Return node embeddings after the final GAT layer (before classifier)."""
        edge_kw  = {'edge_attr': edge_attr} if (self.use_edge_attr and edge_attr is not None) else {}
        identity = self.res_proj(x)
        h = F.elu(self.bn1(self.conv1(x, edge_index, **edge_kw)))
        h = F.elu(self.bn2(self.conv2(h, edge_index, **edge_kw)) + identity)
        return h


# ---------------------------------------------------------------------------
# Focal Loss — down-weights easy examples, forces the model to learn
# hard/minority class patterns instead of just predicting the majority.
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):     #focal loss gives more importance for the rare and less importance to Majority classes
    """
    Focal Loss = -alpha_t * (1 - p_t)^gamma * log(p_t)

    gamma=2 is the standard value from Lin et al. 2017.
    alpha = per-class weights (same as class_weights in CrossEntropyLoss).
    """

    def __init__(self, alpha=None, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.gamma     = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float))
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt      = torch.exp(-ce_loss)                          # p_t
        focal   = (1 - pt) ** self.gamma * ce_loss            # focal factor

        if self.reduction == 'mean':
            return focal.mean()
        elif self.reduction == 'sum':
            return focal.sum()
        return focal


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torch

    model = FixedFaultDetectionGAT(
        num_node_features=38,
        num_edge_features=2,
        hidden_dim=64,
        num_classes=5,
        heads=4,
        dropout=0.2,
        use_edge_attr=True,
    )

    print("=" * 55)
    print("Fixed GAT Model")
    print("=" * 55)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters    : {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Model size (MB)     : {total * 4 / 1024**2:.2f}")

    N, E = 200, 1000
    x          = torch.randn(N, 38)
    edge_index = torch.randint(0, N, (2, E))
    edge_attr  = torch.randn(E, 2)

    out = model(x, edge_index, edge_attr)
    print(f"\nForward pass  :  {x.shape} → {out.shape}  ✅")

    # Test focal loss
    class_weights = [1.0, 2.5, 5.0, 10.0, 20.0]
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    labels    = torch.randint(0, 5, (N,))
    loss      = criterion(out, labels)
    print(f"Focal loss    :  {loss.item():.4f}  ✅")