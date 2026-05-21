"""
Fixed Graph Data Preprocessor
================================
Converts the new snapshot format (each snapshot has its own edge_index/edge_attr)
to PyTorch Geometric Data objects.

Key change from old version:
  - OLD: loaded one shared graph G, applied it to every snapshot
  - NEW: each snapshot carries its own edge_index + edge_attr — just wrap them
"""

import torch
import pickle
from torch_geometric.data import Data


class FixedGraphDataPreprocessor:

    def load_data(self):
        print("Loading processed snapshots...")
        with open('data/timeseries_data.pkl', 'rb') as f:
            train_data = pickle.load(f)
        with open('data/test_data.pkl', 'rb') as f:
            test_data = pickle.load(f)
        with open('data/class_weights.pkl', 'rb') as f:
            class_weights = pickle.load(f)

        print(f"  Train snapshots : {len(train_data)}")
        print(f"  Test  snapshots : {len(test_data)}")
        return train_data, test_data, class_weights

    def snapshot_to_pyg(self, snap: dict) -> Data:
        """
        Each snapshot already has its own edge_index and edge_attr
        pre-computed from its kNN graph — no shared graph needed.
        """
        x          = torch.tensor(snap['features'],   dtype=torch.float)
        y          = torch.tensor(snap['labels'],     dtype=torch.long)
        edge_index = torch.tensor(snap['edge_index'], dtype=torch.long)
        edge_attr  = torch.tensor(snap['edge_attr'],  dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    def create_dataset(self, train_ratio: float = 0.8):
        print("\n" + "="*55)
        print("  FIXED DATASET PREPARATION")
        print("="*55)

        train_data, test_data, class_weights = self.load_data()

        # Split training into train / val (no shuffle — preserve time order)
        n_train    = int(len(train_data) * train_ratio)
        val_data   = train_data[n_train:]
        train_data = train_data[:n_train]

        print(f"\n  Split  →  train {len(train_data)} | val {len(val_data)} | test {len(test_data)}")

        print("\nConverting to PyTorch Geometric...")
        train_graphs = [self.snapshot_to_pyg(s) for s in train_data]
        val_graphs   = [self.snapshot_to_pyg(s) for s in val_data]
        test_graphs  = [self.snapshot_to_pyg(s) for s in test_data]

        print("  Done.")
        return train_graphs, val_graphs, test_graphs, class_weights


if __name__ == "__main__":
    prep = FixedGraphDataPreprocessor()
    train, val, test, weights = prep.create_dataset()

    s = train[0]
    print(f"\nSample graph:")
    print(f"  x          : {s.x.shape}")
    print(f"  edge_index : {s.edge_index.shape}")
    print(f"  edge_attr  : {s.edge_attr.shape}")
    print(f"  y          : {s.y.shape}")
    print(f"  class weights : {[round(w,2) for w in weights]}")