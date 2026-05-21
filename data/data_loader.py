# """
# Fixed NSL-KDD Loader
# =====================
# Key fixes applied:
#   1. kNN graph built per snapshot from feature similarity (not protocol grouping)
#   2. SMOTE for minority oversampling (no exact duplicates)
#   3. Edge attributes recomputed per snapshot, not cached from training indices
#   4. Non-overlapping windows (no data leakage between snapshots)
# """

import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors
from collections import Counter
import pickle

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("WARNING: imbalanced-learn not found. Run: pip install imbalanced-learn")
    print("         Falling back to class-weighted loss only (no oversampling).")


class FixedNSLKDDLoader:

    def __init__(self, train_path='data/KDDTrain+.txt', test_path='data/KDDTest+.txt'):
        self.train_path = train_path
        self.test_path  = test_path

        self.columns = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
            'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
            'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
            'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
            'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
            'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
            'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
            'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
        ]

        self.detailed_attacks = {
            'normal': 'normal',
            'back': 'dos', 'land': 'dos', 'neptune': 'dos', 'pod': 'dos',
            'smurf': 'dos', 'teardrop': 'dos', 'mailbomb': 'dos', 'apache2': 'dos',
            'processtable': 'dos', 'udpstorm': 'dos',
            'ipsweep': 'probe', 'nmap': 'probe', 'portsweep': 'probe',
            'satan': 'probe', 'mscan': 'probe', 'saint': 'probe',
            'ftp_write': 'r2l', 'guess_passwd': 'r2l', 'imap': 'r2l',
            'multihop': 'r2l', 'phf': 'r2l', 'spy': 'r2l', 'warezclient': 'r2l',
            'warezmaster': 'r2l', 'sendmail': 'r2l', 'named': 'r2l',
            'snmpgetattack': 'r2l', 'snmpguess': 'r2l', 'xlock': 'r2l',
            'xsnoop': 'r2l', 'worm': 'r2l',
            'buffer_overflow': 'u2r', 'loadmodule': 'u2r', 'perl': 'u2r',
            'rootkit': 'u2r', 'httptunnel': 'u2r', 'ps': 'u2r',
            'sqlattack': 'u2r', 'xterm': 'u2r'
        }

        self.attack_mapping = {'normal': 0, 'dos': 1, 'probe': 2, 'r2l': 3, 'u2r': 4}
        self.label_encoders = {}
        self.scaler = StandardScaler()

    # ------------------------------------------------------------------
    def load_data(self):
        print("Loading NSL-KDD dataset...")
        train_df = pd.read_csv(self.train_path, names=self.columns, header=None)
        test_df  = pd.read_csv(self.test_path,  names=self.columns, header=None)
        train_df = train_df.drop('difficulty', axis=1)
        test_df  = test_df.drop('difficulty',  axis=1)
        print(f"  Train: {len(train_df)} rows  |  Test: {len(test_df)} rows")
        return train_df, test_df

    def preprocess_labels(self, df):
        df = df.copy()
        df['attack_type']    = df['label'].str.strip().str.lower()
        df['fault_category'] = df['attack_type'].map(
            lambda x: self.attack_mapping.get(self.detailed_attacks.get(x, 'normal'), 0)
        )
        return df

    def encode_categorical(self, train_df, test_df):
        for col in ['protocol_type', 'service', 'flag']:
            le = LabelEncoder()
            le.fit(pd.concat([train_df[col], test_df[col]]))
            train_df = train_df.copy(); test_df = test_df.copy()
            train_df[col] = le.transform(train_df[col])
            test_df[col]  = le.transform(test_df[col])
            self.label_encoders[col] = le
        return train_df, test_df

    def select_features(self):
        return [
            'duration', 'protocol_type', 'service', 'flag',
            'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent',
            'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
            'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
            'num_shells', 'num_access_files',
            'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
            'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
            'srv_diff_host_rate',
            'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
            'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
            'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
            'dst_host_srv_rerror_rate'
        ]

    def normalize_features(self, train_df, test_df, feature_cols):
        train_df = train_df.copy(); test_df = test_df.copy()
        train_df[feature_cols] = self.scaler.fit_transform(train_df[feature_cols])
        test_df[feature_cols]  = self.scaler.transform(test_df[feature_cols])
        return train_df, test_df

    # ------------------------------------------------------------------
    # SMOTE — synthesised minority samples, no exact duplicates
    # ------------------------------------------------------------------
    def balance_with_smote(self, df, feature_cols):
        print("\nBalancing with SMOTE...")
        print("  Original distribution:")
        print(df['fault_category'].value_counts().sort_index().to_string(header=False))

        if not SMOTE_AVAILABLE:
            print("  SMOTE unavailable — returning raw data (use weighted loss).")
            return df

        X = df[feature_cols].values
        y = df['fault_category'].values

        min_count = pd.Series(y).value_counts().min()
        k = min(5, min_count - 1)
        if k < 1:
            print("  A class has only 1 sample — cannot apply SMOTE. Returning raw.")
            return df

        smote = SMOTE(random_state=42, k_neighbors=k)
        X_res, y_res = smote.fit_resample(X, y)

        df_balanced = pd.DataFrame(X_res, columns=feature_cols)
        df_balanced['fault_category'] = y_res

        print("\n  Balanced distribution:")
        print(df_balanced['fault_category'].value_counts().sort_index().to_string(header=False))
        return df_balanced

    # ------------------------------------------------------------------
    #  kNN graph — edges from feature similarity
    # ------------------------------------------------------------------
    def build_knn_graph(self, features: np.ndarray, k: int = 5) -> nx.Graph:
        """
        Each node = one network connection.
        Edge between i and j  =  they are among each other's k nearest
                                  neighbours in feature space (cosine distance).
        Edge weight            =  cosine similarity (higher = more similar).

        This gives the GAT meaningful structure: similar traffic patterns are
        adjacent, so the model can compare borderline cases against their
        closest feature-space neighbours.
        """
        n       = features.shape[0]
        k_use   = min(k, n - 1)

        nbrs = NearestNeighbors(n_neighbors=k_use + 1, metric='cosine').fit(features)
        distances, indices = nbrs.kneighbors(features)

        G = nx.Graph()
        G.add_nodes_from(range(n))

        for i in range(n):
            for pos in range(1, k_use + 1):          # skip pos 0 (self)
                j   = int(indices[i, pos])
                sim = max(0.0, 1.0 - float(distances[i, pos]))
                if not G.has_edge(i, j):
                    G.add_edge(i, j, weight=sim)

        return G

    # ------------------------------------------------------------------
    #  Edge attributes derived from the snapshot's own features
    # ------------------------------------------------------------------
    def graph_to_edge_tensors(self, G: nx.Graph, features: np.ndarray):
        """
        Two edge attributes per edge:
          [0] cosine similarity   (from kNN)
          [1] L2 distance         (normalised to [0, 1])

        Both directions added for bidirectional message passing.
        """
        edges = list(G.edges(data=True))

        if not edges:
            n = features.shape[0]
            return (np.stack([np.arange(n), np.arange(n)]).astype(np.int64),
                    np.ones((n, 2), dtype=np.float32))

        src, dst, attrs = [], [], []

        for u, v, data in edges:
            sim = float(data.get('weight', 0.0))
            l2  = float(np.linalg.norm(features[u] - features[v]))
            for s, d in [(u, v), (v, u)]:
                src.append(s); dst.append(d)
                attrs.append([sim, l2])

        edge_index = np.stack([src, dst]).astype(np.int64)
        edge_attr  = np.array(attrs, dtype=np.float32)

        # Normalise L2 to [0, 1]
        max_l2 = edge_attr[:, 1].max() + 1e-8
        edge_attr[:, 1] /= max_l2

        return edge_index, edge_attr

    # ------------------------------------------------------------------
    #  Non-overlapping windows — no leakage between snapshots
    # ------------------------------------------------------------------
    def create_snapshots(self, df, feature_cols, window_size=200, k_neighbours=5):
        """
        Stride = window_size  →  zero overlap  →  no shared rows between snapshots.
        Each snapshot builds its own kNN graph from its own rows.
        """
        snapshots = []
        n_rows    = len(df)

        for start in range(0, n_rows - window_size + 1, window_size):
            window   = df.iloc[start : start + window_size]
            features = window[feature_cols].values.astype(np.float32)
            labels   = window['fault_category'].values.astype(np.int64)

            G                    = self.build_knn_graph(features, k=k_neighbours)
            edge_index, edge_attr = self.graph_to_edge_tensors(G, features)

            snapshots.append({
                'features'   : features,
                'labels'     : labels,
                'edge_index' : edge_index,
                'edge_attr'  : edge_attr,
                'n_nodes'    : window_size,
                'timestep'   : start // window_size
            })

        return snapshots

    # ------------------------------------------------------------------
    def process_dataset(self, window_size=200, k_neighbours=5):
        print("\n" + "="*60)
        print("  FIXED NSL-KDD PROCESSING PIPELINE")
        print("="*60)

        train_df, test_df = self.load_data()

        print("\nProcessing labels...")
        train_df = self.preprocess_labels(train_df)
        test_df  = self.preprocess_labels(test_df)

        print("\nEncoding categoricals...")
        train_df, test_df = self.encode_categorical(train_df, test_df)

        print("\nNormalising features...")
        feature_cols = self.select_features()
        train_df, test_df = self.normalize_features(train_df, test_df, feature_cols)

        # Class weights (computed before SMOTE, on real distribution)
        counts       = Counter(train_df['fault_category'].values)
        total        = sum(counts.values())
        class_weights = [total / (5 * counts.get(c, 1)) for c in range(5)]
        print(f"\nClass weights: {[round(w, 2) for w in class_weights]}")

        # SMOTE on training data only
        train_balanced = self.balance_with_smote(train_df, feature_cols)

        print("\nBuilding snapshots (kNN graph per window)...")
        train_snaps = self.create_snapshots(train_balanced, feature_cols,
                                            window_size, k_neighbours)
        test_snaps  = self.create_snapshots(test_df,  feature_cols,
                                            window_size, k_neighbours)

        print(f"\n  Train snapshots : {len(train_snaps)}")
        print(f"  Test  snapshots : {len(test_snaps)}")

        # Save
        import os; os.makedirs('data', exist_ok=True)
        with open('data/timeseries_data.pkl', 'wb') as f: pickle.dump(train_snaps, f)
        with open('data/test_data.pkl',       'wb') as f: pickle.dump(test_snaps,  f)
        with open('data/feature_cols.pkl',    'wb') as f: pickle.dump(feature_cols, f)
        with open('data/class_weights.pkl',   'wb') as f: pickle.dump(class_weights, f)

        print("\n  Saved: data/timeseries_data.pkl, test_data.pkl,")
        print("         feature_cols.pkl, class_weights.pkl")
        print("\n" + "="*60)
        print("FIXED PROCESSING COMPLETE!")
        print("="*60)

        return train_snaps, test_snaps, class_weights


if __name__ == "__main__":
    loader = FixedNSLKDDLoader()
    train, test, weights = loader.process_dataset(window_size=200, k_neighbours=5)
    s = train[0]
    print(f"\nSnapshot[0] shapes:")
    print(f"  features   {s['features'].shape}")
    print(f"  edge_index {s['edge_index'].shape}")
    print(f"  edge_attr  {s['edge_attr'].shape}")
    print(f"  labels     {s['labels'].shape}")