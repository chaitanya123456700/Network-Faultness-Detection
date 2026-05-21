"""
Fixed Training Script
======================
Changes from old version:
  1. Uses FocalLoss (with class weights) instead of plain CrossEntropyLoss
  2. Uses FixedFaultDetectionGAT (2 layers, residuals, dropout=0.2)
  3. Loads class weights from data/class_weights.pkl
  4. Cosine annealing LR scheduler (better than StepLR for this task)
  5. Early stopping based on val F1 (not val loss — F1 better reflects minority class perf)
  6. Per-epoch minority-class F1 printed so you can track R2L/U2R progress
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, classification_report

from utils.preprocessing import FixedGraphDataPreprocessor
from models.gat_model     import FixedFaultDetectionGAT, FocalLoss

# ── Config ──────────────────────────────────────────────────────────────────
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE   = 4
EPOCHS       = 80
LR           = 3e-4
HIDDEN_DIM   = 64
HEADS        = 4
DROPOUT      = 0.2
PATIENCE     = 15           # early stopping patience (epochs without val-F1 improvement)
USE_EDGE_ATTR = True
CLASS_NAMES  = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']
# ────────────────────────────────────────────────────────────────────────────


def train_epoch(model, loader, criterion, optimiser):
    model.train()
    total_loss, total_correct, total_nodes = 0.0, 0, 0

    for batch in loader:
        batch = batch.to(DEVICE)
        optimiser.zero_grad()

        edge_attr = batch.edge_attr if USE_EDGE_ATTR else None
        out  = model(batch.x, batch.edge_index, edge_attr, batch.batch)
        loss = criterion(out, batch.y)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()

        total_loss    += loss.item() * batch.num_nodes
        total_correct += (out.argmax(dim=1) == batch.y).sum().item()
        total_nodes   += batch.num_nodes

    return total_loss / total_nodes, total_correct / total_nodes


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    for batch in loader:
        batch = batch.to(DEVICE)
        edge_attr = batch.edge_attr if USE_EDGE_ATTR else None
        out  = model(batch.x, batch.edge_index, edge_attr, batch.batch)
        loss = criterion(out, batch.y)

        total_loss += loss.item() * batch.num_nodes
        all_preds.extend(out.argmax(dim=1).cpu().numpy())
        all_labels.extend(batch.y.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    n          = len(all_labels)

    acc   = (all_preds == all_labels).mean()
    f1_w  = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1_m  = f1_score(all_labels, all_preds, average='macro',    zero_division=0)
    f1_pc = f1_score(all_labels, all_preds, average=None,       zero_division=0,
                     labels=list(range(5)))

    return total_loss / n, acc, f1_w, f1_m, f1_pc, all_preds, all_labels


def main():
    os.makedirs('outputs', exist_ok=True)
    print(f"\nDevice: {DEVICE}")

    # ── Data ────────────────────────────────────────────────────────────────
    prep  = FixedGraphDataPreprocessor()
    train_graphs, val_graphs, test_graphs, class_weights = prep.create_dataset()

    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_graphs,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_graphs,  batch_size=BATCH_SIZE, shuffle=False)

    # ── Model ───────────────────────────────────────────────────────────────
    num_features = train_graphs[0].x.shape[1]
    num_edge_f   = train_graphs[0].edge_attr.shape[1] if USE_EDGE_ATTR else 0

    model = FixedFaultDetectionGAT(
        num_node_features=num_features,
        num_edge_features=num_edge_f,
        hidden_dim=HIDDEN_DIM,
        num_classes=5,
        heads=HEADS,
        dropout=DROPOUT,
        use_edge_attr=USE_EDGE_ATTR,
    ).to(DEVICE)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Loss: FocalLoss with class weights ──────────────────────────────────
    criterion = FocalLoss(alpha=class_weights, gamma=2.0).to(DEVICE)

    # ── Optimiser + LR scheduler ────────────────────────────────────────────
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=EPOCHS, eta_min=1e-6)

    # ── Training loop ───────────────────────────────────────────────────────
    best_val_f1 = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [],
               'val_f1_weighted': [], 'val_f1_r2l': [], 'val_f1_u2r': []}

    print("\n" + "="*70)
    print(f"{'Ep':>4}  {'Tr-Loss':>8}  {'Tr-Acc':>7}  {'Va-Loss':>8}  "
          f"{'Va-Acc':>7}  {'Va-F1w':>7}  {'R2L':>6}  {'U2R':>6}  {'LR':>8}")
    print("="*70)

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimiser)
        va_loss, va_acc, va_f1w, va_f1m, va_f1pc, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step()

        r2l = va_f1pc[3] if len(va_f1pc) > 3 else 0.0
        u2r = va_f1pc[4] if len(va_f1pc) > 4 else 0.0
        lr  = optimiser.param_groups[0]['lr']

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(va_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(va_acc)
        history['val_f1_weighted'].append(va_f1w)
        history['val_f1_r2l'].append(r2l)
        history['val_f1_u2r'].append(u2r)

        print(f"{epoch:>4}  {tr_loss:>8.4f}  {tr_acc:>7.4f}  {va_loss:>8.4f}  "
              f"{va_acc:>7.4f}  {va_f1w:>7.4f}  {r2l:>6.4f}  {u2r:>6.4f}  {lr:>8.2e}")

        # Early stopping on weighted F1
        if va_f1w > best_val_f1:
            best_val_f1 = va_f1w
            patience_counter = 0
            torch.save(model.state_dict(), 'outputs/best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} (no val-F1 improvement for {PATIENCE} epochs).")
                break

    # ── Training curves ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history['train_loss'], label='Train loss')
    axes[0].plot(history['val_loss'],   label='Val loss')
    axes[0].set_title('Loss'); axes[0].legend()

    axes[1].plot(history['train_acc'], label='Train acc')
    axes[1].plot(history['val_acc'],   label='Val acc')
    axes[1].set_title('Accuracy'); axes[1].legend()

    axes[2].plot(history['val_f1_weighted'], label='Val F1 (weighted)')
    axes[2].plot(history['val_f1_r2l'],      label='Val F1 R2L')
    axes[2].plot(history['val_f1_u2r'],      label='Val F1 U2R')
    axes[2].set_title('Minority-class F1'); axes[2].legend()

    plt.tight_layout()
    plt.savefig('outputs/training_curves.png', dpi=150)
    plt.close()
    print("\nTraining curves saved to outputs/training_curves.png")

    # ── Test evaluation ──────────────────────────────────────────────────────
    model.load_state_dict(torch.load('outputs/best_model.pth', map_location=DEVICE))
    te_loss, te_acc, te_f1w, te_f1m, te_f1pc, preds, labels = evaluate(model, test_loader, criterion)

    print("\n" + "="*55)
    print("TEST RESULTS")
    print("="*55)
    print(f"  Loss              : {te_loss:.4f}")
    print(f"  Accuracy          : {te_acc:.4f}")
    print(f"  F1 Weighted       : {te_f1w:.4f}")
    print(f"  F1 Macro          : {te_f1m:.4f}")

    print("\nPer-class F1:")
    for i, name in enumerate(CLASS_NAMES):
        f1 = te_f1pc[i] if i < len(te_f1pc) else 0.0
        bar = '█' * int(f1 * 30)
        print(f"  {name:<12} {f1:.4f}  {bar}")

    print("\nFull classification report:")
    print(classification_report(labels, preds, target_names=CLASS_NAMES, zero_division=0))

    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('outputs/confusion_matrix.png', dpi=150)
    plt.close()
    print("\nConfusion matrix saved to outputs/confusion_matrix.png")
    print("\nDone! Check outputs/")


if __name__ == "__main__":
    main()