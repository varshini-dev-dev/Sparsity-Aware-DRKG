"""
05_sparsity_aware_rgcn.py — Stage 5: sparsity-aware R-GCN (D7, main research
contribution).

Research question:
    "Does a sparsity-aware R-GCN improve drug-candidate ranking for
     low-information diseases compared with TransE, ComplEx and vanilla R-GCN,
     while maintaining performance for better-connected diseases?"

The sparsity-aware model = the vanilla R-GCN base architecture (2 x RGCNConv,
basis decomposition, DistMult-style treats decoder) PLUS two complementary,
INDIVIDUALLY TOGGLEABLE mechanisms (constructor flags), so the SAME class can
be instantiated in every configuration needed for the ablation study (stage 06):

  a) use_self_gate=True — SELF-INFORMATION GATE per node:
         h_v' = beta_v * W_self * h_v  +  (1 - beta_v) * AGG_v
         beta_v = sigmoid(Linear(h_v))   (a learned, per-node scalar)
     Intuition: low-information nodes (few incident edges, e.g. diseases with a
     handful of treatments and compounds with little linkage) cannot trust noisy
     neighborhood aggregation; the model learns to lean on the self feature
     instead. With the flag OFF the layer degenerates to the standard
     R-GCN update h' = W_root h + AGG (identical capacity to stage 04).

  b) use_loss_reweighting=True — LOSS REWEIGHTING per training triple:
         weight(triple) = 1 / log(treatment_degree(disease) + 2)
     (normalized to mean 1 across the training set). This up-weights edges of
     low-degree (low-information) diseases so they are not drowned out by the
     many edges of well-connected diseases.

Both mechanisms ON are trained here and saved as the main proposed model.
The same training routine is shared by stage 06 for the ablation variants.

Run AFTER: 01_data_prep.py
Run BEFORE: 06_ablation.py, 07_evaluation.py
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

from kg_common import (CKPT_DIR, section, require_checkpoints,
                       load_splits, evaluate_stratified, print_results_table,
                       set_seed)

SEED = 42
DIM = 64
NUM_BASES = 30
LR = 0.01
WEIGHT_DECAY = 1e-5
POS_WEIGHT = 10
MAX_EPOCHS = 50
EVAL_EVERY = 5
PATIENCE = 5


class GatedRGCNLayer(nn.Module):
    """One R-GCN layer with an optional per-node self-information gate.

    With use_self_gate=False: out = W_self @ x + RGCNConv-aggregation(x)
    (exactly the standard R-GCN update; same capacity as stage 04).
    With use_self_gate=True: out = beta * (W_self @ x) + (1 - beta) * aggregation,
    beta = sigmoid(Linear(x)) in [0, 1] per node.
    """

    def __init__(self, dim, n_rels, num_bases, use_self_gate):
        super().__init__()
        self.agg = RGCNConv(dim, dim, num_relations=n_rels,
                            num_bases=num_bases, root_weight=False)
        self.self_proj = nn.Linear(dim, dim)
        self.use_self_gate = use_self_gate
        if use_self_gate:
            self.gate = nn.Linear(dim, 1)

    def forward(self, x, edge_index, edge_type):
        agg = self.agg(x, edge_index, edge_type)
        s = self.self_proj(x)
        if not self.use_self_gate:
            return s + agg
        beta = torch.sigmoid(self.gate(x))          # [N, 1]
        return beta * s + (1.0 - beta) * agg


class SparsityAwareRGCN(nn.Module):
    """Vanilla-R-GCN base + toggleable self-information gate + toggleable
    loss reweighting (the sparsity-aware mechanisms)."""

    def __init__(self, n_nodes, n_rels, dim, num_bases, treats_rel,
                 use_self_gate=True, use_loss_reweighting=True):
        super().__init__()
        self.inp = nn.Parameter(torch.empty(n_nodes, dim))
        torch.nn.init.xavier_uniform_(self.inp)
        self.layer1 = GatedRGCNLayer(dim, n_rels, num_bases, use_self_gate)
        self.layer2 = GatedRGCNLayer(dim, n_rels, num_bases, use_self_gate)
        self.rel_vec = nn.Parameter(torch.empty(dim))
        torch.nn.init.normal_(self.rel_vec, std=0.01)
        self.treats_rel = treats_rel
        self.use_self_gate = use_self_gate
        self.use_loss_reweighting = use_loss_reweighting

    def encode(self, edge_index, edge_type):
        x = F.relu(self.layer1(self.inp, edge_index, edge_type))
        return self.layer2(x, edge_index, edge_type)

    def score_pairs(self, z, compound_ids, disease_ids):
        return (z[compound_ids] * self.rel_vec * z[disease_ids]).sum(dim=1)


def train_sparsity_model(ckpt, use_self_gate, use_loss_reweighting, name,
                         max_epochs=MAX_EPOCHS):
    """Shared training routine (also used by stage 06 for the ablations)."""
    n_nodes, n_rel = ckpt["n_nodes"], ckpt["n_subrel"]
    edge_index = ckpt["gnn_edge_index"]
    edge_type = ckpt["gnn_edge_type"]
    train_triples = ckpt["train_triples"]
    val_triples = ckpt["val_triples"]
    neg_compound_ids = ckpt["neg_compound_ids"]
    candidates = ckpt["compound_candidate_ids"]
    known_true_pairs = ckpt["known_true_pairs"]
    treats_rel = ckpt["treats_local_rel"]
    disease_degree = ckpt["disease_node_degree"]

    model = SparsityAwareRGCN(n_nodes, n_rel, DIM, NUM_BASES, treats_rel,
                              use_self_gate=use_self_gate,
                              use_loss_reweighting=use_loss_reweighting)
    opt = torch.optim.Adam(model.parameters(), lr=LR,
                           weight_decay=WEIGHT_DECAY)

    pos_c = train_triples[:, 0]
    pos_d = train_triples[:, 1]
    neg_c = neg_compound_ids
    neg_d = pos_d.unsqueeze(1).expand(-1, neg_c.shape[1])

    # ---- loss reweighting: 1 / log(degree + 2), normalized to mean 1 ----
    w_pos = torch.tensor(
        [1.0 / math.log(disease_degree.get(int(d), 1) + 2)
         for d in pos_d], dtype=torch.float32)
    w_pos = w_pos / w_pos.mean()
    if not use_loss_reweighting:
        w_pos = torch.ones_like(w_pos)
    w_neg = w_pos.unsqueeze(1).expand(-1, neg_c.shape[1])

    last_z = [None]

    def score_fn(cand_ids, disease_id):
        z = last_z[0]
        d = torch.full((len(cand_ids),), int(disease_id), dtype=torch.long)
        return model.score_pairs(z, cand_ids, d)

    best_val, best_state, bad_evals = -1.0, None, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        z = model.encode(edge_index, edge_type)
        s_pos = model.score_pairs(z, pos_c, pos_d)
        s_neg = model.score_pairs(z, neg_c.reshape(-1), neg_d.reshape(-1))
        loss = (POS_WEIGHT * (w_pos * F.binary_cross_entropy_with_logits(
                    s_pos, torch.ones_like(s_pos))).sum()
                + (w_neg.reshape(-1) * F.binary_cross_entropy_with_logits(
                    s_neg, torch.zeros_like(s_neg))).sum()) \
            / (POS_WEIGHT * w_pos.sum() + w_neg.sum())
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"[{name}] Epoch {epoch:>3} | loss {float(loss):.4f}")

        if epoch % EVAL_EVERY == 0 or epoch == max_epochs:
            model.eval()
            with torch.no_grad():
                last_z[0] = model.encode(edge_index, edge_type)
            df_val, _ = evaluate_stratified(
                score_fn, val_triples, ckpt["disease_node_to_group"],
                candidates, known_true_pairs, verbose=False)
            mrr = float(df_val.loc["OVERALL", "MRR"])
            print(f"  [{name}][val] MRR={mrr:.4f} "
                  f"(LOW {df_val.loc['LOW', 'MRR']:.4f} "
                  f"MEDIUM {df_val.loc['MEDIUM', 'MRR']:.4f} "
                  f"HIGH {df_val.loc['HIGH', 'MRR']:.4f})")
            if mrr > best_val:
                best_val, bad_evals = mrr, 0
                best_state = {k: v.detach().clone()
                              for k, v in model.state_dict().items()}
                best_state["epoch"] = epoch
            else:
                bad_evals += 1
                if bad_evals >= PATIENCE:
                    print(f"[{name}] Early stopping at epoch {epoch} "
                          f"(best val MRR {best_val:.4f})")
                    break

    model.load_state_dict({k: v for k, v in best_state.items()
                           if k != "epoch"})
    return model, best_state, best_val


def main():
    section("05: SPARSITY-AWARE R-GCN — SETUP")
    require_checkpoints([CKPT_DIR / "splits_and_mappings.pt"])
    set_seed(SEED)
    ckpt = load_splits()
    print("Mechanisms: use_self_gate=True, use_loss_reweighting=True")
    print("(self-information gate + loss reweighting; see module docstring)")

    section("05: SPARSITY-AWARE R-GCN TRAINING")
    model, best_state, best_val = train_sparsity_model(
        ckpt, use_self_gate=True, use_loss_reweighting=True,
        name="sparsity-aware")

    section("05: SPARSITY-AWARE R-GCN — FINAL VALIDATION RESULTS")
    edge_index = ckpt["gnn_edge_index"]
    edge_type = ckpt["gnn_edge_type"]
    candidates = ckpt["compound_candidate_ids"]
    val_triples = ckpt["val_triples"]

    last_z = [None]

    def score_fn(cand_ids, disease_id):
        z = last_z[0]
        d = torch.full((len(cand_ids),), int(disease_id), dtype=torch.long)
        return model.score_pairs(z, cand_ids, d)

    model.eval()
    with torch.no_grad():
        last_z[0] = model.encode(edge_index, edge_type)
    df_val, _ = evaluate_stratified(
        score_fn, val_triples, ckpt["disease_node_to_group"],
        candidates, ckpt["known_true_pairs"])
    print_results_table("Sparsity-aware R-GCN (validation)", df_val)

    torch.save({"state_dict": {k: v for k, v in model.state_dict().items()},
                "config": {"dim": DIM, "num_bases": NUM_BASES, "lr": LR,
                           "weight_decay": WEIGHT_DECAY,
                           "pos_weight": POS_WEIGHT, "seed": SEED,
                           "use_self_gate": True,
                           "use_loss_reweighting": True,
                           "best_epoch": best_state["epoch"]},
                "best_val_mrr": best_val},
               CKPT_DIR / "sparsity_aware_rgcn.pt")
    print(f"\nSaved: {CKPT_DIR / 'sparsity_aware_rgcn.pt'} "
          f"(best epoch {best_state['epoch']}, val MRR {best_val:.4f})")
    print("STAGE 05 COMPLETE.")


if __name__ == "__main__":
    main()
