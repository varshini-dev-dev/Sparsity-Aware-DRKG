"""
04_vanilla_rgcn.py — Stage 4: vanilla R-GCN baseline (D6).

Architecture (encoder-decoder):
    encoder : input embedding table (all nodes) -> 2 x RGCNConv(64 -> 64,
              basis decomposition, num_bases=30, root weights), message passing
              over the GNN training graph from stage 01 (666,550 edges — gene-gene
              uniformly subsampled for CPU tractability, all other groups full).
    decoder : DistMult-style — score(compound, disease) = sum(z_c * r_treats * z_d)
              with a learned relation vector r_treats.

LOSS (documented choice): binary cross-entropy with logits over the 2,660
positive train treats-pairs vs the 26,600 precomputed negative pairs (stage 01:
compound-side corruption filtered against known true pairs). Positives are
weighted x10 so that each positive contributes as much as its 10 negatives
(otherwise negatives would dominate 10:1).

Validation: filtered MRR on the 729 val pairs every 5 epochs; early stopping
(patience 5 evaluations); best state is restored and saved.

Run AFTER: 01_data_prep.py
Run BEFORE: 07_evaluation.py
"""

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
POS_WEIGHT = 10          # balances 1 positive against its 10 negatives
MAX_EPOCHS = 50
EVAL_EVERY = 5
PATIENCE = 5


class VanillaRGCN(nn.Module):
    """2-layer R-GCN encoder + DistMult-style treats decoder."""

    def __init__(self, n_nodes, n_rels, dim, num_bases, treats_rel):
        super().__init__()
        self.inp = nn.Parameter(torch.empty(n_nodes, dim))
        torch.nn.init.xavier_uniform_(self.inp)
        self.conv1 = RGCNConv(dim, dim, num_relations=n_rels,
                              num_bases=num_bases)
        self.conv2 = RGCNConv(dim, dim, num_relations=n_rels,
                             num_bases=num_bases)
        self.rel_vec = nn.Parameter(torch.empty(dim))
        torch.nn.init.normal_(self.rel_vec, std=0.01)
        self.treats_rel = treats_rel

    def encode(self, edge_index, edge_type):
        x = F.relu(self.conv1(self.inp, edge_index, edge_type))
        return self.conv2(x, edge_index, edge_type)

    def score_pairs(self, z, compound_ids, disease_ids):
        return (z[compound_ids] * self.rel_vec * z[disease_ids]).sum(dim=1)


def main():
    section("04: VANILLA R-GCN — SETUP")
    require_checkpoints([CKPT_DIR / "splits_and_mappings.pt"])
    set_seed(SEED)
    ckpt = load_splits()
    n_nodes, n_rel = ckpt["n_nodes"], ckpt["n_subrel"]
    edge_index = ckpt["gnn_edge_index"]
    edge_type = ckpt["gnn_edge_type"]
    train_triples = ckpt["train_triples"]
    val_triples = ckpt["val_triples"]
    neg_compound_ids = ckpt["neg_compound_ids"]
    candidates = ckpt["compound_candidate_ids"]
    known_true_pairs = ckpt["known_true_pairs"]
    treats_rel = ckpt["treats_local_rel"]
    print(f"GNN training graph: {edge_index.shape[1]:,} edges, "
          f"{n_rel} relations, {n_nodes:,} nodes")
    print(f"Train pairs: {len(train_triples):,} | "
          f"negatives per positive: {neg_compound_ids.shape[1]} | "
          f"val pairs: {len(val_triples):,}")

    model = VanillaRGCN(n_nodes, n_rel, DIM, NUM_BASES, treats_rel)
    opt = torch.optim.Adam(model.parameters(), lr=LR,
                           weight_decay=WEIGHT_DECAY)

    pos_c = train_triples[:, 0]
    pos_d = train_triples[:, 1]
    neg_c = neg_compound_ids            # [n_train, 10]
    neg_d = pos_d.unsqueeze(1).expand(-1, neg_c.shape[1])

    def score_fn(cand_ids, disease_id):
        z = model_last_z
        d = torch.full((len(cand_ids),), int(disease_id), dtype=torch.long)
        return model.score_pairs(z, cand_ids, d)

    section("04: VANILLA R-GCN TRAINING")
    global model_last_z
    best_val, best_state, bad_evals = -1.0, None, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        z = model.encode(edge_index, edge_type)
        s_pos = model.score_pairs(z, pos_c, pos_d)
        s_neg = model.score_pairs(z, neg_c.reshape(-1), neg_d.reshape(-1))
        n_pos, n_neg = len(s_pos), len(s_neg)
        loss = (POS_WEIGHT * F.binary_cross_entropy_with_logits(
                    s_pos, torch.ones_like(s_pos)).sum()
                + F.binary_cross_entropy_with_logits(
                    s_neg, torch.zeros_like(s_neg)).sum()) / (POS_WEIGHT * n_pos + n_neg)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"Epoch {epoch:>3} | loss {float(loss):.4f}")

        if epoch % EVAL_EVERY == 0 or epoch == MAX_EPOCHS:
            model.eval()
            with torch.no_grad():
                model_last_z = model.encode(edge_index, edge_type)
            df_val, _ = evaluate_stratified(
                score_fn, val_triples, ckpt["disease_node_to_group"],
                candidates, known_true_pairs, verbose=False)
            mrr = float(df_val.loc["OVERALL", "MRR"])
            print(f"  [val] MRR={mrr:.4f} "
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
                    print(f"Early stopping at epoch {epoch} "
                          f"(best val MRR {best_val:.4f})")
                    break

    model.load_state_dict({k: v for k, v in best_state.items()
                           if k != "epoch"})

    section("04: VANILLA R-GCN — FINAL VALIDATION RESULTS")
    model.eval()
    with torch.no_grad():
        model_last_z = model.encode(edge_index, edge_type)
    df_val, _ = evaluate_stratified(
        score_fn, val_triples, ckpt["disease_node_to_group"],
        candidates, known_true_pairs)
    print_results_table("Vanilla R-GCN (validation)", df_val)

    torch.save({"state_dict": {k: v for k, v in model.state_dict().items()},
                "config": {"dim": DIM, "num_bases": NUM_BASES, "lr": LR,
                           "weight_decay": WEIGHT_DECAY,
                           "pos_weight": POS_WEIGHT, "seed": SEED,
                           "best_epoch": best_state["epoch"]},
                "best_val_mrr": best_val},
               CKPT_DIR / "vanilla_rgcn.pt")
    print(f"\nSaved: {CKPT_DIR / 'vanilla_rgcn.pt'} (best epoch "
          f"{best_state['epoch']}, val MRR {best_val:.4f})")
    print("STAGE 04 COMPLETE.")


if __name__ == "__main__":
    main()
