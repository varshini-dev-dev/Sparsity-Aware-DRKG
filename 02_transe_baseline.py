"""
02_transe_baseline.py — Stage 2: TransE baseline (D5a).

TransE from scratch in PyTorch:
    score(h, r, t) = -|| h + r - t ||_2   (higher = better)
    loss          = margin ranking: mean max(0, margin + d(pos) - d(neg))

Training graph: the same filtered training graph used by the GNN stages
(stage 01's gnn_edge_index — Compound-Disease, Compound-Gene, Gene-Disease
groups in full, Gene-Gene uniformly subsampled for CPU tractability; held-out
pairs' direct edges already removed by the leakage screen). Using the identical
training graph for every model family keeps the model comparison apples-to-
apples. The target treats edges (train split) are oversampled x20 so the target
relation is not drowned out by ~667K auxiliary triples.

NEGATIVE SAMPLING STRATEGY (documented, per the guide's risk list):
  * For treats-relation triples we corrupt ONLY THE COMPOUND (head) side with a
    uniformly random Compound:: entity — the task ranks compounds for a disease.
    Corruptions that are KNOWN true pairs (train+val+test) for that disease are
    rejected and redrawn (open-world assumption: unknown pairs are not assumed
    false, they are just candidate corruptions).
  * For all other relations we corrupt the tail with a uniformly random node,
    rejecting corruptions that are existing true triples.
  * 10 negatives per positive.

Run AFTER: 01_data_prep.py
Run BEFORE: 07_evaluation.py
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

from kg_common import (CKPT_DIR, section, require_checkpoints,
                       load_splits, evaluate_stratified, print_results_table,
                       set_seed)

SEED = 42
DIM = 64
MARGIN = 1.0
LR = 0.005
NUM_NEGATIVES = 10
BATCH_SIZE = 100_000
MAX_EPOCHS = 40
EVAL_EVERY = 10
PATIENCE = 2          # evaluations without improvement before early stop
TREATS_OVERSAMPLE = 20


def main():
    section("02: TRANSE BASELINE — SETUP")
    require_checkpoints([CKPT_DIR / "splits_and_mappings.pt"])
    set_seed(SEED)
    ckpt = load_splits()
    n_nodes = ckpt["n_nodes"]
    n_rel = ckpt["n_subrel"]
    edge_index, edge_type = ckpt["gnn_edge_index"], ckpt["gnn_edge_type"]
    train_triples = ckpt["train_triples"]
    val_triples = ckpt["val_triples"]
    candidates = ckpt["compound_candidate_ids"]
    known_true_pairs = ckpt["known_true_pairs"]
    treats_rel = ckpt["treats_local_rel"]
    print(f"Nodes: {n_nodes:,} | subgraph relations: {n_rel} | "
          f"training-graph edges: {edge_index.shape[1]:,}")
    print(f"Treats train pairs: {len(train_triples):,} "
          f"(oversampled x{TREATS_OVERSAMPLE}) | val pairs: {len(val_triples):,}")

    # ---- training list: all subgraph triples + oversampled treats pairs ----
    heads = torch.cat([edge_index[0],
                       train_triples[:, 0].repeat(TREATS_OVERSAMPLE)])
    rels = torch.cat([edge_type,
                      torch.full((len(train_triples) * TREATS_OVERSAMPLE,),
                                 treats_rel, dtype=torch.long)])
    tails = torch.cat([edge_index[1],
                       train_triples[:, 1].repeat(TREATS_OVERSAMPLE)])
    n_train = len(heads)
    print(f"Training triples per epoch: {n_train:,}")

    # ---- membership structures for filtered negative sampling ----
    packed_true = np.sort(((heads.numpy().astype(np.int64) * n_rel
                            + rels.numpy()) * n_nodes
                           + tails.numpy()))
    pair_key = np.sort(np.array(
        [c * n_nodes + d for c, d in known_true_pairs], dtype=np.int64))

    def is_true_triple(h, r, t):
        key = (h.astype(np.int64) * n_rel + r) * n_nodes + t
        pos = np.searchsorted(packed_true, key)
        pos = np.clip(pos, 0, len(packed_true) - 1)
        return packed_true[pos] == key

    def is_true_pair(c, d):
        key = c.astype(np.int64) * n_nodes + d
        pos = np.searchsorted(pair_key, key)
        pos = np.clip(pos, 0, len(pair_key) - 1)
        return pair_key[pos] == key

    cand_np = candidates.numpy()

    # ---- model ----
    ent = torch.nn.Parameter(torch.empty(n_nodes, DIM))
    rel = torch.nn.Parameter(torch.empty(n_rel, DIM))
    torch.nn.init.xavier_uniform_(ent)
    torch.nn.init.uniform_(rel, -0.01, 0.01)
    opt = torch.optim.Adam([ent, rel], lr=LR)

    def score_fn(cand_ids, disease_id):
        e_c = ent[cand_ids]
        return -torch.norm(e_c + rel[treats_rel] - ent[disease_id], dim=1)

    section("02: TRANSE TRAINING")
    best_val, best_state, bad_evals = -1.0, None, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        perm = torch.randperm(n_train)
        total_loss, n_batches = 0.0, 0
        for bstart in range(0, n_train, BATCH_SIZE):
            bidx = perm[bstart:bstart + BATCH_SIZE]
            h, r, t = heads[bidx], rels[bidx], tails[bidx]
            if len(bidx) >= 100 and (bstart // BATCH_SIZE) % 10 == 0:
                print(f"  ... epoch {epoch} batch {bstart // BATCH_SIZE + 1}"
                      f"/{math.ceil(n_train / BATCH_SIZE)}")
            # negative sampling (see strategy block in the docstring)
            neg = torch.randint(0, n_nodes, (len(bidx), NUM_NEGATIVES))
            treats_mask = (r == treats_rel)
            if treats_mask.any():
                rows = treats_mask.nonzero().flatten()
                comp_choice = cand_np[np.random.randint(
                    0, len(cand_np), (len(rows), NUM_NEGATIVES))]
                neg[rows] = torch.from_numpy(comp_choice)
            # redraw negatives that hit known true triples/pairs
            h_np, r_np, t_np = h.numpy(), r.numpy(), t.numpy()
            neg_np = neg.numpy()
            for k in range(NUM_NEGATIVES):
                for _ in range(10):  # a handful of redraw rounds suffices
                    rows_t = treats_mask.numpy()
                    bad_t = rows_t & is_true_pair(neg_np[:, k], t_np)
                    bad_o = (~rows_t) & is_true_triple(h_np, r_np, neg_np[:, k])
                    bad = bad_t | bad_o
                    if not bad.any():
                        break
                    nb = np.where(bad)[0]
                    newneg = np.random.randint(0, n_nodes, len(nb))
                    nb_treats = rows_t[nb].astype(bool)
                    if nb_treats.any():
                        newneg[nb_treats] = cand_np[np.random.randint(
                            0, len(cand_np), int(nb_treats.sum()))]
                    neg_np[nb, k] = newneg
            neg = torch.from_numpy(neg_np)
            # margin ranking loss (A = h + r computed once, reused for the
            # positive and all negative distances)
            A = ent[h] + rel[r]
            d_pos = torch.norm(A - ent[t], dim=1)
            d_neg = torch.norm(A.unsqueeze(1) - ent[neg], dim=2)
            loss = F.relu(MARGIN + d_pos.unsqueeze(1) - d_neg).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss)
            n_batches += 1
        with torch.no_grad():
            ent.copy_(F.normalize(ent, dim=1))  # unit-sphere projection
        print(f"Epoch {epoch:>3} | loss {total_loss / n_batches:.4f}")

        if epoch % EVAL_EVERY == 0:
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
                best_state = {"ent": ent.detach().clone(),
                              "rel": rel.detach().clone(),
                              "epoch": epoch}
            else:
                bad_evals += 1
                if bad_evals >= PATIENCE:
                    print(f"Early stopping at epoch {epoch} "
                          f"(best val MRR {best_val:.4f})")
                    break

    if best_state is not None:
        with torch.no_grad():
            ent.copy_(best_state["ent"])
            rel.copy_(best_state["rel"])

    section("02: TRANSE — FINAL VALIDATION RESULTS")
    df_val, _ = evaluate_stratified(
        score_fn, val_triples, ckpt["disease_node_to_group"],
        candidates, known_true_pairs)
    print_results_table("TransE (validation)", df_val)

    out = {"ent": ent.detach(), "rel": rel.detach(), "dim": DIM,
           "treats_rel": treats_rel, "best_val_mrr": best_val,
           "config": {"margin": MARGIN, "lr": LR, "dim": DIM,
                       "num_negatives": NUM_NEGATIVES,
                       "treats_oversample": TREATS_OVERSAMPLE,
                       "seed": SEED}}
    torch.save(out, CKPT_DIR / "transe_model.pt")
    print(f"\nSaved: {CKPT_DIR / 'transe_model.pt'} (best epoch "
          f"{best_state['epoch'] if best_state else '-'}, val MRR {best_val:.4f})")
    print("STAGE 02 COMPLETE.")


if __name__ == "__main__":
    main()
