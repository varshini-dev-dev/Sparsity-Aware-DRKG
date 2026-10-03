"""
03_complex_baseline.py — Stage 3: ComplEx baseline (D5b).

ComplEx from scratch in PyTorch:
    score(h, r, t) = Re( <h, r, conj(t)> )
                   = sum_d [ (h_re*r_re - h_im*r_im) * t_re
                           + (h_re*r_im + h_im*r_re) * t_im ]
    loss          = binary cross-entropy on sigmoid(score) with 1 positive and
                    10 sampled negatives per positive.

Training data / negative sampling: identical strategy to stage 02 (see the
docstring there): the same filtered training graph as the GNN stages
(gnn_edge_index — gene-gene uniformly subsampled in stage 01, all other
groups full) + treats train pairs oversampled x20; compound-side corruption
for treats triples (filtered against known true pairs), tail-side corruption
for auxiliary relations (filtered against true triples).

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
LR = 0.005
WEIGHT_DECAY = 1e-4
NUM_NEGATIVES = 10
BATCH_SIZE = 100_000
MAX_EPOCHS = 40
EVAL_EVERY = 10
PATIENCE = 2
TREATS_OVERSAMPLE = 20


def main():
    section("03: COMPLEX BASELINE — SETUP")
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
          f"subgraph edges: {edge_index.shape[1]:,}")

    heads = torch.cat([edge_index[0],
                       train_triples[:, 0].repeat(TREATS_OVERSAMPLE)])
    rels = torch.cat([edge_type,
                      torch.full((len(train_triples) * TREATS_OVERSAMPLE,),
                                 treats_rel, dtype=torch.long)])
    tails = torch.cat([edge_index[1],
                       train_triples[:, 1].repeat(TREATS_OVERSAMPLE)])
    n_train = len(heads)
    print(f"Training triples per epoch: {n_train:,}")

    packed_true = np.sort(((heads.numpy().astype(np.int64) * n_rel
                            + rels.numpy()) * n_nodes
                           + tails.numpy()))
    pair_key = np.sort(np.array(
        [c * n_nodes + d for c, d in known_true_pairs], dtype=np.int64))

    def is_true_triple(h, r, t):
        key = (h.astype(np.int64) * n_rel + r) * n_nodes + t
        pos = np.clip(np.searchsorted(packed_true, key), 0,
                      len(packed_true) - 1)
        return packed_true[pos] == key

    def is_true_pair(c, d):
        key = c.astype(np.int64) * n_nodes + d
        pos = np.clip(np.searchsorted(pair_key, key), 0, len(pair_key) - 1)
        return pair_key[pos] == key

    cand_np = candidates.numpy()

    # ---- model: complex embeddings as (real, imaginary) halves ----
    ent_re = torch.nn.Parameter(torch.empty(n_nodes, DIM))
    ent_im = torch.nn.Parameter(torch.empty(n_nodes, DIM))
    rel_re = torch.nn.Parameter(torch.empty(n_rel, DIM))
    rel_im = torch.nn.Parameter(torch.empty(n_rel, DIM))
    
    # FIX: Scale down initialization to prevent gradient saturation
    for p in [ent_re, ent_im, rel_re, rel_im]:
        torch.nn.init.normal_(p, std=0.01)
        
    params = [ent_re, ent_im, rel_re, rel_im]
    opt = torch.optim.Adam(params, lr=LR, weight_decay=WEIGHT_DECAY)

    def score_batch(h, r, t):
        hr, hi = ent_re[h], ent_im[h]
        rr, ri = rel_re[r], rel_im[r]
        tr, ti = ent_re[t], ent_im[t]
        return ((hr * rr - hi * ri) * tr + (hr * ri + hi * rr) * ti).sum(dim=-1)

    def score_fn(cand_ids, disease_id):
        return score_batch(cand_ids,
                           torch.full((len(cand_ids),), treats_rel,
                                      dtype=torch.long),
                           torch.full((len(cand_ids),), disease_id,
                                      dtype=torch.long))

    section("03: COMPLEX TRAINING")
    best_val, best_state, bad_evals = -1.0, None, 0
    n_batches_total = math.ceil(n_train / BATCH_SIZE)
    for epoch in range(1, MAX_EPOCHS + 1):
        perm = torch.randperm(n_train)
        total_loss, n_batches = 0.0, 0
        for bstart in range(0, n_train, BATCH_SIZE):
            bidx = perm[bstart:bstart + BATCH_SIZE]
            h, r, t = heads[bidx], rels[bidx], tails[bidx]
            if len(bidx) >= 100 and (bstart // BATCH_SIZE) % 10 == 0:
                print(f"  ... epoch {epoch} batch {bstart // BATCH_SIZE + 1}"
                      f"/{n_batches_total}")
            # ---- negative sampling (same strategy as stage 02) ----
            neg = torch.randint(0, n_nodes, (len(bidx), NUM_NEGATIVES))
            treats_mask = (r == treats_rel)
            if treats_mask.any():
                rows = treats_mask.nonzero().flatten()
                comp_choice = cand_np[np.random.randint(
                    0, len(cand_np), (len(rows), NUM_NEGATIVES))]
                neg[rows] = torch.from_numpy(comp_choice)
            h_np, r_np, t_np = h.numpy(), r.numpy(), t.numpy()
            neg_np = neg.numpy()
            rows_t = treats_mask.numpy()
            for k in range(NUM_NEGATIVES):
                for _ in range(10):
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
            # ---- BCE on 1 positive + NUM_NEGATIVES negatives ----
            pos_score = score_batch(h, r, t)
            neg_score = score_batch(
                h.unsqueeze(1).expand(-1, NUM_NEGATIVES),
                r.unsqueeze(1).expand(-1, NUM_NEGATIVES), neg)
            loss = (F.binary_cross_entropy_with_logits(
                        pos_score, torch.ones_like(pos_score)).sum()
                    + F.binary_cross_entropy_with_logits(
                        neg_score, torch.zeros_like(neg_score)).sum()) \
                / (len(bidx) * (1 + NUM_NEGATIVES))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss)
            n_batches += 1
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
                best_state = {p_name: p.detach().clone()
                              for p_name, p in zip(
                                  ["ent_re", "ent_im", "rel_re", "rel_im"],
                                  params)}
                best_state["epoch"] = epoch
            else:
                bad_evals += 1
                if bad_evals >= PATIENCE:
                    print(f"Early stopping at epoch {epoch} "
                          f"(best val MRR {best_val:.4f})")
                    break

    if best_state is not None:
        with torch.no_grad():
            ent_re.copy_(best_state["ent_re"])
            ent_im.copy_(best_state["ent_im"])
            rel_re.copy_(best_state["rel_re"])
            rel_im.copy_(best_state["rel_im"])
       

    section("03: COMPLEX — FINAL VALIDATION RESULTS")
    df_val, _ = evaluate_stratified(
        score_fn, val_triples, ckpt["disease_node_to_group"],
        candidates, known_true_pairs)
    print_results_table("ComplEx (validation)", df_val)

    out = {"ent_re": ent_re.detach(), "ent_im": ent_im.detach(),
           "rel_re": rel_re.detach(), "rel_im": rel_im.detach(),
           "dim": DIM, "treats_rel": treats_rel, "best_val_mrr": best_val,
           "config": {"lr": LR, "dim": DIM, "num_negatives": NUM_NEGATIVES,
                       "treats_oversample": TREATS_OVERSAMPLE, "seed": SEED}}
    torch.save(out, CKPT_DIR / "complex_model.pt")
    print(f"\nSaved: {CKPT_DIR / 'complex_model.pt'} (best epoch "
          f"{best_state['epoch'] if best_state else '-'}, val MRR {best_val:.4f})")
    print("STAGE 03 COMPLETE.")


if __name__ == "__main__":
    main()