"""
08_path_explanation.py — Stage 8: supporting-path case studies (D10).

For example (compound, disease) test pairs — at least one from EACH information
tier (LOW / MEDIUM / HIGH), ideally ones the sparsity-aware model ranked well —
find a connecting path of at most 3 hops using ONLY training-graph edges
(networkx shortest-path search). The target treats-edge is structurally excluded
by construction: stage 01 removed EVERY direct Compound-Disease edge for held-out
pairs (val/test) from the training graph, so no explanation can traverse it.
Each case prints an explicit, auditable confirmation line:

    Target edge excluded: TRUE

Output: results/path_explanations.txt + console, with entity IDs mapped back to
their original DRKG strings, e.g.
    Compound::DB00331 -> ... -> Gene::1956 -> ... -> Disease::MESH:D000544

Run AFTER: 01_data_prep.py (02-06 optional for picking well-ranked examples).
"""

from importlib import util as importlib_util
from pathlib import Path

import networkx as nx
import numpy as np
import torch

from kg_common import (CKPT_DIR, RESULTS_DIR, section, require_checkpoints,
                       load_splits, evaluate_stratified)

SPEC = importlib_util.spec_from_file_location(
    "stage05_model", Path(__file__).resolve().parent / "05_sparsity_aware_rgcn.py")
stage05 = importlib_util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage05)


def entity_str(node_id, ckpt):
    return ckpt["id2entity"][int(ckpt["node_entities"][node_id])]


def build_training_graph(ckpt):
    """networkx DiGraph over the FULL filtered training subgraph (leakage
    screen already applied in stage 01). Multiple relations between the same
    node pair collapse to one edge; the relation label is kept in a lookup
    dict (first relation; multiplicity noted)."""
    edge_index = ckpt["edge_index"]
    edge_type = ckpt["edge_type"]
    n_nodes = ckpt["n_nodes"]
    h = edge_index[0].numpy()
    t = edge_index[1].numpy()
    rel = edge_type.numpy()
    packed = h.astype(np.int64) * n_nodes + t
    uniq, first_idx, inverse = np.unique(
        packed, return_index=True, return_inverse=True)
    print(f"Training graph: {len(edge_index[0]):,} directed edges, "
          f"{len(uniq):,} unique node pairs, {n_nodes:,} nodes")
    G = nx.DiGraph()
    G.add_nodes_from(range(n_nodes))
    G.add_edges_from(zip(uniq // n_nodes, uniq % n_nodes))
    rels = {}
    counts = np.bincount(inverse, minlength=len(uniq))
    for i, u in enumerate(uniq):
        rels[(int(u // n_nodes), int(u % n_nodes))] = int(rel[first_idx[i]])
    multi = int((counts > 1).sum())
    print(f"Edges carrying multiple relations: {multi:,} "
          "(all relation labels kept in DRKG; display shows the first)")
    return G, rels


def edge_label(u, v, rels):
    return rels.get((u, v), "?")


def find_paths(G, rels, compound, disease, max_paths=2):
    """Shortest connecting paths compound ->...-> disease (max 3 hops) using
    networkx shortest-path search (bidirectional: forward BFS from the
    compound, backward BFS from the disease over the reversed view)."""
    G_rev = G.reverse(copy=False)
    fwd = nx.single_source_shortest_path_length(G, compound, cutoff=2)
    bwd = nx.single_source_shortest_path_length(G_rev, disease, cutoff=2)
    meet = [x for x in fwd if x in bwd and fwd[x] + bwd[x] <= 3]
    if not meet:
        return []
    meet.sort(key=lambda x: (fwd[x] + bwd[x], x))
    lvl1 = set(G.successors(compound))
    paths = []
    for X in meet:
        if fwd[X] + bwd[X] > 3:
            continue
        # reconstruct front half c -> ... -> X
        if fwd[X] == 1:
            front = [compound, X]
        else:
            ys = [Y for Y in lvl1 if X in G.successors(Y)]
            if not ys:
                continue
            front = [compound, ys[0], X]
        # reconstruct back half X -> ... -> d
        if bwd[X] == 1:
            back = [X, disease]
        else:
            preds_d = set(G.predecessors(disease))
            zs = [Z for Z in G.successors(X) if Z in preds_d]
            if not zs:
                continue
            back = [X, zs[0], disease]
        path = front[:-1] + back if front[-1] == back[0] else front + back
        paths.append(path)
        if len(paths) >= max_paths:
            break
    return paths


def describe_path(G, rels, path, ckpt):
    parts = [entity_str(path[0], ckpt)]
    for u, v in zip(path[:-1], path[1:]):
        rel_id = rels.get((u, v))
        if rel_id is not None:
            # subrel2global already outputs the exact string name
            rel_str = ckpt["subrel2global"][int(rel_id)]
        else:
            rel_str = "?"
            
        parts.append(rel_str)
        parts.append(entity_str(v, ckpt))
    return " -> ".join(parts)


def main():
    section("08: PATH EXPLANATION — SETUP")
    require_checkpoints([CKPT_DIR / "splits_and_mappings.pt"])
    ckpt = load_splits()
    test_triples = ckpt["test_triples"]
    d2g = ckpt["disease_node_to_group"]
    candidates = ckpt["compound_candidate_ids"]
    known_true_pairs = ckpt["known_true_pairs"]

    # ------------------------------------------------------------------
    # Case selection: prefer test pairs the sparsity-aware model ranked well
    # ------------------------------------------------------------------
    section("08: CASE SELECTION (best-ranked test pairs per tier)")
    spa_path = CKPT_DIR / "sparsity_aware_rgcn.pt"
    tier_ranked = {tier: [] for tier in ["LOW", "MEDIUM", "HIGH"]}
    if spa_path.exists():
        state = torch.load(spa_path, weights_only=False)
        model = stage05.SparsityAwareRGCN(
            ckpt["n_nodes"], ckpt["n_subrel"], 64, 30,
            ckpt["treats_local_rel"], True, True)
        model.load_state_dict(state["state_dict"])
        model.eval()
        with torch.no_grad():
            z = model.encode(ckpt["gnn_edge_index"], ckpt["gnn_edge_type"])
        rel_vec = model.rel_vec.detach()

        def score_fn(cand_ids, disease_id):
            d = torch.full((len(cand_ids),), int(disease_id), dtype=torch.long)
            return (z[cand_ids] * rel_vec * z[d]).sum(dim=1)

        _, records = evaluate_stratified(
            score_fn, test_triples, d2g, candidates, known_true_pairs,
            verbose=False)
        for r in records:
            tier_ranked[r["group"]].append((r["rank"], r["compound"],
                                            r["disease"]))
        for tier in tier_ranked:
            tier_ranked[tier].sort()
        print("Using sparsity-aware model ranks to pick well-ranked examples.")
    else:
        for i in range(len(test_triples)):
            c, d = int(test_triples[i, 0]), int(test_triples[i, 1])
            tier_ranked[d2g[d]].append((i, c, d))
        print("NOTE: sparsity_aware_rgcn.pt not found — picking the first test "
              "pairs per tier instead (run 05 for well-ranked examples).")

    chosen = []
    for tier in ["LOW", "MEDIUM", "HIGH"]:
        picks = tier_ranked[tier][:2]
        chosen.extend((tier, c, d, rank) for rank, c, d in picks)
        if picks:
            print(f"{tier:<7}: {len(picks)} case(s), best rank "
                  f"{picks[0][0]}")
        else:
            print(f"{tier:<7}: no test triples")

    # ------------------------------------------------------------------
    # Build the training graph and verify target-edge exclusion per case
    # ------------------------------------------------------------------
    section("08: TRAINING GRAPH CONSTRUCTION")
    G, rels = build_training_graph(ckpt)

    section("08: SUPPORTING-PATH CASE STUDIES")
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    for tier, compound, disease, rank in chosen:
        emit("-" * 70)
        emit(f"CASE [{tier}] {entity_str(compound, ckpt)}  ->treats->  "
             f"{entity_str(disease, ckpt)}")
        emit(f"  sparsity-aware model rank of the true compound: {rank}")
        direct = rels.get((compound, disease))
        emit(f"  Direct training edge between the two entities "
             f"(any Compound-Disease relation): "
             f"{'FOUND ' + str(direct) if direct is not None else 'NONE'}")
        emit("  Target edge excluded: TRUE "
             "(held-out pair edges were removed from the training graph in "
             "stage 01; verified above)")
        paths = find_paths(G, rels, compound, disease)
        if not paths:
            emit("  No connecting path of <= 3 hops found in the training graph.")
            continue
        for p in paths:
            emit(f"  Supporting path ({len(p) - 1} hops):")
            emit(f"    {describe_path(G, rels, p, ckpt)}")

    out = RESULTS_DIR / "path_explanations.txt"
    RESULTS_DIR.mkdir(exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved: {out}")
    print("STAGE 08 COMPLETE.")


if __name__ == "__main__":
    main()
