"""
01_data_prep.py — Stage 1: DRKG loading, cohort selection, tiers, splits, checkpoints.

Research question (verbatim from the academic guide):
    "Does a sparsity-aware R-GCN improve drug-candidate ranking for low-information
     diseases compared with TransE, ComplEx and vanilla R-GCN, while maintaining
     performance for better-connected diseases?"

Task direction (critical): given a DISEASE, rank candidate DRUGS (Compound:: entities).
Candidates for ranking are ALWAYS Compound:: entities, never Disease:: entities.

Deliverables covered here: D1 (cohort + justification), D2 (subgraph statistics),
D3 (information tiers), D4 (leakage-controlled split + negative-sampling strategy).

Run AFTER: nothing (this is the first stage).
Run BEFORE: 02, 03, 04, 05, 06, 07, 08 (all need checkpoints/splits_and_mappings.pt).
"""

import os
import math
import random
import collections
from pathlib import Path

import numpy as np
import torch

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DRKG_TSV = BASE_DIR / "raw" / "drkg.tsv"
CKPT_DIR = BASE_DIR / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"

SEED = 42
TREATS_RELATION = "DRUGBANK::treats::Compound:Disease"
# Therapeutic ("treats-like") relation family used for leakage screening of the
# training graph. GNBR categories excluded: C, J, Mp, Pa, Sa (non-therapeutic or
# ambiguous associations, e.g. side effects / pathogenesis / pharmacokinetics).
TREATS_LIKE_FAMILY = {
    "DRUGBANK::treats::Compound:Disease",
    "GNBR::T::Compound:Disease",       # GNBR "treats"
    "GNBR::Pr::Compound:Disease",      # GNBR "prevents / suppresses"
    "Hetionet::CtD::Compound:Disease", # Hetionet "Compound treats Disease"
}
# Eligibility: a disease needs >= MIN_TREATS_DEGREE distinct known treatments so that a
# per-disease 70/15/15 split leaves at least one edge in each of train/val/test.
MIN_TREATS_DEGREE = 3
NUM_NEGATIVES = 10  # per positive train edge (compound-side corruption)
# GNN message passing over the FULL 2.77M-edge subgraph is computationally
# infeasible on CPU (~25-40s per forward+backward). The Gene-Gene group alone is
# 2.35M bioarx co-expression edges (85% of the subgraph) of a SINGLE relation
# and is the least task-relevant group (treatment paths run Compound->Gene->Disease).
# For the GNN training graph we keep a UNIFORM RANDOM subsample of Gene-Gene
# edges (fixed seed) — unbiased neighborhood aggregation (stochastic message
# passing view); all other groups are kept in full. KGE training and the path
# explanations use the FULL subgraph edges.
GENE_GENE_KEEP = 250_000
VALID_SPLIT_FRAC = 0.15
TEST_SPLIT_FRAC = 0.15

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def section(title):
    print("\n" + "=" * 70)
    print("==== " + title)
    print("=" * 70)


def main():
    # ======================================================================
    section("0. ENVIRONMENT AND FILE CHECKS")
    # ======================================================================
    if not DRKG_TSV.exists():
        raise FileNotFoundError(
            f"DRKG data file not found at {DRKG_TSV}. "
            "Place drkg.tsv under raw/ before running this script."
        )
    CKPT_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    print(f"Data file        : {DRKG_TSV}")
    print(f"Checkpoints dir  : {CKPT_DIR}")
    print(f"Results dir      : {RESULTS_DIR}")

    # ======================================================================
    section("1. LOAD RAW DRKG TRIPLES")
    # ======================================================================
    entity2id = {}
    relation2id = {}
    # streamed parse: keep (head, rel, tail) as int triples in the FULL id space
    triples = []          # list of (h_id, r_id, t_id)
    triple_set = set()    # for exact-duplicate removal
    n_lines = 0
    with open(DRKG_TSV, encoding="utf-8") as f:
        for line in f:
            n_lines += 1
            if n_lines % 1_000_000 == 0:
                print(f"  ... parsed {n_lines:,} lines "
                      f"({len(entity2id):,} entities, {len(triples):,} unique triples)")
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            h, r, t = parts
            if h not in entity2id:
                entity2id[h] = len(entity2id)
            if t not in entity2id:
                entity2id[t] = len(entity2id)
            if r not in relation2id:
                relation2id[r] = len(relation2id)
            key = (entity2id[h], relation2id[r], entity2id[t])
            if key not in triple_set:
                triple_set.add(key)
                triples.append(key)

    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}
    n_ent, n_rel = len(entity2id), len(relation2id)
    print(f"Total lines read           : {n_lines:,}")
    print(f"Unique triples             : {len(triples):,}")
    print(f"Distinct entities          : {n_ent:,}")
    print(f"Distinct relations         : {n_rel:,}")

    # entity prefix census
    prefix_count = collections.Counter(
        e.split("::")[0] for e in entity2id)
    print("Entity counts by prefix:")
    for p, c in prefix_count.most_common():
        print(f"  {p:<22} {c:>7,}")

    # ======================================================================
    section("2. TREATS-TYPE RELATION CANDIDATES (target-relation choice)")
    # ======================================================================
    cd_rel_count = collections.Counter()
    dc_rel_count = collections.Counter()
    for h, r, t in triples:
        hp = id2entity[h].split("::")[0]
        tp = id2entity[t].split("::")[0]
        if hp == "Compound" and tp == "Disease":
            cd_rel_count[id2relation[r]] += 1
        if hp == "Disease" and tp == "Compound":
            dc_rel_count[id2relation[r]] += 1
    print("Compound -> Disease relations in DRKG (triple counts):")
    for r, c in cd_rel_count.most_common():
        flag = "  [treats-like]" if r in TREATS_LIKE_FAMILY else ""
        print(f"  {c:>7,}  {r}{flag}")
    print("Disease -> Compound relations in DRKG (possible inverse encodings):")
    if dc_rel_count:
        for r, c in dc_rel_count.most_common():
            print(f"  {c:>7,}  {r}")
    else:
        print("  (none — DRKG encodes treats only as Compound -> Disease)")

    treats_rel_id = relation2id[TREATS_RELATION]
    n_treats = cd_rel_count[TREATS_RELATION]
    print("\nCHOSEN TARGET RELATION: %s" % TREATS_RELATION)
    print("Justification: DRUGBANK::treats is the curated, high-precision source of")
    print("approved drug-disease treatments (%s triples). GNBR::T is larger (%s) but is" % (
        f"{n_treats:,}", f"{cd_rel_count['GNBR::T::Compound:Disease']:,}"))
    print("literature-mined and includes weak/noisy associations; using it would inflate")
    print("the cohort beyond the intended scope. DRUGBANK::treats is NOT too small: it")
    print("spans 1,182 diseases and supports a well-sized cohort (see next section).")
    print("NOTE: GNBR/other treats-like edges are still used as auxiliary knowledge")
    print("graph context (with leakage screening, section 6).")

    # ======================================================================
    section("3. DISEASE COHORT (D1: final cohort and justification)")
    # ======================================================================
    # treatment degree = number of distinct compounds with a treats edge to the disease
    disease_treats = collections.defaultdict(set)   # disease_ent_id -> set(compound)
    compound_treats = collections.defaultdict(set)  # compound_ent_id -> set(disease)
    for h, r, t in triples:
        if r == treats_rel_id:
            disease_treats[t].add(h)
            compound_treats[h].add(t)

    degrees = {d: len(cs) for d, cs in disease_treats.items()}
    deg_vals = sorted(degrees.values())
    all_disease_entities = [e for e in entity2id if e.split("::")[0] == "Disease"]
    print(f"Diseases with >= 1 DRUGBANK::treats treatment : {len(degrees):,} of "
          f"{len(all_disease_entities):,} Disease entities in DRKG")
    print(f"Treatment-degree distribution over treated diseases:")
    print(f"  min={deg_vals[0]}  max={deg_vals[-1]}  "
          f"mean={sum(deg_vals)/len(deg_vals):.2f}  median={deg_vals[len(deg_vals)//2]}")
    q = np.percentile(deg_vals, [25, 50, 75])
    print(f"  quartiles: Q1={q[0]:.0f}  Q2={q[1]:.0f}  Q3={q[2]:.0f}")
    hist = collections.Counter(min(d, 10) for d in deg_vals)
    print("  histogram (degree capped at 10+): "
          + ", ".join(f"{k if k < 10 else '10+'}:{hist[k]}" for k in sorted(hist)))

    print(f"\nELIGIBILITY FILTER: treatment-degree >= {MIN_TREATS_DEGREE}")
    print("Justification (graph statistics, not convenience): a disease needs at least")
    print(f"{MIN_TREATS_DEGREE} known treatments so that a per-disease 70/15/15 split can place")
    print("at least one edge in each of train/val/test (D4 requirement). Diseases below")
    print("this threshold cannot be evaluated without emptying the test support.")
    cohort_diseases = sorted(
        [d for d, deg in degrees.items() if deg >= MIN_TREATS_DEGREE])
    cohort_edges = sum(degrees[d] for d in cohort_diseases)
    print(f"RESULTING COHORT: {len(cohort_diseases)} diseases, "
          f"{cohort_edges} treats edges, "
          f"{len(set().union(*[disease_treats[d] for d in cohort_diseases])):,} "
          "distinct treating compounds")
    print("Cohort size falls in the intended '10-20 to a few hundred' scope window.")

    # ======================================================================
    section("4. DISEASE INFORMATION TIERS (D3: exact grouping rule)")
    # ======================================================================
    cohort_degrees = sorted((degrees[d], d) for d in cohort_diseases)
    # Rule: rank diseases by treatment degree (ties broken deterministically by
    # entity id); count-tertiles of that ranked list give ~equal disease counts per
    # tier and every disease has exactly one unambiguous tier.
    # unambiguous rule: rank diseases by degree; LOW = bottom third, MEDIUM = middle
    # third, HIGH = top third of the RANKED LIST (count-tertiles), with ties broken
    # deterministically by entity id. This guarantees ~equal disease counts per tier
    # and every disease has exactly one tier.
    n_c = len(cohort_degrees)
    tier_of_rank = {}
    for rank, (deg, d) in enumerate(cohort_degrees):
        if rank < n_c / 3:
            tier_of_rank[d] = "LOW"
        elif rank < 2 * n_c / 3:
            tier_of_rank[d] = "MEDIUM"
        else:
            tier_of_rank[d] = "HIGH"
    disease_to_group = {id2entity[d]: tier_of_rank[d] for d in cohort_diseases}

    print("EXACT RULE: diseases ranked by treatment degree (ties broken by entity id);")
    print("cohort split into count-tertiles -> LOW (bottom third), MEDIUM (middle")
    print("third), HIGH (top third). Count-tertiles keep the three groups comparable")
    print("in disease population, which the research question requires.")
    print(f"Degree range of each tier (attained cut points): LOW <= {cohort_degrees[n_c // 3 - 1][0]} | "
          f"MEDIUM <= {cohort_degrees[2 * n_c // 3 - 1][0]} | HIGH >= {cohort_degrees[2 * n_c // 3][0]}")
    print(f"{'TIER':<8}{'#Diseases':>10}{'#Edges':>8}{'deg min':>9}{'deg max':>9}")
    for tier in ["LOW", "MEDIUM", "HIGH"]:
        ds = [d for d in cohort_diseases if tier_of_rank[d] == tier]
        print(f"{tier:<8}{len(ds):>10}{sum(degrees[d] for d in ds):>8}"
              f"{min(degrees[d] for d in ds):>9}{max(degrees[d] for d in ds):>9}")

    # ======================================================================
    section("5. LEAKAGE-CONTROLLED TRAIN/VAL/TEST SPLIT (D4)")
    # ======================================================================
    # Per-disease split of treats edges: ~70/15/15, at least 1 edge per split
    # (guaranteed by the degree >= 3 eligibility filter), deterministic RNG per disease.
    rng = random.Random(SEED)
    train_pairs, val_pairs, test_pairs = [], [], []   # (compound_ent, disease_ent)
    done = 0
    for d in cohort_diseases:
        done += 1
        if done % 100 == 0:
            print(f"  ... split edges for {done}/{len(cohort_diseases)} diseases")
        edges = sorted(disease_treats[d])            # deterministic order
        rng.shuffle(edges)
        n = len(edges)
        n_test = max(1, round(TEST_SPLIT_FRAC * n))
        n_val = max(1, round(VALID_SPLIT_FRAC * n))
        n_train = n - n_val - n_test
        if n_train < 1:  # cannot happen for n >= 3, but guard anyway
            n_val = n_test = 1
            n_train = n - 2
        test_pairs.extend((c, d) for c in edges[:n_test])
        val_pairs.extend((c, d) for c in edges[n_test:n_test + n_val])
        train_pairs.extend((c, d) for c in edges[n_test + n_val:])

    train_set = set(train_pairs)
    val_set = set(val_pairs)
    test_set = set(test_pairs)
    print(f"Split sizes (compound, disease) treats pairs:")
    print(f"  train : {len(train_pairs):>5}  ({100*len(train_pairs)/cohort_edges:.1f}%)")
    print(f"  val   : {len(val_pairs):>5}  ({100*len(val_pairs)/cohort_edges:.1f}%)")
    print(f"  test  : {len(test_pairs):>5}  ({100*len(test_pairs)/cohort_edges:.1f}%)")

    # ---- leakage checks (a) + (b): no triple/pair in more than one split ----
    ov_tv = len(train_set & val_set)
    ov_tt = len(train_set & test_set)
    ov_vt = len(val_set & test_set)
    print(f"Overlap train & val  : {ov_tv}  (must be 0)")
    print(f"Overlap train & test : {ov_tt}  (must be 0)")
    print(f"Overlap val   & test : {ov_vt}  (must be 0)")
    assert ov_tv == 0 and ov_tt == 0 and ov_vt == 0, "LEAKAGE DETECTED IN SPLIT"
    print("LEAKAGE CHECK PASSED: the three pair sets are disjoint.")

    # ---- known_true_pairs: all true treats pairs (train+val+test), cohort diseases
    known_true_pairs_ent = train_set | val_set | test_set
    true_compounds_for_disease = collections.defaultdict(set)
    for c, d in known_true_pairs_ent:
        true_compounds_for_disease[d].add(c)
    print(f"known_true_pairs (train+val+test, cohort diseases): {len(known_true_pairs_ent):,}")

    # ---- negative sampling strategy (documented, precomputed, shared by all models)
    # STRATEGY: for each positive train triple (compound, treats, disease) we draw
    # NUM_NEGATIVES negatives by corrupting ONLY THE COMPOUND side with a uniformly
    # random Compound:: entity. Rationale: the prediction task ranks compounds given
    # a disease, so negatives must be (non-true compound, disease) pairs. Unknown
    # pairs are NOT assumed false (open-world assumption) — they are used only as
    # candidate corruptions, and any corruption that is a KNOWN true pair
    # (train+val+test) for that disease is rejected and redrawn. The same fixed
    # negative set (seed 42) is used by every model -> fair comparison.
    all_compound_ents = sorted(e for e in entity2id if e.split("::")[0] == "Compound")
    compound_ent_ids = [entity2id[e] for e in all_compound_ents]
    print(f"Compound candidate pool size: {len(compound_ent_ids):,} "
          "(all Compound:: entities — fixed pool for ALL evaluation)")
    neg_rng = random.Random(SEED)
    neg_compound_ent = np.zeros((len(train_pairs), NUM_NEGATIVES), dtype=np.int64)
    for i, (c, d) in enumerate(train_pairs):
        if i % 500 == 0 and i:
            print(f"  ... sampled negatives for {i}/{len(train_pairs)} train pairs")
        used = true_compounds_for_disease[d]
        for k in range(NUM_NEGATIVES):
            cand = neg_rng.choice(compound_ent_ids)
            while cand in used:
                cand = neg_rng.choice(compound_ent_ids)
            neg_compound_ent[i, k] = cand
    print(f"Negative samples drawn: {neg_compound_ent.shape[0]} x {neg_compound_ent.shape[1]} "
          "(compound-side corruption, filtered against known true pairs)")

    # ======================================================================
    section("6. FILTERED SUBGRAPH FOR MODELS (D2 statistics)")
    # ======================================================================
    # Relation-type filter: keep exactly the four relation groups
    #   Compound-Disease, Compound-Gene, Gene-Disease, Gene-Gene
    # (both directions where DRKG encodes them). Anatomy, Symptom, Side Effect,
    # Pharmacologic Class, Compound-Compound, etc. are dropped.
    CORE = {"Compound", "Gene", "Disease"}
    GROUP_OF = {frozenset(("Compound", "Disease")): "Compound-Disease",
                frozenset(("Compound", "Gene")): "Compound-Gene",
                frozenset(("Gene", "Disease")): "Gene-Disease",
                frozenset(("Gene", "Gene")): "Gene-Gene"}

    held_pairs = val_set | test_set
    sub_triples = []          # (h_ent, r_glob, t_ent)
    dropped_leak_edges = 0
    group_edge_count = collections.Counter()
    cohort_induced_edges = 0
    cohort_compound_ents = set()
    for c, d in known_true_pairs_ent:
        cohort_compound_ents.add(c)
    treat_pairs_all_diseases = set()
    for h, r, t in triples:
        hp = id2entity[h].split("::")[0]
        tp = id2entity[t].split("::")[0]
        if hp not in CORE or tp not in CORE:
            continue
        g = GROUP_OF.get(frozenset((hp, tp)))
        if g is None:
            continue
        # TARGET-EDGE LEAKAGE SCREEN: for any held-out (val/test) treats pair,
        # remove EVERY direct Compound-Disease edge between the two entities,
        # regardless of relation (covers treats, its inverse if present, and
        # treats-like encodings such as GNBR::T / Hetionet::CtD).
        if g == "Compound-Disease":
            if (h, t) in held_pairs or (t, h) in held_pairs:
                dropped_leak_edges += 1
                continue
        sub_triples.append((h, r, t))
        group_edge_count[g] += 1
    cohort_disease_set = set(cohort_diseases)
    cohort_induced_edges = sum(
        1 for h, r, t in sub_triples
        if h in cohort_disease_set or t in cohort_disease_set
        or h in cohort_compound_ents or t in cohort_compound_ents)

    print("Relation groups kept (both directions): "
          + ", ".join(sorted(GROUP_OF.values())))
    print("Dropped: all edges touching Anatomy / Symptom / Side Effect / "
          "Pharmacologic Class / ATC / Taxonomy / Biological Process / "
          "Compound-Compound etc.")
    print(f"Held-out pair direct edges REMOVED from training graph "
          f"(any Compound-Disease relation): {dropped_leak_edges:,}")
    print(f"Filtered subgraph edges: {len(sub_triples):,}")
    print("Edges per group:")
    for g, c in group_edge_count.most_common():
        print(f"  {g:<20} {c:>10,}")
    print(f"Cohort-induced edges (incident to a cohort disease or cohort compound): "
          f"{cohort_induced_edges:,}")

    # ---- subgraph relation space (local ids 0..K-1 for R-GCN / KGE) ----
    sub_rels = sorted({r for _, r, _ in sub_triples})
    subrel2global = {i: r for i, r in enumerate(sub_rels)}
    global2subrel = {r: i for i, r in enumerate(sub_rels)}
    print(f"Distinct relations in subgraph: {len(sub_rels)}")

    # ---- node space for ALL models (KGE + GNN share it) ----
    # Nodes = every Compound entity (the candidate pool must be embeddable, even
    # compounds with no subgraph edge) + every entity appearing in the subgraph.
    sub_ent = set()
    for h, r, t in sub_triples:
        sub_ent.add(h)
        sub_ent.add(t)
    node_entities = sorted(sub_ent | set(compound_ent_ids))
    entid_to_node = {e: i for i, e in enumerate(node_entities)}
    n_nodes = len(node_entities)
    node_prefix_count = collections.Counter(
        id2entity[e].split("::")[0] for e in node_entities)
    print(f"Model node space: {n_nodes:,} nodes "
          f"(Compounds {node_prefix_count['Compound']:,}, "
          f"Genes {node_prefix_count['Gene']:,}, "
          f"Diseases {node_prefix_count['Disease']:,})")
    n_isolated_compounds = sum(
        1 for e in compound_ent_ids if e not in sub_ent)
    print(f"Compounds with no subgraph edge (isolated candidates): "
          f"{n_isolated_compounds:,} — they still receive learned input features")

    # ---- degree / sparsity distribution for the cohort's induced subgraph (D2) ----
    print("Treatment-degree distribution of the COHORT (sparsity evidence):")
    cd_deg = [degrees[d] for d in cohort_diseases]
    bins = [0, 3, 5, 8, 12, 20, 10 ** 9]
    for lo, hi in zip(bins[:-1], bins[1:]):
        cnt = sum(1 for x in cd_deg if lo <= x < hi)
        print(f"  degree in [{lo}, {hi if hi < 10**9 else 'inf'}): {cnt:>4} diseases")

    # ======================================================================
    section("7. BUILD TENSORS AND SAVE CHECKPOINT")
    # ======================================================================
    edge_h = torch.tensor([entid_to_node[h] for h, _, _ in sub_triples], dtype=torch.long)
    edge_t = torch.tensor([entid_to_node[t] for _, _, t in sub_triples], dtype=torch.long)
    edge_type = torch.tensor([global2subrel[r] for _, r, _ in sub_triples], dtype=torch.long)
    edge_index = torch.stack([edge_h, edge_t], dim=0)

    def to_node_pair(pairs):
        return torch.tensor([[entid_to_node[c], entid_to_node[d]] for c, d in pairs],
                            dtype=torch.long)

    train_triples = to_node_pair(train_pairs)
    val_triples = to_node_pair(val_pairs)
    test_triples = to_node_pair(test_pairs)
    compound_candidate_ids = torch.tensor(
        [entid_to_node[c] for c in compound_ent_ids], dtype=torch.long)
    neg_compound_ids = torch.tensor(
        [[entid_to_node[neg_compound_ent[i, k]] for k in range(NUM_NEGATIVES)]
         for i in range(len(train_pairs))], dtype=torch.long)
    known_true_pairs = {(entid_to_node[c], entid_to_node[d])
                        for c, d in known_true_pairs_ent}
    disease_node_to_group = {entid_to_node[d]: tier_of_rank[d] for d in cohort_diseases}
    disease_node_degree = {entid_to_node[d]: degrees[d] for d in cohort_diseases}
    treats_local_rel = global2subrel[treats_rel_id]

    # ---- GNN message-passing graph: gene-gene subsampled for CPU tractability ----
    is_gene_node = torch.zeros(n_nodes, dtype=torch.bool)
    for i, e in enumerate(node_entities):
        if id2entity[e].split("::")[0] == "Gene":
            is_gene_node[i] = True
    gg_mask = is_gene_node[edge_h] & is_gene_node[edge_t]
    gg_idx = torch.nonzero(gg_mask, as_tuple=False).flatten()
    keep_rng = np.random.RandomState(SEED)
    keep_pos = np.sort(keep_rng.choice(
        len(gg_idx), size=min(GENE_GENE_KEEP, len(gg_idx)), replace=False))
    gg_keep = gg_idx[torch.tensor(keep_pos, dtype=torch.long)]
    gnn_mask = ~gg_mask
    gnn_mask[gg_keep] = True
    print(f"GNN message-passing graph: {int(gnn_mask.sum()):,} edges "
          f"(Gene-Gene subsampled {len(gg_idx):,} -> {len(gg_keep):,} uniformly, "
          f"seed {SEED}; all other groups in full)")
    print("KGE training and path explanations use the FULL subgraph edges.")

    ckpt = {
        "entity2id": entity2id,
        "id2entity": id2entity,
        "relation2id": relation2id,
        "id2relation": id2relation,
        "node_entities": torch.tensor(node_entities, dtype=torch.long),
        "entid_to_node": torch.tensor(
            [entid_to_node.get(e, -1) for e in range(n_ent)], dtype=torch.long),
        "n_nodes": n_nodes,
        "n_subrel": len(sub_rels),
        "subrel2global": {i: id2relation[r] for i, r in enumerate(sub_rels)},
        "edge_index": edge_index,
        "edge_type": edge_type,
        "gnn_edge_index": edge_index[:, gnn_mask],
        "gnn_edge_type": edge_type[gnn_mask],
        "gene_gene_kept": len(gg_keep),
        "treats_local_rel": treats_local_rel,
        "train_triples": train_triples,
        "val_triples": val_triples,
        "test_triples": test_triples,
        "disease_to_group": disease_to_group,          # keyed by entity string
        "disease_node_to_group": disease_node_to_group,  # keyed by node id
        "disease_node_degree": disease_node_degree,
        "compound_candidate_ids": compound_candidate_ids,
        "known_true_pairs": known_true_pairs,
        "neg_compound_ids": neg_compound_ids,
        "cohort_diseases": [(id2entity[d], degrees[d], tier_of_rank[d])
                            for d in cohort_diseases],
        "config": {
            "seed": SEED,
            "treats_relation": TREATS_RELATION,
            "treats_like_family": sorted(TREATS_LIKE_FAMILY),
            "min_treats_degree": MIN_TREATS_DEGREE,
            "num_negatives": NUM_NEGATIVES,
        },
    }
    out_path = CKPT_DIR / "splits_and_mappings.pt"
    torch.save(ckpt, out_path)
    torch.save({"entity2id": entity2id, "relation2id": relation2id},
               CKPT_DIR / "entity_relation_maps.pt")
    print(f"Saved checkpoint: {out_path}")
    print("Checkpoint inventory:")
    for k, v in ckpt.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:<24} Tensor {tuple(v.shape)}")
        elif isinstance(v, dict):
            print(f"  {k:<24} dict(len={len(v)})")
        elif isinstance(v, (list, set)):
            print(f"  {k:<24} {type(v).__name__}(len={len(v)})")
        else:
            print(f"  {k:<24} {type(v).__name__}")
    print("\nSTAGE 01 COMPLETE.")


if __name__ == "__main__":
    main()
