"""
kg_common.py — shared utilities for the sparsity-aware drug-repurposing pipeline.

Contains the ONE reusable, correct evaluation path used by every stage:

    filtered_rank(...)        — rank of the true compound among all Compound::
                                candidates for one (compound, treats, disease)
                                test triple, with standard filtered setting.
    evaluate_stratified(...)  — MRR / Hits@1/3/10 / mean & median rank plus
                                SUPPORT COUNTS, overall and per LOW/MEDIUM/HIGH
                                information tier.

TASK DIRECTION (do not get this backwards): given a DISEASE, rank candidate
DRUGS (Compound:: entities). The candidate pool is ALWAYS compound_candidate_ids
(all Compound:: nodes) — never Disease:: entities.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

BASE_DIR = Path(__file__).resolve().parent
CKPT_DIR = BASE_DIR / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"
SPLITS_CKPT = CKPT_DIR / "splits_and_mappings.pt"
DEVICE = "cpu"


def section(title):
    print("\n" + "=" * 70)
    print("==== " + title)
    print("=" * 70)


def require_checkpoints(files):
    """Verify required checkpoint files exist; raise a clear error naming the
    earlier pipeline stage that must be run first."""
    for path in files:
        if not Path(path).exists():
            stage = "01_data_prep.py"
            name = Path(path).name
            for prefix, stage_name in [
                ("transe", "02_transe_baseline.py"),
                ("complex", "03_complex_baseline.py"),
                ("vanilla", "04_vanilla_rgcn.py"),
                ("sparsity_aware", "05_sparsity_aware_rgcn.py"),
                ("ablation", "06_ablation.py"),
            ]:
                if name.startswith(prefix):
                    stage = stage_name
            raise FileNotFoundError(
                f"Required checkpoint '{path}' is missing.\n"
                f"-> Run '{stage}' first to produce it, then re-run this stage."
            )


def load_splits(path=SPLITS_CKPT):
    """Load the stage-01 checkpoint."""
    require_checkpoints([path])
    return torch.load(path, weights_only=False)


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)


# ==========================================================================
# EVALUATION (the single, correct, reusable evaluation code path)
# ==========================================================================
def filtered_rank(score_fn, true_compound_id, disease_id,
                  compound_candidate_ids, known_true_pairs):
    """
    Standard filtered ranking of the TRUE COMPOUND for a held-out
    (compound, treats, disease) triple.

    Given a disease, the candidate pool is compound_candidate_ids — ALL
    Compound:: entities (never Disease:: entities). Before ranking we REMOVE
    every OTHER compound that is a known-true treatment for this same disease
    (train+val+test positives — the standard filtered KG setting), so that the
    model is not penalized for ranking another known drug above the true one.
    The true compound itself is NEVER filtered out.

    Args:
        score_fn:          callable (cand_ids: LongTensor, disease_id: int)
                           -> FloatTensor of scores, HIGHER = better.
        true_compound_id:  node id of the true compound.
        disease_id:        node id of the disease.
        compound_candidate_ids: LongTensor of ALL Compound:: node ids.
        known_true_pairs:  either a set of (compound_id, disease_id) pairs, or a
                           dict {disease_id: set(compound_id)} (faster, built by
                           evaluate_stratified).

    Returns:
        int: 1-indexed rank of the true compound.
    TIE-BREAKING: pessimistic — every candidate whose score EQUALS the true
    compound's score is counted ahead of it. (Exact ties are rare with trained
    embeddings; the pessimistic convention avoids optimistic bias.)
    """
    if isinstance(known_true_pairs, dict):
        known_drugs = known_true_pairs.get(disease_id, set())
    else:
        known_drugs = {c for c, d in known_true_pairs if d == disease_id}
    known_drugs.discard(true_compound_id)

    base = np.asarray(compound_candidate_ids)
    if len(known_drugs):
        keep = ~np.isin(base, np.fromiter(known_drugs, dtype=np.int64,
                                          count=len(known_drugs)))
    else:
        keep = np.ones(len(base), dtype=bool)
    cand_ids = torch.from_numpy(base[keep])
    with torch.no_grad():
        scores = score_fn(cand_ids, disease_id)
    true_mask = cand_ids == true_compound_id
    s_true = float(scores[true_mask][0])
    rank = 1 + int((scores > s_true).sum()) + int((scores == s_true).sum()) - 1
    # (subtract 1: the true compound itself is among the == s_true entries)
    return int(rank)


def evaluate_stratified(score_fn, test_triples, disease_to_group,
                        compound_candidate_ids, known_true_pairs,
                        verbose=True):
    """
    Evaluate a model on test treats-triples, overall AND per information tier.

    Args:
        score_fn:              callable (cand_ids, disease_id) -> scores (higher=better).
        test_triples:          LongTensor [n, 2] of (compound_node, disease_node).
        disease_to_group:      dict {disease_node_id: 'LOW'|'MEDIUM'|'HIGH'}.
        compound_candidate_ids: LongTensor of ALL Compound:: node ids.
        known_true_pairs:      set of (compound_id, disease_id) true treats pairs
                               (train+val+test).

    Returns:
        (df, records) where df is a pandas DataFrame with rows OVERALL/LOW/
        MEDIUM/HIGH and columns MRR, Hits@1, Hits@3, Hits@10, mean_rank,
        median_rank, support; records is a list of per-triple dicts
        {compound, disease, group, rank, reciprocal_rank} (for statistical tests).
    """
    kt = {}
    for c, d in known_true_pairs:
        kt.setdefault(d, set()).add(c)
    cand_np = np.asarray(compound_candidate_ids)

    records = []
    n = len(test_triples)
    for i in range(n):
        if verbose and (i + 1) % 100 == 0:
            print(f"  ... evaluated {i + 1}/{n} test triples")
        c, d = int(test_triples[i, 0]), int(test_triples[i, 1])
        r = filtered_rank(score_fn, c, d, cand_np, kt)
        records.append({
            "compound": c, "disease": d,
            "group": disease_to_group[d],
            "rank": r, "reciprocal_rank": 1.0 / r,
        })

    def summarize(recs):
        ranks = np.array([x["rank"] for x in recs])
        rrs = np.array([x["reciprocal_rank"] for x in recs])
        return {
            "MRR": rrs.mean(),
            "Hits@1": float((ranks <= 1).mean()),
            "Hits@3": float((ranks <= 3).mean()),
            "Hits@10": float((ranks <= 10).mean()),
            "mean_rank": ranks.mean(),
            "median_rank": float(np.median(ranks)),
            "support": len(recs),
        }

    rows = [("OVERALL", summarize(records))]
    for tier in ["LOW", "MEDIUM", "HIGH"]:
        rows.append((tier, summarize(
            [x for x in records if x["group"] == tier])))
    df = pd.DataFrame({tier: vals for tier, vals in rows}).T
    return df, records


def print_results_table(model_name, df):
    print(f"\nStratified results — {model_name}")
    print(f"{'group':<8}{'MRR':>8}{'H@1':>8}{'H@3':>8}{'H@10':>8}"
          f"{'mean_rank':>11}{'median_rank':>13}{'support':>9}")
    for tier, row in df.iterrows():
        print(f"{tier:<8}{row['MRR']:>8.4f}{row['Hits@1']:>8.3f}"
              f"{row['Hits@3']:>8.3f}{row['Hits@10']:>8.3f}"
              f"{row['mean_rank']:>11.1f}{row['median_rank']:>13.1f}"
              f"{int(row['support']):>9}")
