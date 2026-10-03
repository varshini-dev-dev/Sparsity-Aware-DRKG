"""
07_evaluation.py — Stage 7: final stratified evaluation (D5/D6/D7/D8/D9).

Loads the stage-01 checkpoint and ALL trained models from stages 02-06 and
evaluates every model on the test split with the ONE reusable, correct
evaluation path (kg_common.filtered_rank / evaluate_stratified):

    given a DISEASE, rank all Compound:: candidates (never Disease::), with
    standard filtered setting (other known-true drugs of the same disease are
    removed from the candidate pool; the true compound is kept).

Outputs:
    results/stratified_results.csv      one row per model x group
                                        (OVERALL/LOW/MEDIUM/HIGH) with MRR,
                                        Hits@1/3/10, mean/median rank, SUPPORT.
    results/statistical_comparison.csv  Wilcoxon signed-rank tests + paired
                                        sign tests, sparsity-aware vs vanilla
                                        R-GCN, overall and per group.
Also prints the final summary table.

Run AFTER: 01-06. Fully standalone: no retraining, results in seconds.
"""

from importlib import util as importlib_util
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

from kg_common import (CKPT_DIR, RESULTS_DIR, section, require_checkpoints,
                       load_splits, evaluate_stratified, print_results_table)

SPEC = importlib_util.spec_from_file_location(
    "stage05_model", Path(__file__).resolve().parent / "05_sparsity_aware_rgcn.py")
stage05 = importlib_util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage05)


def load_model_modules():
    spec04 = importlib_util.spec_from_file_location(
        "stage04_model", Path(__file__).resolve().parent / "04_vanilla_rgcn.py")
    stage04 = importlib_util.module_from_spec(spec04)
    spec04.loader.exec_module(stage04)
    return stage04


def build_score_fns(ckpt):
    """Return {model_name: score_fn} for every trained model.

    Every score_fn maps (candidate compound ids, disease id) -> scores with
    HIGHER = better. Candidates are always Compound:: nodes.
    """
    score_fns = {}

    # ---------------- TransE ----------------
    tr = torch.load(CKPT_DIR / "transe_model.pt", weights_only=False)
    ent, rel = tr["ent"], tr["rel"]

    def transe_score(cand_ids, disease_id):
        e_c = ent[cand_ids]
        return -torch.norm(e_c + rel[tr["treats_rel"]] - ent[disease_id], dim=1)
    score_fns["TransE"] = transe_score

    # ---------------- ComplEx ----------------
    cx = torch.load(CKPT_DIR / "complex_model.pt", weights_only=False)
    er, ei = cx["ent_re"], cx["ent_im"]
    rr = cx["rel_re"][cx["treats_rel"]]
    ri = cx["rel_im"][cx["treats_rel"]]

    def complex_score(cand_ids, disease_id):
        hr, hi = er[cand_ids], ei[cand_ids]
        tr_, ti_ = er[disease_id], ei[disease_id]
        return ((hr * rr - hi * ri) * tr_ + (hr * ri + hi * rr) * ti_).sum(dim=-1)
    score_fns["ComplEx"] = complex_score

    # ---------------- GNN models (vanilla, sparsity-aware, ablations) -------
    edge_index = ckpt["gnn_edge_index"]
    edge_type = ckpt["gnn_edge_type"]
    n_nodes, n_rel = ckpt["n_nodes"], ckpt["n_subrel"]
    treats_rel = ckpt["treats_local_rel"]

    def gnn_score(state, cls, use_gate, use_rw):
        model = cls(n_nodes, n_rel, 64, 30, treats_rel,
                    use_self_gate=use_gate, use_loss_reweighting=use_rw)
        model.load_state_dict(state["state_dict"])
        model.eval()
        with torch.no_grad():
            z = model.encode(edge_index, edge_type)
        rel_vec = model.rel_vec.detach()

        def score(cand_ids, disease_id):
            return (z[cand_ids] * rel_vec * z[disease_id]).sum(dim=1)
        return score

    stage04 = load_model_modules()

    van = torch.load(CKPT_DIR / "vanilla_rgcn.pt", weights_only=False)
    van_model = stage04.VanillaRGCN(n_nodes, n_rel, 64, 30, treats_rel)
    van_model.load_state_dict(van["state_dict"])
    van_model.eval()
    with torch.no_grad():
        van_z = van_model.encode(edge_index, edge_type)

    def vanilla_score(cand_ids, disease_id):
        return (van_z[cand_ids] * van_model.rel_vec.detach()
                * van_z[disease_id]).sum(dim=1)
    score_fns["Vanilla R-GCN"] = vanilla_score

    # FIX: Now loading optimized_sparsity.pt
    spa = torch.load(CKPT_DIR / "optimized_sparsity.pt", weights_only=False)
    score_fns["Sparsity-aware R-GCN"] = gnn_score(
        spa, stage05.SparsityAwareRGCN, True, True)
        
    for ab_file, label, gate, rw in [
        ("ablation_selfgate_only", "Ablation: self-gate only", True, False),
        ("ablation_reweighting_only", "Ablation: reweighting only", False, True),
        ("ablation_neither", "Ablation: neither", False, False),
    ]:
        st = torch.load(CKPT_DIR / f"{ab_file}.pt", weights_only=False)
        score_fns[label] = gnn_score(st, stage05.SparsityAwareRGCN, gate, rw)

    return score_fns


def statistical_tests(records_a, records_b):
    """Wilcoxon signed-rank test + paired sign test on paired per-triple
    reciprocal ranks (records aligned by (compound, disease))."""
    a = {x["disease"] * 10 ** 7 + x["compound"]: x["reciprocal_rank"]
         for x in records_a}
    b = {x["disease"] * 10 ** 7 + x["compound"]: x["reciprocal_rank"]
         for x in records_b}
    keys = sorted(set(a) & set(b))
    da = np.array([a[k] for k in keys])
    db = np.array([b[k] for k in keys])
    diff = da - db
    if np.allclose(diff, 0):
        w_p = float("nan")
    else:
        w_p = float(stats.wilcoxon(da, db, zero_method="wilcox").pvalue)
    wins = int((diff > 0).sum())
    losses = int((diff < 0).sum())
    n_eff = wins + losses
    s_p = float(stats.binomtest(wins, n_eff, 0.5).pvalue) if n_eff else float("nan")
    return {"wilcoxon_p": w_p, "sign_p": s_p, "wins": wins,
            "losses": losses, "n_pairs": len(keys)}


def main():
    section("07: FINAL EVALUATION — SETUP")
    require_checkpoints([
        CKPT_DIR / "splits_and_mappings.pt",
        CKPT_DIR / "transe_model.pt",
        CKPT_DIR / "complex_model.pt",
        CKPT_DIR / "vanilla_rgcn.pt",
        CKPT_DIR / "optimized_sparsity.pt",  # FIX: Required checkpoint updated
        CKPT_DIR / "ablation_selfgate_only.pt",
        CKPT_DIR / "ablation_reweighting_only.pt",
        CKPT_DIR / "ablation_neither.pt",
    ])
    ckpt = load_splits()
    test_triples = ckpt["test_triples"]
    d2g = ckpt["disease_node_to_group"]
    candidates = ckpt["compound_candidate_ids"]
    known_true_pairs = ckpt["known_true_pairs"]
    print(f"Test triples: {len(test_triples):,} | "
          f"candidate pool: {len(candidates):,} Compound:: entities")
    print("Per-group test support (reported alongside every metric below):")
    for tier in ["LOW", "MEDIUM", "HIGH"]:
        n_t = sum(1 for i in range(len(test_triples))
                  if d2g[int(test_triples[i, 1])] == tier)
        print(f"  {tier:<7} {sum(1 for v in d2g.values() if v == tier)} diseases, "
              f"{n_t} test triples")

    score_fns = build_score_fns(ckpt)

    section("07: TEST-SET EVALUATION (all models, stratified)")
    all_rows = []
    per_model_records = {}
    for name, fn in score_fns.items():
        print(f"\n>>> {name}")
        df, records = evaluate_stratified(
            fn, test_triples, d2g, candidates, known_true_pairs)
        per_model_records[name] = records
        print_results_table(name, df)
        for tier, row in df.iterrows():
            all_rows.append({"model": name, "group": tier, **row.to_dict()})

    results = pd.DataFrame(all_rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    out1 = RESULTS_DIR / "stratified_results.csv"
    results.to_csv(out1, index=False)
    print(f"\nSaved: {out1}")

    section("07: STATISTICAL COMPARISON — sparsity-aware vs vanilla R-GCN")
    spa = per_model_records["Sparsity-aware R-GCN"]
    van = per_model_records["Vanilla R-GCN"]
    stat_rows = [dict(comparison="sparsity_vs_vanilla", group="OVERALL",
                      **statistical_tests(spa, van))]
    for tier in ["LOW", "MEDIUM", "HIGH"]:
        s = [x for x in spa if x["group"] == tier]
        v = [x for x in van if x["group"] == tier]
        stat_rows.append(dict(comparison="sparsity_vs_vanilla", group=tier,
                              **statistical_tests(s, v)))
    stat_df = pd.DataFrame(stat_rows)
    out2 = RESULTS_DIR / "statistical_comparison.csv"
    stat_df.to_csv(out2, index=False)
    print(stat_df.to_string(index=False))
    print(f"Saved: {out2}")

    section("07: FINAL SUMMARY (master table — D5/D6/D7/D8/D9 evidence)")
    print(results.to_string(index=False))
    print("\nNOTE: every metric is reported per LOW/MEDIUM/HIGH information tier"
          " (with support counts) — never an overall score alone.")
    print("STAGE 07 COMPLETE.")


if __name__ == "__main__":
    main()