"""
06_ablation.py — Stage 6: ablation study (D9).

Demonstrates the contribution of each sparsity-aware mechanism by training the
SAME model class as stage 05 (imported from 05_sparsity_aware_rgcn.py — no
reimplementation, no second evaluation code path) in three configurations:

    (a) self-gate only                (use_self_gate=True,  use_loss_reweighting=False)
    (b) loss reweighting only         (use_self_gate=False, use_loss_reweighting=True)
    (c) neither (== vanilla capacity) (use_self_gate=False, use_loss_reweighting=False)

Config (c) is a SANITY CHECK that the gated/gated-off layer degenerates to the
standard R-GCN update — it is NOT a replacement for stage 04's independently
trained vanilla R-GCN result.

Run AFTER: 01_data_prep.py (and 05 must exist as the class definition)
Run BEFORE: 07_evaluation.py
"""

import importlib.util
from pathlib import Path

import torch

from kg_common import CKPT_DIR, section, require_checkpoints, load_splits, set_seed

SEED = 42

SPEC = importlib.util.spec_from_file_location(
    "stage05_model", Path(__file__).resolve().parent / "05_sparsity_aware_rgcn.py")
stage05 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage05)


def main():
    section("06: ABLATION STUDY — SETUP")
    require_checkpoints([CKPT_DIR / "splits_and_mappings.pt"])
    set_seed(SEED)
    ckpt = load_splits()
    print("Using the SAME SparsityAwareRGCN class and training routine as stage 05")
    print("(loaded from 05_sparsity_aware_rgcn.py via importlib).")

    configs = [
        ("ablation_selfgate_only", True, False, "self-gate only"),
        ("ablation_reweighting_only", False, True, "loss-reweighting only"),
        ("ablation_neither", False, False, "neither (sanity check)"),
    ]
    for ckpt_name, gate, reweight, desc in configs:
        section(f"06: ABLATION — ({desc}) gate={gate}, reweighting={reweight}")
        model, best_state, best_val = stage05.train_sparsity_model(
            ckpt, use_self_gate=gate, use_loss_reweighting=reweight,
            name=desc)
        out = CKPT_DIR / f"{ckpt_name}.pt"
        torch.save({"state_dict": {k: v for k, v in model.state_dict().items()},
                    "config": {"dim": stage05.DIM, "num_bases": stage05.NUM_BASES,
                               "lr": stage05.LR, "seed": SEED,
                               "use_self_gate": gate,
                               "use_loss_reweighting": reweight,
                               "best_epoch": best_state["epoch"]},
                    "best_val_mrr": best_val}, out)
        print(f"Saved: {out} (best epoch {best_state['epoch']}, "
              f"val MRR {best_val:.4f})")

    print("\nSTAGE 06 COMPLETE.")


if __name__ == "__main__":
    main()
