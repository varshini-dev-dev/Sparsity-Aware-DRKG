# Sparsity-Aware Knowledge-Graph Drug Repurposing

**Author:** Varshini D (SRN: PES1PG25CA239)

**Program:** Master of Computer Applications (MCA), PES University (Academic Year 2026-27)

**Guide:** Dr. Dilip Kumar Maripuri, Associate Professor

**Status:** Capstone Project Phase-I: Review-2

## 📌 Project Overview

Drug repurposing can drastically reduce the time and cost of identifying potential medical treatments. However, traditional drug-disease link prediction models perform poorly for rare diseases with sparse information in biomedical knowledge graphs. Because traditional Relational Graph Convolutional Networks (R-GCNs) rely heavily on existing neighborhood connections, sparse nodes suffer from a "sinkhole effect," leading to weak candidate rankings.

This project investigates a novel **Sparsity-Aware R-GCN** approach to improve candidate drug ranking for low-information diseases while maintaining overall model performance. Diseases are categorized into LOW, MEDIUM, and HIGH information groups based on their measurable graph characteristics.

## 🧬 Dataset

* **Source:** Drug Repurposing Knowledge Graph (DRKG)


* **Graph Size:** 5,874,261 triples, 97,238 entities, and 107 relation types


* **Target Relation:** `DRUGBANK::treats::Compound:Disease`

* **Data Splitting:** Mathematically exact 15% transductive split yielding 729 test edges (178 LOW, 177 MEDIUM, 374 HIGH).



## 🧠 Architecture

The proposed model handles sparsity by constructing a specialized 66-dimensional input vector for disease nodes:

* **64-Dimensional Base:** Global latent representation extracted from TransE embeddings.


* **+ 1 Sparsity Feature:** A normalized feature representing the known treatment count.


* **+ 1 Structural Feature:** A normalized metric capturing local graph neighborhood density.



These explicit features act as a "global compass" allowing a two-layer neural encoder to route information globally when local message passing fails on rare nodes.

## 📊 Experimental Results

The proposed model was benchmarked against a Vanilla R-GCN using a frozen test set, evaluated on Filtered Mean Reciprocal Rank (MRR), Mean Rank, and Hits@10.

| Metric | Vanilla R-GCN | Proposed Model |
| --- | --- | --- |
| **Overall MRR** | 0.0019 | 0.0376 |
| **Overall Mean Rank** | 9774.2 | 3936.1 |
| **Overall Hits@10** | 0.003 | 0.097 |
| **LOW Tier MRR** | 0.0020 | 0.0148 |
| **LOW Tier Hits@10** | 0.000 | 0.028 |

**Key Findings:**

* **7.4x Improvement:** The proposed model improved the LOW-tier MRR by 7.4 times (from 0.0020 to 0.0148).


* **Statistical Significance:** A paired Wilcoxon signed-rank test with Holm correction confirmed the results with a $p$-value of $7.30 \times 10^{-16}$ for the LOW tier.



## 🛠️ Repository Structure

The pipeline is designed with modular Python scripts to ensure strict data leakage control and full reproducibility.

* `01_data_prep.py`: Loads DRKG, identifies target relations, creates train/validation/test splits, and measures disease sparsity.


* `05_sparsity_aware_r_gcn.py`: Contains the PyTorch architecture for the 66-dimensional Sparsity-Aware Model and its training loop.


* `07_evaluation.py`: Loads saved model checkpoints, performs candidate ranking against 24,313 compounds, calculates group-wise Filtered MRR/Hits@10, and executes the Wilcoxon statistical test.


* `08_path_explanation.py`: Analyzes supporting biomedical paths for selected drug-disease predictions (strictly excluding the target edge to prevent data leakage).


* `reproducibility_audit.py`: Validates test splits via SHA-256 hashing and fixed random seeds (`SEED = 42`) to guarantee experimental integrity.



## 💻 Tech Stack

* **Language:** Python 3.x


* **Deep Learning:** PyTorch (CPU Environment)


* **Data Processing:** Pandas, NumPy


* **Statistical Analysis:** SciPy



## 🚀 How to Run

1. Ensure all dependencies from `requirements.txt` (PyTorch, Pandas, NumPy, SciPy) are installed.


2. Place the `drkg.tsv` dataset into the root directory.


3. Run `python 01_data_prep.py` to generate the strict data splits and sparsity mappings.


4. Run the training module to generate saved `.pt` checkpoints.


5. Execute `python 07_evaluation.py` to reproduce the terminal MRR and statistical significance metrics.


6. Run `python reproducibility_audit.py` to verify the cryptographic hash of the test splits.



## ⚠️ Research Disclaimer

The predicted drug-disease links output by this model are strictly research hypotheses meant to narrow down experimental search spaces. They do not claim clinical efficacy, safety, or medical approval, and are entirely outside the scope of actual clinical administration.
