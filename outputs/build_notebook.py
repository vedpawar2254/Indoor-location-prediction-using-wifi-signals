"""Assemble the Phase 1 deliverable notebook from the lifecycle modules.

The real logic lives in src/part*.py and scripts/*.py (already tested).
The notebook imports and calls those modules so there is a single source of truth.
Each lifecycle stage gets an executive markdown narrative + a code cell that runs it,
reproducing every number, table, and figure inline.

Key contributors:
  - Member 1 (M1): Data setup, quality checks, EDA, and USERID-grouped folds.
  - Member 2 (M2): Missing-signal handling, RSSI representations, engineered features, fold-safe transforms, and kNN feature baselines.
"""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()
cells = []


def md(text):
    cells.append(new_markdown_cell(text.strip("\n")))


def code(src):
    cells.append(new_code_cell(src.strip("\n")))


# ---------------------------------------------------------------- title ----
md(r"""
# Indoor Positioning via Wi-Fi Signal Strength — Phase 1 (AML Baseline)

**Project 9 — Indoor Location Prediction Using Wi-Fi Signals**  
Dataset: **UJIIndoorLoc** (UCI ML Repository, DOI: 10.24432/C5MS59).  
Targets: **building** (3 classes) & **floor** (5 classes) classification + **longitude/latitude** continuous 2D regression.

This notebook represents the verified **Phase 1 deliverable**: the foundational Applied Machine Learning (AML) baseline evaluating independent tabular models (Random Forest and XGBoost) per target. Phase 2 will replace these with a shared multitask deep-learning trunk; the gap this baseline demonstrates — no information sharing across coupled spatial targets — is precisely what Phase 2 addresses.

### Team Work Order & Phase 1 Responsibilities:
- **Member 1 (Data & Infrastructure Lead)**: Data setup, quality checks (duplicates, dead APs, near-zero RSSI), EDA, and USERID-grouped folds.
- **Member 2 (Features & Representation Lead)**: Missing-signal handling (+100 to -105 dBm sentinel), three RSSI representations, 936 engineered features, fold-safe transforms, and kNN feature ablation baselines.
- **Member 3 (Classification Lead)**: Independent building and floor classification with Random Forest and XGBoost.
- **Member 4 (Regression & Routing Lead)**: Multi-output coordinate regression (LONGITUDE, LATITUDE) and spatial error analysis.
- **Member 5 (Evaluation & Integration Lead)**: Unified evaluation suite, coordinate MAE reporting, EvAAL-style combined error scoring, and device robustness analysis.

---

### Machine Learning Lifecycle Walkthrough:
1. **Problem Framing, Data Loading & EDA** (Quality checks, sparsity, near-zero RSSI audit, phone overlap)
2. **Preprocessing** (Train-only deduplication, dead AP removal, fixed sentinel remapping, train-only scaling)
3. **Feature Engineering & Representations** (936 features: RSSI, binary absence flags, physical aggregates, discrete fingerprint)
4. **Model Training** (Random Forest & XGBoost × 3 tasks = 6 estimators)
5. **Evaluation** (Building/floor accuracy, explicit coordinate MAE, Euclidean error, custom hierarchical score)
6. **Visualization** (Floor confusion matrices, coordinate error maps, error vector displacement, CDF comparison)
7. **Device Robustness** (Handset hardware shift: seen vs unseen phone evaluation)
""")

code(r"""
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), "src"))
sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
%matplotlib inline
from IPython.display import Image, display

def show(*names):
    for n in names:
        p = os.path.join("outputs", n)
        if os.path.exists(p):
            display(Image(filename=p))
        else:
            print(f"File not found: {p}")
""")

# ---------------------------------------------------------- part 1 ----
md(r"""
## Part 1 — Problem Framing, Data Loading & EDA (Member 1 Lead)

**Framing:** 520 Wi-Fi access-point RSSI readings → predict building (3 classes), floor (5 classes), and coordinates (x, y). The baseline treats targets independently without multitask weight sharing.

**Metrics (Fixed Upfront):**
- Building & floor classification accuracy + floor macro-F1
- Coordinate regression MAE (LONGITUDE, LATITUDE, and Average MAE)
- Positioning error: Euclidean distance in meters (mean, median, P75, P90)
- Custom hierarchical combined score: `Euclidean_error + 4.0 * (floor_wrong) + 50.0 * (building_wrong)`

**Split Strategy:**
- **Official Held-Out Test Set (`validationData.csv`):** 1,111 samples collected months later by different operators/devices. We never re-normalize the validation set on its own statistics.
- **USERID GroupKFold:** 5-fold cross-validation partitioned strictly by `USERID` on the training set (generated via `scripts/build_splits.py`), guaranteeing zero user leakage between training and validation folds.
""")

code("import part1_eda\npart1_eda.main()")

md("**EDA Visualizations:**")
code('show("eda_spatial_layout.png", "eda_floor_per_building.png", "eda_aps_per_sample.png")')

md(r"""
*Key EDA Insights:*
1. **Spatial Layout:** The three buildings are physically isolated, making building classification near-perfect (>99.7%).
2. **Signal Sparsity:** Signals are **96.5% sparse** — the median sample hears only ~17 of 520 APs. 55 WAPs are never heard in training data.
3. **Data Quality & Near-Zero RSSI Audit:** 31 readings occur in the anomalous `[-5, 0]` dBm range (antenna saturation / physical proximity).
4. **Phone Overlap Fact:** Validation is a **mixed seen/unseen handset evaluation**. PHONEIDs 13 and 14 appear in both sets (accounting for 397 validation rows), while 714 rows come from 9 phones unseen in training.
""")

# ---------------------------------------------------------- part 2 ----
md(r"""
## Part 2 — Preprocessing (Members 1 & 2)

Transforms are fit on **training data only** and applied unchanged to validation:

1. **Deduplication:** 637 exact-duplicate scans in `trainingData.csv` are removed (19,937 → 19,300), preventing repeated samples from distorting diagnostics. Validation data is kept completely untouched.
2. **Sentinel Remapping:** UJIIndoorLoc uses `+100 dBm` for undetected APs. We remap this to a fixed physical floor `NO_SIGNAL_FILL = -105 dBm` (just below the weakest real observation of -104 dBm).
3. **Dead AP Removal:** 55 WAPs never heard in training are dropped, reducing dimensionality to 465 active WAPs.
4. **Train-Only Scaling:** `StandardScaler` is fitted strictly on the training set and applied to validation, preserving real distribution shifts.
""")

code("import part2_preprocess\npart2_preprocess.main()")

# ---------------------------------------------------------- part 3 ----
md(r"""
## Part 3 — Feature Engineering & Representations (Member 2 Lead)

On top of the 465 kept RSSI columns, we construct three complementary representations:

1. **465 Binary Absence Indicators (`WAP###_absent`):** While standard scaling is an invertible linear transform, decision trees make axis-aligned splits on single features. After standardization, missing values map to varying continuous floats across APs. Explicit binary flags provide an immediate, single-split boolean representation of AP visibility.
2. **5 Physical Signal Aggregates:** Derived from raw dBm readings before scaling:
   - `signal_count`: count of APs heard (RSSI > -105 dBm)
   - `strongest_signal`: maximum heard RSSI (closest AP)
   - `mean_rssi` / `std_rssi` / `range_rssi`: dispersion and dynamic range of heard signals
3. **1 Discrete Fingerprint:** `strongest_wap_id` (index of loudest AP, -1 if no signal heard).

Total Feature Space: **936 features** (465 RSSI + 465 absence + 5 aggregates + 1 ordinal).

### Feature Ablation & kNN Baselines:
Evaluated on three representation modes (A1: RSSI only, A2: RSSI + Absence, A3: Full 936 features).
""")

code("import part3_features\npart3_features.main()")

code("import run_knn_baselines\nrun_knn_baselines.run_ablation_and_knn()")

md(r"""
*Ablation Read:* The full engineered feature set (**A3: 936 features**) improves kNN floor accuracy from **80.65% to 84.34%** and reduces mean positioning error to **11.77 m** (P95 error down to 37.75 m), confirming the empirical value of the summary statistics.
""")

# ---------------------------------------------------------- part 4 ----
md(r"""
## Part 4 — Model Training: AML Baseline

Two model families × three tasks = **6 independent estimators**:
- **Random Forest:** `n_estimators=200`, `n_jobs=-1` (native multi-output coordinate regression).
- **XGBoost:** `n_estimators=300`, `max_depth=8`, `learning_rate=0.1`, `subsample=0.9`, `colsample_bytree=0.8` (wrapped in `MultiOutputRegressor` for 2D coordinates).

Trained on the 936-feature design matrix. Models and validation predictions are saved to `outputs/` for zero-leakage evaluation.
""")

code("import part4_train\npart4_train.main()")

# ---------------------------------------------------------- part 5 ----
md(r"""
## Part 5 — Evaluation (Explicit Coordinate MAE & Metrics)

Full metric suite evaluated on the official 1,111-row validation set:
- Explicit per-axis Coordinate MAE (`LONGITUDE`, `LATITUDE`, and average MAE)
- Positioning error (Euclidean distance: mean, median, P75, P90)
- EvAAL custom hierarchical score: `Euclidean + 4 m * (wrong floor) + 50 m * (wrong building)`
- Breakdown of positioning error conditioned on building/floor correctness
""")

code("import part5_evaluate\npart5_evaluate.main()")

md(r"""
*Evaluation Reads:*
- **Building Classification:** Near perfect for both RF (99.91%) and XGBoost (99.73%).
- **Floor Classification:** XGBoost wins floor accuracy (**91.99%** vs 90.73%, macro-F1 0.916).
- **Coordinates:** RandomForest dominates continuous coordinates (**mean 11.58 m, median 6.86 m, average MAE 7.49 m** vs XGBoost mean 14.52 m, median 10.06 m, average MAE 9.18 m).
- **Coupling Effect:** Positioning error jumps ~2.5× (from ~10 m to ~25–28 m) when building or floor is misclassified, providing direct empirical proof that the three targets are tightly coupled.
- **Combined Row:** A post-hoc exploratory combination routing each target to its stronger family achieves 99.91% building, 91.99% floor, and 11.58 m coordinate error (11.95 m combined score).
""")

# ---------------------------------------------------------- part 6 ----
md(r"""
## Part 6 — Visualization (Figure Deliverables)

Visual diagnostic deliverables:
1. Floor confusion matrices
2. Coordinate error maps (spatial distribution of errors)
3. Error vector displacements (worst 10% errors from true to predicted coordinates)
4. Error Cumulative Distribution Function (CDF) comparison
""")

code("import part6_visualize\npart6_visualize.main()")

md("**Floor Confusion Matrices:**")
code('show("confusion_floor_RandomForest.png", "confusion_floor_XGBoost.png")')

md("**Coordinate Error Maps & Worst-Case Vector Displacements:**")
code('show("error_map_RandomForest.png", "error_map_XGBoost.png", "error_vectors_RandomForest.png", "error_vectors_XGBoost.png")')

md("**Coordinate Error CDF (P(error ≤ d)):**")
code('show("error_cdf_comparison.png")')

md(r"""
*Visual Diagnostic Reads:*
- Floor confusions are **mostly adjacent** (inter-floor RF bleed), with rare corner cases (e.g. true floor 4 predicted as floor 0 due to limited floor 4 samples in Building 2).
- Highest coordinate errors concentrate along building outer edges and corners where trilateration coverage weakens.
- The CDF shows RandomForest consistently outperforming XGBoost across all percentiles.
""")

# ---------------------------------------------------------- part 7 ----
md(r"""
## Part 7 — Device Robustness (Hardware Shift Analysis)

Real-world Wi-Fi positioning is sensitive to device antenna calibration.
Validation data contains **11 handsets**:
- **Shared (Seen) Handsets:** Phones 13 and 14 (397 rows)
- **Unseen Handsets:** 9 phones (714 rows)
""")

code("import part7_robustness\npart7_robustness.main()")

md("**Per-Handset Error Distribution:**")
code('show("robustness_by_phone.png")')

md(r"""
*Robustness Reads:*
- Handset variance is substantial (~4× spread between Phone 9 at 7.65 m and Phone 14 at 30.45 m for RF).
- Performance on shared phones (RF mean 10.13 m) is slightly lower than on unseen phones (RF mean 12.39 m), providing descriptive evidence of device-level domain shift.
- Aggregate metrics alone mask this per-device risk.
""")

# ---------------------------------------------------------- summary ----
md(r"""
## Phase 1 Summary & Transition to Phase 2

| Metric | Random Forest | XGBoost | Combined (Post-Hoc) |
|---|---|---|---|
| **Building Accuracy** | **99.91%** | 99.73% | **99.91%** |
| **Floor Accuracy** | 90.73% | **91.99%** | **91.99%** |
| **Floor Macro-F1** | 0.893 | **0.916** | **0.916** |
| **Coordinate MAE (Lon / Lat / Avg)** | **8.19 / 6.80 / 7.49 m** | 9.82 / 8.54 / 9.18 m | **8.19 / 6.80 / 7.49 m** |
| **Coordinate Error (Mean / Median / P90)** | **11.58 / 6.86 / 24.99 m** | 14.52 / 10.06 / 30.38 m | **11.58 / 6.86 / 24.99 m** |
| **Custom Combined Score (Mean / Median)** | 12.00 / 7.02 m | 14.97 / 10.37 m | **11.95 / 7.00 m** |

### Key Motivations for Phase 2 Multitask Deep Learning:
1. **Target Coupling:** Independent tabular models cannot share latent spatial representations; coordinate error spikes when floor is misclassified.
2. **Complementary Strengths:** RF excels at continuous spatial regression, while gradient boosting excels at discrete floor boundaries.
3. **Phase 2 Architecture:** A shared deep-learning MLP trunk with joint classification and regression heads will explicitly leverage inter-target dependencies while sharing representations across building, floor, and coordinate spaces.
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    },
    "language_info": {"name": "python"},
}

out = "Phase1_AML_Baseline.ipynb"
with open(out, "w") as f:
    nbf.write(nb, f)
print(f"Successfully assembled {out} with {len(cells)} cells.")
