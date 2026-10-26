#!/usr/bin/env python3
"""Master pipeline runner for Indoor Location Prediction (Phase 1 AML Baseline).

Executes the entire lifecycle sequentially:
  1. Data verification & quality checks (Part 1 EDA)
  2. USERID GroupKFold splits generation (build_splits.py)
  3. Preprocessing & scaling verification (Part 2)
  4. Feature engineering & ablation (Part 3 & run_knn_baselines.py)
  5. Baseline model training: RF & XGBoost (Part 4)
  6. Evaluation & explicit Coordinate MAE (Part 5)
  7. Visual diagnostic figures generation (Part 6)
  8. Device robustness & seen/unseen handset analysis (Part 7)
  9. Deliverable notebook assembly (build_notebook.py)

Usage:
  python scripts/run_pipeline.py
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable


def run_step(step_name: str, cmd: list):
    print("\n" + "=" * 75)
    print(f"PIPELINE STEP: {step_name}")
    print("=" * 75)
    t0 = time.time()
    env = {**subprocess.os.environ, "PYTHONPATH": str(ROOT / "src") + ":" + str(ROOT / "scripts")}
    res = subprocess.run(cmd, cwd=str(ROOT), env=env)
    elapsed = time.time() - t0
    if res.returncode != 0:
        print(f"\n✗ Error in step '{step_name}' (exit code {res.returncode})", file=sys.stderr)
        sys.exit(res.returncode)
    print(f"✓ Step completed in {elapsed:.2f}s")


def main():
    print("=" * 75)
    print("STARTING COMPLETE PHASE 1 AML BASELINE PIPELINE")
    print(f"Root: {ROOT}")
    print(f"Python: {PYTHON}")
    print("=" * 75)
    t_start = time.time()

    run_step("1. Data Download & Verification", [PYTHON, "scripts/download_data.py"])
    run_step("2. USERID-Grouped Folds Generation (M1)", [PYTHON, "scripts/build_splits.py"])
    run_step("3. Exploratory Data Analysis & Sparsity Audit (M1)", [PYTHON, "src/part1_eda.py"])
    run_step("4. Preprocessing & Dead AP Filtering (M1/M2)", [PYTHON, "src/part2_preprocess.py"])
    run_step("5. Feature Engineering & Absence Indicators (M2)", [PYTHON, "src/part3_features.py"])
    run_step("6. Feature Ablation & kNN Baselines (M2)", [PYTHON, "scripts/run_knn_baselines.py"])
    run_step("7. Model Training: RF & XGBoost Baseline", [PYTHON, "src/part4_train.py"])
    run_step("8. Unified Evaluation & Coordinate MAE", [PYTHON, "src/part5_evaluate.py"])
    run_step("9. Diagnostic Visualizations", [PYTHON, "src/part6_visualize.py"])
    run_step("10. Device Robustness: Seen vs Unseen Handsets", [PYTHON, "src/part7_robustness.py"])
    run_step("11. Notebook Compilation", [PYTHON, "outputs/build_notebook.py"])

    total_time = time.time() - t_start
    print("\n" + "=" * 75)
    print(f"✓ ALL PHASE 1 PIPELINE STEPS COMPLETED SUCCESSFULLY IN {total_time:.1f}s!")
    print("Outputs available in outputs/ and data/splits/")
    print("=" * 75)


if __name__ == "__main__":
    main()
