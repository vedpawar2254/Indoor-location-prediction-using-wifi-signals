#!/usr/bin/env python3
"""Master one-command pipeline runner for Phase 2 (Deep Learning & Hybrid).

Executes all Phase 2 modules in clean isolated subprocesses (preserving Phase 1
artifacts while preventing macOS OpenMP collisions between PyTorch and XGBoost):
  1. 5-Fold USERID GroupKFold DL Training (SingleTaskMLP, MTL_MLP, SAE_MTL, CNN1D_MTL, HierResMTL)
  2. Controlled Ablation Studies (Representations, Hygiene, Regularization, Mechanism)
  3. ML + DL Hybrid Modeling & Nested OOF Validation (HierResMTL-Hybrid)
  4. Unified Evaluation, Bootstrap CIs, McNemar Tests & Device Robustness
  5. Diagnostic & Comparative Visualization Suite
  6. Deliverable Notebook Assembly & Execution (`Phase2_DL_Hybrid.ipynb`)
"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable


def run_step(step_name: str, cmd: list):
    print("\n" + "=" * 75)
    print(f"PHASE 2 STEP: {step_name}")
    print("=" * 75)
    t0 = time.time()
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    res = subprocess.run(cmd, cwd=str(ROOT), env=env)
    if res.returncode != 0:
        print(f"\n✗ Error in step '{step_name}' (exit {res.returncode})", file=sys.stderr)
        sys.exit(res.returncode)
    print(f"✓ Completed in {time.time() - t0:.1f}s")


def main():
    t_start = time.time()
    print("=" * 75)
    print("STARTING COMPLETE PHASE 2 DEEP LEARNING & HYBRID PIPELINE")
    print("=" * 75)

    run_step("1. 5-Fold USERID CV DL Benchmark (5 Architectures)", [PYTHON, "src/phase2_train_cv.py"])
    run_step("2. Controlled Representation & Regularization Ablations", [PYTHON, "src/phase2_ablations.py"])
    run_step("3. ML + DL Hybrid Integration & Nested OOF Validation", [PYTHON, "src/phase2_hybrid.py"])
    run_step("4. Unified Evaluation, Bootstrap CIs & Device Robustness", [PYTHON, "src/phase2_evaluate.py"])
    run_step("5. Phase 2 Visualization Suite", [PYTHON, "src/phase2_visualize.py"])
    if (ROOT / "outputs" / "build_phase2_notebook.py").exists():
        run_step("6. Phase 2 Notebook Compilation", [PYTHON, "outputs/build_phase2_notebook.py"])

    print("\n" + "=" * 75)
    print(f"✓ ALL PHASE 2 STEPS COMPLETED IN {time.time() - t_start:.1f}s!")
    print("=" * 75)


if __name__ == "__main__":
    main()

