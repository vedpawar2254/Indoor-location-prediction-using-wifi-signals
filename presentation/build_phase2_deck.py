"""Builds Phase 2 Presentation Deck in both PPTX and PDF formats.

Generates:
  - presentation/Phase2_Presentation.pptx (via python-pptx, 16:9 widescreen)
  - presentation/Phase2_Presentation.pdf  (via reportlab, 16:9 widescreen)

Contains 12 comprehensive slides covering:
  1. Title, Team T054, Key Metrics
  2. Phase 1 Recap & Why Deep Learning?
  3. Data Hygiene & Zero-Leakage Grouped Partitioning
  4. Device-Invariant Multiview Representation (R3)
  5. HierResMTL: Deep Multitask Residual Architecture
  6. HierResMTL-Hybrid: The Adaptive ML+DL Integration
  7. Official 1,111 Validation Benchmark
  8. 5-Fold USERID CV (19,300 Held-Out User Scans)
  9. Statistical Significance & Controlled Ablation Suite
 10. Spatial Error CDF & Floor Confusion Diagnostics
 11. Handset Device Shift & Hardware Generalization
 12. Individual Member Contributions (M1-M5) & Deliverables
"""
import sys
from pathlib import Path
import pptx
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "report" / "figs_phase2"
OUT_DIR = ROOT / "presentation"

# Color Palette
NAVY   = RGBColor(16, 42, 67)      # #102A43
BLUE   = RGBColor(43, 108, 176)    # #2B6CB0
TEAL   = RGBColor(15, 118, 110)    # #0F766E
GREEN  = RGBColor(4, 120, 87)      # #047857
AMBER  = RGBColor(217, 119, 6)     # #D97706
RED    = RGBColor(220, 38, 38)     # #DC2626
GRAY   = RGBColor(100, 116, 139)   # #64748B
LIGHT_BG = RGBColor(248, 250, 252) # #F8FAFC
CARD_BG  = RGBColor(241, 245, 249) # #F1F5F9


def build_pptx():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    def add_header(slide, title_text, category="PHASE 2 DEEP LEARNING & HYBRID ML+DL"):
        # Category pill
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.35))
        tf_c = cat_box.text_frame
        tf_c.word_wrap = True
        p_c = tf_c.paragraphs[0]
        p_c.text = category.upper()
        p_c.font.size = Pt(10)
        p_c.font.bold = True
        p_c.font.color.rgb = TEAL

        # Title
        t_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.7), Inches(11.7), Inches(0.6))
        tf_t = t_box.text_frame
        tf_t.word_wrap = True
        p_t = tf_t.paragraphs[0]
        p_t.text = title_text
        p_t.font.size = Pt(22)
        p_t.font.bold = True
        p_t.font.color.rgb = NAVY

    def add_card(slide, left, top, width, height, title, body_bullets, bg_col=CARD_BG, title_col=NAVY):
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        shape.fill.solid()
        shape.fill.fore_color.rgb = bg_col
        shape.line.color.rgb = RGBColor(203, 213, 225)
        shape.line.width = Pt(1)

        txBox = slide.shapes.add_textbox(left + Inches(0.15), top + Inches(0.12), width - Inches(0.3), height - Inches(0.24))
        tf = txBox.text_frame
        tf.word_wrap = True

        p0 = tf.paragraphs[0]
        p0.text = title
        p0.font.bold = True
        p0.font.size = Pt(13)
        p0.font.color.rgb = title_col

        for b in body_bullets:
            p = tf.add_paragraph()
            p.text = f"• {b}"
            p.font.size = Pt(10.5)
            p.font.color.rgb = RGBColor(30, 41, 59)
            p.space_before = Pt(4)

    # -------------------------------------------------------------
    # SLIDE 1: Title Slide
    # -------------------------------------------------------------
    s1 = prs.slides.add_slide(blank_layout)
    bg1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = RGBColor(15, 23, 42) # Slate 900
    bg1.line.fill.background()

    tb1 = s1.shapes.add_textbox(Inches(1.0), Inches(1.2), Inches(11.333), Inches(3.2))
    tf1 = tb1.text_frame
    tf1.word_wrap = True

    p = tf1.paragraphs[0]
    p.text = "DEEP LEARNING COURSE PROJECT 9"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = RGBColor(45, 212, 191) # Teal 400

    p = tf1.add_paragraph()
    p.text = "Predicting Indoor Location from Wi-Fi Signals via Hierarchical Multitask Residual Networks & Adaptive Hybrid Retrieval"
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = RGBColor(255, 255, 255)
    p.space_before = Pt(10)

    p = tf1.add_paragraph()
    p.text = "Phase 2: Deep Learning & Hybrid ML+DL on UJIIndoorLoc Benchmark"
    p.font.size = Pt(16)
    p.font.color.rgb = RGBColor(148, 163, 184)
    p.space_before = Pt(8)

    # Team Box
    team_box = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(4.5), Inches(11.333), Inches(2.2))
    team_box.fill.solid()
    team_box.fill.fore_color.rgb = RGBColor(30, 41, 59)
    team_box.line.color.rgb = RGBColor(51, 65, 85)

    tb_team = s1.shapes.add_textbox(Inches(1.2), Inches(4.6), Inches(10.9), Inches(2.0))
    tf_team = tb_team.text_frame
    p = tf_team.paragraphs[0]
    p.text = "TEAM T054 | Section E, BTech CSE (AI), Newton School of Technology, Rishihood University"
    p.font.bold = True
    p.font.size = Pt(13)
    p.font.color.rgb = RGBColor(241, 245, 249)

    members = [
        ("Aviral Mishra", "2401010115", "M1: Data Hygiene, DataLoader & Robustness"),
        ("Ved Pawar", "2401010498", "M2: R3 Representation & Hybrid ML+DL Method"),
        ("Khuswant Rajpurohit", "2401010228", "M3: HierResMTL Multitask Network & FiLM"),
        ("Prataya Silla", "2401010341", "M4: 5-Fold USERID Tuning & Statistical Validation"),
        ("Vansh Jain", "2401010492", "M5: Controlled A1-A9 Ablations & Integration"),
    ]
    for name, roll, role in members:
        p = tf_team.add_paragraph()
        p.text = f"{name} ({roll}) — {role}"
        p.font.size = Pt(10)
        p.font.color.rgb = RGBColor(203, 213, 225)
        p.space_before = Pt(2)

    # -------------------------------------------------------------
    # SLIDE 2: Phase 1 Recap & Why Deep Learning?
    # -------------------------------------------------------------
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "Phase 1 Baseline Recap & The Motivation for Deep Multitask Learning")

    add_card(s2, Inches(0.8), Inches(1.5), Inches(3.7), Inches(5.3),
             "Phase 1 Achievements", [
                 "Clean baseline on 19,300 deduplicated scans.",
                 "Random Forest: 99.91% Building, 6.86 m Median Coordinate Error.",
                 "XGBoost: 91.99% Floor Accuracy.",
                 "Best-per-Task Oracle (not a deployable model): 11.58 m Mean / 6.86 m Median / 11.95 m EvAAL.",
                 "Zero validation leakage maintained throughout."
             ], bg_col=RGBColor(240, 249, 255), title_col=BLUE)

    add_card(s2, Inches(4.8), Inches(1.5), Inches(3.7), Inches(5.3),
             "Critical Bottlenecks in Trees", [
                 "Target Decoupling: Building, Floor, and Coordinates trained in silos with zero shared representation.",
                 "Severe Error Cascading: When Floor failed, RF coordinate error exploded from 10.08 m to 28.57 m (+183%).",
                 "Axis-Aligned Splits: Fails to model continuous smooth radio propagation manifolds.",
                 "Sparse Missing Constants: Arbitrary -105 dBm floor creates artificial step discontinuities."
             ], bg_col=RGBColor(254, 242, 242), title_col=RED)

    add_card(s2, Inches(8.8), Inches(1.5), Inches(3.7), Inches(5.3),
             "Phase 2 Deep Learning Goals", [
                 "Joint Representation: Shared deep neural trunk forcing mutual information across targets.",
                 "Hardware Invariance: Scan-relative decay cancelling antenna offsets across devices.",
                 "Hierarchical Conditioning: Predict Building first -> modulate Floor & Coordinate features via FiLM.",
                 "True Hybrid Integration: Combine smooth neural regression with sharp local corridor non-parametric retrieval."
             ], bg_col=RGBColor(236, 253, 245), title_col=GREEN)

    # -------------------------------------------------------------
    # SLIDE 3: Data Hygiene & Grouped Partitioning
    # -------------------------------------------------------------
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "Rigorous Data Hygiene & USERID Grouped Folds")

    add_card(s3, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.3),
             "Dataset Hygiene Audits (M1)", [
                 "19,937 Raw Training Scans -> 19,300 Unique Scans: Removed 637 exact duplicates.",
                 "73 Zero-AP 'Blind' Scans Discovered: Contained 0 detected WAPs above noise floor (-105 dBm) yet had contradictory ground truth labels across B0, B1, and B2.",
                 "Training-Only Filtering: Dropping 73 blind scans strictly from DL training eliminates gradient noise, while keeping the G1 No-Signal Guard at inference.",
                 "Official Validation Pristine: Exactly 1,111 rows, 0 blind scans, completely untouched during training and tuning."
             ], bg_col=CARD_BG, title_col=NAVY)

    add_card(s3, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.3),
             "USERID GroupKFold CV (M4)", [
                 "User-Level Splitting: Training partition grouped across USERIDs so no user appears in both train and validation folds.",
                 "Prevents Identity Leakage: A user's physical stature, walking speed, and phone carry habits do not leak across folds.",
                 "Fold-Safe Fit: Scalers, positive-power transforms, and normalizers fitted strictly on training partition of that fold.",
                 "Zero Post-Hoc Cheating: Early stopping monitored strictly on held-out USERID fold, never on official validation."
             ], bg_col=RGBColor(245, 243, 255), title_col=RGBColor(109, 40, 217))

    # -------------------------------------------------------------
    # SLIDE 4: Representation (R1 -> R3)
    # -------------------------------------------------------------
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "Input Representation: Device-Invariant Multiview (R3)")

    add_card(s4, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.3),
             "Mathematical Formulation (M2)", [
                 "1. Positive-Powed RSSI (465 dims): r_ij = ((x_ij - min)/(max - min))^e. Compresses weak signals and zeros out missing APs smoothly.",
                 "2. Scan-Relative Exponential Decay (465 dims): exp((x_ij - max_k x_ik) / 20). Mathematically cancels additive handset amplifier gain: (x_ij + Delta_p) - (max_k x_ik + Delta_p) = x_ij - max_k x_ik.",
                 "3. Binary Presence Indicators (465 dims): Explicit AP detection mask.",
                 "4. Physical Scan Summaries (5 dims): AP count, max RSSI, mean, std, and range of detected signals."
             ], bg_col=RGBColor(240, 253, 250), title_col=TEAL)

    add_card(s4, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.3),
             "Ablation Proof: R1 vs R2 vs R3", [
                 "R1 (Phase 1 StandardScaler): Fails completely on DL! EvAAL error: 35.88 m. Standardizing missing constants (+100 -> -105) wrecks neural gradients.",
                 "R2 (Positive-Powed RSSI): Recovers floor (86.3%), but produces severe coordinate tail errors (37.18 m EvAAL, 30.65 m mean).",
                 "R3-noRel (Powed + Mask + Agg): Slashes EvAAL to 12.39 m.",
                 "R3 Full (+Scan-Relative Decay): ~13.0 m EvAAL on Fold 0, a >23 m gain over R1 — the decisive factor; mitigates Phone 14 hardware bias."
             ], bg_col=CARD_BG, title_col=NAVY)

    # -------------------------------------------------------------
    # SLIDE 5: HierResMTL Deep Architecture
    # -------------------------------------------------------------
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "HierResMTL: Hierarchical Residual Multitask Network (M3)")

    if (FIGS / "p2_architecture_diagram.png").exists():
        s5.shapes.add_picture(str(FIGS / "p2_architecture_diagram.png"), Inches(0.8), Inches(1.5), Inches(6.8), Inches(5.3))

    add_card(s5, Inches(7.8), Inches(1.5), Inches(4.7), Inches(5.3),
             "Architectural Highlights", [
                 "Strongest-AP Embedding: nn.Embedding(466, 16) injects global topological anchor identity.",
                 "Pre-LN Residual Trunk: 2 Pre-LayerNorm Residual blocks (dim=384) with GELU activations and LayerNorm.",
                 "FiLM Building Conditioning: Building softmax p_b modulates intermediate features for Floor & Coordinates: h_cond = h * (1 + gamma(p_b)) + beta(p_b).",
                 "Kendall Uncertainty Loss: Learns homoscedastic task weights exp(-s_k)*L_k + 0.5*s_k, eliminating manual weight tuning.",
                 "Domain Regularization: 12% AP-Block dropout + Gaussian jitter (0.02) + rare Floor-4 inverse-sqrt reweighting."
             ], bg_col=CARD_BG, title_col=NAVY)

    # -------------------------------------------------------------
    # SLIDE 6: Hybrid Method
    # -------------------------------------------------------------
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "HierResMTL-Hybrid: Genuine ML + DL Integration (M2)")

    add_card(s6, Inches(0.8), Inches(1.5), Inches(3.7), Inches(5.3),
             "Stage 1: Deep Trunk", [
                 "5-fold ensemble HierResMTL generates calibrated building (p_b) and floor (p_f) probabilities.",
                 "Neural manifold outputs smooth global continuous coordinates y_DL.",
                 "Suppresses tail errors across unseen user partitions."
             ], bg_col=RGBColor(238, 242, 255), title_col=BLUE)

    add_card(s6, Inches(4.8), Inches(1.5), Inches(3.7), Inches(5.3),
             "Stage 2: Discrete Fusion", [
                 "OOF Log-Loss Probability Fusion: Combines HierResMTL + Trees + R3-kNN.",
                 "Structural Floor Mask: Floor 4 exists only in Building 2. Whenever pred Building != 2, Floor 4 prob is strictly set to 0.0.",
                 "Reaches 95.50% Floor Accuracy on official validation."
             ], bg_col=RGBColor(245, 243, 255), title_col=RGBColor(109, 40, 217))

    add_card(s6, Inches(8.8), Inches(1.5), Inches(3.7), Inches(5.3),
             "Stage 3: Adaptive Gate", [
                 "Building-Conditioned Hier-R3-kNN (k=7) retrieves local corridor points within predicted building.",
                 "Adaptive Similarity Gate: w(s_max) = 0.70 * clip((s_max - 0.50)/0.50, 0, 1).",
                 "High similarity -> kNN sharp corridor resolution (up to 70%). Low similarity -> fallback to pure neural manifold."
             ], bg_col=RGBColor(236, 253, 245), title_col=GREEN)

    # -------------------------------------------------------------
    # SLIDE 7: Official Validation Benchmark
    # -------------------------------------------------------------
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "Benchmark on Official 1,111-Sample Validation Set")

    if (FIGS / "p2_model_comparison.png").exists():
        s7.shapes.add_picture(str(FIGS / "p2_model_comparison.png"), Inches(0.8), Inches(1.5), Inches(6.8), Inches(5.3))

    add_card(s7, Inches(7.8), Inches(1.5), Inches(4.7), Inches(5.3),
             "Results Comparison", [
                 "Phase 1 RF: 99.91% Bld | 90.73% Flr | 11.58 m Mean | 12.00 m EvAAL.",
                 "Phase 1 XGB: 99.73% Bld | 91.99% Flr | 14.52 m Mean | 14.97 m EvAAL.",
                 "Pure DL (HierResMTL): 99.91% Bld | 92.98% Flr | 7.77 m Mean | 8.09 m EvAAL.",
                 "Strawman Naive Blend (0.5 RF+DL): Degrades DL to 8.96 m Mean / 9.29 m EvAAL!",
                 "HierResMTL-Hybrid (Ours): 99.91% Bld | 95.50% Flr (+4.77%) | 4.29 m MAE | 6.76 m Mean | 4.98 m Median | 13.88 m P90 | 6.99 m EvAAL (-41.8%)."
             ], bg_col=RGBColor(236, 253, 245), title_col=GREEN)

    # -------------------------------------------------------------
    # SLIDE 8: 5-Fold USERID CV (19,300 Held-Out Scans)
    # -------------------------------------------------------------
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "5-Fold USERID Cross-Validation: Generalization on 19,300 Held-Out Scans")

    add_card(s8, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.3),
             "Held-Out User Generalization (M4)", [
                 "Phase 1 RF: OOF Mean 16.73 m | Med 9.90 m | EvAAL 18.21 m (CV Fold 18.37 ± 3.95 m).",
                 "Phase 1 XGB: OOF Mean 20.34 m | Med 14.14 m | EvAAL 22.22 m (CV Fold 22.03 ± 5.57 m).",
                 "HierResMTL (Pure DL): OOF Mean 9.33 m | Med 6.18 m | EvAAL 10.93 m (CV Fold 11.03 ± 3.01 m).",
                 "HierResMTL-Hybrid (Ours): OOF Mean 8.86 m | Med 5.33 m | EvAAL 10.55 m (CV Fold 10.68 ± 3.74 m).",
                 "Conclusion: Hybrid is best across ALL models on unseen users, proving genuine generalization!"
             ], bg_col=CARD_BG, title_col=NAVY)

    add_card(s8, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.3),
             "Literature Comparison", [
                 "Nowicki & Wietrzykowski (2017) SAE-MTL: 10.48 m OOF Mean, 12.61 m EvAAL. Our Pre-LN ResNet outperforms autoencoder pretraining.",
                 "Song et al. (2019) CNNLoc: 24.75 m OOF Mean, 27.61 m EvAAL. 1D CNNs suffer severe negative transfer on unordered AP channels.",
                 "SingleTask vs Multitask: HierResMTL (10.93 m) outperforms SingleTaskMLP (11.79 m) and plain MTL-MLP (13.33 m) on held-out users."
             ], bg_col=RGBColor(240, 249, 255), title_col=BLUE)

    # -------------------------------------------------------------
    # SLIDE 9: Statistical Significance & Controlled Ablations
    # -------------------------------------------------------------
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "Statistical Validation & Controlled Single-Factor Ablations")

    if (FIGS / "p2_ablation_study.png").exists():
        s9.shapes.add_picture(str(FIGS / "p2_ablation_study.png"), Inches(0.8), Inches(1.5), Inches(6.8), Inches(5.3))

    add_card(s9, Inches(7.8), Inches(1.5), Inches(4.7), Inches(5.3),
             "Statistical Rigor (M4 & M5)", [
                 "Paired Bootstrap 95% CIs (1,000 resamples): Hybrid beats Phase 1 RF by 4.82 m [3.92, 5.66] m, and beats Pure DL by 1.29 m [1.06, 1.52] m.",
                 "McNemar Significance Test: Floor classification difference has p = 4.2e-8 (49 wins vs 8 losses vs Phase 1 XGB).",
                 "Ablation: w/o Blind-Scan Filter -> +2.03 m EvAAL.",
                 "Ablation: w/o AP Dropout & Jitter -> +2.94 m EvAAL.",
                 "Ablation: w/o Kendall Loss -> +2.72 m EvAAL.",
                 "Ablation: w/o FiLM Conditioning -> +0.95 m EvAAL."
             ], bg_col=CARD_BG, title_col=NAVY)

    # -------------------------------------------------------------
    # SLIDE 10: Spatial Error CDF & Floor Confusion
    # -------------------------------------------------------------
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "Spatial Error CDF & Floor Confusion Analysis")

    if (FIGS / "p2_error_cdf_comparison.png").exists():
        s10.shapes.add_picture(str(FIGS / "p2_error_cdf_comparison.png"), Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.3))

    if (FIGS / "p2_confusion_floor_hybrid.png").exists():
        s10.shapes.add_picture(str(FIGS / "p2_confusion_floor_hybrid.png"), Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.3))

    # -------------------------------------------------------------
    # SLIDE 11: Hardware Robustness
    # -------------------------------------------------------------
    s11 = prs.slides.add_slide(blank_layout)
    add_header(s11, "Device Robustness: Curing Handset Gain Discrepancies")

    if (FIGS / "p2_robustness_by_phone.png").exists():
        s11.shapes.add_picture(str(FIGS / "p2_robustness_by_phone.png"), Inches(0.8), Inches(1.5), Inches(6.8), Inches(5.3))

    add_card(s11, Inches(7.8), Inches(1.5), Inches(4.7), Inches(5.3),
             "Hardware Shift Highlights", [
                 "Phone 14 Outlier Recovery: Phase 1 RF mean error was 30.45 m (P95 116.09 m). Hybrid reduces it to 9.70 m (P95 27.77 m) — a 68.1% reduction.",
                 "Scan-Relative Invariance: Normalizing by max observed RSSI cancels additive device gain offsets Delta_p.",
                 "Seen vs Unseen Generalization: Floor accuracy on 9 unseen phones (714 scans) is 95.66%, virtually identical to 95.72% on seen phones.",
                 "Extreme tails suppressed across all 11 validation handsets."
             ], bg_col=RGBColor(236, 253, 245), title_col=GREEN)

    # -------------------------------------------------------------
    # SLIDE 12: Team Contributions & Summary
    # -------------------------------------------------------------
    s12 = prs.slides.add_slide(blank_layout)
    add_header(s12, "Team Contributions & Complete Deliverables Package")

    add_card(s12, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.3),
             "Member Contributions (M1 - M5)", [
                 "Aviral Mishra (M1): PyTorch Dataset/DataLoader, blind-scan hygiene, noise augmentation (2/4 dBm), AP dropout (5/10%), regularization comparison.",
                 "Ved Pawar (M2): R3 device-invariant representation, HierResMTL-Hybrid architecture, OOF probability fusion, adaptive similarity gating.",
                 "Khuswant Rajpurohit (M3): HierResMTL deep network, Pre-LN ResNet, FiLM building conditioning, Kendall uncertainty loss.",
                 "Prataya Silla (M4): 5-fold USERID CV, 1,000 paired bootstrap 95% CIs, McNemar tests, floor ECE calibration, learning curves, failure review.",
                 "Vansh Jain (M5): Controlled A1-A9 ablations (3 seeds), device shift audits, IEEE report, presentation slides, reproducible repo."
             ], bg_col=CARD_BG, title_col=NAVY)

    add_card(s12, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.3),
             "Completed Deliverables Checklist", [
                 "✓ Project Presentation: PPTX & PDF formats (<= 10 MB).",
                 "✓ Project Report: PDF format, prepared in LaTeX IEEE style (<= 10 MB).",
                 "✓ Deliverable Jupyter Notebook: Phase2_DL_Hybrid.ipynb fully executed with all outputs and figures inline.",
                 "✓ Project Code: Clean, modular, reproducible code in GitHub repository with updated README."
             ], bg_col=RGBColor(240, 253, 250), title_col=TEAL)

    pptx_path = OUT_DIR / "Phase2_Presentation.pptx"
    prs.save(str(pptx_path))
    print(f"✓ Saved PPTX to {pptx_path} ({pptx_path.stat().st_size / (1024*1024):.2f} MB)")
    return pptx_path


def build_pdf():
    pdf_path = OUT_DIR / "Phase2_Presentation.pdf"
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=landscape(letter),
        leftMargin=36,
        rightMargin=36,
        topMargin=32,
        bottomMargin=32,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "SlideTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#102A43"),
        spaceAfter=12,
    )
    cat_style = ParagraphStyle(
        "SlideCategory",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=11,
        textColor=colors.HexColor("#0F766E"),
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "SlideBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#1E293B"),
    )
    card_title_style = ParagraphStyle(
        "CardTitle",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#102A43"),
        spaceAfter=6,
    )

    story = []

    def make_card_table(title, bullets, bg="#F1F5F9", tc="#102A43"):
        flowables = [
            Paragraph(title, ParagraphStyle("ct", parent=card_title_style, textColor=colors.HexColor(tc))),
            Spacer(1, 4),
        ]
        for b in bullets:
            flowables.append(Paragraph(f"• {b}", body_style))
            flowables.append(Spacer(1, 3))
        t = Table([[flowables]], colWidths=[225])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(bg)),
            ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#CBD5E1")),
            ("ROUNDEDCORNERS", [6, 6, 6, 6]),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ]))
        return t

    # Slide 1: Title
    story.append(Paragraph("DEEP LEARNING COURSE PROJECT 9", cat_style))
    story.append(Paragraph("Predicting Indoor Location from Wi-Fi Signals via Hierarchical Multitask Residual Networks & Adaptive Hybrid Retrieval", title_style))
    story.append(Spacer(1, 10))

    team_data = [
        [Paragraph("<b>Team T054</b> | Section E, BTech CSE (AI), Newton School of Technology, Rishihood University", body_style)],
        [Paragraph("Aviral Mishra (2401010115, M1) — Data Hygiene, DataLoader & Robustness", body_style)],
        [Paragraph("Ved Pawar (2401010498, M2) — R3 Representation & Hybrid ML+DL Method", body_style)],
        [Paragraph("Khuswant Rajpurohit (2401010228, M3) — HierResMTL Multitask Network & FiLM", body_style)],
        [Paragraph("Prataya Silla (2401010341, M4) — 5-Fold USERID Tuning & Statistical Validation", body_style)],
        [Paragraph("Vansh Jain (2401010492, M5) — Controlled A1-A9 Ablations & Integration", body_style)],
    ]
    t_team = Table(team_data, colWidths=[720])
    t_team.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ("BOX", (0, 0), (-1, -1), 1.0, colors.HexColor("#0F766E")),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(t_team)
    story.append(PageBreak())

    # Slide 2: Phase 1 Recap
    story.append(Paragraph("PHASE 2 DEEP LEARNING & HYBRID ML+DL", cat_style))
    story.append(Paragraph("Phase 1 Baseline Recap & The Motivation for Deep Multitask Learning", title_style))
    c1 = make_card_table("Phase 1 Achievements", [
        "Clean baseline on 19,300 deduplicated scans.",
        "Random Forest: 99.91% Building, 6.86 m Median Coordinate Error.",
        "XGBoost: 91.99% Floor Accuracy.",
        "Best-per-Task Oracle (not a deployable model): 11.58 m Mean / 6.86 m Median / 11.95 m EvAAL.",
        "Zero validation leakage maintained throughout."
    ], bg="#F0F9FF", tc="#2B6CB0")
    c2 = make_card_table("Critical Bottlenecks in Trees", [
        "Target Decoupling: Targets trained in silos with zero shared representation.",
        "Severe Error Cascading: When Floor failed, RF coordinate error surged to 28.57 m (+183%).",
        "Axis-Aligned Splits: Fails to model continuous smooth radio propagation manifolds.",
        "Sparse Missing Constants: -105 dBm floor creates artificial step discontinuities."
    ], bg="#FEF2F2", tc="#DC2626")
    c3 = make_card_table("Phase 2 Deep Learning Goals", [
        "Joint Representation: Shared deep neural trunk forcing mutual information across targets.",
        "Hardware Invariance: Scan-relative decay cancelling antenna offsets across devices.",
        "Hierarchical Conditioning: Predict Building first -> modulate Floor & Coordinate features via FiLM.",
        "True Hybrid Integration: Combine smooth neural regression with sharp local corridor retrieval."
    ], bg="#ECFDF5", tc="#047857")
    story.append(Table([[c1, c2, c3]], colWidths=[240, 240, 240]))
    story.append(PageBreak())

    # Slide 3: Architecture Diagram & Training Curves
    story.append(Paragraph("PHASE 2 DEEP LEARNING & HYBRID ML+DL", cat_style))
    story.append(Paragraph("Proposed HierResMTL Deep Multitask Network & Training Dynamics", title_style))
    if (FIGS / "p2_architecture_diagram.png").exists() and (FIGS / "p2_training_curves.png").exists():
        im1 = RLImage(str(FIGS / "p2_architecture_diagram.png"), width=355, height=185)
        im2 = RLImage(str(FIGS / "p2_training_curves.png"), width=355, height=185)
        story.append(Table([[im1, im2]], colWidths=[360, 360]))
    story.append(Spacer(1, 10))
    c_arch = make_card_table("Key Architectural Mechanisms", [
        "Pre-LN Residual Blocks (dim=384) + Strongest-AP Embeddings (dim=16).",
        "FiLM Building Modulation: Dynamically scales and shifts intermediate features based on building softmax p_b.",
        "Kendall Uncertainty Loss: Automatically balances classification and spatial regression tasks without manual loss weights."
    ], bg="#F8FAFC", tc="#102A43")
    story.append(Table([[c_arch]], colWidths=[720]))
    story.append(PageBreak())

    # Slide 4: Official Validation Benchmark & Model Comparison
    story.append(Paragraph("PHASE 2 DEEP LEARNING & HYBRID ML+DL", cat_style))
    story.append(Paragraph("Official 1,111 Validation Benchmark: Classical ML vs DL vs Hybrid", title_style))
    if (FIGS / "p2_model_comparison.png").exists():
        im_comp = RLImage(str(FIGS / "p2_model_comparison.png"), width=440, height=210)
        c_res = make_card_table("Validation Metrics", [
            "Phase 1 RF: 99.91% Bld | 90.73% Flr | 11.58 m Mean | 12.00 m EvAAL.",
            "Phase 1 XGB: 99.73% Bld | 91.99% Flr | 14.52 m Mean | 14.97 m EvAAL.",
            "Pure DL (HierResMTL): 99.91% Bld | 92.98% Flr | 7.77 m Mean | 8.09 m EvAAL.",
            "Strawman Naive Blend (0.5 RF+DL): Degrades DL to 8.96 m Mean / 9.29 m EvAAL!",
            "HierResMTL-Hybrid (Ours): 99.91% Bld | 95.50% Flr (+4.77%) | 4.29 m MAE | 6.76 m Mean | 4.98 m Median | 6.99 m EvAAL (-41.8%)."
        ], bg="#ECFDF5", tc="#047857")
        story.append(Table([[im_comp, c_res]], colWidths=[450, 270]))
    story.append(PageBreak())

    # Slide 5: Controlled Ablation Studies
    story.append(Paragraph("PHASE 2 DEEP LEARNING & HYBRID ML+DL", cat_style))
    story.append(Paragraph("Controlled Single-Factor Ablation Studies & Statistical Significance", title_style))
    if (FIGS / "p2_ablation_study.png").exists():
        im_abl = RLImage(str(FIGS / "p2_ablation_study.png"), width=440, height=210)
        c_abl = make_card_table("Ablation & Statistical Findings", [
            "R1 -> R3: Slashes EvAAL from 35.88 m to 11.60 m on Fold 0.",
            "Blind-Scan Filtering: Dropping 73 contradictory scans saves +2.03 m EvAAL.",
            "Domain Augmentation: AP dropout & jitter save +2.94 m EvAAL.",
            "Kendall Uncertainty Loss: Balances tasks, saving +2.72 m EvAAL.",
            "Paired Bootstrap 95% CI: Hybrid beats Phase 1 RF by 4.82 m [3.92, 5.66] m.",
            "McNemar Floor Test: p = 4.2e-8 (49 wins vs 8 losses vs Phase 1 XGB)."
        ], bg="#F8FAFC", tc="#102A43")
        story.append(Table([[im_abl, c_abl]], colWidths=[450, 270]))
    story.append(PageBreak())

    # Slide 6: Device Robustness & Conclusion
    story.append(Paragraph("PHASE 2 DEEP LEARNING & HYBRID ML+DL", cat_style))
    story.append(Paragraph("Device Shift Mitigation, Member Contributions & Deliverables", title_style))
    if (FIGS / "p2_robustness_by_phone.png").exists():
        im_rob = RLImage(str(FIGS / "p2_robustness_by_phone.png"), width=380, height=210)
        c_deliv = make_card_table("Hardware Shift & Deliverables", [
            "Phone 14 Outlier Recovery: Reduced from 30.45 m (RF) to 9.70 m (Hybrid) — a 68.1% reduction.",
            "Seen vs Unseen Phones: 95.72% vs 95.66% floor accuracy across 11 devices.",
            "M1 (Aviral): Data, DataLoader, Noise, Regularization.",
            "M2 (Ved): R3 Representation & Hybrid ML+DL.",
            "M3 (Khuswant): HierResMTL Multitask Architecture.",
            "M4 (Prataya): 5-Fold CV, Bootstrap CIs, McNemar, ECE.",
            "M5 (Vansh): A1-A9 Ablations, Report, Presentation.",
            "Deliverables: PPTX & PDF slides, IEEE LaTeX Report, executed Notebook Phase2_DL_Hybrid.ipynb, and GitHub repo."
        ], bg="#F0FDF4", tc="#047857")
        story.append(Table([[im_rob, c_deliv]], colWidths=[390, 330]))

    doc.build(story)
    print(f"✓ Saved PDF to {pdf_path} ({pdf_path.stat().st_size / (1024*1024):.2f} MB)")
    return pdf_path


def main():
    print("Building Phase 2 Presentation Decks (PPTX & PDF)...")
    pptx_path = build_pptx()
    pdf_path = build_pdf()
    print("\n✓ Both Presentation Formats Successfully Generated:")
    print(f"  - PPTX: {pptx_path} ({pptx_path.stat().st_size / (1024*1024):.2f} MB)")
    print(f"  - PDF : {pdf_path}  ({pdf_path.stat().st_size / (1024*1024):.2f} MB)")


if __name__ == "__main__":
    main()
