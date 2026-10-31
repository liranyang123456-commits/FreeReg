# FreeReg: Free-Coordinate 3D Model Registration and Stable Display for AR/MR

FreeReg registers a known 3D model (a CAD mesh or a CT-derived airway) to an
RGB-D or monocular image stream and drives a depth-aware AR/MR composite. It
combines a two-source correspondence engine with contour-gated arbitration,
rigid-plus-warp (RTW) deformation with an optional observability penalty, a
manifold SE(3) filter with innovation gating, and a Gaussian-splatting display
path.

Repository: <https://github.com/liranyang123456-commits/FreeReg> (release tag: `vr-r1`).

## Repository Structure

| Path | Description |
|---|---|
| `src/` | Core library implementation |
| `scripts/` | Evaluation drivers, multi-seed runner, statistical aggregation, and CUDA timing |
| `patches/` | Windows build patches for external transfer probes (FoundationPose, Endo3R) |
| `requirements.txt` | Python package dependencies |

### Core Modules (`src/`)

| Module | Role |
|---|---|
| `src/geometry.py` | SE(3)/SO(3) exponential and logarithm, pinhole projection, back-projection |
| `src/pose_estimation.py` | Kabsch/Umeyama, EPnP+RANSAC, FPFH+RANSAC, multi-start and trimmed ICP |
| `src/appearance_engine.py` | ROI-SIFT texture correspondences against held-out reference views |
| `src/render_compare_engine.py` | Render-and-compare matcher (standalone probe) |
| `src/pipeline.py` | RGB-D registration with contour-gated arbitration and the hybrid engine |
| `src/nonrigid_rtw.py` | FFD rigid-plus-warp solver, fixed-correspondence depth Jacobian, observability penalty |
| `src/se3_kalman_filter.py` | Constant-velocity SE(3) filter with component-normalized innovation gate |
| `src/egomotion_free_registration.py` | Variant that keeps the model-to-world pose and predicts only camera ego-motion |
| `src/rasterize.py`, `src/gaussian_renderer.py` | Z-buffer splatting and the PyTorch EWA Gaussian splatter |
| `src/scene_graph.py` | Multi-model scene graph and depth-aware compositing |
| `src/centerline.py`, `src/ct_loader.py` | CT airway segmentation, centerline extraction, and 2D-3D centerline registration |
| `src/freereg_net.py` | FreeRegNet learned initializer (rigid BOP objects) |
| `src/metrics.py` | ADD/ADD-S, reprojection error, TRE, jitter, occlusion-edge accuracy |

## Environment Setup

All primary and repeated-seed runs used Windows 11, Python 3.9.25, PyTorch
2.8.0 with CUDA 12.8, and an NVIDIA GeForce RTX 5090 Laptop GPU:

```bash
pip install -r requirements.txt
```

For deterministic execution across runs, set:

```bash
set PYTHONHASHSEED=0
set KMP_DUPLICATE_LIB_OK=TRUE
set PYTHONIOENCODING=utf-8
set CUBLAS_WORKSPACE_CONFIG=:4096:8
```

The CUDA tile-rasterizer benchmark (`scripts/time_3dgs_cuda_rasterizer.py`) uses
an environment with PyTorch 2.7.0 (CUDA 12.8) and `gsplat==1.5.3` compiled for
the local GPU. The transfer probes additionally use FoundationPose and Endo3R
with the patches in `patches/`; the monocular depth-degradation study uses
Depth Anything V2.

## Datasets

Public datasets:
- **BOP Linemod and T-LESS**: <https://bop.felk.cvut.cz/datasets/> (default path in scripts: `D:\bop\lm`, `D:\bop\tless`)
- **SCARED**: <https://endovissub2019-scared.grand-challenge.org/> (set `FREEREG_SCARED_ROOT` or place under `E:\MIS_Datasets\SCARED`)
- **C3VD**: <https://durrlab.github.io/C3VD/> (default path in scripts: `E:\Diff_Rending_Re_3D\dataset`)

Clinical CT volumes and bronchoscopy videos are subject to institutional
data-governance restrictions and are not publicly distributed. Scripts that use
them read `FREEREG_ION_ROOT`, `FREEREG_RF_ROOT`, `FREEREG_CT_EXTRACT_ROOT`, and
`FREEREG_BRONCHOSCOPY_VIDEO`.

## Step-by-Step Reproduction

All commands below are run from the repository root. Generated CSV/JSON
metrics and figures are written under `outputs/`.

### 1. Primitive and Geometric Validation

```bash
python scripts/run_unit_tests.py
```

### 2. Rigid Registration (BOP Linemod, T-LESS, and Correspondence Ablations)

```bash
# Geometric engine (classic) and hybrid engine (contour + depth arbitration)
python scripts/run_bop_eval.py --engine classic --scenes 1 2 3 4 5 --frames 25 --seed 0 --out outputs/bop_eval_results.csv
python scripts/run_bop_eval.py --engine hybrid --n_refs 40 --scenes 1 2 3 4 5 --frames 25 --use_contour --seed 0 --out outputs/bop_eval_hybrid_full.csv

# Matched contour-arbitration ablation (depth-only vs contour + depth, seeds 0/1/2)
python scripts/run_bop_eval.py --engine hybrid --n_refs 40 --no-use_contour --seed 0
python scripts/run_bop_eval.py --engine hybrid --n_refs 40 --use_contour --seed 0

# Self-implemented BOP protocol (MSSD / MSPD / VSD), T-LESS, standalone probes, and FreeRegNet cascade
python scripts/run_bop_official_eval.py --scenes 1 2 3 4 5 --frames 25
python scripts/run_tless_eval.py --scenes 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 --frames 15
python scripts/run_correspondence_benchmark.py
python scripts/run_freereg_net_train.py
python scripts/run_freereg_cascade.py --scenes 1 2 --max 150 --with_engine
```

### 3. Non-Rigid RTW Registration and Observability Analysis

```bash
# 71-frame C3VD-derived FEM cohort (base RTW solver)
python scripts/run_real_colon_eval.py --scenes 24 --seed 0

# Controlled RTW and observability spectrum / penalty ablation
python scripts/run_rtw_eval.py
python scripts/run_observability_ablation.py --seed 17
python scripts/run_observability_curve.py
python scripts/run_stereo_observability.py
python scripts/run_stereo_recovery.py
```

### 4. Temporal Filtering and SCARED Supplied-Trajectory Perturbation

```bash
# Controlled CAD trajectory and virtual-video tracking + SE(3) Kalman filter
python scripts/run_temporal_eval.py --seed 0

# SCARED filter-and-display test on supplied camera poses with controlled perturbation (not blind endoscopic tracking)
python scripts/run_scared_video_ar.py --seed 0
python scripts/run_tracking_jitter_compare.py --force
```

### 5. 3DGS / 4DGS Display, CUDA Tile Rasterizer Timing, and Multi-Model Occlusion

```bash
python scripts/run_3dgs_eval.py --seed 0 --out outputs/multirun/render_3dgs/seed0
python scripts/run_4dgs_eval.py --seed 0
python scripts/time_3dgs_cuda_rasterizer.py --gaussians outputs/multirun/render_3dgs/seed0/gaussians.pt
python scripts/run_ar_visual_quant.py
python scripts/run_scene_graph_eval.py
```

### 6. CT Airway Fly-Through, Centerline Registration, Robustness, and Transfer Probes

```bash
python scripts/run_multipatient_eval.py --patients 16 --seed 0
python scripts/run_centerline_eval.py
python scripts/run_depth_degradation.py
python scripts/run_occlusion_robustness.py --scenes 1 2 3 4 5
python scripts/run_init_robustness.py --scenes 1 2 --frames 8
python scripts/run_fp_on_clinical_synth.py
python scripts/run_endo3r_on_clinical_synth.py
```

### 7. Multi-Seed Driver and Statistical Aggregation (Mean ± SD and Bootstrap CIs)

```bash
# Run all stochastic protocols with seeds 0, 1, 2 (and observability ablation with seeds 17, 18, 19)
python scripts/run_vr_revision_multiseed.py --seeds 0 1 2
python scripts/run_vr_revision_multiseed.py --tasks obs_ablation --seeds 17 18 19

# Aggregate multi-seed mean ± SD (-> outputs/stats/multiseed_summary.{json,md})
python scripts/analyze_vr_multiseed.py

# Compute 95% bootstrap CIs and RTW failure breakdown (-> outputs/stats/revision_stats.{json,md})
python scripts/analyze_vr_revision_stats.py
```
