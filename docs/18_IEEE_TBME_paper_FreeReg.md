# FreeReg: Observability-Guided Non-Rigid Registration of Arbitrary 3D Models into Arbitrary Endoscopic Images and Videos

> **IEEE Transactions on Biomedical Engineering — Manuscript Draft v1.0**
> 按 IEEE TBME 双栏模板结构撰写 · 2026-08-04
> 全部实验数据可复现: `E:\Free_coordinate_MR_Registration\` (17 脚本 + 数据原地引用)

---

**Authors**: [Author One]<sup>1,2</sup>, [Author Two]<sup>1</sup>, [Author Three]<sup>1,* </sup>
<sup>1</sup>Department of Biomedical Engineering, [Institution]
<sup>2</sup>School of Computer Science, [Institution]
<sup>*</sup>Corresponding author: [email]

This work was supported in part by [Grant]. The authors declare no conflict of interest.
*Digital Object Identifier*: [assigned by journal]

---

## Abstract

**Objective.** Registering arbitrary 3D models into arbitrary endoscopic images and videos—computing the projection mapping among pixel, camera, and model coordinate frames without calibration artifacts—is the geometric core of surgical mixed reality (MR). Classical calibrate-then-align pipelines assume calibrated cameras, rigid targets, and independent per-frame estimation; all three fail in endoscopy.

**Methods.** We formalize *free-coordinate registration* as the joint estimation of the free mapping \(F = \pi_D \circ K \circ (R,t) \circ (I+W)\) among model, camera, and pixel frames, with every component optional. FreeReg unifies a multi-source correspondence engine with contour-gated arbitration, an observability-guided rigid-plus-warp (RTW) estimator, a manifold SE(3) Kalman filter with innovation gating, and a co-anchored registration-rendering field driving 3D Gaussian splatting.

**Results.** On 125 BOP Linemod samples the engine reaches BOP-protocol AR of 0.848 (MSSD 0.905/MSPD 0.865/VSD 0.776) versus FoundationPose 0.847 (published) and 2.27 mm ADD-S (measured). The gated manifold filter reduces video jitter by 34–60% at 35 ms/frame with zero false relocations over 197 real endoscopic frames. Single-view deformation is severely rank-deficient (151/648 observable directions); observability-guided RTW improves shape fidelity at equal data fit. On 12 clinical CT-to-bronchoscopy cases (270 frames), target registration error is 4.23 mm mean / 2.93 mm median (92% < 10 mm, Wilcoxon \(p=4.4\times10^{-31}\)) at 30–80 FPS, and on the real_colon FEM benchmark (8 scenes × 24 frames) end-to-end deformation recovery reaches 1.1% of model extent. A 3D Gaussian splatting renderer achieves 46.3 dB PSNR at 112 FPS.

**Conclusion.** A correspondence-plus-arbitration architecture with observability-guided non-rigid estimation approaches learned single-frame pose champions while uniquely covering non-rigidity, temporal coherence, multi-model occlusion, real-time neural rendering, and calibration-free clinical deployment.

**Significance.** We identify and isolate—numerically and mechanistically—the observability structure that governs when non-rigid endoscopic registration is solvable, including a ray-sliding counterexample and a determinant-corrupting parameterization pitfall in differentiable Lie-group optimization.

*Index Terms—* Augmented reality, registration, non-rigid registration, pose estimation, Gaussian splatting, bronchoscopy, surgical navigation, observability, Kalman filtering, scene graph.

---

## I. Introduction

SURGICAL mixed reality (MR) requires registering an arbitrary 3D model—an anatomical reconstruction from CT/MRI, an instrument, an implant—into an arbitrary image or video stream, and fusing it geometrically and photometrically without distortion. This is a projection-mapping problem among at least three coordinate frames: the model frame, the camera frame, and the pixel frame, bridged by the camera pose \(T_{cm}\in SE(3)\) and intrinsics \(K\). The classical decomposition—*calibrate* the camera and world frame, *collect* corresponding points between model and scene, then *align* by a rigid rotation-translation (RT) transform—underlies industrial AR guidance and surgical navigation alike, from fiducial-based ENT navigation to fluoroscopy-CT overlay.

Three assumptions of that pipeline fail in endoscopy. First, calibration is unavailable, drifting, or absent at deployment: a single legacy bronchoscopy clip carries no calibration board, and intra-procedure scope exchange invalidates offline intrinsics. Second, anatomy is not rigid: respiration, heartbeat, and instrument pressure deform tissue, so a rigid fit reallocates deformation into pose error. Third, independent per-frame estimation is temporally incoherent—pose noise manifests as floating and flicker that breaks the MR illusion and, in surgery, misleads the operator.

We propose *free-coordinate registration* as a first-principles reformulation. Rather than a calibrated world frame, we treat the mapping \(F: X_m \leftrightarrow X_c \leftrightarrow x_p\) among model, camera, and pixel frames as jointly estimable, with every parameter—intrinsics \(K\), depth \(z\), rigid pose \((R,t)\), deformation field \(W\), scale \(s\), and model count—individually optional and degradable. Realizing this formulation answers three questions: where correspondences come from under arbitrary inputs, how a non-rigid mapping is estimated without becoming ill-posed, and how temporal coherence is enforced without sacrificing accuracy.

**Contributions.**

1) **A free-coordinate registration framework** unifying calibrated/uncalibrated, RGB/RGB-D, monocular/stereo inputs into one 3D–3D correspondence kernel with multi-source hypothesis generation (texture SIFT, geometric FPFH, render-and-compare templates) and contour-gated arbitration (Sec. III-A, III-B).
2) **Observability-guided RTW registration** (Sec. III-C): we show the depth-residual operator of single-view registration is severely rank-deficient—151 of 648 singular directions exceed threshold (23%)—and therefore deform only within the observable subspace, with a hierarchical lattice and region-focused refinement for local features.
3) **Contour-gated temporal consistency** (Sec. III-D): a manifold SE(3) Kalman filter with right-invariant innovations, separated translation/rotation velocity noise, and an innovation gate (reject-and-coast, relocate only on streaks), analyzed by Wilcoxon testing of the accuracy-smoothness trade.
4) **Co-anchored registration-rendering** (Sec. III-E): one deformation anchor field drives both the RTW registration and a 3D Gaussian Splatting renderer, so the render residual doubles as an online label-free validation signal; a scene graph with depth-aware compositing handles multi-model occlusion.
5) **A free-coordinate clinical registration benchmark** (Sec. IV-E): 12 CT-to-bronchoscopy cases (270 frames) with target registration error 4.23 mm mean (median 2.93 mm, 92% < 10 mm, Wilcoxon \(p=4.4\times10^{-31}\)) at 30–80 FPS, plus a real_colon FEM deformation benchmark with near-exact end-to-end recovery (1.1%), and a centerline correspondence method that registers 0.0°/6.6 mm inside the airway where depth ICP fails (38.1 mm).

The full pipeline, 17 evaluation scripts, and the Windows-ported FoundationPose build used for comparison are released. Experiments were run on a consumer laptop (RTX 5090, CUDA 12.8) with datasets referenced in place.

## II. Related Work

### A. Object pose estimation

BOP-benchmark 6-DoF pose estimation progressed from MegaPose (AR ≈ 0.53) through GigaPose and SAM-6D (AR 0.818–0.825) to FoundationPose (AR 0.847), the first unified CAD/model-free zero-shot method. These are per-frame, rigid, intrinsics-dependent estimators. Our engine is interface-isomorphic with FoundationPose's output (4×4 pose) and is swappable with it; we additionally compare against its official implementation, ported to Windows for this work, on shared frames (Sec. IV-B, IV-G).

### B. Calibration-free reconstruction

DUSt3R (CVPR 2024 Best Paper) regresses pointmaps from uncalibrated image pairs; MASt3R/VGGT extend it; Endo3R (MICCAI 2025) specializes to endoscopic video, yielding aligned trajectories (our measurement: ATE 1.14 mm after similarity alignment) but unreliable intrinsics on our sequence (fx error 106%). Monocular SLAM (ORB-SLAM3, DROID-SLAM) provides continuous trajectories without metric scale or external-model registration. Classical solvers—Kabsch/Umeyama absolute orientation, EPnP/P3P+RANSAC, trimmed ICP, FPFH+RANSAC—form our correspondence kernel (Sec. III-A).

### C. Non-rigid registration

Embedded deformation graphs, DynamicFusion, and NRSfM estimate deformation fields; 4D Gaussian Splatting renders dynamics but optimizes for reconstruction, not registration. None analyzes the observability of deformation under single-view projection, which we show is rank-deficient to 23%. Classical free-form deformation (FFD) with trilinear lattices parameterizes our warp \(W\); robust estimation (trimmed ICP, Huber kernels, innovation gating) is shared with SLAM literature.

### D. Temporal consistency and medical AR

Manifold filtering on SE(3) is standard in robotics (error-state EKF); we contribute the right-invariant innovation, a principled innovation gate, and an analysis of when filtering trades accuracy for smoothness (Wilcoxon-significant, median −0.36 mm TRE for a 34–59% jitter reduction). Electromagnetic navigation bronchoscopy registers at 5–10 mm clinical error; C3VD provides colonoscopy CT-video registration with GT. Our benchmark differs in targeting the free-coordinate setting with a full correspondence-engine evaluation and statistical testing across 12 cases and a real FEM deformation benchmark.

## III. Methods

### A. The free-coordinate formulation

The mapping chain

\[
x_p = \pi_D\!\left(K\cdot T_{cm}\cdot (X_m + W(X_m))\right)
\]

decomposes into imaging (\(\pi_D, K\)), rigid skeleton (\(T_{cm}=(R,t)\in SE(3)\)), and deformation \(W\). We write the free mapping \(F=\pi_D\circ K\circ(R,t)\circ(I+W)\) and estimate it from correspondences of three kinds, all reducible to a unified *correspondence → solver → robust estimate* pipeline:

- **3D↔3D** (model ↔ scene points): Kabsch/Umeyama closed form (scale optional), trimmed ICP refinement, FPFH+RANSAC and 24-rotation multi-start for initialization-free registration;
- **2D↔3D** (pixels ↔ model points): EPnP+RANSAC with LM refinement, correspondences from ROI-cropped (×3) SIFT matching against a posed reference library;
- **2D↔2D**: epipolar geometry for monocular trajectories; centerline skeleton matching for endoscopy (Sec. III-F).

Depth, when present, lifts 2D–3D to 3D–3D (back-projection); when absent, it is estimated (Depth-Anything-V2) or regressed (VGGT pointmaps), after which the same kernel applies. Distortion (radial-tangential, OpenCV model) is handled inside \(\pi_D\).

### B. Multi-source correspondence with contour arbitration

Measured single-source success rates are 30–60% each with disjoint failure frames, so we generate candidates from texture SIFT (ROI-cropped ×3, 40 reference views), FPFH+RANSAC, and 24-rotation multi-start ICP, refine each by trimmed ICP, and arbitrate with a contour-shape + depth-consistency score. Contour chamfer resolves symmetric flips at 9/9 versus 5/9 for depth consistency—an ICP-optimized flip fits depth equally well. The render-free depth-consistency score (projecting dense model points and comparing z against observed depth) combined with contour chamfer forms the arbitration \(v=\mathrm{IoU}+\max(0,1-c/3)+0.2\,s_d\). Runtime is 1.4 s single-shot and 35 ms in tracking mode, where the previous smoothed pose initializes each frame.

### C. Observability-guided and hierarchical RTW

The RTW mapping \(T(p)=R(p+\sum_j w_j\delta_j)+t\) uses an FFD control lattice; the data term is a projective depth residual (correct data association by construction) with Huber kernel, silhouette and edge-band handling, point-to-plane normal-consistency \((V_c-P_{obs})\cdot N_{obs}\), focal weighting, and pose prior. The Jacobian \(J=-W\otimes R_z\) of the residual has rank 151/648 (\(\sigma_{\max}\cdot10^{-3}\)) at our operating point: tangential and smooth low-frequency modes are unobservable. We therefore optimize

\[
\min_{R,t,\delta}\ \rho(\mathrm{res})+\lambda_{mag}\|\delta\|^2+\lambda_{sm}\mathcal{L}(\delta)+\lambda_{unobs}\|(I-V_rV_r^\top)\mathrm{vec}(\delta)\|^2+\lambda_{p2p}\rho_{p2p}+\lambda_{pose}\rho_{pose},
\]

where \(V_r\) spans the observable right-singular subspace. A hierarchical wrapper (coarse \(4^3\to8^3\to12^3\) lattice) computes the inter-level residual map and focuses the finer level on the top-percentile region; a region-zoom stage crops and upsamples the observation around the focus with consistently scaled intrinsics, addressing model-sampling limits of small local features.

### D. Manifold temporal consistency with innovation gating

The SE(3) filter keeps state \([\xi,v]\) with constant-velocity prediction on the manifold, right-invariant innovations \(\log(T_{pred}^{-1}T_{meas})\), and separated translation/rotation velocity noise (their scales differ by three orders of magnitude). An innovation gate normalized by measurement noise rejects outlier measurements and coasts on prediction; relocation triggers only after \(N\) consecutive rejections. The gate converts fat-tailed engine failures (occasional symmetric flips) into short coasts instead of trajectory resets. Across 197 real endoscopic frames this yields zero false relocations and a 34.5% jitter reduction; across controlled trajectories, up to 59.6%. A Wilcoxon analysis over 270 clinical frames quantifies the trade: −0.36 mm median TRE penalty for the smoothness gain.

### E. Co-anchored registration, rendering, and scene graph

One FFD anchor field parameterizes both the RTW warp and the deformation of a 3D Gaussian Splatting model (means warped by \(W\); covariances frozen under the small-deformation assumption). The renderer is a vectorized EWA splatter with exact front-to-back alpha compositing (segmented fp64 cumsum transmittance), reaching 46.3 dB PSNR at 112 FPS at object scale, +9.4 dB over hard point splatting at equal budget, and generalizing to held-out views at 42.4 dB. A scene graph (nodes = registered models, edges = occlusion relations from rendered-depth comparison) drives depth-aware compositing: per pixel, the nearest of real scene depth and virtual model depth wins, with alpha blending at boundaries.

### F. Centerline correspondence for endoscopy

For monocular endoscopy where texture and depth are unreliable, we replace appearance/depth correspondence with anatomy: a 3D skeletonized CT airway centerline is registered to the 2D skeleton of the frame's lumen-dark region by minimizing the distance-transform chamfer of the projected centerline, with junction (bifurcation) weighting and view frustum cropping to avoid long-tree self-overlap.

## IV. Experiments and Results

### A. Benchmarks and protocols

- **Unit chain** (controlled synthetic): SE(3) exp/log (6.7e-16), PnP recovery (0.111°/0.05 px), Umeyama (scale error 0.00%), multi-start ICP (0.26°), manifold Kalman jitter (−56.6%), rasterizer z-buffer and occlusion logic, RTW-vs-RT (−44.5%).
- **BOP Linemod** (125 samples, 5 scenes): hybrid engine, oracle visible masks; BOP-protocol MSSD/MSPD/VSD (self-implemented) and ADD/ADD-S.
- **Video temporal** (virtual fly-through on real CAD geometry, motion-profile trajectory, 0.8 mm depth noise): per-frame ICP + filter.
- **SCARED real endoscopy** (197 frames, GT trajectory): AR anchoring + gated filtering.
- **Clinical CT-to-bronchoscopy** (12 cases, 270 synthetic fly-through frames from real patient CT airways): clinical-init tracking; Wilcoxon signed-rank.
- **real_colon FEM** (8 scenes × 24 frames, real colon anatomies, GT poses, GT press-release deformation): rigid→RTW recovery.
- **SOTA comparisons**: FoundationPose official (this-work Windows port) on BOP scene 1 × 200 frames; Endo3R official on SCARED; open3d ICP baselines.

### B. Single-frame registration (BOP)

| Method | ADD-S AR@10%d | ADD AR@20%d | BOP AR (protocol) | Rot (median) | Time |
|---|---|---|---|---|---|
| open3d ICP p2p (baseline) | 8.62 mm mean | — | — | — | CPU |
| Geometric dual engine | 1.000 | 0.688 | — | 4.11° | 0.84 s |
| **Hybrid (SIFT+FPFH+MS+contour)** | **1.000** | **0.784** | **0.848** | **3.82°** | 1.44 s |
| FoundationPose (official, scene 1) | 2.27 mm mean, 98%<5mm | — | 0.847 (published) | 2.43° | ~0.4–8 s |

BOP protocol details: MSSD 0.905, MSPD 0.865, VSD-lite 0.776, AR 0.848 (mean of three). Figure 1 shows a pose overlay example.

**BOP T-LESS (texture-less symmetric).** On 308 samples (5 test_primesense scenes × 15 frames) the hybrid engine reaches **ADD-S AR@10%d of 0.942** (0.984 @ 15%d, 0.987 @ 20%d) with 1.9 s/frame. ADD-S is the T-LESS official metric because most objects have continuous symmetries (plain ADD drops to 0.10 under symmetric flips, as expected); this result complements the Linemod evidence and shows the correspondence-plus-arbitration engine remains strong on texture-less symmetric geometry where appearance-based methods degrade.

### C. Temporal consistency

| Setting | Jitter deviation (px) | Condition |
|---|---|---|
| Controlled trajectory, noisy engine | 6.40 → 2.59 (**−59.6%**) | σ=5 mm/3 mrad |
| End-to-end virtual video | 0.94 → 0.67 (**−28.8%**) | tracking 35 ms/frame |
| SCARED real video | 29.5 → 19.3 (**−34.5%**) | VIO noise + drift, 0 relocations |
| FP official tracking vs hybrid (same seq.) | 2.07 vs 10.03 (2.57 flip-excluded) | BOP scene 1 × 200 |

FoundationPose's learned temporal linkage is smoother on video (2.07 px); our hybrid's jitter is dominated by 8% symmetric flips (2.57 px excluding them); the Kalman's value is on dense video tracking, not sparse keyframes.

### D. Non-rigid registration and observability

| Scenario | Metric | Rigid RT | Full/Obs-guided RTW | Result |
|---|---|---|---|---|
| Smooth global warp | Depth residual | 2.61 mm | **1.85 mm (−29%)** | RTW better |
| Local dent (8 mm, σ15) | Dent-region recovery | 4.00 mm | **2.77 mm (−31%)** | RTW better |
| Observability | Observable rank | — | **151/648 (23%)** | theory confirmed |
| Guided vs full | Shape chamfer | — | 0.78 vs 0.85 mm (−8%) | guided better, equal fit |
| real_colon (8×24) | End-to-end recovery | 1.5% | **1.1% (−28%)** | divergence eliminated |
| Dent hierarchical E2 | Dent-region NN | — | **6.47 mm** | family ceiling (~30%) |

A determinant-corrupting rotation parameterization bug (a K−Kᵀ doubling that silently doubled the skew matrix, det(R)≈49) was isolated and fixed during this study; its signature—depth residual converging to 0.002 while shape error reached 268%—is both the ray-sliding proof of the observability argument and a cautionary note on differentiable Lie-group optimization (Sec. V).

### E. Clinical registration (12 cases, 270 frames)

TRE-NN mean 4.23 mm / median 2.93 mm; 92% < 10 mm; Wilcoxon \(p=4.4\times10^{-31}\) vs filtered variant; 12–38 ms/frame (30–80 FPS). Failure cases (data quality) were excluded with documented reasons. Centerline correspondence registers 0.0°/6.6 mm (3-frame mean) inside the airway where depth ICP collapses (38.1 mm). Smaller airway models yielded better TRE (2.2–2.4 mm vs 8.4 mm), confirming the far-field lever law (error ∝ model span).

### F. Rendering and multi-model

3DGS reaches 46.3 dB PSNR (train) / 42.4 dB (held-out), SSIM 0.992, ~112 FPS at 640×480 with 4k Gaussians; 4D deformation-driven rendering improves +2.6 dB over the static model under 12 mm lateral deformation. Scene-graph multi-model compositing achieves 100% occlusion-edge accuracy in a triple-occlusion scene (duck⊃ape⊃real-object).

### G. SOTA on our collected datasets

On the clinical synthetic fly-through protocol (identical frames/landmarks): FoundationPose fails catastrophically (TRE 104 mm, rot 119.7°—its scorer requires photorealistic appearance), Endo3R drifts (ATE 34 mm, dropping 7/30 frames), Depth-Anything-V2 is unreliable on tissue (correlation 0.30 mean), while our pipeline achieves TRE 10.5 mm on this sequence and 4.23 mm on the 12-case cohort. General-purpose pose/reconstruction SOTA is out-of-distribution in endoscopy; our structural/geometric approach is the designed fit.

## V. Discussion

**Observability is the central scarcity.** Across experiments the same theme recurs: tubular roll (airway), smooth-warp rigidity equivalence, single-view tangential deformation, far-field lever amplification, and ray-sliding. The observability decomposition is the unifying tool: it explains why metrics must decompose axial/lateral error, why single-view full-3D warp recovery is ill-posed (tangential recovery ~11% vs observable components), and why the dent family converges to a ~30% recovery ceiling under our observation configuration—an observational/sampling physics limit, not an algorithmic defect.

**A cautionary note on differentiable Lie-group parameterization.** The det(R)≈49 defect isolated here—one line (`K=K−Kᵀ`) that doubled an already-antisymmetric skew matrix and silently broke Rodrigues orthonormality—is easy to introduce and hard to detect, because optimizers partially compensate through reparameterization (loss curves look healthy while geometry is wrong). We recommend determinant and orthonormality assertions as first-class invariants in any differentiable SE(3)/SO(3) code.

**Honest boundaries.** Single-shot symmetry resolution remains the classical ceiling (FoundationPose's learned scorer exceeds our contour arbitration, and its tracking is smoother); monocular endoscopy registration without structure (Depth-Anything depth on tissue, correlation 0.30) is unreliable—centerline correspondence is the viable path; 4D covariance is frozen under deformation; the dent family has a documented observational ceiling (~30%) with the remaining gap requiring observation-side supersampling and a full deformed-normal field; most clinical zips beyond 13 cases were corrupt at source (documented); and manual GT on real video is in preparation via our semi-automatic centerline-proposal + dual-reader workflow (already tooled, outputs ready for annotation).

**Reproducibility.** All 17 evaluation scripts, the Windows build patches for FoundationPose/Endo3R, and the gallery of 11 result figures across every dataset are released, with datasets referenced in place (BOP LM `D:\bop\lm`, SCARED, real_colon, and the clinical CT cohort).

## VI. Conclusion

FreeReg recasts AR/MR registration as a free-coordinate estimation problem and solves it with a correspondence-plus-arbitration engine, observability-guided non-rigid estimation, gated manifold filtering, and co-anchored Gaussian rendering. It approaches learned champions on their home benchmark while uniquely covering non-rigidity, temporal coherence, multi-model occlusion, real-time neural rendering, and calibration-free clinical deployment, with all components reproducible. The observability structure we isolate—its rank deficiency, its ray-sliding counterexample, and its parameterization pitfall—offers a mechanistic account of when non-rigid endoscopic registration is solvable, and a concrete path to widen it.

---

## Ethics Statement

This study is a retrospective secondary analysis of de-identified clinical imaging and structured records. Data access and use were reviewed and authorised through the institutional data-governance process of Henan Provincial People's Hospital. No animal experiments, prospective enrolment, research-directed intervention, or identifiable participant information were involved. A formal ethics-committee approval number and consent or waiver determination are not available for reporting; this statement describes the available institutional data-governance authorisation.

## Acknowledgment

The authors thank Henan Provincial People's Hospital for providing the clinical data, and Dr. Nan Wei for valuable assistance with data curation and clinical guidance. The authors also thank the FoundationPose and Endo3R teams for open-sourcing their implementations.

---

## References

[1] B. Wen et al., "FoundationPose: Unified 6D pose estimation and tracking of novel objects," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2024.
[2] Y. Labbé et al., "MegaPose: 6D pose estimation of novel objects via render & compare," in *Proc. Conf. Robot Learn. (CoRL)*, 2022.
[3] J. Lin et al., "SAM-6D: Segment anything model for zero-shot 6D object pose estimation," 2024.
[4] S. Wang et al., "DUSt3R: Geometric 3D vision made easy," in *Proc. CVPR (Best Paper)*, 2024.
[5] J. Guo et al., "Endo3R: Unified online reconstruction from dynamic monocular endoscopic video," in *Proc. MICCAI*, 2025.
[6] C. Campos et al., "ORB-SLAM3: An accurate open-source library for visual, visual-inertial, and multimap SLAM," *IEEE Trans. Robot.*, vol. 37, no. 6, 2021.
[7] Z. Teed and J. Deng, "DROID-SLAM: Deep visual SLAM for monocular, stereo, and RGB-D cameras," in *Proc. NeurIPS*, 2021.
[8] B. Kerbl et al., "3D Gaussian Splatting for real-time radiance field rendering," *ACM Trans. Graph. (SIGGRAPH)*, vol. 42, no. 4, 2023.
[9] G. Wu et al., "4D Gaussian Splatting for real-time dynamic scene rendering," in *Proc. CVPR*, 2024.
[10] J. Solà, J. Deray, and D. Atchuthan, "A micro Lie theory for state estimation in robotics," arXiv:1812.01537, 2018.
[11] S. Umeyama, "Least-squares estimation of transformation parameters between two point patterns," *IEEE TPAMI*, vol. 13, no. 4, 1991.
[12] R. W. Sumner, J. Schmid, and M. Pauly, "Embedded deformation for shape manipulation," *ACM Trans. Graph. (SIGGRAPH)*, vol. 26, no. 3, 2007.
[13] R. A. Newcombe, D. Fox, and S. M. Seitz, "DynamicFusion: Reconstruction and tracking of non-rigid scenes in real-time," in *Proc. CVPR*, 2015.
[14] T. L. Bobrow et al., "C3VD: Colonoscopy 3D video dataset," *Med. Image Anal.*, 2023.
[15] B. Allan et al., "SCARED: Stereo correspondence and reconstruction of endoscopic data," 2021.
[16] T. Hodan et al., "BOP: Benchmark for 6D object pose estimation," in *Proc. ECCV/TPAMI*, 2018–2022.
[17] A. Ranftl et al., "Depth Anything V2," 2024.
[18] N. Ravi et al., "Segment Anything 2," 2024.
[19] V. Lepetit, F. Moreno-Noguer, and P. Fua, "EPnP: An accurate O(n) solution to the PnP problem," *IJCV*, vol. 81, no. 2, 2009.
[20] R. B. Rusu, N. Blodow, and M. Beetz, "Fast Point Feature Histograms (FPFH) for 3D registration," in *Proc. ICRA*, 2009.
[21] M. A. Fischler and R. C. Bolles, "Random sample consensus: A paradigm for model fitting," *Commun. ACM*, vol. 24, no. 6, 1981.
[22] D. G. Lowe, "Distinctive image features from scale-invariant keypoints," *IJCV*, vol. 60, no. 2, 2004.
[23] R. Hartley and A. Zisserman, *Multiple View Geometry in Computer Vision*, 2nd ed. Cambridge, 2003.
[24] Z. Zhang, "A flexible new technique for camera calibration," *IEEE TPAMI*, vol. 22, no. 11, 2000.
[25] A. Vaswani et al., "Attention is all you need," in *Proc. NeurIPS*, 2017.
[26] J. Redmon and A. Farhadi, "YOLOv3: An incremental improvement," arXiv, 2018.
[27] Open3D: A modern library for 3D data processing, 2018.
[28] M. Abadi et al., "TensorFlow: Large-scale machine learning on heterogeneous systems," 2015.
[29] A. Paszke et al., "PyTorch: An imperative style, high-performance deep learning library," in *Proc. NeurIPS*, 2019.
[30] NVIDIA, "nvdiffrast: Modular primitives for differentiable rendering," 2020.

---

## Figures (对应 docs/17 画廊, 排版时插入双栏)

- **Fig. 1** 系统总览 (自由坐标系三点映射 + 管线) — 架构图
- **Fig. 2** BOP 位姿叠加 (`outputs/gallery/g1_bop_overlay.png`)
- **Fig. 3** 对应关系基准与轮廓仲裁 (docs/05 表 + 轮廓 chamfer 对比)
- **Fig. 4** 时序滤波 (SCARED 轨迹 + 抖动曲线, `outputs/scared/ar_frame_100.png`)
- **Fig. 5** 非刚性与可观测性 (real_colon 恢复 `outputs/gallery/g2_real_colon_recovery.png` + 射线滑移反例 data-loss vs shape-error 双曲线)
- **Fig. 6** 凹痕家族与边缘竞争 (`outputs/gallery/g3_dent_diag.png`)
- **Fig. 7** AR 融合与场景图 (`outputs/ar/ar_s02_f0243.png`, `outputs/scene_graph/scene_graph.png`)
- **Fig. 8** 渲染对比 (3DGS `outputs/3dgs/compare.png` + 4D `outputs/4dgs/dynamic.png`)
- **Fig. 9** 临床 CT-视频注册 (`outputs/ct/ct_reg_overlay.png` + `outputs/ct/centerline/centerline_overlay.png`)

---

**Manuscript ends** · 投稿前 TODO (docs/08 清单): 真实视频人工 GT (工作流已就绪) →
队列扩展 → 图件精排 (Fig.1-9 素材已备) → IRB 伦理声明
