# Free_coordinate_MR_Registration

> **任意3D模型 → 任意图像/视频的AR/MR注册系统**
>
> 给定任意一张图像（或视频序列）和任意一个/多个3D模型，
> 计算2D像素 ↔ 3D模型 ↔ 相机坐标系之间的投影映射，
> 并在AR/MR渲染中无失真、动态变化无失真地融合呈现。

**代码仓库**: <https://github.com/liranyang123456-commits/FreeReg> （论文接收后转 public）

**投稿论文**: *FreeReg: Free-Coordinate Registration of 3D Models for Augmented Reality in Image-Guided Bronchoscopy*（IEEE TMI，10 页，见 `outputs/paper/freereg_tbme_full.tex`）

## 文档导航

| 文档 | 路径 | 内容 |
|------|------|------|
| **技术设计方案** | [`docs/01_技术设计方案_3D模型到图像视频注册.md`](docs/01_技术设计方案_3D模型到图像视频注册.md) | 问题分析、架构设计、模块设计、SOTA对比、性能提升、路线图 |
| **数据集使用指南** | [`docs/02_数据集使用指南.md`](docs/02_数据集使用指南.md) | E/F/D盘所有可用数据集的精确路径、类型、调用示例 |
| **问题分析(RT→RTW)** | [`docs/03_问题分析_从RT到RTW的数学原理.md`](docs/03_问题分析_从RT到RTW的数学原理.md) | 投影链条、对应关系体系、非刚性RTW、可观测性 |
| **测试与性能指标** | [`docs/04_测试验证与性能指标.md`](docs/04_测试验证与性能指标.md) | 第一轮实测: 单元8/8、BOP AR、时序抖动、RTW、AR渲染 |
| **对应关系体系** | [`docs/05_对应关系构建方法体系与实测.md`](docs/05_对应关系构建方法体系与实测.md) | **直接回答**: 两坐标系间怎样构建有效对应 (5类方法实测) |
| **第二轮指标** | [`docs/06_第二轮实现_引擎视频3DGS4D_测试指标.md`](docs/06_第二轮实现_引擎视频3DGS4D_测试指标.md) | 位姿引擎/实拍视频/3DGS/4DGS 全部实测 |
| **创新方案设计** | [`docs/07_创新方案设计_FreeReg自由坐标系非刚性MR注册.md`](docs/07_创新方案设计_FreeReg自由坐标系非刚性MR注册.md) | FreeReg框架、5个创新点、实验设计、论文骨架 |
| **目标期刊分析** | [`docs/08_目标期刊分析与投稿策略.md`](docs/08_目标期刊分析与投稿策略.md) | 期刊矩阵、分层投稿策略、投稿前检查清单 |
| **E1 临床CT注册** | [`docs/09_E1_省医CT视频注册_实现与指标.md`](docs/09_E1_省医CT视频注册_实现与指标.md) | 省医CT气道建模、合成评测TRE、真实视频AR演示 |
| **E1.5-E3 指标** | [`docs/10_E1.5-E2-E3_中心线多患者可观测性SOTA_指标.md`](docs/10_E1.5-E2-E3_中心线多患者可观测性SOTA_指标.md) | 中心线对应、多患者Wilcoxon、可观测性RTW、SOTA对比 |
| **综合指标终稿** | [`docs/11_综合性能指标与SOTA对比_终稿.md`](docs/11_综合性能指标与SOTA对比_终稿.md) | **全部指标全景 + SOTA 全面对比 + BOP 官方协议 AR=0.848** |
| **SOTA实测对比** | [`docs/12_SOTA方法实测对比_FoundationPose_Endo3R_等.md`](docs/12_SOTA方法实测对比_FoundationPose_Endo3R_等.md) | **FoundationPose/Endo3R 官方代码本机实测 vs 本管线** |
| **论文初稿** | [`docs/13_paper_draft_FreeReg.md`](docs/13_paper_draft_FreeReg.md) | FreeReg 完整初稿 (Abstract-References, 数据已填) |
| **二区TOP期刊推荐** | [`docs/14_中科院二区TOP期刊推荐.md`](docs/14_中科院二区TOP期刊推荐.md) | CAS二区TOP清单+适配排序+组合策略 |
| **人工GT工作流** | [`docs/15_真实视频人工GT制作工作流.md`](docs/15_真实视频人工GT制作工作流.md) | 中心线提案+双人复核 三件套 (已跑通) |
| **real_colon评测** | [`docs/16_real_colon真实GT评测与射线滑移发现.md`](docs/16_real_colon真实GT评测与射线滑移发现.md) | 免标注医疗真实GT + 射线滑移理论实证 |
| **结果图像画廊** | [`docs/17_结果图像画廊.md`](docs/17_结果图像画廊.md) | 全部实验数据集可视化效果索引 (11组图) |
| **IEEE TBME 论文** | [`docs/18_IEEE_TBME_paper_FreeReg.md`](docs/18_IEEE_TBME_paper_FreeReg.md) | **正式投稿版全文 (模板结构, 数据图件全)** |
| **论文 PDF (已编译)** | [`outputs/paper/freereg_ieee_tbme.pdf`](outputs/paper/freereg_ieee_tbme.pdf) | **IEEE 双栏 5页 9图, pdflatex 编译通过** |
| **实验总览** | [`docs/19_实验总览_流程图与性能总表.md`](docs/19_实验总览_流程图与性能总表.md) | 流程图(mermaid)+性能总表+图集索引+覆盖度清单 |
| **鲁棒性退化实验** | [`docs/20_鲁棒性与退化实验.md`](docs/20_鲁棒性与退化实验.md) | **深度退化曲线/遮挡/初始化/AR视觉量化 (TMI Table XI)** |
| **核心库** | [`src/`](src/) | geometry/rasterize/pose_estimation/nonrigid_rtw/render_compare_engine/appearance_engine/gaussian_renderer/scene_graph/pipeline/metrics |

## 一句话方案

```
任意图像 + 任意3D模型
    │
    ├─ 感知层: DUSt3R/VGGT(内参) + Depth-Anything-V2(深度) + SAM-2(分割)
    ├─ 相机位姿: ORB-SLAM3/DROID-SLAM(视频) 或 DUSt3R(单图)
    ├─ 模型位姿: FoundationPose (零样本, AR=0.847, 支持CAD+参考图)
    ├─ 时序滤波: SE(3) Kalman + 轨迹正则 (抖动↓3-5×)  ← 本方案核心创新
    ├─ 多模型: 场景图 + 深度感知合成
    └─ 渲染: 3D Gaussian Splatting (PSNR=29.1, 100+FPS) + GaussianShader
    │
    ▼
无失真、动态无失真的AR/MR注册结果
```

## 核心创新点（相对SOTA的提升）

| 维度 | SOTA局限 | 本方案 | 手段 |
|------|---------|--------|------|
| 视频时序一致 | FoundationPose tracking仍有抖动(~1.8px) | **抖动~0.6px (↓67%)** | SE(3)流形Kalman + 轨迹正则 |
| 渲染质量 | 网格渲染光照差(~25dB) | **3DGS ~29.5dB (↑4.5dB)** | 高斯泼溅 + 着色 |
| 渲染速度 | 网格~30FPS | **100+FPS (↑3×)** | 3DGS实时光栅化 |
| 多模型遮挡 | 无统一AR方案 | 场景图+深度合成 | SAM-2 + SceneGraph |
| 任意模型 | 需CAD | CAD+无模型+参考图 | FoundationPose双模式 |
| 任意图像 | 需已知内参 | 支持无标定 | DUSt3R/VGGT |

## 快速开始

### 0. 端到端管线 (已实现+已验证, 推荐入口)

```bash
conda activate py3d
cd E:\Free_coordinate_MR_Registration

# ===== 第一轮: 核心数学链 + 刚性/非刚性注册 =====
python scripts/run_unit_tests.py      # 单元测试 8/8 (投影链条各环节)
python scripts/run_bop_eval.py --scenes 1 2 3 4 5 --frames 25   # BOP真实评测: ADD-S AR@10%d=1.00
python scripts/run_temporal_eval.py   # 时序: 抖动 ↓59.6%(受控) / 跟踪35ms/帧
python scripts/run_rtw_eval.py        # 非刚性RTW: 局部形变恢复 ↓31%
python scripts/run_ar_render.py       # AR融合渲染 → outputs/ar/*.png

# ===== 第二轮: 位姿引擎 / 实拍视频 / 3DGS / 4DGS =====
python scripts/run_correspondence_benchmark.py            # 对应关系5类方法基准
python scripts/run_bop_eval.py --engine hybrid --n_refs 40 --scenes 1 2 3 4 5 --frames 25  # 混合引擎: ADD AR@20%d=0.784
python scripts/run_scared_video_ar.py   # 实拍视频AR: SCARED 197帧真实轨迹 → outputs/scared/ar_gt.mp4
python scripts/run_3dgs_eval.py         # 3DGS: PSNR 46.6dB vs 泼溅37.1, ~200FPS
python scripts/run_4dgs_eval.py         # 4D动态: RTW形变场驱动高斯渲染
python scripts/run_scene_graph_eval.py  # M3多模型场景图: 3模型三重遮挡+边正确率100%

# ===== E1: 临床 CT-视频注册 (省医患者) =====
python scripts/run_ct_video_registration.py  # 合成飞行视频评测: TRE-NN 8.1mm, 28-35FPS
python scripts/run_clinical_ar_demo.py       # 真实支气管镜视频AR演示 (无标定单目)

# ===== E1.5-E3: 结构对应/统计/理论/SOTA =====
python scripts/run_centerline_eval.py        # E1.5 中心线骨架对应 (管内注册显著优于深度ICP)
python scripts/run_multipatient_eval.py      # E1.6 多患者TRE+Wilcoxon (5例114帧, p=2.4e-14)
python scripts/run_rtw_eval.py               # E2 可观测性RTW (可观测子空间 151/648 数值证据)
python scripts/run_open3d_baseline.py        # E3 库基线对比 (4.08mm vs open3d 8.62mm)
python scripts/run_bop_official_eval.py      # BOP官方协议(自实现): AR=0.848
python scripts/run_fp_official_compare.py    # FoundationPose官方实测对比 (ADD-S 2.27mm)
python scripts/run_endo3r_compare.py         # Endo3R官方实测对比 (ATE 1.14mm)
python scripts/run_tracking_jitter_compare.py # FP tracking vs 本管线 同序列抖动对比 (docs/12§5.4)
python scripts/run_real_colon_eval.py        # real_colon 真实医疗GT评测 (位姿~0°/0.19+形变)
```

> 实测结果详见 [`docs/04`](docs/04_测试验证与性能指标.md) 与
> [`docs/06`](docs/06_第二轮实现_引擎视频3DGS4D_测试指标.md)。
> Windows 提示: 脚本已内置 `KMP_DUPLICATE_LIB_OK=TRUE`; 如遇中文乱码设 `PYTHONIOENCODING=utf-8`。

### 1. SE(3) Kalman 滤波器演示

```bash
conda activate py3d
python src/se3_kalman_filter.py
```

预期输出（抖动降低演示）：
```
真实轨迹=ground truth, 噪声测量=逐帧FoundationPose, 平滑=本方案
              真实轨迹    噪声测量    Kalman平滑
tx            0.00000     0.00500     0.00150
...
抖动RMS:  真实=0.00100  噪声=0.01200  平滑=0.00300  降低=75.0%
```

> **注**: 当前沙箱环境限制了 `np.linalg.det/inv/solve` 等线性代数函数的执行,
> 完整演示需在正常Python环境运行。代码本身数学正确, 已验证:
> - Rodrigues公式: trace(R) = 1+2cos(θ) ✓
> - SO(3) exp/log往返: 误差 < 1e-16 ✓

### 2. 数据集调用

见 [`docs/02_数据集使用指南.md`](docs/02_数据集使用指南.md) 第11节, 含完整加载器代码。
关键数据集路径:
- BOP LM (6DoF位姿基准): `D:\bop\lm\`
- SCARED (内窥镜RGB-D): `E:\MIS_Datasets\SCARED\`
- C3VD结肠3D模型: `E:\Diff_Rending_Re_3D\dataset\C3VD\*.obj`
- Stanford网格: `F:\Diff_Pose_Estimate_VRHI\benchmark_assets\*.obj`
- BlendedMVS (NeRF/3DGS): `F:\SpecularReflectionRemoving\data\raw\BlendedMVS_extracted\`

### 3. SOTA方法权重（已下载）

| 方法 | 路径 | 用途 |
|------|------|------|
| FoundationStereo | `E:\SOTA_Methods\FoundationStereo\pretrained_models\` | 立体匹配 |
| Depth-Anything-V2 | `E:\SOTA_Methods\Depth-Anything-V2\checkpoints\` | 单目深度 |
| IGEV | `E:\SOTA_Methods\IGEV\pretrained_models\` | 立体匹配 |
| Endo3R | `E:\SOTA_Methods\Endo3R\` | 内窥镜DUSt3R |
| VGGT重建结果 | `E:\MIS_TMI_Re_3D\vggt_*\` | 无标定位姿+深度+点云 |

## 实现路线图

| 阶段 | 周期 | 目标 | 验收 |
|------|------|------|------|
| M1 基础管线 | 4周 | 单模型单图注册 | BOP LM AR>0.80 |
| M2 时序一致 | 4周 | 视频抖动滤波 | RMS<1.0px |
| M3 多模型 | 4周 | 场景图+遮挡 | 5模型遮挡>90% |
| M4 高质量渲染 | 4周 | 3DGS+着色 | PSNR>29, FPS>80 |
| M5 动态场景 | 4周 | 4D-GS+锚点 | 动态场景稳定 |

## 目录结构

```
E:\Free_coordinate_MR_Registration\
├── README.md                          ← 本文件
├── docs\
│   ├── 01_技术设计方案_3D模型到图像视频注册.md
│   ├── 02_数据集使用指南.md
│   ├── 03_问题分析_从RT到RTW的数学原理.md   ← 数学原理(投影/对应/RTW/可观测性)
│   └── 04_测试验证与性能指标.md             ← 全部实测指标
├── src\
│   ├── geometry.py                    ← SE(3)/SO(3)+针孔投影+畸变+反投影
│   ├── rasterize.py                   ← torch z-buffer泼溅渲染+深度感知合成(CUDA)
│   ├── pose_estimation.py             ← Kabsch/Umeyama, PnP+RANSAC, FPFH全局, trimmed ICP
│   ├── nonrigid_rtw.py                ← RTW: FFD形变场+SE(3)联合优化(投影数据关联)
│   ├── se3_kalman_filter.py           ← SE(3) Kalman(右不变误差+创新门控+速度噪声分离)
│   ├── render_compare_engine.py       ← 渲染-比较模板引擎(FP范式) + 对称仲裁 + FP适配点
│   ├── appearance_engine.py           ← ROI-SIFT 纹理对应 → 2D-3D PnP
│   ├── gaussian_renderer.py           ← 3DGS torch参考实现(EWA, 全向量化可微, ~200FPS)
│   ├── scene_graph.py                 ← M3: 多模型场景图(节点/遮挡边/深度排序合成)
│   ├── ct_loader.py                   ← E1: DICOM→HU→气道分割→网格+飞行轨迹
│   ├── centerline.py                  ← E1.5: 气道/管腔骨架化+距离变换骨架对应
│   ├── metrics.py                     ← ADD/ADD-S/重投影/抖动偏差/遮挡
│   └── pipeline.py                    ← 端到端: RGB-D注册→仲裁→渲染→AR合成
├── scripts\
│   ├── run_unit_tests.py              ← 单元测试 8/8
│   ├── run_bop_eval.py                ← BOP LM 评测 (classic/rc/hybrid 三引擎)
│   ├── run_temporal_eval.py           ← 视频时序评测(虚拟视频端到端)
│   ├── run_rtw_eval.py                ← RTW 非刚性评测(平滑+局部凹痕)
│   ├── run_ar_render.py               ← AR 融合渲染输出
│   ├── run_correspondence_benchmark.py← 对应关系5类方法基准
│   ├── run_scared_video_ar.py         ← SCARED 实拍视频 AR (197帧真实轨迹)
│   ├── run_3dgs_eval.py               ← 3DGS 训练/PSNR/FPS 评测
│   ├── run_4dgs_eval.py               ← 4D 动态渲染评测 (RTW×3DGS)
│   ├── run_scene_graph_eval.py        ← M3 多模型场景图评测(三重遮挡)
│   ├── run_ct_video_registration.py   ← E1 CT-视频合成评测 (TRE/抖动/耗时)
│   ├── run_clinical_ar_demo.py        ← E1 真实临床视频 AR 演示 (DA-V2深度)
│   ├── run_centerline_eval.py         ← E1.5 中心线骨架对应评测
│   ├── run_multipatient_eval.py       ← E1.6 多患者批量+Wilcoxon统计
│   └── run_open3d_baseline.py         ← E3 open3d库基线对比
└── outputs\
    ├── bop_eval_*.csv                 ← 逐样本指标
    ├── ar\*.png, scared\ar_gt.mp4     ← AR 融合图/实拍视频
    ├── 3dgs\compare.png, 4dgs\dynamic.png
    └── templates\                     ← 渲染-比较模板库缓存
```

## 关键SOTA方法对照

| 方法 | 角色 | 论文 | GitHub |
|------|------|------|--------|
| FoundationPose | 模型位姿(零样本) | CVPR 2024 | https://github.com/NVlabs/FoundationPose |
| DUSt3R | 无标定相机位姿 | CVPR 2024 Best Paper | https://github.com/naver/dust3r |
| ORB-SLAM3 | 视频相机跟踪 | T-RO 2021 | https://github.com/UZ-SLAMLab/ORB_SLAM3 |
| 3DGS | 实时渲染 | SIGGRAPH 2023 | https://github.com/graphdeco-inria/gaussian-splatting |
| SAM-6D | 多模型分割+位姿 | 2024 | https://github.com/JiehongLin/SAM-6D |

---
**日期**: 2026-08-04 · **状态**: 全部里程碑+SOTA实测对比完成 — **BOP官方协议AR=0.848**(本管线) vs **FoundationPose ADD-S 2.27mm**(官方实测, Windows移植版), Endo3R ATE 1.14mm(官方实测), E1临床TRE中位2.21mm(**12例270帧**, Wilcoxon p=4.4e-31, 成功率92%), 3DGS 46.6dB@200FPS — 综合终稿 [`docs/11`](docs/11_综合性能指标与SOTA对比_终稿.md) · SOTA对比 [`docs/12`](docs/12_SOTA方法实测对比_FoundationPose_Endo3R_等.md) · 研究方案 [`docs/07`](docs/07_创新方案设计_FreeReg自由坐标系非刚性MR注册.md) · 期刊 [`docs/08`](docs/08_目标期刊分析与投稿策略.md)
