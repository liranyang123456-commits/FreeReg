# 中科院二区 TOP 期刊推荐 (适配本工作)

> **日期**: 2026-08-04 · **注意**: 中科院分区每年更新 (2023/2025 版),
> 投稿前请以最新官方分区表 (中国科学院文献情报中心) 复核;
> 以下适配排序基于本工作证据结构 (docs/08/11/12/13)。

---

## 1. 推荐清单 (按适配度排序)

| # | 期刊 | CAS分区 | IF≈ | 适配度 | 与本工作的契合点 |
|---|------|--------|-----|--------|-----------------|
| 1 | **Computers in Biology and Medicine (CBM)** | **二区 TOP** | ~7 | ★★★★★ | 方法+系统+临床统计, 本工作全部证据直接对口 (docs/08 首选) |
| 2 | **IEEE Trans. Biomedical Engineering (TBME)** | **二区 TOP** | ~4.5 | ★★★★ | 注册+滤波+内镜导航的工程内核; 难度为二区里最高一档, 需真实 GT 补强 |
| 3 | **Computer Methods and Programs in Biomedicine (CMPB)** | **二区 TOP** | ~6.1 | ★★★★ | 软件系统+可复现包 (17脚本+Windows移植), 系统型贡献友好 |
| 4 | **IEEE Robotics and Automation Letters (RA-L)** | **二区 TOP** | ~4.3 | ★★★☆ | 手术机器人导航角度 (跟踪 35ms/0翻转), 短文快, 可转 ICRA |
| 5 | **Physics in Medicine & Biology (PMB)** | **二区 TOP** | ~3.5 | ★★★☆ | 医学物理/配准建模, RTW+可观测性理论可契合 |
| 6 | **Biomedical Signal Processing and Control (BSPC)** | **二区 TOP** | ~5 | ★★★ | 若突出 SE(3) Kalman 滤波/时序处理主线 |
| 7 | **Artificial Intelligence in Medicine (AIM)** | 二区 TOP (近年波动) | ~6 | ★★★ | 医疗 AI 方法, 需强化学习式成分 (如神经评分/学习对应) |
| 8 | **Neurocomputing** | 二区 TOP | ~6 | ★★☆ | 偏学习范式, 本工作为几何/结构方法, 契合弱 |
| 9 | **IEEE TVCG** | 一区或二区 TOP (视年份) | ~4.7 | ★★★★ | MR 系统+3DGS 渲染主线时的最优图形学出口 (部分年份列一区 TOP) |
| 10 | **Engineering Applications of AI (EAAI)** | 二区 TOP | ~8 | ★★☆ | 泛 AI 应用, 叙事需 AI 化 |

> 注: **IEEE TMI / MedIA 为一区 TOP** (不在本清单, 属冲顶路径, 见 docs/08 决策树);
> IJCARS (手术导航对口) 为三区非TOP, 见刊快但不在二区TOP范畴。

## 2. 选择建议 (基于证据结构)

```
证据强项 = 临床统计(12例 Wilcoxon) + SOTA对比(FP/Endo3R实测) + 全链路系统
证据弱项 = 无真实视频人工GT (制作中, 见 docs/15 工作流) + 单中心数据

首选:        CBM (#1) —— 证据已足, 周期适中, 决策树见 docs/08
稳妥系统向:  CMPB (#3) —— 若审稿更认可软件系统贡献
工程强化向:  TBME (#2) —— 需补真实 GT 后再投
快速机器人向: RA-L (#4) —— 短文快通道, 占坑优先
```

**推荐组合**: CBM 正投 (现在) + 真实 GT 完成后冲 TBME 或走
MICCAI'27 短文 → TMI(一区TOP) 扩展 (docs/08 §2-3)。

> **2026-08-04 更新**: 作者已选定 **IEEE TBME** 为目标期刊。
> 执行路径: 100帧双人真实GT (docs/15, 进行中) → 重投影误差+分层统计
> → docs/13 补真实GT实验段 → 投 TBME; CBM 转为备选第二志愿。

---

**文档结束** · 投稿决策树见 `docs/08` · 初稿见 `docs/13`
