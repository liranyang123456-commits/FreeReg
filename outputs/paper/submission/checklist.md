# FreeReg 投稿清单 (IEEE TMI Submission Checklist)

**投稿目标**: IEEE Transactions on Medical Imaging (TMI)
**标题**: FreeReg: Free-Coordinate Registration of Arbitrary 3D Models to Arbitrary Image Streams for Augmented and Mixed Reality
**对应作者**: Ranyang Li (lry@haut.edu.cn)

---

## 1. 格式合规 Format Compliance

| 项目 | 要求 | 当前 | 状态 |
|---|---|---|---|
| 页数 (含参考文献) | ≤10 (regular paper) | 10 | ✅ |
| 摘要词数 | ≤250 words | 250 | ✅ |
| 正文模板 | ieeecolor2 (IEEE TMI) | ✓ | ✅ |
| 图数量 | — | 9 (Fig.1–9) | ✅ |
| 表数量 | — | 15 (Table I–XV) | ✅ |
| 参考文献 | 全部被正文引用 | 39 条, 无缺失/多余 | ✅ |
| 未定义引用/交叉引用 | 无 | 0 | ✅ |
| 图超出栏宽 (overfull) | 无内容性 overfull | 仅模板 LOGO | ✅ |

## 2. 内容完整性 Content Integrity

| 项目 | 状态 |
|---|---|
| 所有实验数字与可复现运行一致 (已逐一核验) | ✅ |
| real-colon 52% 发散已诚实归因 (初值对称翻转) | ✅ |
| synthetic RTW 数字改为可复现值 (不可复现的已移除) | ✅ |
| Fig.6 为真实实验 (非手绘) | ✅ |
| Fig.5 证据类别已区分 (定性真实视频 vs 定量合成) | ✅ |
| observability 梯度 + stereo 提升已加入 (Fig.6) | ✅ |
| 伦理声明 (Ethics Statement) | ✅ |
| 致谢 (基金号) | ✅ |

## 3. 投稿材料 Submission Materials

| 文件 | 位置 | 状态 |
|---|---|---|
| 主文稿 PDF | `outputs/paper/freereg_tbme_full.pdf` | ✅ |
| LaTeX 源 | `outputs/paper/freereg_tbme_full.tex` | ✅ |
| Cover letter | `submission/cover_letter.txt` | ✅ |
| Highlights | `submission/highlights.txt` | ✅ |
| 代码/复现 | github.com/liranyang123456-commits/FreeReg | ✅ |

## 4. 投稿前最后确认 (作者自查)

- [ ] 作者顺序与单位最终确认
- [ ] 基金号确认 (6240074117 / 2022BS076 / 2022BS075 / 21421329 / VRLAB2023C03 / 2024TXTD18)
- [ ] 通讯作者邮箱确认
- [ ] TMI ScholarOne 在线系统信息填写 (关键词/领域/审稿人建议)
- [ ] 图件分辨率 ≥300 dpi (印刷要求)
- [ ] 利益冲突声明 (如有)
- [ ] 建议/回避审稿人名单 (如需要)

## 5. 备注

- 论文已重新定位为**通用 free-coordinate AR/MR 注册方法**，医学（支气管镜）作为最强临床验证场景。
- 所有关键数字 (4.23mm TRE, 0.848 AR, 0.921 T-LESS, 1.1% colon, 23%/41% observability) 均与 `outputs/` 下可复现运行一致。
- 参考文献 39 条与正文一一对应。
