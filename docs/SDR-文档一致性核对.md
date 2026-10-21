# SDR-Voxel R-CNN：文档 ↔ 代码 一致性核对（v2）

> 核对基准：`scp-innovations/最终方案.html`（v2，16 处修订 + 5.3 开关表 + 5.4 监控表）
> 核对方式：`python tools/check_sdr_conformance.py`，只读 yaml 与源码，**不需要 torch**
> 结论：**64 项检查：63 PASS / 0 FAIL / 1 WARN（WARN 是文档 5.5 里待你确认的 Car-only 口径，不是代码错）**

本文档记录这一轮"让代码真的符合文档、并且看得见"改了什么。上一轮（v1→v2）改的是
公式与配置；这一轮改的是**实现缺口**与**可观测性**。

---

## 1. 本轮修掉的 7 处"文档有、代码缺/不一致"

| # | 位置 | 原状 | 现在 | 为什么必须改 |
|---|---|---|---|---|
| 1 | 文档 2.5 可消融项「曲率门控开/关」 | **代码里没有这个开关**，门控恒定生效 | `MODEL.KD.CURVATURE_GATE: True`；关掉时 `w ≡ 1` | 文档把"门控值多少"列为可消融项，但代码跑不出这一组，等于该结论无法验证 |
| 2 | 文档 5.4 `msdge_gate_entropy` | 只在 `_compute_kd_loss()` 里写日志，**KD 一关就没有** | 抽出 `_msdge_tb()`，与 ② 完全解耦 | `sdr_nokd` / `sdr_noms` 两组恰恰要回答"① 有没有学到东西"，指标不能绑在 ② 的开关上 |
| 3 | 文档 5.4 `cgrd_gate_eff ≈ 1` | 直接报 w 的绝对值，**从 0.5 起步**，看不出是否塌缩 | 以初始化值 σ(0)=0.5 归一，`≈1` 才成立 | 原来那条"≈1 平稳"在代码里永远不成立，日志读起来像故障 |
| 4 | 文档 5.4 `cgrd_theta_mean / cgrd_logmag_mean` | 只在 `riemann/cosine` 下计算 | 四种 `DIST_TYPE` 全算 | `--preset sdr_mse`（l2n）是最需要看这两项的那一组，原实现里它反而没有 |
| 5 | 文档 5.3 默认值 | `data_processor` 仍是旧默认：`STRATIFIED=True`、`TARGET_KEEP_RATIO=None`、`BEAM_SCORE=mean` | 与 5.3 表对齐：`False / 0.25 / topk` | yaml 里写全了才正确，只写 `NAME` 就悄悄退回旧行为 —— 这类"配置漏写"最难查 |
| 6 | 文档修订 14「测试期无 GT」 | KITTI 的 val 划分**带标注**，采样器照用 GT 框 → 评测时"用标注保车" | 测试期丢弃 GT（`USE_GT_IN_TEST: False`，默认开） | 这是信息泄漏，不只是不一致：val AP 会虚高，且与部署分布不同 |
| 7 | 文档 1.5 可消融项「软门控 vs 硬切换」 | 无实现 | `MODEL.VFE.GATE_HARD: False`（True 时 argmax one-hot） | 文档 1.4 第 2 条"软配比优于硬切换"需要这一组对照才能成立 |

附带两处小补：

- **构建期打印 MS-DGE 参数量**（文档修订 5 要求"以构建期日志为准"）：现在会打印
  `MSDGE 尺度=[1,8,16,32](0.05/0.40/0.80/1.60 m) … 参数=0.110M` 与
  `净增参数≈0.136M(含首层 4→64 通道)`。按通道逐层累加的结果是 **+0.110 M**（本体）
  / **+0.136 M**（含首层卷积），文档里原写的 +0.16 M 已同步修正。
- **EIP-BS 的运行时统计**进 batch：`eipbs_keep_ratio / eipbs_fg_keep_ratio /
  eipbs_beams / eipbs_ms`。无 GT 时前景保留率记 **-1 哨兵**（不是 nan），
  模型侧只对有 GT 的样本求均值，避免 nan 污染整条曲线。

---

## 2. 训练期：现在能看到什么

每 `logger_iter_interval`（默认 50）次迭代，终端/日志两行：

```
[SDR] iter 500    | roi_ratio=0.0350 gate_eff=1.020 theta_mean=0.1100 logmag_mean=0.0500 weight=0.980 gate_entropy=0.830
                  | loss_rod=0.0210 loss_ford=0.0090 loss_rho=0.0040 loss_total=0.0340 to_det_ratio=0.020 eipbs_keep_ratio=0.250 eipbs_fg_keep_ratio=0.790 eipbs_beams=60.0 eipbs_ms=47.5
```

- 第一行 = 文档 5.4 的 6 个数，顺序固定；
- 第二行 = 三项损失分量 + 蒸馏/检测比值 + ③ 的四个运行时统计；
- 基线训练**不出现**这两行（`tb_dict` 里没有这些键，函数直接返回）；
- `sdr_nokd` 只打有值的部分（`gate_entropy` + `eipbs_*`），这正是它的意义；
- tqdm 模式另外把 `kd / rod / g_eff / H` 放进进度条。

同时逐迭代落盘：`iter_log.csv`（全部 `tb_dict` 标量）、`training_log.csv`、
`val_metrics.csv`（需 `--eval_interval`）、`tensorboard/`。

---

## 3. 训练后：自动生成三个文件

`tools/train.py` 在评测结束后自动调用 `tools/plot_curves.py`：

| 文件 | 内容 |
|---|---|
| `training_curves.png` | loss / lr / AP / AP50-95 / P-R-F1 / 损失分项（所有模型都有） |
| `sdr_diagnostics.png` | 9 格诊断面板（仅 SDR 组） |
| `sdr_summary.json` | 每个指标首值/末值/均值/极值，可直接抄进论文表格 |

9 格依次是：① 蒸馏三项 → ② λ 与门控 → ① 门控熵与四尺度权重 → ② θ vs |log r| →
掩码与 ③ 保留率 → ③ 存活线束 → KD/检测比值 → ③ 单帧耗时 → 验证 AP。

示例（**合成数据，仅示意版式，不是真实结果**）：
`docs/sdr_diagnostics_example_synthetic.png`

关闭：`--no_plot_after_train`；训练中断后补图：
`python tools/plot_curves.py --exp_dir <输出目录> --skip_head 500`。

---

## 4. 两项已定型的决策 + 一条保留的 WARN

`check_sdr_conformance.py` 现在 **64 项 / 63 PASS / 0 FAIL / 1 WARN**（新增了
`CLASS_PRIORITY` 长度一致性检查、教师 ckpt 存在性检查）。

> **修掉的一个严重 bug**：`Checker.add()` 的 `level` 默认值原来写成了 `'PASS'`，
> 于是 `ck.true()/ck.eq()` 的**任何失败都会被渲染成 `[PASS]` 并计入通过数** ——
> 等于把整张检查表变成"永远绿"。发现问题的方式：新增的教师 ckpt 断言明明是
> `False`（配置当时还是 `null`），输出却是 `[PASS]`。现默认值改为 `FAIL`。
> 修复后复跑，其余断言没有新的失败项，说明之前的通过结论仍成立。

### 4.1 `W_CLASS` → 取 `0`（已写进 yaml 默认值）

| | `W_CLASS: 0`（现行默认） | `W_CLASS: 0.3` |
|---|---|---|
| 训练期车体点保留 | ≈0.35 | ≈0.99 |
| 测试期车体点保留 | ≈0.35 | ≈0.35（无 GT） |
| 训练/测试分布 | **一致** | 训练比测试容易 → val AP 虚高 |
| 论文风险 | 无 | 审稿人问"测试期你怎么拿到 GT"即失守 |

取 0 的代价是训练期前景点也少，可能整体掉几个点。**这是正确方向的掉点**：原来那 0.99 是借了标注。
如果实测 `W_CLASS: 0` 掉得太多（>0.5 AP），才把 0.3 作为附录里的"GT 引导采样式增广"单独报一组，
并在论文里写清"仅训练期生效、测试期强制 0"（`USE_GT_IN_TEST: false` 已保证这一点）。

### 4.2 Car-only 口径 → `CLASS_PRIORITY: [1.0]`（已改）

`_box_relations()` 里 `prios = class_priority[labels - 1]`，单类训练时所有 GT 的 `label` 都是 1，
只会取列表第 1 项。原来写 `[0.6, 1.0, 1.0]` 时 `W_CLASS=0.3` 的实际权重是
`0.3 × 0.6 = 0.18`，和文档写的 0.3 不符、别人也复现不出来，故改为 `[1.0]`。

**建议**：主表按 KITTI 标准 3 类（Car/Pedestrian/Cyclist）训练，此时用 `[0.6, 1.0, 1.0]`，
行人/骑行者分支才真正起作用；Car-only 只作快速验证与消融。这也是剩下那条 WARN 的内容。

---

## 5. 教师权重：CG-RD 的前置条件

### 5.1 用官方预训练 ckpt（标准 KD 做法）

`BEVTeacher` 只建 **VFE + 3D 骨干 + MapToBEV + 2D 骨干**（不含检测头），所以教师只需要
一个"已经很强的稠密 Voxel R-CNN 前段"。这就是知识蒸馏的常规设定：**教师是现成模型，
学生从零开始学稀疏输入**。

本项目已配置为 OpenPCDet 官方 **KITTI Car-only** 权重：

```
MODEL.TEACHER.PRETRAINED_PATH: checkpoints/voxel_rcnn_84.54.pth
```

| 教师来源 | 优点 | 问题 | 结论 |
|---|---|---|---|
| **官方 `voxel_rcnn_car_84.54.pth`** | 现成的强模型，符合 KD 标准设定；**Car-only，与本项目 `CLASS_NAMES: ['Car']` 完全同构**；不用自己训 80 epoch | 训练增强管线与官方不同（但"教师更强"正是 KD 想利用的，不构成问题） | **默认主结果** |
| 自己 `--preset baseline` 训的 ckpt | 与学生同数据、同增强，消融变量最少 | 要多一次 80 epoch；且教师强度取决于你复现得好不好 | 备选 / 附加消融（"同管线教师"对照） |
| 改进模型自己的 ckpt | 不用额外训练 | 教师也看过稀疏输入，"稠密→稀疏"通道消失，② 立论不成立 | **不要用** |

### 5.2 已实测：`voxel_rcnn_84.54.pth` 与本仓库逐层对齐

```
格式            : legacy，裸 state_dict（无 model_state 包装）
张量总数        : 257
按顶层前缀分布  : roi_head 94 / backbone_2d 84 / backbone_3d 72 / dense_head 6 / global_step 1
关键通道        : backbone_3d 首层输入 4（MeanVFE 输出） ✓
                  backbone_2d 首层输入 256（NUM_BEV_FEATURES） ✓
                  backbone_3d 输出 128（HeightCompression 的 C） ✓
                  dense_head.conv_cls 输出 2 → 1 类 × 2 锚 → Car-only ✓
教师侧张量 156 个，全部命中 → PASS
```

两个容易误判的点（第一版体检脚本就在这里误报 FAIL）：

- ckpt 里**没有 `vfe.*`**：`MeanVFE` 是逐体素取均值，**没有可学习参数**；
- ckpt 里**没有 `map_to_bev.*`**：`HeightCompression` 只做 dense + reshape，**也没有参数**。

所以教师侧应有的 156 个张量 = `backbone_3d` 72 + `backbone_2d` 84，与 ckpt 完全一致。

另外三个数据配置（`kitti_dataset.yaml` / `_sdr` / `_sparse`）的
`POINT_CLOUD_RANGE` 与 `VOXEL_SIZE` 都是 `[0,-40,-3,70.4,40,1]` / `[0.05,0.05,0.1]`，
BEV 网格尺寸相同，教师与学生的特征图逐格对应，不需要插值。

### 5.3 半随机教师：最贵的失败模式

`load_teacher_from_baseline()` 是"按名 + 形状匹配，匹配不上静默跳过"。填错路径的后果是：

- 教师冻结在**随机初始化**上；
- `cgrd_theta_mean` 照样从大变小（学生在拟合噪声），9 格诊断图看起来完全正常；
- 最终 AP 只等于稀疏基线 → 会误判成"蒸馏没用"，白跑 80 epoch。

所以做了四道防线：构建期 WARNING（含修复顺序）、命中率 <90% 判为加载失败、
BEV 尺寸不一致时打印一次插值警告、独立体检脚本（无 torch 也能跑）。

### 5.4 验收顺序

```bash
# 1) 体检 ckpt（不需要 torch、不需要 spconv）
python tools/check_teacher_ckpt.py --ckpt checkpoints/voxel_rcnn_84.54.pth \
  --config tools/cfgs/kitti_models/sdr_voxel_rcnn_car.yaml
#    PASS → 教师 156/156 命中
#    WARN → 含 kd_criterion.*（改进模型自身，自蒸馏口径不成立）/ 通道对不上
#    FAIL → 没有任何 backbone 张量（不是 Voxel R-CNN 权重）

# 2) 开跑，核对这一行
python tools/train.py --preset sdr
#    [SDR-VoxelRCNN] teacher 权重已载入: checkpoints/voxel_rcnn_84.54.pth  (156/156 个张量匹配, 100%)

# 3) 人工兜底（2 分钟）：真实教师的 cgrd_theta_mean 会从 ~0.9 明显降到 0.3 以下并持续走低；
#    随机教师只在常数附近抖动、不下降。
```

**写论文时注意两点**：① baseline 那一行必须用**你自己复现的** 64 线基线数字，不能直接写
84.54（那是教师权重自带的成绩，训练配方不同）；② 所有对比组共用同一个教师，否则增益不可比。

---

## 6. 装好 torch + spconv 后的验收顺序

```bash
# 1) 静态一致性（不需要 torch，随时可跑）
python tools/check_sdr_conformance.py -v

# 2) 链路自检：配置 → 数据 → EIP-BS → 前反向 → 推理（5 步）
python tools/check_sdr_setup.py --preset sdr
#    第 4 步应额外打印：5.4 六个监控量「齐全」、门控熵、近/中/远权重、EIP-BS 保留率/耗时

# 3) 跑 1 个 epoch 看终端是否出现 [SDR] 行，并在结束后确认出图
python tools/train.py --preset sdr --epochs 1
ls output/cfgs/kitti_models/sdr_voxel_rcnn_car/default/   # 应有 sdr_diagnostics.png / sdr_summary.json
```

判读优先级（来自文档 5.4 + 5.5）：

- `eipbs_keep_ratio` 必须严格 ≈0.250（修订 11）；偏离说明预算对齐没生效；
- `gate_entropy` 恒为 1 → ① 没学到东西，先查 `DENSITY_PRIOR` / `GATE_PRIOR_DIRECT`；
- `gate_eff` 单调下滑 → ② 门控在塌缩，确认 `GATE_NORM: mean`；
- `logmag >> theta` → 调 `RADIAL_WEIGHT` 或切 `cosine`；
- `to_det_ratio > 1` → 降 `LOSS_WEIGHT`。

最后一句仍然成立：**代理指标 ≠ AP**。上面所有数字只说明"模块在正常工作"，
"到底值多少 AP"必须靠 `--preset sparse` vs `--preset sdr` vs `--preset sdr_mse` 说话。

---

## 7. 全链路审计（数据 → ckpt → 训练 → 显示 → 出图）

第 7 轮把"从 dataloader 到出图"的每条路都走了一遍源码，又抓到 **4 个会在真实训练里
直接报错或悄悄污染对比的 bug**，另有 3 处体验问题。

| # | 位置 | 症状 | 修法 |
|---|---|---|---|
| 1 | `pcdet/datasets/dataset.py` `collate_batch` | **`--preset sparse` 第一帧就崩**：`WRITE_TO_MAIN` 分支会写 `voxel_num_points_true`，它每帧长度不同，却不在拼接白名单里 → 落到 `np.stack` → `TypeError: Error in collate_batch` | 把 `voxel_num_points_true` 加进拼接列表（与 `voxel_num_points` 同处理） |
| 2 | `kitti_dataset_sparse.yaml` | sparse 对照组仍用**三类**采样组 `Car:20/Ped:15/Cyc:15`，而 baseline/sdr 是 Car-only `Car:15` → 对照组每帧多贴 5 个车，"sparse vs sdr 只差保住了哪些点"不成立 | 改成 `Car:5` / `Car:15` / `LIMIT_WHOLE_SCENE: False`，三组增强完全一致（已写进断言） |
| 3 | `detector3d_template.load_params_from_file` | 用**官方裸 `state_dict`**（如 `voxel_rcnn_84.54.pth`）初始化时报 `KeyError: 'model_state'` | 兼容两种：有 `model_state` 取它，否则整个 dict 就是 state_dict |
| 4 | `train_utils.train_one_epoch` | `use_logger_to_record=True`（默认）时 **tqdm 不前进**、`disp_dict` 里的 `kd/rod/g_eff/H` 永远不显示 | `pbar.update()` 与 `set_postfix` 移出分支，两种记录模式都更新 |
| 5 | `tools/test.py` | 没有 `--preset`，评测必须手写 `--cfg_file`，消融组还容易漏掉 `DIST_TYPE/ENABLE` 覆写 | 加 `--preset`，与 `train.py` 共用同一套覆写与 tag 后缀 |
| 6 | `test.py` `repeat_eval_ckpt` | `train.py` 的 `--max_waiting_mins` 默认 0，训练结束仍固定空等 30 s，看着像卡死 | 已评测过且预算用尽时直接 break |
| 7 | `plot_curves.component_keys` | `cgrd_*`/`msdge_*`/`eipbs_*` 混进 "Loss components" 面板，把 `rpn_loss` 压成一条直线 | 按前缀过滤，诊断量只进九格面板 |

### 7.1 新增：训练结束的检测效果图（此前整条链路缺这一段）

`tools/train.py` 之前只出统计图，**没有任何一张"框画得对不对"的图**；`tools/demo.py` 走 open3d
交互窗口、不落盘，服务器上跑不起来。现在新增 `tools/visualize_detections.py`：

- 每帧一张三联图：`GT`（红）/ `Pred`（蓝，按 score 深浅）/ **放大叠加**（GT+Pred+分数）；
- 另出 `detections_overview.png`（2×K 总览）与 `detections_summary.json`（帧号、点数、保留率、预测数、最高分）；
- 画的是 `test_set` 的点云，即 **EIP-BS 之后的稀疏输入**，与评测/部署同口径；
- 只用 matplotlib（Agg）落盘，**不需要 open3d / 显示器**。

**帧号只随机一次**：首次按 `--vis_seed` 抽 `--vis_num` 帧写进 `output/vis_frames.json`，
之后 baseline / sparse / sdr / sdr_mse 各组都读它 —— 同一批场景才能并排比较。
`--vis_frames 192 238 318` 可显式指定并覆盖；删清单或 `--reset_frames` 才重新随机；
清单帧号超出当前数据集（例如从完整 KITTI 切到 500 帧子集）会自动重抽并打印提示。

### 7.2 通道/特征对齐的静态复核

`check_sdr_conformance.py` 新增了三条只靠 yaml 就能算的断言：

1. **BEV 通道链**：`z = 40+1 → 21 → 11 → 6 → 2`（conv2/3/4 k3s2p1，conv_out k3s2p0），
   `128 × 2 = 256 == MAP_TO_BEV.NUM_BEV_FEATURES`，否则 `BaseBEVBackbone` 首次卷积就会报尺寸不符；
2. **师生 2D 骨干超参一致**（`LAYER_NUMS/STRIDES/NUM_FILTERS/UPSAMPLE_*`）→ `f_t` 与 `f_s` 同层同分辨率，
   `CGRDLoss` 里那条"尺寸不一致就双线性插值"的兜底不会被触发；
3. **三组 gt_sampling 完全一致** → baseline / sparse / sdr 之间只剩"输入"这一个自变量。
