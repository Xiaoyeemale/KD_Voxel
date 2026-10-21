# SDR-Voxel R-CNN 使用说明

> **SDR** = **S**parse input · **D**ensity-gated encoding · **R**iemannian distillation
> 三个字母对应三个创新点：EIP-BS（数据侧）→ MS-DGE（网络侧）→ CG-RD（训练侧）。

---

## 1. 一句话切换

在项目根目录（`KD_Voxel/`）下：

```bash
# 0) 环境装好后先自检一遍（配置 / 数据 / EIP-BS / 前反向 / 推理，5 步）
python tools/check_sdr_setup.py --preset sdr        # 只看配置不碰数据：加 --config_only

# 基线 Voxel R-CNN（与原仓库行为完全一致，稠密 64 线）
python tools/train.py --preset baseline

# 改进模型 SDR-Voxel R-CNN（③+①+② 全开）
python tools/train.py --preset sdr

# —— 消融组（同样一条命令，输出目录自动分开，不会互相覆盖 ckpt）——
python tools/train.py --preset sparse     # 基线模型 + 均匀抽线到 25%（等点数稀疏对照）
python tools/train.py --preset sdr_noms   # 去掉 ①（VFE 换回 MeanVFE）
python tools/train.py --preset sdr_nokd   # 去掉 ②（不蒸馏）
python tools/train.py --preset sdr_mse    # ② 换成归一化 L2 蒸馏（② 的对照组）
```

`sparse` 是审稿人一定会问的那组：同样的点数预算、同样未经改动的基线模型，
只有"保住了哪些点"不同。`sdr_mse` 则是回答"② 相对普通 MSE 蒸馏到底值多少"的关键一行。

等价写法（不依赖 `--preset`）：

```bash
python tools/train.py --cfg_file tools/cfgs/kitti_models/voxel_rcnn_car.yaml
python tools/train.py --cfg_file tools/cfgs/kitti_models/sdr_voxel_rcnn_car.yaml
```

`--preset` 只是把上面的 `--cfg_file` 映射成了几个词（外加几个消融开关的覆写），
其余参数（`--batch_size`、`--epochs`、`--extra_tag` 等）照旧可用。任意配置项也可以用
`--set` 现场改，例如验证"门控是不是在起作用"：

```bash
python tools/train.py --preset sdr --set MODEL.KD.GATE_NORM count
python tools/train.py --preset sdr --set MODEL.VFE.DENSITY_PRIOR no_count
python tools/train.py --preset sdr --set MODEL.VFE.SCALE_FACTORS "[8,16,32]"
```

---

## 2. 三个创新点分别落在哪里

| 创新点 | 环节 | 落地位置 | 触发方式 |
|---|---|---|---|
| ③ EIP-BS 边缘重要性保持渐进波束采样 | 数据输入 | `pcdet/utils/eipbs_utils.py` + `DataProcessor.eip_bs_voxelize` | `kitti_dataset_sdr.yaml` 的 `DATA_PROCESSOR` 里多一项 |
| ① MS-DGE 多尺度密度门控体素编码 | 点→体素编码 | `pcdet/models/backbones_3d/vfe/msdge_vfe.py`（注册名 `MSDGE`） | `MODEL.VFE.NAME: MSDGE` |
| ② CG-RD 曲率门控黎曼蒸馏 | 训练期损失 | `pcdet/models/kd/cgrd_loss.py` + `pcdet/models/kd/teacher_branch.py` | `MODEL.KD` + `MODEL.TEACHER` |

训练时的数据流：

```
同一帧稠密点云
   ├─(原样)──────────────► transform_points_to_voxels ──► 教师分支(冻结) ──► f_t
   └─(EIP-BS 64→16 线)───► eip_bs_voxelize ─────────────► 学生分支(可训练) ──► f_s + 检测结果
                                                              └── CG-RD(f_t, f_s) ──► 蒸馏损失
```

学生分支就是最终部署的模型；教师分支只在训练时存在，推理时不跑（省一半算力）。

---

## 3. 与原独立脚本的两处口径适配（重要）

两个文件都是**框架内适配版**，数学与超参不变，只改了口径：

### 3.1 MS-DGE：Pillar → 3D Voxel

原版 `创新点1_MS_DGE.py` 自己做柱体化（z 塌缩，`voxel_coords` 的 z 恒为 0），这无法直接
接进 Voxel R-CNN 的 3D 稀疏骨干（`VoxelBackBone8x` 需要在 41 层 z 上做 `stride(2,2,2)`，
z 只有 1 层时 `conv4` 的 `padding=(0,1,1)` 会把 z 压成 0）。

适配版改为：**不做体素化，直接消费 `data_processor` 已产生的 3D 体素**，
`voxel_coords` 原样透传；多尺度池化只在 **BEV 平面（x/y）** 上做，z 不变。
接口 `get_output_feature_dim()` 返回 `OUTPUT_CHANNELS`，`forward` 写回
`batch_dict['voxel_features']` —— 与 `MeanVFE` 完全对位，骨干/检测头/ROI 头一行都不用改。

尺度倍率的换算（基础体素 0.05 m）：

| `SCALE_FACTORS` | 实际尺度 | 说明 |
|---|---|---|
| `[1, 8, 16, 32]`（默认） | 0.05（原生恒等）/ 0.40 / 0.80 / 1.60 m | **推荐**。原版是 `voxel_size=0.4, scale_factors=(1,2,4)`，本项目基础体素 0.05 m，倍率需 ×8 才落到同一组绝对尺度；另外保留 `factor=1` 的恒等分支，让 3D 骨干的 stride-1/2 与 ROI 的 `x_conv2` 仍能拿到 0.05 m 细节 |
| `[8, 16, 32]` | 0.40 / 0.80 / 1.60 m | 严格三尺度，**但没有任何分支保留原生分辨率**：0.05~0.4 m 的细节被均值抹平，实测会让小目标掉点，仅建议做消融 |
| `[1, 8, 16]` | 0.05 / 0.40 / 0.80 m | 更轻的三尺度变体 |

> 注意：`SCALE_FACTORS[0]` 是**真的参与池化**的（实现里每个尺度都按自己的倍率池化后回填），
> 只有填 `1` 时该分支才是恒等映射。文档 1.5 节写明「尺度数（2/3/4）」均可消融，
> 默认取 4 档 = 原生 + 文档三尺度，是唯一同时满足「绝对尺度对齐」与「不丢原生细节」的取法。

### 3.2 EIP-BS：dataloader 内部调用

原版是独立脚本（`EIPBS(points, gt_boxes)`）。适配版保持同一签名，只在外面加了一层
`DataProcessor.eip_bs_voxelize`：对 `data_dict['points']`（增强后的稠密点云）降线束，
再用**独立的 `VoxelGenerator`** 把结果体素化，写出 `points_kd / voxels_kd /
voxel_coords_kd / voxel_num_points_kd`。稠密那一套键完全不动。

两个可选实现（`USE_NUMBA`）：

- `true`  —— 逐点 kNN 曲率（原版精度，单帧约几十万次邻域搜索）
- `false` —— 网格协方差近似（全向量化，同一 0.5 m 单元内曲率相同，默认）

两者打分排序高度相关，`false` 在多进程 dataloader 下明显更快。

**真实帧实测**（KITTI `000000.bin`，115 384 点，单核 CPU）：

| `USE_NUMBA` | 单帧耗时 | 输出点数 | 压缩比 |
|---|---|---|---|
| `false`（默认） | **0.11 s** | 49 771 | 0.43 |
| `true` | 2.5 s | 49 778 | 0.43 |

两者输出点数几乎一致，说明网格近似没破坏"哪些点重要"的排序；但 kNN 版慢 20 倍，
只建议在离线做消融/画图时开。

⚠️ **压缩比不是 25%（=16/64）**，这是算法本身的设计，不是 bug：`KEEP_RATIO_PER_DROP: 0.2`
会对每条被丢掉的线束救援其中打分最高的 20% 点，所以最终点数 ≈ 16/64 + (48/64)×0.2 ≈ 0.40。
实测 0.43 略高，是因为 HDL-64E 各线束点数不均匀（下扫线束点数更多）。
若论文里需要严格报"16 线"的稀疏度，把 `KEEP_RATIO_PER_DROP` 设为 `0`（实测约 0.37）。

前景点保留率实测 ≈ **1.00**（框内点几乎全被救援保留），说明"边缘/前景优先"这一设计成立。

---

## 4. 教师权重怎么来（必做）

CG-RD 的教师是一个**已经很强的稠密 Voxel R-CNN**，学生从零开始学稀疏输入 —— 这是知识蒸馏
的标准设定。配置里已经指向 OpenPCDet 官方 KITTI **Car-only** 权重：

```yaml
MODEL:
    TEACHER:
        FREEZE: True
        PRETRAINED_PATH: checkpoints/voxel_rcnn_84.54.pth
```

已实测：该 ckpt 与本仓库教师逐层对齐，**教师侧 156 个张量全部命中（100%）**。

### 4.1 换权重前后各跑一次体检

```bash
python tools/check_teacher_ckpt.py --ckpt checkpoints/voxel_rcnn_84.54.pth \
  --config tools/cfgs/kitti_models/sdr_voxel_rcnn_car.yaml
```

- `PASS` → 可用。会打印教师侧张量数、`backbone_3d` 首层输入通道（应为 4）、
  `backbone_2d` 首层输入通道（应等于 `NUM_BEV_FEATURES`，本配置 256）、以及 ckpt 的类别口径；
- `WARN` → 含 `kd_criterion.*`（这是**改进模型自身**，自蒸馏口径不成立）或通道对不上；
- `FAIL` → 没有任何 `backbone_*` 张量，不是 Voxel R-CNN 权重。

这个脚本**不需要 torch、不需要 spconv**（torch 装坏时自动退到纯 pickle 解析，只读键名与形状），
所以环境还没配好时就能先确认权重对不对。

### 4.2 两个容易误判的点

- ckpt 里**没有 `vfe.*`**：`MeanVFE` 是逐体素取均值，**本身没有可学习参数**；
- ckpt 里**没有 `map_to_bev.*`**：`HeightCompression` 只做 dense + reshape，**也没有参数**。

所以"教师侧 156 个张量"= `backbone_3d`(72) + `backbone_2d`(84)，少一个都值得查。
开跑后应看到：

```
[SDR-VoxelRCNN] teacher 权重已载入: checkpoints/voxel_rcnn_84.54.pth  (156/156 个张量匹配, 100%)
```

命中率 <90% 时加载函数会判为失败并打印警告（剩余张量仍是随机初始化，蒸馏目标不可信）。

### 4.3 也可以用自己的基线 ckpt

```bash
python tools/train.py --preset baseline --extra_tag base80
# 然后把 PRETRAINED_PATH 改成那份 checkpoint_epoch_80.pth
```

好处是教师与学生同数据、同增强，消融变量更少；代价是多一次 80 epoch。两种都能用，
但**同一张表里所有对比组必须共用同一个教师**，否则增益不可比。

### 4.4 人工兜底验收（2 分钟）

用随机教师跑 50 iter 做对照：真实教师的 `cgrd_theta_mean` 会从 ~0.9 明显降到 0.3 以下并持续
走低；随机教师的 θ 只在常数附近抖动、不下降。这一步能挡掉"白跑 80 epoch 才发现教师没加载"。

---

## 5. 关键超参

`tools/cfgs/kitti_models/sdr_voxel_rcnn_car.yaml`：

```yaml
MODEL:
  VFE:
    NAME: MSDGE
    SCALE_FACTORS: [1, 8, 16, 32]   # 0.05(恒等) / 0.4 / 0.8 / 1.6 m
    OUTPUT_CHANNELS: 64             # = 3D 稀疏骨干的输入通道
    FINE_RESIDUAL: True             # 精细分支残差+BN零初始化 => 初始化即恒等
    GATE_PRIOR_DIRECT: True         # 密度/距离先验直连 softmax（零初始化旁路）
    DENSITY_PRIOR: full             # full / no_count / count_only / none（消融）
  TEACHER:
    FREEZE: True
    PRETRAINED_PATH: checkpoints/voxel_rcnn_84.54.pth   # 官方 Car-only 权重，教师侧 156/156 命中
  KD:
    ROI_DILATE: 0.5          # M_roi 的 GT 框外扩（米）
    GATE_NORM: mean          # mean=防门控塌缩（默认）| count=文档原式，仅做对照
    DIST_TYPE: riemann       # riemann / cosine / l2n / l2（l2 系为普通蒸馏对照）
    RADIAL_WEIGHT: 1.0       # |log r| 项权重，看日志 theta vs logmag 决定是否下调
    RAMPUP_ITERS: 500        # 前 N 次迭代 λ 线性爬升，避免早期 KD 压过定位梯度
    LOSS_WEIGHTS: {rod: 1.0, ford: 0.5, rho: 0.2}   # 文档 4 节的建议初值
    LOSS_WEIGHT: 1.0         # 蒸馏权重 λ（爬升后的目标值）
```

`tools/cfgs/dataset_configs/kitti_dataset_sdr.yaml`：

```yaml
- NAME: eip_bs_voxelize
  SRC_BEAMS: 64
  DST_BEAMS: 16
  STAGES: [64, 48, 32, 16]
  W_CURVATURE: 0.4
  W_DENSITY: 0.3
  W_CLASS: 0.3             # ⚠ 依赖 GT 框：训练有、测试无（见下）
  TARGET_KEEP_RATIO: 0.25  # 等点数预算，= 真实 16 线的点数比例
  ALIGN_KEEP_STRUCTURE: False   # 预算怎么花；True = 保住"哪 16 条线束"
  STRATIFIED: False        # 分层丢线；等预算下实测有害，仅"16 线保真"口径才开
  BEAM_SCORE: topk         # mean(文档原式) / top / topk / max / count
  MAX_POINTS_PER_VOXEL: 8      # 稠密分支仍是 5
  SAVE_TRUE_COUNTS: True       # 另存真实点数，供 MS-DGE 的密度先验/池化权重使用
```

> `W_CLASS` 是这套方法里**最容易翻车**的一个开关：类别先验要用 GT 框，训练期有、测试期没有。
> 实测同等预算下训练期车体点保留 0.99、测试期 0.35。要么在论文里把它定位成"训练期稀疏增强"
> 并说明差异，要么设 `W_CLASS: 0` 让两边一致。详见 `SDR-理论-实现对照与优化.md` §2、§7。

消融只需改配置，不用改代码：

| 想关掉 | 改法 |
|---|---|
| ③ EIP-BS（学生改用稠密点云） | `eip_bs_voxelize.ENABLED: False` |
| ① MS-DGE（学生改用 MeanVFE） | `--preset sdr_noms` |
| ② CG-RD（只做单模型训练） | `--preset sdr_nokd` |
| ② 换成普通蒸馏（关键对照） | `--preset sdr_mse` |
| ③ 换成均匀抽线（纯稀疏基线） | `--preset sparse` |

---

## 6. 基线为什么不受影响

1. `VoxelRCNN` 类、`voxel_rcnn_car.yaml`、`kitti_dataset.yaml` **一字未改**；
2. 新代码全部是新文件，`__init__.py` 里只是多注册了两个名字（`MSDGE`、`SDRVoxelRCNN`）；
3. `DataProcessor` 新增的方法只有在数据配置列出 `eip_bs_voxelize` 时才会执行；
4. `collate_batch` 只是把 `voxels_kd / points_kd` 等键名加进了特判列表 ——
   基线根本不产生这些键，走不到分支；
5. `train.py` 的 `--preset` 是可选项，不传时行为与之前完全一致。

---

## 7. 训练期能看到什么、训练后自动出什么图

### 7.1 终端：每 `logger_iter_interval`（默认 50）次迭代两行

```
[SDR] iter 500    | roi_ratio=0.0350 gate_eff=1.020 theta_mean=0.1100 logmag_mean=0.0500 weight=0.980 gate_entropy=0.830
                  | loss_rod=0.0210 loss_ford=0.0090 loss_rho=0.0040 loss_total=0.0340 to_det_ratio=0.020 eipbs_keep_ratio=0.250 eipbs_fg_keep_ratio=0.790 eipbs_beams=60.0 eipbs_ms=47.5
```

第一行就是文档 5.4 的六个必看量，顺序固定；第二行是三项损失分量、蒸馏/检测比值与 ③ 的运行时统计。

- 基线训练**不会**出现这两行（`tb_dict` 里没有 `cgrd_*` / `msdge_*` / `eipbs_*`，函数直接返回）；
- `--preset sdr_nokd`（只有 ①③）只打第二行里有值的部分，`gate_entropy` 依然在 —— 这是它存在的意义；
- tqdm 模式下另外把 `kd / rod / g_eff / H` 四个短键放进进度条，避免把进度条撑爆。

各项的健康区间、异常时改哪个配置，见 `scp-innovations/最终方案.html` 5.4 节。

### 7.2 落盘：三个 csv + tensorboard

训练一开始就在输出目录写，崩溃也不丢：

| 文件 | 粒度 | 内容 |
|---|---|---|
| `iter_log.csv` | 每迭代一行 | total loss + **全部 tb_dict 标量**（含 `cgrd_*` / `msdge_*` / `eipbs_*`） |
| `training_log.csv` | 每 epoch 一行 | train_loss / lr |
| `val_metrics.csv` | 每次验证一行 | AP / AP50-95 / P-R-F1（需 `--eval_interval > 0`） |
| `tensorboard/` | 每迭代 | 同上，`tensorboard --logdir <输出目录>` 直接看 |

### 7.3 训练结束自动出图

`tools/train.py` 在评测结束后自动调用 `tools/plot_curves.py`，输出三个文件：

```
<输出目录>/training_curves.png     loss / lr / AP / P-R-F1 / 损失分项      （所有模型都有）
<输出目录>/sdr_diagnostics.png     ① 门控 · ② 蒸馏 · ③ 采样 的 9 格诊断面板 （仅 SDR 组）
<输出目录>/sdr_summary.json        每个指标的首值/末值/均值/极值（可直接抄进论文表格）
```

`sdr_diagnostics.png` 的九格分别是：

1. ② 三项蒸馏损失（log 轴）；
2. ② λ 爬升曲线 + `gate_eff` + `gate_mean`（λ 应在 500 iter 内升到设定值；`gate_eff` 应稳在 1 附近）；
3. ① 门控熵 + 四个尺度的平均权重（熵恒为 1 = 门控没学到东西；权重应随训练分化）；
4. ② 方向项 θ/π 与径向项 |log r|（后者大一个量级就把 `RADIAL_WEIGHT` 调小或切 `cosine`）；
5. 掩码覆盖率 `roi_ratio` / `fore_ratio` 与 ③ 的整体/前景保留率；
6. ③ 存活线束数（≈16 = "16 线保真"口径；≈60 = 全局花预算口径）；
7. 蒸馏项 / 检测损失（> 1 说明蒸馏压过检测，应降 `LOSS_WEIGHT`）；
8. ③ 单帧采样耗时（dataloader 侧真实开销）；
9. 验证 AP（有 `--eval_interval` 时）。

不想自动出图：`--no_plot_after_train`。事后补图（例如训练被手动中断）：

```bash
python tools/plot_curves.py --exp_dir output/cfgs/kitti_models/sdr_voxel_rcnn_car/default
python tools/plot_curves.py --exp_dir <目录> --skip_head 500   # 汇总时跳过 ramp-up
```

### 7.4 训练结束抽 K 帧出检测效果图（帧号只随机一次，之后固定）

`tools/train.py` 评测完会再调 `tools/visualize_detections.py`，随机抽 `--vis_num`（默认 3）帧画图。
每帧一张**三联图**：

```
左   学生输入的点云（EIP-BS 之后的 25% 稀疏点，按高度上色）+ GT 框（红）
中   同一批点云 + 预测框（蓝，按 score 深浅）
右   放大叠加：GT（红）与 Pred（蓝）画在一起，并标出每个框的分数
```

外加 `detections_overview.png`（2×K 总览，上排 GT 下排 Pred）与 `detections_summary.json`
（抽到哪几帧、各帧点数/保留率/预测数/最高分）。

**帧号规则（这条最重要）**：

| 情况 | 行为 |
|---|---|
| 第一次跑 | 用 `--vis_seed` 随机抽 `--vis_num` 帧，写进 `<repo>/output/vis_frames.json` |
| 之后再训练任何一组 | 直接读清单，**不重新随机** —— baseline / sparse / sdr / sdr_mse 出的是同一批场景 |
| 想换一批 | `--vis_frames 192 238 318` 显式指定（会覆盖清单），或删掉清单文件 |
| 清单帧号超出当前数据集 | 自动重抽并打印提示（例如从完整 KITTI 切到 500 帧子集时） |

为什么必须固定：不固定的话各组看到的是不同帧，"改进模型框得更准"就没法并排比较。

几个细节：

- 图里画的是 **test_set**（`training=False`）的点云，所以学生输入也是稀疏的，和评测/部署同口径；
- 只用 matplotlib（Agg 后端）落盘，**不需要 open3d、不需要显示器**，服务器上直接出 PNG；
- 关闭：`--no_vis_after_train`；换输出目录/清单：`--vis_frames_file <path>`。

事后单独出图（例如想看某个 epoch 的框）：

```bash
python tools/visualize_detections.py --preset sdr \
    --ckpt output/cfgs/kitti_models/sdr_voxel_rcnn_car/default/ckpt/checkpoint_epoch_80.pth \
    --num_samples 3 --seed 0

# 指定帧 / 重抽一批
python tools/visualize_detections.py --preset sdr --ckpt <ckpt> --frames 12 340 512
python tools/visualize_detections.py --preset sdr --ckpt <ckpt> --reset_frames --num_samples 4
```

### 7.5 一条命令核对"代码是否真的按文档改了"

```bash
python tools/check_sdr_conformance.py        # 64 项：63 PASS / 0 FAIL / 1 项已知 WARN
python tools/check_sdr_conformance.py -v     # 附带每项的说明
```

逐条对应 `最终方案.html` 的 16 处修订与 5.3 开关表：尺度、零初始化、真实点数、
分母 Σ(M_roi·w)、asin clamp、`TARGET_KEEP_RATIO`、`STRATIFIED`、`BEAM_SCORE`、
6 个预设、测试期不用 GT、6 个监控量是否进日志、训练后是否出图与检测效果图、
BEV 通道链 128×2=256、师生 2D 骨干是否一致、三组增强是否只有"输入"这一个自变量……
不通过会直接打印"期望 / 实际"。它只读 yaml 和源码，**不需要 torch**，所以在没有 GPU
的机器上也能跑。

---

## 8. 文件清单

新增：

```
pcdet/models/backbones_3d/vfe/msdge_vfe.py     创新点① MS-DGE（VFE，注册名 MSDGE）
pcdet/utils/eipbs_utils.py                     创新点③ EIP-BS
pcdet/models/kd/__init__.py
pcdet/models/kd/cgrd_loss.py                    创新点② CG-RD
pcdet/models/kd/teacher_branch.py               教师分支 BEVTeacher
pcdet/models/detectors/sdr_voxel_rcnn.py        SDRVoxelRCNN（注册名 SDRVoxelRCNN）
tools/cfgs/dataset_configs/kitti_dataset_sdr.yaml
tools/cfgs/dataset_configs/kitti_dataset_sparse.yaml   纯稀疏基线（均匀抽线到 25%）
tools/cfgs/kitti_models/sdr_voxel_rcnn_car.yaml
tools/cfgs/kitti_models/voxel_rcnn_car_sparse.yaml     纯稀疏基线（基线模型 + 稀疏输入）
tools/check_sdr_setup.py                        链路自检（5 步，出问题会直接提示下一步）
tools/check_sdr_conformance.py                  文档 ↔ 代码 一致性检查（64 项，不需 torch）
tools/check_teacher_ckpt.py                     教师 ckpt 体检（开跑蒸馏前先跑一次）
tools/bench_eipbs.py                            采样策略实测（纯 numpy，不需要 torch）
tools/plot_curves.py                            训练曲线 + SDR 诊断图 + json 汇总
tools/visualize_detections.py                   随机抽 K 帧出检测效果图（帧号首次固定，不需 open3d）
docs/SDR-理论-实现对照与优化.md                  理论问题 → 建议 → 代码落点（含实测表）
docs/SDR-文档一致性核对.md                       7 处实现缺口修复 + W_CLASS/Car-only 决策 + 教师验收
docs/SDR-训练与验证命令.md                        训练/评测/出图的完整命令手册（含 8 条常见坑）
```

改动（均为追加式，不改原有行为）：

```
pcdet/models/backbones_3d/vfe/__init__.py       注册 MSDGE
pcdet/models/detectors/__init__.py              注册 SDRVoxelRCNN
pcdet/models/detectors/detector3d_template.py   兼容官方"裸 state_dict"权重（无 model_state 包装）
pcdet/datasets/processor/data_processor.py      新增 eip_bs_voxelize（含 WRITE_TO_MAIN / LOG_STATS）
pcdet/datasets/dataset.py                       collate_batch 增加 5 个键名（含 voxel_num_points_true）
tools/train.py                                  新增 --preset（6 个预设）+ 训练后自动出图/出检测图
tools/test.py                                   新增 --preset（与 train.py 同一套覆写）
tools/train_utils/train_utils.py                新增 [SDR] 周期日志行；进度条在两种记录模式下都前进
```

文档与结构图（在代码的同级目录 `../scp-innovations/`，不属于训练链路，删掉不影响训练）：

```
../scp-innovations/最终方案.html                    标准说明文档（v2.2，含 0.1/0.2/0.3 三节结构图）
../scp-innovations/assets/fig-1-overview.svg      图 0-1 整体双通路：教师全冻结 / 学生 EIP-BS+MS-DGE / 中间三张创新点卡
../scp-innovations/assets/fig-2-cgrd.svg          图 0-2 CG-RD 算子级：1×1 适配器 → 逐格几何量 → 掩码×门控 → 三项损失
../scp-innovations/assets/fig-3-input.svg         图 0-3 输入侧：EIP-BS 五步采样 + MS-DGE 四尺度门控
../scp-innovations/_shared/make_arch_figure.py    三张图的唯一生成源（改图只改这里，再 --write 回写 HTML）
```

重画/微调结构图：

```bash
python ../scp-innovations/_shared/make_arch_figure.py           # 只预览：打印布局警告 + 重写 assets/*.svg
python ../scp-innovations/_shared/make_arch_figure.py --write   # 同时回写 最终方案.html（幂等，连跑两次结果一致）
```

脚本自带字符宽度估算，框内文字超宽会打印 `[WARN] 文本溢出` 并自动降字号；正常情况下收尾输出 `[OK] 没有文本溢出提醒`。

---

## 9. 已知边界

- **必须先把算子编译好**（`python setup.py develop`）并装好 `spconv`，`SDRVoxelRCNN`
  依赖 `VoxelBackBone8x` 的 `spconv.SubMConv3d`；
- 教师分支使显存接近翻倍，`BATCH_SIZE_PER_GPU` 建议先试 1；
- 教师是 `FREEZE: True` 时，前向内部会强制 `teacher.eval()`（BN 用 running stats），
  即使外层 `model.train()` 也不会污染教师 BN；
- `TEACHER.FREEZE: False` 时教师会参与训练（协同训练模式），此时 `get_master` 不会
  过滤教师参数，学习率与主网络一致，一般不建议。
- 教师是 `self.teacher` 注册在模型里的子模块，所以 SDR 的 ckpt **会包含教师权重**，
  文件体积约为基线的 2 倍（`teacher.*` 前缀）。这是刻意的：评测/续训时不必再带一份
  基线 ckpt。基线的 ckpt 不受任何影响。
- 教师分支只在 `self.training and MODEL.KD.ENABLE` 时前向，**评测/部署时完全不跑**，
  推理算力与学生分支一致。
